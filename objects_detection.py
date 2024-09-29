import os
from collections import defaultdict
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import json
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, image_folder, processor, img_size):
        self.image_folder = image_folder
        self.processor = processor
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Resize(img_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "img_tensor": inputs.pixel_values.squeeze(),
            "img_name": img_name,
        }

class ObjectDetector:
    def __init__(self, pixel_per_side=256):
        # you can specify the revision tag if you don't want the timm dependency
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = (pixel_per_side, pixel_per_side)

    def detect(self, batch):
        bsz = batch["img_tensor"].shape[0]
        img_tensor = batch["img_tensor"].to(self.model.device)
        outputs = self.model(pixel_values=img_tensor)
        target_sizes = torch.tensor([self.img_size] * bsz).to(self.model.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)

        all_bbox_list = []
        for result in results:
            bbox_list = []
            for box in result["boxes"]:
                box = [int(i) for i in box.tolist()]
                bbox_list.append(','.join(map(str, box)))
            all_bbox_list.append(bbox_list)

        return all_bbox_list

def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, dict):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, torch.Tensor):
        try:
            return default_collate(batch)
        except RuntimeError:
            # If tensors are of different sizes, stack them instead
            return torch.stack(batch, dim=0)
    else:
        return default_collate(batch)

if __name__ == "__main__":
    # Usage
    image_folder = "/home/stevexu/data/imagenet_val"
    output_folder = "/home/stevexu/data/processed_imagenet/val"
    draw_img = False
    batch_size = 64
    batch_num_limit = None
    pixel_per_side = 256
    detector = ObjectDetector(pixel_per_side)

    os.makedirs(output_folder, exist_ok=True)

    dataset = ImageDataset(image_folder, detector.processor, detector.img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)

    all_bbox_dict = {}

    for i, batch in tqdm(enumerate(dataloader), desc="Processing images"):
        if batch_num_limit and i >= batch_num_limit:
            break
        bbox_lists = detector.detect(batch)

        for img_name, bbox_list in zip(batch["img_name"], bbox_lists):
            all_bbox_dict[img_name] = bbox_list
    # Save all bounding box data to a JSON file
    json_filename = "all_bboxes.json"
    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(all_bbox_dict, json_file, indent=4)
    print(f"Saved all bounding box data to {json_path}")