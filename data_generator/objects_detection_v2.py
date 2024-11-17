import json
import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
prompt = "<OD>"

class ImageNetDataset(Dataset):
    def __init__(self, image_folder, processor, prompt):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.processor = processor
        self.prompt = prompt

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(text=self.prompt, images=image, return_tensors="pt")
        return {
            'image_name': image_name,
            'image_size': torch.tensor([image.width, image.height]),
            'input_ids': inputs['input_ids'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze()
        }

def process_batch(batch, model, processor, all_bbox_dict):
    input_ids = batch['input_ids'].to(device)
    pixel_values = batch['pixel_values'].to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)

    for i, text in enumerate(generated_text):
        image_name = batch['image_name'][i]
        image_size = batch['image_size'][i]
        result = processor.post_process_generation(text, task="<OD>", image_size=(image_size[0], image_size[1]))

        # Calculate the area of the image
        image_area = image_size[0] * image_size[1]
        
        # Filter bboxes that are larger than 10% of the image area
        filtered_bbox_list = []
        for bbox in result[prompt]["bboxes"]:
            x1, y1, x2, y2 = bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area > 0.1 * image_area:
                filtered_bbox_list.append(','.join(map(str, bbox)))
        
        # Only add to all_bbox_dict if there are valid bboxes
        if filtered_bbox_list:
            all_bbox_dict[image_name] = filtered_bbox_list

def main():
    image_folder = "/workspace/imagenet_val"
    output_folder = "/home/stevexu/data/processed_imagenet/val"
    os.makedirs(output_folder, exist_ok=True)
    batch_size = 4 
    batch_num_limit = 5

    dataset = ImageNetDataset(image_folder, processor, prompt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_bbox_dict = {}
    
    for i, batch in enumerate(tqdm(dataloader, desc="Processing images")):
        if batch_num_limit and i >= batch_num_limit:
            break
        process_batch(batch, model, processor, all_bbox_dict)

    # Save all bounding box data to a JSON file
    json_filename = "all_bboxes.json"
    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(all_bbox_dict, json_file, indent=4)
    print(f"Generated {sum([len(bbox) for bbox in all_bbox_dict.values()])} bboxes")
    print(f"Saved all bounding box data to {json_path}")

if __name__ == "__main__":
    main()