import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import shutil
import cv2
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import ImageDraw
from utils import make_dir

class ImageProcessor:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        self.prompt = "<OD>"

        self.augmentations = T.Compose([
            T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.0), shear=5),
            T.RandomPerspective(distortion_scale=0.1, p=0.5),
            # T.CenterCrop(int(min(image.size) * 0.9)),  # Ensure entire object is visible
            # T.Resize(image.size),  # Resize back to original size
        ])

    def center_main_object(self, image, bbox):
        if bbox is None:
            return image
        
        x, y, w, h = bbox
        center_x, center_y = image.size[0] // 2, image.size[1] // 2
        
        shift_x = center_x - (x + w // 2)
        shift_y = center_y - (y + h // 2)
        
        return F.affine(image, angle=0, translate=(shift_x, shift_y), scale=1, shear=0)

    def post_process(self, augmented_image):
        inputs = self.processor(text=self.prompt, images=augmented_image, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(generated_text, task="<OD>", image_size=(augmented_image.width, augmented_image.height))
        
        # Choose the box with the biggest area
        bboxes = result[self.prompt]["bboxes"]
        biggest_bbox = max(bboxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

        # Create a mask for the biggest bounding box
        mask = Image.new('L', augmented_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(biggest_bbox, fill=255)

        # Create a white background image
        white_background = Image.new('RGB', augmented_image.size, (255, 255, 255))

        # Paste the object onto the white background using the mask
        white_background.paste(augmented_image, (0, 0), mask)

        return white_background


    def process_images(self, img, bbox, num_augmentations=5):
        centered_image = self.center_main_object(img, bbox)
        
        augmented_images = []
        for _ in range(num_augmentations):
            augmented_image = self.augmentations(centered_image)
            augmented_image = self.post_process(augmented_image)
            augmented_images.append(augmented_image)
        
        return augmented_images

# Usage example
if __name__ == "__main__":
    data_dir = "/home/stevexu/data/imagenet_val"
    output_dir = "/home/stevexu/data/processed_imagenet/val"
    foreground_dir = os.path.join(output_dir, "foreground")
    bbox_path = "/home/stevexu/data/processed_imagenet/val/all_bboxes.json"

    garment_dir = os.path.join(output_dir, "garment")
    make_dir(garment_dir)

    processor = ImageProcessor()
    img_num_limit = None

    agmt_times = 3
    with open(os.path.join(output_dir, 'all_bboxes.json'), 'r') as f:
        all_bboxes = json.load(f)
    
    # Process and save the images
    for i, img_name in enumerate(os.listdir(data_dir)):
        if img_num_limit and i > img_num_limit:
            break
        if img_name not in all_bboxes:
            continue
        file_name = img_name.split('.')[0]
        bbox_list = all_bboxes[img_name]
        for i, bbox in enumerate(bbox_list):
            bbox = list(map(float, bbox.split(',')))
            fgrd_img_name = f"{file_name}_mask_{i}_foreground.png"
            fgrd_img = Image.open(os.path.join(foreground_dir, fgrd_img_name)).convert('RGB')
            augmented_img_list = processor.process_images(fgrd_img, bbox, agmt_times)
            for j, agmt_img in enumerate(augmented_img_list):
                agmt_img_name = f"{file_name}_mask_{i}_agmt_{j}.png"
                agmt_img.save(os.path.join(garment_dir, agmt_img_name))
            print(f"Processed and saved: {len(augmented_img_list)} augmented images")