import os
import torch
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import shutil
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import make_dir
load_dotenv()

# Config sam2
CHECKPOINT = "/home/stevexu/VSprojects/sag_imgnet/checkpoints/sam2_hiera_large.pt"
CONFIG = "configs/sam2/sam2_hiera_l.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam2 = build_sam2(CONFIG, CHECKPOINT, device=device, apply_postprocessing=False)
sam2.to(device=device)
predictor = SAM2ImagePredictor(sam2)

def load_image_batch(image_paths, batch_size):
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        images.append(img_np)
        if len(images) == batch_size:
            yield images
            images = []
    if images:
        yield images

def process_batch(predictor, image_batch, bboxes_batch):
    predictor.set_image_batch(image_batch)
    # Check and adjust bboxes for each image in the batch
    adjusted_bboxes_batch = []
    for img, bboxes in zip(image_batch, bboxes_batch):
        if len(bboxes) == 0:
            # If bboxes is empty, create a bbox covering the entire image
            h, w = img.shape[:2]
            bboxes = np.array([[0, 0, w, h]])
        else:
            # Ensure bboxes is a numpy array
            bboxes = np.array(bboxes)
        adjusted_bboxes_batch.append(bboxes)
    
    # Replace the original bboxes_batch with the adjusted one
    bboxes_batch = adjusted_bboxes_batch
    # TODO: why GPU usage increases with batch size when predict_batch looks a sequential process?
    all_masks, all_ious, _ = predictor.predict_batch(
        box_batch=bboxes_batch,
        multimask_output=False,
        return_logits=False
    )
    return all_masks

def main():
    data_dir = "/home/stevexu/data/generated_cup/trained-output/images"
    output_dir = "/home/stevexu/data/generated_cup/trained-output/val"
    model_dir = os.path.join(output_dir, "model")
    make_dir(model_dir)
    inpaint_dir = os.path.join(output_dir, "inpaint")
    make_dir(inpaint_dir)
    foreground_dir = os.path.join(output_dir, "foreground")
    make_dir(foreground_dir)

    # Load bounding boxes
    with open(os.path.join(output_dir, 'all_bboxes.json'), 'r') as f:
        all_bboxes = json.load(f)

    # Get all image files
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.JPEG', '.jpg', '.png'))]
    
    batch_size = 16  # Adjust based on your GPU memory
    batch_num_limit = None
    
    for i in tqdm(range(0, len(image_files), batch_size)):
        if batch_num_limit and i >= batch_num_limit:
            break
        batch_files = image_files[i:i+batch_size]
        batch_files = [f for f in batch_files if f in all_bboxes]
        if not batch_files:
            # not bboxes for this batch
            continue
        image_paths = [os.path.join(data_dir, f) for f in batch_files]
        
        print(f"Processing batch {i//batch_size + 1}")
        print(f"Batch files: {batch_files}")
        
        image_batch = list(load_image_batch(image_paths, batch_size))[0]
        bboxes_batch = [np.array([list(map(float, box['bbox'].split(','))) for box in all_bboxes[f]]) for f in batch_files]
        
        masks_batch = process_batch(predictor, image_batch, bboxes_batch)
        
        for j, (file, masks) in enumerate(zip(batch_files, masks_batch)):
            file_name = os.path.splitext(file)[0]
            for k, mask in enumerate(masks):
                foreground_name = f"{file_name}_mask_{k}_foreground.png"
                background_name = f"{file_name}_mask_{k}_background.png"
                
                # Ensure mask is boolean
                mask = torch.from_numpy(mask).bool()
                
                # Read the original image
                original_img = Image.open(image_paths[j]).convert("RGB")
                original_tensor = torch.from_numpy(np.array(original_img)).permute(2, 0, 1).float()
                
                # Apply mask to original image for foreground and background
                foreground_tensor = torch.where(mask.repeat(3, 1, 1), 
                                                original_tensor, 
                                                torch.full_like(original_tensor, 255))
                background_tensor = torch.where(~mask.repeat(3, 1, 1), 
                                                original_tensor, 
                                                torch.full_like(original_tensor, 128))
                
                # Convert tensors back to PIL images
                foreground_img = Image.fromarray(foreground_tensor.permute(1, 2, 0).byte().numpy())
                background_img = Image.fromarray(background_tensor.permute(1, 2, 0).byte().numpy())
                
                # Save as images
                foreground_img.save(os.path.join(foreground_dir, foreground_name))
                background_img.save(os.path.join(inpaint_dir, background_name))
                shutil.copy(image_paths[j], os.path.join(model_dir, file))

if __name__ == "__main__":
    main()