import json
import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
from torchvision import transforms, models
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50').to(device)
model.eval()
classifier = models.resnet50(pretrained=True).to(device)
classifier.eval()
classifier_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
class ImageNetDataset(Dataset):
    def __init__(self, image_folder, prompt_folder, processor):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.processor = processor
        
    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            'image_name': image_name,
            'image_path': image_path,
            'pixel_values': inputs['pixel_values'].squeeze(),
        }
    
    def __len__(self):
        return len(self.image_files)

def get_bbox_confidence(image, bbox, classifier, transform):
    # Extract bbox coordinates and convert to integers
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop the bbox region from the image
    bbox_image = image.crop((x1, y1, x2, y2))
    
    # Preprocess the cropped image for the classifier
    bbox_tensor = transform(bbox_image).unsqueeze(0).to(device)
    
    # Get classifier prediction
    with torch.no_grad():
        output = classifier(bbox_tensor)
        confidence = F.softmax(output, dim=1)
        
    # Get confidence score for cup class (assume class_idx is 968 for cup)
    # You might need to adjust this index based on your classifier's classes
    cup_confidence = confidence[0][968].item()  # 968 is the ImageNet index for 'cup'
    return cup_confidence

def process_batch(batch, model, processor, classifier, all_bbox_dict):
    pixel_values = batch['pixel_values'].to(device)
    
    # Get the original PIL images for cropping
    original_images = [Image.open(img_path).convert('RGB') 
                      for img_path in batch['image_path']]

    with torch.no_grad():
        outputs = model(pixel_values)

    # Process DETR outputs
    for i, image in enumerate(original_images):
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9  # Confidence threshold
        
        bboxes = outputs.pred_boxes[0, keep].cpu()
        filtered_bbox_list = []
        
        # Convert relative coordinates to absolute
        h, w = image.size
        for bbox in bboxes:
            x_c, y_c, width, height = bbox.tolist()
            x1 = int((x_c - width/2) * w)
            y1 = int((y_c - height/2) * h)
            x2 = int((x_c + width/2) * w)
            y2 = int((y_c + height/2) * h)
            
            # Get classifier confidence
            confidence = get_bbox_confidence(image, (x1, y1, x2, y2), classifier, classifier_transform)
            
            # Only keep bbox if confidence exceeds threshold
            if confidence > 0.5:  # Adjust threshold as needed
                filtered_bbox_list.append(','.join(map(str, [x1, y1, x2, y2])))
        
        # Only add to all_bbox_dict if there are valid bboxes
        if filtered_bbox_list:
            all_bbox_dict[batch['image_name'][i]] = filtered_bbox_list

def main():
    image_folder = "/home/stevexu/data/generated_cup/trained-output/split_right_images"
    prompt_folder = "/home/stevexu/data/generated_cup/trained-output/prompts"
    output_folder = "/home/stevexu/data/generated_cup/trained-output/val"
    os.makedirs(output_folder, exist_ok=True)
    batch_size = 8
    batch_num_limit = None

    dataset = ImageNetDataset(image_folder, prompt_folder, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_bbox_dict = {}
    
    for i, batch in enumerate(tqdm(dataloader, desc="Processing images")):
        if batch_num_limit and i >= batch_num_limit:
            break
        process_batch(batch, model, processor, classifier, all_bbox_dict)

    # Save all bounding box data to a JSON file
    json_filename = "all_bboxes.json"
    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, 'w') as json_file:
        json.dump(all_bbox_dict, json_file, indent=4)
    print(f"Generated {sum([len(bbox) for bbox in all_bbox_dict.values()])} bboxes")
    print(f"Saved all bounding box data to {json_path}")

if __name__ == "__main__":
    main()