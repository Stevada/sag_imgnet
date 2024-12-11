import json
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
from torchvision import transforms, models
import torch.nn.functional as F
import shutil

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# ImageNet classifier for filtering bboxes
classifier = models.resnet50(pretrained=True).to(device)
classifier.eval()
classifier_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
class ImageNetDataset(Dataset):
    def __init__(self, image_folder, prompt_folder, processor):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.prompt_files = [f for f in os.listdir(prompt_folder) if f.lower().endswith(('.txt'))]
        self.processor = processor
        self.prompt_folder = prompt_folder
        self.target_object_list = ['car', 'cherry', "tree", 'fish', 'dog', 'pineapple', 'strawberry', 'watermelon', 'flower']
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        prompt_name = self.prompt_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        # TODO: Resize image to 512x512. Don't forget to reshape the mask.
        # image = self.transform(image)
        
        # TODO: handel the prompt
        with open(os.path.join(self.prompt_folder, prompt_name), 'r') as file:
            prompt = file.read().strip()
        # prompt_changed = False
        # for target_object in self.target_object_list:
        #     if target_object in prompt:
        #         prompt = target_object
        #         prompt_changed = True
        #         break
        # if not prompt_changed:
        #     raise ValueError(f"Target object not found in prompt: {prompt}")

        # prompt = "locate the objects in the image"
        # prompt = "locate the cup in the image"
        # prompt = "cup"
        # prompt = "<OD>"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return {
            'image_name': image_name,
            'image_path': image_path,
            'image_size': torch.tensor([image.width, image.height]),
            'input_ids': inputs['input_ids'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'prompt': prompt
        }

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

def process_batch(batch, model, processor, all_bbox_dict):
    input_ids = batch['input_ids'].to(device)
    pixel_values = batch['pixel_values'].to(device, torch_dtype)
    # Get the original PIL images for cropping
    original_images = [Image.open(img_path).convert('RGB') 
                      for img_path in batch['image_path']]
    
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
        prompt = batch['prompt'][i]
        task="<OD>"
        result = processor.post_process_generation(text, task=task, image_size=(image_size[0], image_size[1]))

        # Calculate the area of the image
        image_area = image_size[0] * image_size[1]
        
        filtered_bbox_list = []
        prev_bbox = None
        for bbox, label in zip(result[task]["bboxes"], result[task]["labels"]):
            x1, y1, x2, y2 = bbox
            
            # Filter bboxes by image area
            # bbox_area = (x2 - x1) * (y2 - y1)
            # if bbox_area < 0.3 * image_area:
            #     continue

            # Filter bboxes by classifier confidence
            # image = original_images[i]
            # confidence = get_bbox_confidence(image, (x1, y1, x2, y2), classifier, classifier_transform)
            # if confidence < 0.5:  # Adjust threshold as needed
            #     continue
            
            # Filter bboxes by label
            if "cup" not in label:
                continue

            if prev_bbox:
                # Convert bboxes to tensors and compute cosine similarity
                bbox_tensor = torch.tensor(bbox)
                prev_bbox_tensor = torch.tensor(prev_bbox)
                cos_sim = F.cosine_similarity(bbox_tensor.float().unsqueeze(0), prev_bbox_tensor.float().unsqueeze(0))
                if cos_sim > 0.95:  # Threshold for considering bboxes too similar
                    continue
            
            filtered_bbox_list.append({
                'bbox': ','.join(map(str, bbox)),
                'label': label,
                'prompt': prompt
            })
            prev_bbox = bbox
            
        # Only add to all_bbox_dict if there are valid bboxes
        if filtered_bbox_list:
            all_bbox_dict[image_name] = filtered_bbox_list

def draw_bboxes_and_save(image_folder, bbox_dict, output_folder):
    """
    Draw bounding boxes on images and save them to output folder.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Try to load a font, fallback to default if not found
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    for image_name, records in bbox_dict.items():
        # Load image
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Draw each bbox
        bboxes = [record['bbox'] for record in records]
        labels = [record['label'] for record in records]
        for i, bbox_str in enumerate(bboxes):
            x1, y1, x2, y2 = map(float, bbox_str.split(','))
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Draw label
            label = labels[i]
            text_bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(text_bbox, fill='red')
            draw.text((x1, y1-25), label, fill='white', font=font)
        
        # Save annotated image
        output_path = os.path.join(output_folder, f"annotated_{image_name}")
        image.save(output_path)

def main():
    image_folder = "/home/stevexu/data/generated_cup/trained-output/images"
    prompt_folder = "/home/stevexu/data/generated_cup/trained-output/prompts"
    output_folder = "/home/stevexu/data/generated_cup/trained-output/val"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    batch_size = 1
    batch_num_limit = None

    dataset = ImageNetDataset(image_folder, prompt_folder, processor)
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

    # Draw and save annotated images
    visualization_folder = os.path.join(output_folder, "visualizations")
    draw_bboxes_and_save(image_folder, all_bbox_dict, visualization_folder)
    print(f"Saved annotated images to {visualization_folder}")

if __name__ == "__main__":
    main()