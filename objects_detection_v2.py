import json
from PIL import ImageDraw
from PIL import Image
import os

import torch
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

prompt = "<OD>"

image_folder = "/home/stevexu/data/imagenet_val"
output_folder = "data/objects_detection_v2"
os.makedirs(output_folder, exist_ok=True)
image_num_limit = 10   
all_bbox_dict = {}
for i, image_name in enumerate(os.listdir(image_folder)):
    if i >= image_num_limit:
        break
    image = Image.open(os.path.join(image_folder, image_name))
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

    # save bbox
    bbox_list = []
    for box in result[prompt]["bboxes"]:
        bbox_list.append(','.join(map(str, box)))
    all_bbox_dict[image_name] = bbox_list

    # draw bbox
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(result[prompt]["bboxes"]):
        draw.rectangle(box, outline="red")
        label = result[prompt]["labels"][i]
        draw.text((box[0], box[1]), label, fill="red")
    image.save(os.path.join(output_folder, image_name))

# Save all bounding box data to a JSON file
json_filename = "all_bboxes.json"
json_path = os.path.join(output_folder, json_filename)
with open(json_path, 'w') as json_file:
    json.dump(all_bbox_dict, json_file, indent=4)
print(f"Saved all bounding box data to {json_path}")