import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import json

class ObjectDetector:
    def __init__(self):
        # you can specify the revision tag if you don't want the timm dependency
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def detect(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        bbox_list = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )
            bbox_list.append(box)
        return bbox_list

if __name__ == "__main__":
    # Usage
    img_num_limit = 10
    detector = ObjectDetector()
    image_folder = "/home/stevexu/data/imagenet_val"
    output_folder = "objects_detection_output"
    os.makedirs(output_folder, exist_ok=True)
    # image_folder = "/home/stevexu/data/viton_for_mage_10/model"
    for i, img_name in enumerate(os.listdir(image_folder)):
        if i >= img_num_limit:
            break
        # Display predictions
        img = Image.open(os.path.join(image_folder, img_name))
        bbox_list = detector.detect(img)

        draw = ImageDraw.Draw(img)
        for box in bbox_list:
            draw.rectangle(box, outline="red", width=3)
        
        # Save the image with bounding boxes
        output_path = os.path.join(output_folder, f"bbox_{img_name}")
        img.save(output_path)
        print(f"Saved image with bounding boxes to {output_path}")
        
        # Save the bounding box list to a JSON file
        os.makedirs(output_folder, exist_ok=True)
        json_filename = f"bbox_{os.path.splitext(img_name)[0]}.json"
        json_path = os.path.join(output_folder, json_filename)
        with open(json_path, 'w') as json_file:
            json.dump(bbox_list, json_file)
        print(f"Saved bounding box data to {json_path}")