import numpy as np
from PIL import Image
import os
from transformers import ViTImageProcessor, ViTForImageClassification
import torch


class ImgClassifier:

    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model.eval()
    
    def classify(self, img, return_logits=False):
        inputs = self.processor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # model predicts one of the 1000 ImageNet classes
        # predicted_class_idx = logits.argmax(-1).item()
        # print("Predicted class:", self.model.config.id2label[predicted_class_idx])
        
        if return_logits:
            return logits.detach().squeeze(0)
        else:
            return None

class ObjectDetector:
    def __init__(self):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    def detect(self, img):
        # Convert PIL Image to tensor
        img_tensor = ToTensor()(img).unsqueeze(0)
        
        # Perform inference
        results = self.model(img_tensor)
        
        # Process results
        detections = results.xyxy[0].numpy()
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            print(f"Detected {self.model.names[int(cls)]} with confidence {conf:.2f} at location [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        
        return detections
    
if __name__ == "__main__":
    # Usage
    img_num_limit = 10
    classifier = ImgClassifier()
    # image_folder = "/home/stevexu/data/imagenet_val"
    image_folder = "/home/stevexu/data/viton_for_mage_10/model"
    for i, img_name in enumerate(os.listdir(image_folder)):
        if i >= img_num_limit:
            break
        # Display predictions
        img = Image.open(os.path.join(image_folder, img_name))
        classifier.classify(img)
        