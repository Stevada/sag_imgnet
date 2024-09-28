import os
import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import json
import shutil
import random
from image_classification import ImgClassifier
from scipy.spatial.distance import cosine

# Constants
SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
IMAGENET_DIR = "/home/stevexu/data/imagenet_val"
OUTPUT_DIR = "outputs_sort_by_area"

class ObjectExtractor:
    def __init__(self, keep_top_k: int=3):
        self.classifier = ImgClassifier()
        self.keep_top_k = keep_top_k
    
    def extract_objects(self, image, masks, mode: str="classifier", **kwargs):
        assert mode in ["classifier", "mask_area"], "mode must be 'classifier' or 'mask_area'"
        if mode == "classifier":
            return self.__extract_objects_with_classifier(image, masks, **kwargs)
        else:
            return self.__extract_objects_with_keys(image, masks, **kwargs)

    def __extract_objects_with_classifier(self, image, masks, **kwargs):
        """
        Extract objects with classifier.
        """
        objects = []
        similarities = []
        bbox = kwargs.get("bbox", None)

        # Classify the original image
        orig_img_width, orig_img_height = image.shape[:2]
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            assert x1 < x2 and y1 < y2, "bbox is invalid"
            image = image[y1:y2, x1:x2]
            for mask in masks:
                mask['segmentation'] = mask['segmentation'][y1:y2, x1:x2]
        
        original_img = Image.fromarray(image)
        original_logits = self.classifier.classify(original_img, return_logits=True)
        original_predicted_class_idx = original_logits.argmax(-1).item()
        
        for i, mask in enumerate(masks):
            m = mask['segmentation']
            obj = image.copy()
            msk = image.copy()
            obj[~m] = 0  # Set background to black
            msk[m] = 0  # Set mask to black

            # Classify the masked image
            selected_img = Image.fromarray(obj)
            selected_logits = self.classifier.classify(selected_img, return_logits=True)
            selected_softmax = torch.nn.functional.softmax(selected_logits.unsqueeze(0), dim=1)

            # Compute similarity
            # similarity = torch.log(selected_softmax[0, original_predicted_class_idx]).item()
            similarity = torch.nn.functional.cosine_similarity(original_logits.unsqueeze(0), selected_logits.unsqueeze(0), dim=1).item()
            
            objects.append((obj, msk))
            similarities.append(similarity)

        # Return the object with the highest similarity
        best_object_indices = np.argsort(similarities)[-self.keep_top_k:][::-1]
        print(f"highest similarity: {[similarities[idx] for idx in best_object_indices]}")
        return [objects[idx] for idx in best_object_indices]


    def __extract_objects_with_keys(self, image, masks, **kwargs):
        """
        Extract objects with keys.
        """
        objects = []
        # Sort masks by area in descending order
        key = kwargs.get("key", "area")
        assert key in masks[0], f"{key} is not in masks"
        masks = sorted(masks, key=lambda x: x[key], reverse=True)

        for i, mask in enumerate(masks):
            if self.keep_top_k and i >= self.keep_top_k:
                break

            m = mask['segmentation']
            obj = image.copy()
            msk = image.copy()
            obj[~m] = 0  # Set background to white
            msk[m] = 0  # Set mask to black
            
            objects.append((obj, msk))

        return objects
    
    
def save_object(obj, output_path):
    Image.fromarray(obj).save(output_path)
    

if __name__ == "__main__":
    data_dir = "/home/stevexu/VSprojects/sag_imgnet/objects_detection_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    extractor = ObjectExtractor(keep_top_k=1)
    extract_mode = "mask_area"
    
    file_names = [name for name in os.listdir(data_dir) if name.endswith(".JPEG")]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for img_name in file_names:
        src_img_name = img_name.replace("bbox_", "")
        # for debugging
        # if src_img_name != "ILSVRC2012_val_00002499_n03085013.JPEG":
        #     continue
        img_path = os.path.join(IMAGENET_DIR, src_img_name)
        bbox_path = os.path.join(data_dir, img_name.replace(".JPEG", ".json"))
        # load image
        with Image.open(img_path) as img:
            img = np.array(img.convert("RGB"))
            
        # load bbox
        with open(bbox_path, "r") as f:
            bbox_list = json.load(f)

        # copy image to output dir
        # shutil.copy(img_path, os.path.join(OUTPUT_DIR, os.path.basename(img_path)))

        # generate mask for the whole image without bbox
        if len(bbox_list) == 0:
            mask_generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=32,
                points_per_batch=256,
            )
            masks = mask_generator.generate(img)
            masks = mask_generator.generate(img)
            print(f"generate {len(masks)} masks")
            # objects = extract_objects_with_classifier(classifier, img, masks, keep_top_k=1)
            objects = extractor.extract_objects(img, masks, mode=extract_mode)
            
            # save extracted objects
            for j, (obj, msk) in enumerate(objects):
                obj_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_obj_{j}_obj.png")
                msk_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_obj_{j}_mask.png")
                save_object(obj, obj_path)
                save_object(msk, msk_path)
        else:
            # generate mask for each bbox by sampling points in the bbox
            for i, bbox in enumerate(bbox_list):
                # Normalize the bounding box coordinates
                img_height, img_width = img.shape[:2]
                x1, y1, x2, y2 = bbox
                normalized_bbox = [
                    x1 / img_width,
                    y1 / img_height,
                    x2 / img_width,
                    y2 / img_height
                ]
                
                # Sample random points in the normalized bbox
                bbox_width = normalized_bbox[2] - normalized_bbox[0]
                bbox_height = normalized_bbox[3] - normalized_bbox[1]
                points_num = 256
                x = np.random.uniform(normalized_bbox[0], normalized_bbox[2], points_num)
                y = np.random.uniform(normalized_bbox[1], normalized_bbox[3], points_num)
                sampled_points = np.column_stack((x, y)).tolist()
                normalized_bbox_points = np.array(sampled_points)

                # generate mask for the bbox
                mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=None,
                    points_per_batch=256,
                    point_grids=np.array([sampled_points]),
                )
                masks = mask_generator.generate(img)
                print(f"generate {len(masks)} masks")
                # objects = extractor.extract_objects(img, masks, mode="classifier", bbox=bbox)
                objects = extractor.extract_objects(img, masks, mode=extract_mode)

                # Save extracted objects
                for j, (obj, msk) in enumerate(objects):
                    obj_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_bbox_{i}_obj_{j}_obj.png")
                    msk_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_bbox_{i}_obj_{j}_mask.png")
                    save_object(obj, obj_path)
                    save_object(msk, msk_path)
