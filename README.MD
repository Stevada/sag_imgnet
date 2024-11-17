# Steps for generating data
1. run `objects_detection_v2.py` to get bboxes of objects in ImageNet images
2. run `mask_imgnet.py` to mask ImageNet images
3. (Optional) run `data_augmentation.py` to wrap foreground objects with garment images
4. (Optional) run `data_pair_preparation.py` to generate training data pairs

# Outputs structure
- `all_bboxes.json`: bboxes of objects in a image, {"`img_name`.JPEG": [bbox1, bbox2,...]}
- `foreground`: masked objects, "`img_name`_mask_`i`_foreground.png"
- `inpaint`: masked background, "`img_name`_mask_`i`_background.png"
- `model`: original image, "`img_name`.JPEG"