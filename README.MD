# Steps for generating data
1. go to `data_generator` folder
2. run `objects_detection_v3.py` to get bboxes of objects in ImageNet images
3. run `mask_imgnet.py` to mask ImageNet images
4. (Optional) run `data_augmentation.py` to wrap foreground objects with garment images
5. (Optional) run `data_pair_preparation.py` to generate training data pairs list in `data_pairs.txt`

# Outputs structure
- `all_bboxes.json`: bboxes of objects in a image, {"`img_name`.JPEG": [bbox1, bbox2,...]}
- `foreground`: masked objects, "`img_name`_mask_`i`_foreground.png"
- `inpaint`: masked background, "`img_name`_mask_`i`_background.png"
- `model`: original image, "`img_name`.JPEG"
- `mask`: mask image, "`img_name`_mask_`i`_mask.png"
- `visualization`: visualization of bounding boxes, "annoated_`img_name`.png"