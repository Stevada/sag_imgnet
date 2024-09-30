import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import supervision as sv
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import List, Optional, Tuple

from supervision.annotators.base import ImageType
from supervision.utils.conversion import pillow_to_cv2
from dotenv import load_dotenv

load_dotenv()

# Constants
CHECKPOINT = "/home/stevexu/VSprojects/sag_imgnet/checkpoints/sam2_hiera_large.pt"
MODEL_TYPE = "vit_h"
IMAGENET_DIR = "/home/stevexu/data/imagenet_val"
CONFIG = "configs/sam2/sam2_hiera_l.yaml"

def plot_images_grid(
    images: List[ImageType],
    grid_size: Tuple[int, int],
    titles: Optional[List[str]] = None,
    size: Tuple[int, int] = (12, 12),
    cmap: Optional[str] = "gray",
    show: bool = True,
    save: bool = False,
    output_path: Optional[str] = None
) -> None:
    """
    Plots images in a grid using matplotlib.

    Args:
       images (List[ImageType]): A list of images as ImageType
             is a flexible type, accepting either `numpy.ndarray` or `PIL.Image.Image`.
       grid_size (Tuple[int, int]): A tuple specifying the number
            of rows and columns for the grid.
       titles (Optional[List[str]]): A list of titles for each image.
            Defaults to None.
       size (Tuple[int, int]): A tuple specifying the width and
            height of the entire plot in inches.
       cmap (str): the colormap to use for single channel images.

    Raises:
       ValueError: If the number of images exceeds the grid size.

    Examples:
        ```python
        import cv2
        import supervision as sv
        from PIL import Image

        image1 = cv2.imread("path/to/image1.jpg")
        image2 = Image.open("path/to/image2.jpg")
        image3 = cv2.imread("path/to/image3.jpg")

        images = [image1, image2, image3]
        titles = ["Image 1", "Image 2", "Image 3"]

        %matplotlib inline
        plot_images_grid(images, grid_size=(2, 2), titles=titles, size=(16, 16))
        ```
    """
    nrows, ncols = grid_size

    for idx, img in enumerate(images):
        if isinstance(img, Image.Image):
            images[idx] = pillow_to_cv2(img)

    if len(images) > nrows * ncols:
        raise ValueError(
            "The number of images exceeds the grid size. Please increase the grid size"
            " or reduce the number of images."
        )

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)

    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            if images[idx].ndim == 2:
                ax.imshow(images[idx], cmap=cmap)
            else:
                ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])

        ax.axis("off")
    
    if show:
        plt.show()
    if save:
        plt.savefig(output_path)

def save_object(obj, output_path):
    Image.fromarray(obj).save(output_path)
    

if __name__ == "__main__":
    data_dir = "/home/stevexu/VSprojects/sag_imgnet/data/objects_detection_v2"
    output_dir = "data/outputs_v2/advanced_obj_detection_sam2"

    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2 = build_sam2(CONFIG, CHECKPOINT, device=device, apply_postprocessing=False)
    sam2.to(device=device)
    predictor = SAM2ImagePredictor(sam2)
    bbox_path = os.path.join(data_dir, "all_bboxes.json")
    # load bbox
    bbox_dict = json.load(open(bbox_path, "r"))
    file_names = [name for name in os.listdir(data_dir) if name.endswith(".JPEG")]
    for i, img_name in enumerate(file_names):
        src_img_name = img_name.replace("bbox_", "")
        img_path = os.path.join(IMAGENET_DIR, src_img_name)
        
        # load image
        image_bgr = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # generate mask for the whole image without bbox
        bbox_list = bbox_dict[src_img_name]
        for i, bbox in enumerate(bbox_list):
            bbox_list[i] = list(map(float, bbox.split(",")))
        bbox_array = np.array(bbox_list)
        predictor.set_image(image_rgb)
        if bbox_list:
            masks, scores, logits = predictor.predict(
                box=bbox_array,
                multimask_output=False
                )
            box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            
            if bbox_array.shape[0] != 1:
                masks = np.squeeze(masks)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks.astype(bool)
            )
            source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

            plot_images_grid(
                images=[source_image, segmented_image],
                grid_size=(1, 2),
                titles=['source image', 'segmented image'],
                show=False,
                save=True,
                output_path=os.path.join(output_dir, f"{src_img_name}_segmented_bbox_{i}.jpg")
            )
        else:
            masks, scores, logits = predictor.predict(
                multimask_output=False
            )
            box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
            
            detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks),
                    mask=masks.astype(bool)
                )
            source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

            plot_images_grid(
                images=[source_image, segmented_image],
                grid_size=(1, 2),
                titles=['source image', 'segmented image'],
                show=False,
                save=True,
                output_path=os.path.join(output_dir, f"{src_img_name}_segmented.jpg")
            )
            

