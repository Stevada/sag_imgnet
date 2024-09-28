from transformers import AutoImageProcessor, SuperPointForKeypointDetection
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import shutil
class SuperPointDetector:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
        # self.model.eval()
        # self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def detect_keypoints(self, image):
        inputs = self.processor(image, return_tensors="pt")
        outputs = self.model(**inputs)
        img_height, img_width = inputs['pixel_values'].squeeze(0).shape[-2:]

        # Normalize keypoints
        keypoints = outputs.keypoints.squeeze(0).detach().cpu().numpy()
        keypoints[:, 0] /= img_width
        keypoints[:, 1] /= img_height
        scores = outputs.scores.squeeze(0).detach().cpu().numpy()

        return {
            "keypoints": keypoints,
            "scores": scores
        }
    
def draw_keypoints(img, keypoints, scores, output_path, name):
    # Create a new figure with the same size as the input image
    fig_size = (img.width / 100, img.height / 100)
    fig, ax = plt.subplots(figsize=fig_size, dpi=100)
    
    # Display the image
    ax.imshow(img)
    
    # Plot the keypoints
    ax.scatter(
        keypoints[:, 0]*img.width,
        keypoints[:, 1]*img.height,
        c=scores * 100,
        s=scores * 50,
        alpha=0.8
    )
    
    # Remove axes and set limits to match image dimensions
    ax.axis('off')
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)  # Reverse Y-axis to match image coordinates
    
    # Save the figure without extra padding
    plt.savefig(f"{output_path}/{name}.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    # Usage
    img_num_limit = 10
    detector = SuperPointDetector()
    # image_folder = "/home/stevexu/data/viton_for_mage_10/model"
    image_folder = "/home/stevexu/data/imagenet_val"
    output_folder = "objects_superpoint"
    os.makedirs(output_folder, exist_ok=True)

    for i, img_name in enumerate(os.listdir(image_folder)):
        if i >= img_num_limit:
            break
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path)
        shutil.copy(img_path, os.path.join(output_folder, img_name))

        # get keypoints
        results = detector.detect_keypoints(img)

        # save keypoints
        draw_keypoints(img, results["keypoints"], results["scores"], output_folder, f"{img_name}_keypoints")