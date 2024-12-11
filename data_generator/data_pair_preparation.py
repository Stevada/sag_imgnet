"""Generate a txt file of data pairs for training"""

import os
from PIL import Image
from collections import defaultdict
import re

def extract_indices(file_name):
    mask_match = re.search(r'mask_(\d+)', file_name)
    agmt_match = re.search(r'agmt_(\d+)', file_name)
    return mask_match.group(1) if mask_match else None, agmt_match.group(1) if agmt_match else None

def generate_data_pairs(data_dir, output_file):
    model_dir = os.path.join(data_dir, "model")
    garment_dir = os.path.join(data_dir, "garment")
    inpaint_dir = os.path.join(data_dir, "inpaint")

    # Use a nested defaultdict for efficient data organization
    data_dict = defaultdict(lambda: defaultdict(lambda: {'garments': set(), 'inpaint': False}))

    # Process garment and inpaint directories
    for directory, is_garment in [(garment_dir, True), (inpaint_dir, False)]:
        for file_name in os.listdir(directory):
            img_name = file_name.split("_mask")[0]
            mask_idx, agmt_idx = extract_indices(file_name)
            if is_garment:
                data_dict[img_name][mask_idx]['garments'].add(agmt_idx)
            else:
                data_dict[img_name][mask_idx]['inpaint'] = True

    # Generate results
    results = []
    for model_file in os.listdir(model_dir):
        orig_img_size = Image.open(os.path.join(model_dir, model_file)).size
        img_name = model_file.split('.')[0]
        for mask_idx, data in data_dict[img_name].items():
            if data['inpaint']:
                inpaint_name = f"{img_name}_mask_{mask_idx}_background.png"
                for agmt_idx in data['garments']:
                    garment_name = f"{img_name}_mask_{mask_idx}_agmt_{agmt_idx}.png"
                    results.append((model_file, inpaint_name, garment_name, ','.join(list(map(str, orig_img_size)))))

    # Write results to file
    with open(output_file, "w") as f:
        for result in results:
            f.write(f"{result[0]} {result[1]} {result[2]} {result[3]}\n")

    print(f"Total {len(results)} processed images")

if __name__ == "__main__":
    data_dir = "/home/stevexu/data/processed_imagenet/val"
    output_file = os.path.join(data_dir, "data_pairs.txt")
    generate_data_pairs(data_dir, output_file)
