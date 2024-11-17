import os 
import json 

data_dir = "/home/stevexu/data/processed_imagenet/val"
# %%
foreground_count = len(os.listdir(os.path.join(data_dir, "foreground")))
background_count = len(os.listdir(os.path.join(data_dir, "inpaint")))
garment_count = len(os.listdir(os.path.join(data_dir, "garment")))
model_count = len(os.listdir(os.path.join(data_dir, "model")))

print(f"Foreground count: {foreground_count}")
print(f"Inpaint count: {background_count}")
print(f"Garment count: {garment_count}")
print(f"Model count: {model_count}")
print(f"diff: {foreground_count - background_count}")

# %%
with open(os.path.join(data_dir, "all_bboxes.json"), "r") as f:
    all_bboxes = json.load(f)

box_num_list = []
for k, v in all_bboxes.items():
    if len(v) > 1:
        box_num_list.append(len(v))
    else:
        box_num_list.append(1)

print(f"number of bboxes: {sum(box_num_list)}")

# %%

