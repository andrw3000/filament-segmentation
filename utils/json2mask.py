import os
import cv2
import json
import numpy as np
from tqdm import tqdm

root_dir = '/Users/Holmes/Research/IDSAI/PROOF/filament-segmentation'
data_dir = os.path.join(root_dir, 'data/tomograms2D/tf1')
img_dir = os.path.join(data_dir, 'png-original')
json_file = os.path.join(data_dir, 'polygons-tf1.json')

imask_dir = os.path.join(data_dir, 'png-masks/instance')
smask_dir = os.path.join(data_dir, 'png-masks/semantic')
if not os.path.exists(imask_dir):
    os.makedirs(imask_dir)
if not os.path.exists(smask_dir):
    os.makedirs(smask_dir)

single_file = next(os.walk(img_dir))[2][0]
single_img = cv2.imread(os.path.join(img_dir, single_file))
mask_height, mask_width = single_img.shape[0], single_img.shape[1]
# image_files = [os.path.join(img_dir, file)
#                for file in os.listdir(img_dir) if file.endswith('.png')]

count = 0  # Count of total images saved
file_bbs = {}  # Dictionary containing polygon coordinates for mask

# Read JSON file
with open(json_file) as f:
    data = json.load(f)


# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
    try:
        x_points = \
            data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
        y_points = \
            data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
    except:
        print("No BB. Skipping", key)
        return

    all_points = []
    for x, y in zip(x_points, y_points):
        all_points.append([x, y])

    file_bbs[key] = all_points


for itr in data:
    file_name_json = data[itr]["filename"]
    sub_count = 0  # Contains count of masks for a single ground truth image
    
    if len(data[itr]["regions"]) > 1:
        for _ in range(len(data[itr]["regions"])):
            key = file_name_json[:-4] + "*" + str(sub_count+1)
            add_to_dict(data, itr, key, sub_count)
            sub_count += 1
    else:
        add_to_dict(data, itr, file_name_json[:-4], 0)

print("\nTotal number of polygons in dictionary: ", len(file_bbs))

for file_name in os.listdir(img_dir):
    if file_name.endswith('.png'):
        instance_dir = os.path.join(imask_dir, file_name[:-4])
        curr_img = os.path.join(img_dir, file_name)
        
# For each dictionary entry, generate instance mask and save
for itr in tqdm(file_bbs):
    num_masks = itr.split("*")
    mask_folder = os.path.join(imask_dir, num_masks[0])
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    mask = np.zeros((mask_height, mask_width))
    try:
        arr = np.array(file_bbs[itr])
    except:
        print("Not found:", itr)
        continue
    count += 1
    cv2.fillPoly(mask, [arr], color=(255,))

    if len(num_masks) > 1:
        cv2.imwrite(os.path.join(mask_folder, itr.replace("*", "_") + ".png"),
                    mask)
    else:
        cv2.imwrite(os.path.join(mask_folder, itr + ".png"), mask)

print("Instances saved:", count)

# Generate semantic masks

for dir_name in tqdm(os.listdir(imask_dir)):
    dir_path = os.path.join(imask_dir, dir_name)
    smask = np.zeros((mask_height, mask_width))
    for img_name in os.listdir(dir_path):
        imask = cv2.imread(os.path.join(dir_path, img_name),
                           cv2.IMREAD_GRAYSCALE) / 255
        smask += imask
    smask = (smask > 0).astype(float) * 255
    cv2.imwrite(os.path.join(smask_dir, dir_name + ".png"), smask)
