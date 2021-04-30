import os
import numpy as np
import cv2

# get data path
ANN_PATH = "D:/DAVIS/Annotations/480p/"
ann_list = os.listdir(ANN_PATH)

# video annotation list : bbox/mask
# [video_num, bbox_num]
# [video_num, mask_num]
video_bbox_list = []
video_mask_list = []
# get video annotation : bbox/mask
for index, i in enumerate(ann_list):
    ann_dir_path = ANN_PATH + i + "/"
    ann_list = os.listdir(ann_dir_path)
    img = cv2.imread(ann_dir_path + str(ann_list[0]))
    num_color = np.c_[np.unique(img.reshape(-1,3), axis=0, return_counts=1)]
    bbox_list = []
    mask_list = []
    for j in range(1, len(num_color), 1):
        mask = np.where((img[:, :, 0] == num_color[j][0]) & (img[:, :, 1] == num_color[j][1]) & (img[:, :, 2] == num_color[j][2]), 255, 0)
        mask = mask.astype(np.uint8)
        mask_list.append(mask)
        x, y = np.where(mask)
        bbox_list.append([y.min(), x.min(), y.max(), x.max()])
    video_bbox_list.append(np.array(bbox_list))
    video_mask_list.append(np.array(mask_list))

video_mask_list = np.array(video_mask_list)
video_bbox_list = np.array(video_bbox_list)
np.save("./davis/mask.npy", video_mask_list)
np.save("./davis/bbox.npy", video_bbox_list)
