import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from skimage.color import gray2rgb
from sklearn.metrics import jaccard_score
import time
# my function
sys.path.append('./')
import utils.function as function
from segmentation.grab_cut import Grabcut
from segmentation.snake import Snake
from segmentation.my_approach import AE_Segmentation

# get data path
IMG_PATH = "D:/SegTrackv2/JPEGImages/"
img_dir_list = os.listdir(IMG_PATH)
function.get_sorting_list(img_dir_list, "./img_dir_list.txt")
img_list = []

BBOX_PATH = "./segtrack/bbox/"
bbox_dir_list = os.listdir(BBOX_PATH)
function.get_sorting_list(bbox_dir_list, "./bbox_dir_list.txt")
bbox_list = []

GT_PATH = "D:/SegTrackv2/GroundTruth/"
gt_dir_list = os.listdir(GT_PATH)
function.get_sorting_list(gt_dir_list, "./gt_dir_list.txt")
gt_img_list = []

# img
for i in range(0, 14, 1):
    a = os.listdir(IMG_PATH + img_dir_list[i])
    function.get_sorting_list(a, "./a.txt")
    for j in range(0, len(a), 1):
        a[j] = IMG_PATH + img_dir_list[i] + "/" + a[j]
    if i == 2 or i == 3 or i == 4 or i == 7 or i == 9:
        img_list.append(a)
        img_list.append(a)
    elif i == 11:
        img_list.append(a)
        img_list.append(a)
        img_list.append(a)
        img_list.append(a)
        img_list.append(a)
        img_list.append(a)
    else:
        img_list.append(a)

# bbox
for i in range(0, 14, 1):
    bbox_total_data_list = os.listdir(BBOX_PATH + bbox_dir_list[i])
    function.get_sorting_list(bbox_total_data_list, "./bbox_total_data_list.txt")
    for j in range(0, len(bbox_total_data_list), 1):
        bbox_data_list = np.load(BBOX_PATH + bbox_dir_list[i] + "/" + bbox_total_data_list[j], allow_pickle=True)
        bbox_list.append(bbox_data_list)

# gt
for i in range(0, 14, 1):
    if i == 2 or i == 3 or i == 4 or i == 7 or i == 9 or i == 11:
        sub_dir = os.listdir(GT_PATH + gt_dir_list[i])
        function.get_sorting_list(sub_dir, "./sub_dir.txt")
        for j in range(0, len(sub_dir), 1):
            sub_path = GT_PATH + gt_dir_list[i] + "/" + sub_dir[j]
            a = os.listdir(GT_PATH + gt_dir_list[i] + "/" + sub_dir[j])
            function.get_sorting_list(a, "./a2.txt")
            sub_list = []
            for k in range(0, len(a), 1):
                sub_list.append(GT_PATH + gt_dir_list[i] + "/" + sub_dir[j] + "/" +  a[k])
            gt_img_list.append(sub_list)
    else:
        sub_list = []
        a = os.listdir(GT_PATH + gt_dir_list[i])
        function.get_sorting_list(a, "./a3.txt")
        for k in range(0, len(a), 1):
            sub_list.append(GT_PATH + gt_dir_list[i] + "/" +  a[k])
        gt_img_list.append(sub_list)

# data tarnsformation
data_transformation = transforms.Compose([
            transforms.ToTensor(),
            ])

grabcut = Grabcut()
grabcut_iou_list = []
grabcut_iou_sub_list = []
snake = Snake()
snake_iou_list = []
snake_iou_sub_list = []
My_Approach = AE_Segmentation()
My_Approach_iou_list = []
My_Approach_iou_sub_list = []


time1 = time.time()
for i in range(0, len(img_list), 1):
    if i < -1:
        continue
    for j in range(0, len(img_list[i]), 1):
        img = Image.open(img_list[i][j])
        img = img.convert('RGB')
        img = data_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=torch.float32)
        gt_img = Image.open(gt_img_list[i][j])
        gt_img = gt_img.convert('RGB')
        gt_img = data_transformation(gt_img)
        gt_img = torch.unsqueeze(gt_img, 0)
        gt_img = gt_img.to(dtype=torch.float32)
        bbox = bbox_list[i][j]
        x, y, w, h = float(bbox[0]), float(bbox[1]) \
                ,float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])
        # grid
        grid = function.get_grid(img.shape[3], img.shape[2], x + w/2, y + h/2, 2*w, 2*h, 128, 128)
        grid = grid.to(dtype=torch.float32)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros")
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        # search_pil.save("./img" + str(i) + "_" + str(j) + ".jpg")
        grid = function.get_grid(gt_img.shape[3], gt_img.shape[2], x + w/2, y + h/2, 2*w, 2*h, 128, 128)
        grid = grid.to(dtype=torch.float32)
        mask = torch.nn.functional.grid_sample(gt_img, grid, mode="bilinear", padding_mode="zeros")
        mask_pil = torchvision.transforms.ToPILImage()(mask[0].detach().cpu())
        # mask_pil.save("./mask" + str(i) + "_" + str(j) + ".jpg")
        mask_np = np.array(mask_pil)
        mask_np = mask_np.mean(axis = 2)
        mask_np /= 255
        mask_np = mask_np.astype(np.uint8)
        # grabcut
        grabcut_result = grabcut.get_mask(np.array(search_pil), j)
        iou_i = np.logical_and(grabcut_result, mask_np)
        iou_u = np.logical_or(grabcut_result, mask_np)
        iou_i = np.where(iou_i == True, 1, 0)
        iou_u = np.where(iou_u == True, 1, 0)
        if iou_u.sum() > 0:
            iou = iou_i.sum() / iou_u.sum()
        else:
            iou = 0.0
        grabcut_iou_sub_list.append(iou)
        # Snake
        snake_result = snake.get_mask(np.array(search_pil))
        iou_i = np.logical_and(snake_result, mask_np)
        iou_u = np.logical_or(snake_result, mask_np)
        iou_i = np.where(iou_i == True, 1, 0)
        iou_u = np.where(iou_u == True, 1, 0)
        if iou_u.sum() > 0:
            iou = iou_i.sum() / iou_u.sum()
        else:
            iou = 0.0
        snake_iou_sub_list.append(iou)
        # Model 1
        img_batch = function.get_image_batch_with_translate_augmentation(img, 4, x, y, w, 128, h, 128, torch.float32)
        # if j % 5 == 0:
        My_Approach.train(img_batch, search)
        result = My_Approach.inference(search, j)
        iou_i = np.logical_and(result, mask_np)
        iou_u = np.logical_or(result, mask_np)
        iou_i = np.where(iou_i == True, 1, 0)
        iou_u = np.where(iou_u == True, 1, 0)
        if iou_u.sum() > 0:
            iou = iou_i.sum() / iou_u.sum()
        else:
            iou = 0.0
        My_Approach_iou_sub_list.append(iou)

    grabcut_iou_list.append(grabcut_iou_sub_list)
    snake_iou_list.append(snake_iou_sub_list)
    My_Approach_iou_list.append(My_Approach_iou_sub_list)
    if i == 7:
       break

# grabcut
avg = 0
for i in range(0, len(grabcut_iou_list), 1):
    avg += sum(grabcut_iou_list[i]) / len(grabcut_iou_list[i])
print(avg / len(grabcut_iou_list))

# snake
avg = 0
for i in range(0, len(snake_iou_list), 1):
    avg += sum(snake_iou_list[i]) / len(snake_iou_list[i])
print(avg / len(snake_iou_list))

# snake
avg = 0
for i in range(0, len(My_Approach_iou_list), 1):
    avg += sum(My_Approach_iou_list[i]) / len(My_Approach_iou_list[i])
print(avg / len(My_Approach_iou_list))

print(time.time() - time1)
