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
import configparser
import shutil
# my function
sys.path.append('./')
import utils.function as function
from segmentation.grab_cut import Grabcut
from segmentation.snake import Snake
from segmentation.my_approach_superpixel import AE_Segmentation2

# get data path
IMG_PATH = "D:/SegTrackv2/JPEGImages/"
img_dir_list = os.listdir(IMG_PATH)
img_dir_list.sort()
img_list = []

BBOX_PATH = "./segtrack/bbox/"
bbox_dir_list = os.listdir(BBOX_PATH)
bbox_dir_list.sort()
bbox_list = []

GT_PATH = "D:/SegTrackv2/GroundTruth/"
gt_dir_list = os.listdir(GT_PATH)
gt_dir_list.sort()
gt_img_list = []

config = configparser.ConfigParser()
config.read('./config/example.ini')
start_video_num = int(config['video']['start'])
end_video_num = int(config['video']['end'])

t_now = time.time()
os.mkdir("./output/"+str(t_now))
file1 = open("./output/" + str(t_now) + "/out.txt", "w")
shutil.copy2('./config/example.ini', './output/' + str(t_now) + '/example.ini')

# img
for i in range(0, 14, 1):
    a = os.listdir(IMG_PATH + img_dir_list[i])
    a.sort()
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
    bbox_total_data_list.sort()
    for j in range(0, len(bbox_total_data_list), 1):
        bbox_data_list = np.load(BBOX_PATH + bbox_dir_list[i] + "/" + bbox_total_data_list[j], allow_pickle=True)
        bbox_list.append(bbox_data_list)

# gt
for i in range(0, 14, 1):
    if i == 2 or i == 3 or i == 4 or i == 7 or i == 9 or i == 11:
        sub_dir = os.listdir(GT_PATH + gt_dir_list[i])
        sub_dir.sort()
        for j in range(0, len(sub_dir), 1):
            sub_path = GT_PATH + gt_dir_list[i] + "/" + sub_dir[j]
            a = os.listdir(GT_PATH + gt_dir_list[i] + "/" + sub_dir[j])
            a.sort()
            sub_list = []
            for k in range(0, len(a), 1):
                sub_list.append(GT_PATH + gt_dir_list[i] + "/" + sub_dir[j] + "/" +  a[k])
            gt_img_list.append(sub_list)
    else:
        sub_list = []
        a = os.listdir(GT_PATH + gt_dir_list[i])
        a.sort()
        for k in range(0, len(a), 1):
            sub_list.append(GT_PATH + gt_dir_list[i] + "/" +  a[k])
        gt_img_list.append(sub_list)

# data tarnsformation
data_transformation = transforms.Compose([
            transforms.ToTensor(),
            ])

# model 1
My_Approach = AE_Segmentation2()
My_Approach_mask_iou_list = []
My_Approach_mask_iou_sub_list = []
My_Approach_bbox_iou_list = []
My_Approach_bbox_iou_sub_list = []

time1 = time.time()
for i in range(0, len(img_list), 1):
    if i < start_video_num:
        continue
    for j in range(0, len(img_list[i]) - 1, 1):
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
        next_bbox = bbox_list[i][j + 1]
        x, y, w, h = float(bbox[0]), float(bbox[1]) \
                ,float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])

        if j == 0:
            img_save = img.clone()
            gt_img_save = gt_img.clone()
            pre_x, pre_y, pre_w, pre_h = x, y, w, h
            grid_save = function.get_grid(img_save.shape[3], img_save.shape[2], pre_x + pre_w/2, pre_y + pre_h/2, 2*pre_w, 2*pre_h, 128, 128)
            grid_save = grid_save.to(dtype=torch.float32)
            continue

        # grid previous
        grid = function.get_grid(img_save.shape[3], img_save.shape[2], pre_x + pre_w/2, pre_y + pre_h/2, 2*pre_w, 2*pre_h, 128, 128)
        grid = grid.to(dtype=torch.float32)
        previous = torch.nn.functional.grid_sample(img_save, grid, mode="bilinear", padding_mode="zeros")
        previous_pil = torchvision.transforms.ToPILImage()(previous[0].detach().cpu())
        # grid current
        grid = function.get_grid(img.shape[3], img.shape[2], pre_x + pre_w/2, pre_y + pre_h/2, 2*pre_w, 2*pre_h, 128, 128)
        grid = grid.to(dtype=torch.float32)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros")
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        # search_pil.save("./img" + str(i) + "_" + str(j) + ".jpg")
        # gt previous
        grid = function.get_grid(gt_img_save.shape[3], gt_img_save.shape[2], pre_x + pre_w/2, pre_y + pre_h/2, 2*pre_w, 2*pre_h, 128, 128)
        grid = grid.to(dtype=torch.float32)
        previous_mask = torch.nn.functional.grid_sample(gt_img_save, grid, mode="bilinear", padding_mode="zeros")
        previous_mask_pil = torchvision.transforms.ToPILImage()(previous_mask[0].detach().cpu())
        # previous_mask_pil.save("./mask" + str(i) + "_" + str(j) + ".jpg")
        previous_mask_np = np.array(previous_mask_pil)
        previous_mask_np = previous_mask_np.mean(axis = 2)
        previous_mask_np /= 255
        previous_mask_np = previous_mask_np.astype(np.uint8)
        # gt current
        grid = function.get_grid(gt_img.shape[3], gt_img.shape[2], pre_x + pre_w/2, pre_y + pre_h/2, 2*pre_w, 2*pre_h, 128, 128)
        grid = grid.to(dtype=torch.float32)
        mask = torch.nn.functional.grid_sample(gt_img, grid, mode="bilinear", padding_mode="zeros")
        mask_pil = torchvision.transforms.ToPILImage()(mask[0].detach().cpu())
        # mask_pil.save("./mask" + str(i) + "_" + str(j) + ".jpg")
        mask_np = np.array(mask_pil)
        mask_np = mask_np.mean(axis = 2)
        mask_np /= 255
        mask_np = mask_np.astype(np.uint8)

        try:
            gt_l, gt_t, gt_r, gt_b = function.get_x_y_w_h(mask_np)
        except:
            gt_l, gt_t, gt_r, gt_b = 0.0, 0.0, 0.0, 0.0

        if j == 1:
            mask_np1 = mask_np.copy()
            gt_l1, gt_t1, gt_r1, gt_b1 = gt_l, gt_t, gt_r, gt_b
            search1 = search.clone()

        # Model 1
        img_batch = function.get_image_batch_with_translate_augmentation(img_save, 4, pre_x, pre_y, pre_w, 128, pre_h, 128, torch.float32)
        flag = My_Approach.train(img_batch, previous, grid, i, j)

        if flag:
            result = My_Approach.inference_fore(search, grid, i, j)
        else:
            result = My_Approach.inference(search, grid, i, j)
        
        iou_i = np.logical_and(result, mask_np)
        iou_u = np.logical_or(result, mask_np)
        iou_i = np.where(iou_i == True, 1, 0)
        iou_u = np.where(iou_u == True, 1, 0)
        if iou_u.sum() > 0:
            iou = iou_i.sum() / iou_u.sum()
        else:
            iou = 0.0

        My_Approach_mask_iou_sub_list.append(iou)

        try:
            pred_l, pred_t, pred_r, pred_b = function.get_x_y_w_h(result)
        except:
            pred_l, pred_t, pred_r, pred_b = 0.0, 0.0, 0.0, 0.0
        x_left = max(pred_l, gt_l)
        y_top = max(pred_t, gt_t)
        x_right = min(pred_r, gt_r)
        y_bottom = min(pred_b, gt_b)
        iw = np.maximum(x_right - x_left + 1., 0.)
        ih = np.maximum(y_bottom - y_top + 1., 0.)

        intersection_area = iw * ih
        union_area = ((gt_r - gt_l + 1.) * (gt_b - gt_t + 1.) +
            (pred_r - pred_l + 1.) * (pred_b - pred_t + 1.) -
            intersection_area)
        if float(union_area) > 0.0:
            bbox_iou = intersection_area / union_area
        else:
            bbox_iou = 1.0

        My_Approach_bbox_iou_sub_list.append(bbox_iou)

        img_save = img.clone()
        gt_img_save = gt_img.clone()
        pre_x, pre_y, pre_w, pre_h = x, y, w, h

    My_Approach_mask_iou_list.append(My_Approach_mask_iou_sub_list)
    My_Approach_bbox_iou_list.append(My_Approach_bbox_iou_sub_list)
    print("finish")
    if i == end_video_num:
       break

# model 1
avg = 0
for i in range(0, len(My_Approach_mask_iou_list), 1):
    avg += sum(My_Approach_mask_iou_list[i]) / len(My_Approach_mask_iou_list[i])
print("model_mask")
print(avg / len(My_Approach_mask_iou_list))
file1.writelines("model_mask")
file1.writelines("\n")
file1.writelines(str(avg / len(My_Approach_mask_iou_list)))
file1.writelines("\n")

avg = 0
for i in range(0, len(My_Approach_bbox_iou_list), 1):
    avg += sum(My_Approach_bbox_iou_list[i]) / len(My_Approach_bbox_iou_list[i])
print("model_bbox")
print(avg / len(My_Approach_bbox_iou_list))
file1.writelines("model_bbox")
file1.writelines("\n")
file1.writelines(str(avg / len(My_Approach_bbox_iou_list)))

file1.close()
print(time.time() - time1)
