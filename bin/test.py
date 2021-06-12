# library
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import os
import cv2
import numpy as np
import json
import time
# my function
import src.My_Tracker as Tracker

GOT_data_path = "D:/GOT/test/"
GOT_data_list =  os.listdir(GOT_data_path)

for index_2, i in enumerate(GOT_data_list):
    time1 = time.time()
    if index_2 < 20:
        continue
    if index_2 == 9:
        assert False
    # Path setting
    img_path = "D:/GOT/test/" + i + "/"
    gt_path = "D:/GOT/test/" + i + "/groundtruth.txt"

    # Read image directory as a list & Remove unused files in list
    img_list = os.listdir(img_path)
    # img_list.remove("absence.label")
    # img_list.remove("cover.label")
    # img_list.remove("cut_by_image.label")
    # img_list.remove("meta_info.ini")
    img_list.remove("groundtruth.txt")

    gt = open(gt_path)
    # rect = gt.readline().split(',')
    # x = int(float(rect[0]))
    # y = int(float(rect[1]))
    # w, h = int(float(rect[2])), int(float(rect[3]))
    tracker = Tracker.FCAE_tracker()

    for index, i in enumerate(img_list):
        img = Image.open(img_path+"/"+i)
        # rect = gt.readline().split(',')
        # real_x = int(float(rect[0]))
        # real_y = int(float(rect[1]))
        # real_w, real_h = int(float(rect[2])), int(float(rect[3]))
        if index == 0:
            rect = gt.readline().split(',')
            real_x = int(float(rect[0]))
            real_y = int(float(rect[1]))
            real_w, real_h = int(float(rect[2])), int(float(rect[3]))
            tracker.tracker_init(img, real_x, real_y, real_w, real_h, 500, index_2)
        else:
            x, y, w, h, flag = tracker.tracker_inference_for_eval(img, index_2)
            print(time1 - time.time())
            # if index % 3 == 0:
            # if init_num < 0:
            #     init_num = 0
            # for j in range(init_num, index, 1):
            if flag:
                tracker.tracker_update(index_2, 0)

    print(index_2)
    print(time1 - time.time())
