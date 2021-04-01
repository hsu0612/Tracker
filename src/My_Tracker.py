# library
import os
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from scipy import signal
from sklearn import manifold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
# mine
from src.model import FCNet
from src.model import Discriminator
import utils.function as function
from utils.gaussian import g_kernel
# Function
class FCAE_tracker():
    def __init__(self):
        np.random.seed(999)
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        torch.backends.cudnn.deterministic = True
        self.device = 'cuda'
        self.data_type = torch.float32
        self.threshold_for_background = 0.05
        self.threshold_for_foreground = 0.95
        # check img num
        self.check_num = 7
    def tracker_init(self, img, x, y, w, h, number_of_frame, video_num):
        # input data init
        data_transformation = transforms.Compose([
            transforms.ToTensor(),
            ])
        img = data_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=self.data_type)

        # set x, y, w, h 
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        # model init
        self.model_discriminator = Discriminator().to(self.device, dtype=self.data_type)
        self.model_discriminator.train()
        self.model_background = FCNet().to(self.device, dtype=self.data_type)
        self.model_background.train()

        # background

        # image batch
        image_batch = function.get_image_batch_with_translate_augmentation(img, 4, x, y, w, 128, h, 128, self.data_type)

        # memory
        self.memory = torch.zeros(number_of_frame, 4*4, 3, 128, 128)
        self.memory[0] = image_batch

        # count image
        self.count_image = 0

        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)

        # input for model init
        image_batch = image_batch.to(self.device, dtype=self.data_type)

        # train
        for i in range(0, 500, 1):
            # optimizer init
            optimizer.zero_grad()
            pred, feature_map = self.model_background(image_batch)
            background_diff = torch.abs(pred - image_batch)
            for index1, i in enumerate(range(-32, 32, 16)):
                for index2, j in enumerate(range(-32, 32, 16)):
                    background_diff[index1*4+index2, :, 32+j:96+j, 32+i:96+i] = 0.0
            background_diff_loss = background_diff.mean()
            loss = background_diff_loss
            loss.backward()
            optimizer.step()
        print("background finish !!!")

        # check image_batch
        # for index1, i in enumerate(range(-64, 64, 32)):
        #     for index2, j in enumerate(range(-64, 64, 32)):
        #         # get the cropped img
        #         search_pil = torchvision.transforms.ToPILImage()(image_batch[index1*4+index2].detach().cpu())
        #         search_pil.save("./test" + str(index1*4+index2) +".jpg")

        # check search
        # search_pil = torchvision.transforms.ToPILImage()(image_batch[self.check_num].detach().cpu())
        # search_pil.save("./search_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_background(image_batch)
        # pred_pil = torchvision.transforms.ToPILImage()(pred[self.check_num].detach().cpu())
        # pred_pil.save("./pred_img_with_background_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        error_map = torch.abs(pred - image_batch)
        error_map = error_map.mean(axis = 1)
        # function.write_heat_map(error_map[self.check_num].detach().cpu().numpy(), self.count_image, "./error_background_" + str(video_num) + "_")

        # gaussian_map
        gaussian_map = torch.from_numpy(g_kernel)
        gaussian_map = gaussian_map.to(self.device, dtype=self.data_type)
        gaussian_map = torch.unsqueeze(gaussian_map, 0)
        gaussian_map_mask_center= torch.zeros(error_map.shape)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                gaussian_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = gaussian_map
        gaussian_map_mask_center = torch.unsqueeze(gaussian_map_mask_center, 1)

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                threshold_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = threshold_map[index1*4+index2, 32+j:96+j, 32+i:96+i] + gaussian_map
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        threshold_map[threshold_map > 1.0] = 1.0
        self.threshold_map_temp = threshold_map.clone()
        # function.write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        # check mask
        # mask = np.zeros((128, 128, 3))
        # search_np = np.array(search_pil)
        # for i in range(0, 3, 1):
        #     mask[:, :, i] = np.where(threshold_map[self.check_num][0].detach().cpu().detach().numpy() == 1.0, search_np[:, :, i], 0.0)
        # search_with_mask = Image.fromarray(mask.astype("uint8"))
        # search_with_mask = data_transformation(search_with_mask)
        # search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
        # search_with_mask.save("./mask_" + str(video_num) + ".jpg")

        # discreminator

        # optimizer init
        optimizer = optim.Adam(self.model_discriminator.parameters(), lr = 1e-4)

        # loss function init
        criterion_bec_loss = nn.BCELoss()

        # train
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, pred_seg, feature_map = self.model_discriminator(image_batch)
            correlation_loss = criterion_bec_loss(pred, gaussian_map_mask_center.to(self.device, dtype=self.data_type))
            seg_loss = criterion_bec_loss(pred_seg, threshold_map.to(self.device, dtype=self.data_type))
            loss = correlation_loss + seg_loss
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")

        # check pred
        # with torch.no_grad():
        #     pred, pred_seg, feature_map = self.model_discriminator(image_batch)
        # pred_pil = torchvision.transforms.ToPILImage()(pred[self.check_num].detach().cpu())
        # pred_pil.save("./pred_img_with_foreground_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        # error_map_fore = pred
        # error_map_fore = error_map_fore.mean(axis = 0)
        # error_map_fore = error_map_fore.mean(axis = 0)
        # error_map_fore = error_map_fore.mean(axis = 1)
        # function.write_heat_map(error_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_foregroud" + str(video_num) + "_")

        # check threshold map
        # threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)
        # threshold_map_fore[threshold_map_fore!=0.0] = 1.0
        # function.write_heat_map(threshold_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./threshold_foregroud" + str(video_num) + "_")

        self.count_image += 1
    def tracker_inference_for_eval(self, img, video_num, flag):
        # input data init
        data_transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        img = data_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=self.data_type)

        # set model
        self.model_discriminator = self.model_discriminator.to(self.device, dtype=self.data_type)
        self.model_discriminator.eval()

        # Get search grid
        grid = function.get_grid(img.shape[3], img.shape[2], self.x + self.w/2, self.y + self.h/2, 2*self.w, 2*self.h, 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros")
        search = search.to(self.device, dtype=self.data_type)

        # inference
        with torch.no_grad():
            pred, pred_seg, feature_map = self.model_discriminator(search)

        # check search
        # search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        # search_pil.save("./search_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check pred
        # img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # img_pil.save("./pred_fore_img_" + str(video_num) + "_" + str(self.count_image) + ".jpg")
        
        # error map
        error_map = pred
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        # function.write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map
        threshold_map = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        # function.write_heat_map(threshold_map, self.count_image, "./threshold_map_fore_" + str(video_num) + "_")

        # error map with seg
        error_map = pred_seg
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        # function.write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map with seg
        threshold_map_seg = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        # function.write_heat_map(threshold_map_seg, self.count_image, "./threshold_map_fore_seg" + str(video_num) + "_")
        
        # get x, y, w, h
        threshold_map = threshold_map.astype(np.uint8)
        threshold_map_seg = threshold_map_seg.astype(np.uint8)
        self.x, self.y, self.w, self.h, flag = function.get_obj_x_y_w_h(threshold_map, threshold_map_seg, self.x, self.y, self.w, self.h, img, self.device, self.data_type, self.model_discriminator, search)

        # image_batch
        image_batch = function.get_image_batch_with_translate_augmentation(img, 4, self.x, self.y, self.w, 128, self.h, 128, self.data_type)

        if flag:
            # memory
            self.memory[self.count_image] = image_batch

        # function.write_tracking_result(img.detach().cpu().numpy(), self.x, self.y, self.count_image, self.w, self.h, "./" + str(video_num) + "_")

        self.count_image+=1

        return self.x, self.y, self.w, self.h
    
    def tracker_update(self, video_num):
        # model init
        self.model_background.train()
        self.model_discriminator.train()
        # train data init
        self.training_set = torch.cat((self.memory[self.count_image-1], self.memory[self.count_image-2], self.memory[0]), 0)
        self.training_set = self.training_set.to(self.device, dtype=self.data_type)
        
        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)

        # train
        for i in range(0, 100, 1):
            # opt init
            optimizer.zero_grad()
            pred, feature_map = self.model_background(self.training_set)
            background_diff = torch.abs(pred - self.training_set)
            for index1, i in enumerate(range(-32, 32, 16)):
                for index2, j in enumerate(range(-32, 32, 16)):
                    background_diff[index1*4+index2, :, 32+j:96+j, 32+i:96+i] = 0.0
                    background_diff[index1*4+index2+16, :, 32+j:96+j, 32+i:96+i] = 0.0
                    background_diff[index1*4+index2+32, :, 32+j:96+j, 32+i:96+i] = 0.0
            background_diff_loss = background_diff.mean()
            loss = background_diff_loss
            # if i % 100 == 0:
            #     print(loss)
            loss.backward()
            optimizer.step()
        print("background finish !!!")

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_background(self.training_set)
        # pred_pil = torchvision.transforms.ToPILImage()(pred[self.check_num].detach().cpu())
        # pred_pil.save("./pred_img_with_background_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        error_map = torch.abs(pred - self.training_set)
        error_map = error_map.mean(axis = 1)
        # function.write_heat_map(error_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_background_" + str(video_num) + "_")

        # gaussian_map
        gaussian_map = torch.from_numpy(g_kernel)
        gaussian_map = gaussian_map.to(self.device, dtype=self.data_type)
        gaussian_map = torch.unsqueeze(gaussian_map, 0)
        gaussian_map_mask_center= torch.zeros(error_map.shape)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                gaussian_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = gaussian_map
                gaussian_map_mask_center[index1*4+index2+16, 32+j:96+j, 32+i:96+i] = gaussian_map
                gaussian_map_mask_center[index1*4+index2+32, 32+j:96+j, 32+i:96+i] = gaussian_map
        gaussian_map_mask_center = torch.unsqueeze(gaussian_map_mask_center, 1)
        gaussian_map_mask_center[gaussian_map_mask_center > 1.0] = 1.0

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                threshold_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = threshold_map[index1*4+index2, 32+j:96+j, 32+i:96+i] + gaussian_map
                threshold_map_mask_center[index1*4+index2+16, 32+j:96+j, 32+i:96+i] = threshold_map[index1*4+index2+16, 32+j:96+j, 32+i:96+i] + gaussian_map
                threshold_map_mask_center[index1*4+index2+32, 32+j:96+j, 32+i:96+i] = threshold_map[index1*4+index2+32, 32+j:96+j, 32+i:96+i] + gaussian_map
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        threshold_map[threshold_map > 1.0] = 1.0
        threshold_map[32:] = self.threshold_map_temp
        # function.write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        # foreground
        # optimizer init
        optimizer = optim.Adam(self.model_discriminator.parameters(), lr = 1e-4)
        # loss function init
        criterion_bec_loss = nn.BCELoss()
        # train
        for i in range(0, 100, 1):
            optimizer.zero_grad()
            pred, pred_seg, feature_map = self.model_discriminator(self.training_set)
            correlation_loss = criterion_bec_loss(pred, gaussian_map_mask_center.to(self.device, dtype=self.data_type))
            seg_loss = criterion_bec_loss(pred_seg, threshold_map.to(self.device, dtype=self.data_type))
            loss = correlation_loss + seg_loss
            # if i % 100 == 0:
            #     print(loss)
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")
        