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
        self.model_foreground = Discriminator().to(self.device, dtype=self.data_type)
        self.model_foreground.train()
        self.model_background = FCNet().to(self.device, dtype=self.data_type)
        self.model_background.train()
        # background
        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)
        # image batch
        image_batch = function.get_image_batch_with_translate_augmentation(img, 4, x, y, w, 192, h, 192, self.data_type)
        # memory
        self.memory = torch.zeros(number_of_frame, 4*4, 3, 192, 192)
        self.memory[0] = image_batch
        image_batch = image_batch.to(self.device, dtype=self.data_type)
        # count image
        self.count_image = 0
        # train
        for i in range(0, 1, 1):
            # optimizer init
            optimizer.zero_grad()
            pred, feature_map = self.model_background(image_batch)
            background_diff = torch.abs(pred - image_batch)
            for index1, i in enumerate(range(-64, 64, 32)):
                for index2, j in enumerate(range(-64, 64, 32)):
                    background_diff[index1*4+index2, :, 64+j:128+j, 64+i:128+i] = 0.0
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

        gaussian_map = torch.from_numpy(g_kernel)
        gaussian_map = gaussian_map.to(self.device, dtype=self.data_type)
        gaussian_map = torch.unsqueeze(gaussian_map, 0)

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        for index1, i in enumerate(range(-64, 64, 32)):
            for index2, j in enumerate(range(-64, 64, 32)):
                threshold_map_mask_center[index1*4+index2, 64+j:128+j, 64+i:128+i] = gaussian_map# * threshold_map[index1*4+index2, 64+j:128+j, 64+i:128+i]
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        # g_filter = torch.from_numpy(g_kernel)
        # g_filter = g_filter.to(dtype=self.data_type)
        # g_filter = torch.unsqueeze(g_filter, 0)
        # g_filter = torch.unsqueeze(g_filter, 0)
        # threshold_map = F.conv2d(threshold_map, g_filter, padding=3)
        threshold_map[threshold_map > 1.0] = 1.0
        # function.write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        # check mask
        # mask = np.zeros((192, 192, 3))
        # search_np = np.array(search_pil)
        # for i in range(0, 3, 1):
        #     mask[:, :, i] = np.where(threshold_map[self.check_num][0].detach().cpu().detach().numpy() == 1.0, search_np[:, :, i], 0.0)
        # search_with_mask = Image.fromarray(mask.astype("uint8"))
        # search_with_mask = data_transformation(search_with_mask)
        # search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
        # search_with_mask.save("./mask_" + str(video_num) + ".jpg")

        # foreground
        # optimizer init
        optimizer = optim.Adam(self.model_foreground.parameters(), lr = 1e-4)
        # loss function init
        criterion_bec_loss = nn.BCELoss()
        # train
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, feature_map = self.model_foreground(image_batch)
            bce_loss = criterion_bec_loss(pred, threshold_map.to(self.device, dtype=self.data_type))
            loss = bce_loss
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_foreground(image_batch)
        pred_pil = torchvision.transforms.ToPILImage()(pred[self.check_num].detach().cpu())
        # pred_pil.save("./pred_img_with_foreground_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        error_map_fore = pred
        # error_map_fore = error_map_fore.mean(axis = 0)
        # error_map_fore = error_map_fore.mean(axis = 0)
        error_map_fore = error_map_fore.mean(axis = 1)
        # function.write_heat_map(error_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_foregroud" + str(video_num) + "_")

        # # check threshold map
        # threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)
        # threshold_map_fore[threshold_map_fore!=0.0] = 1.0
        # function.write_heat_map(threshold_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./threshold_foregroud" + str(video_num) + "_")

        # # second stage
        # # optimizer init
        # optimizer_back = optim.Adam(list(self.model_background.parameters()), lr = 1e-4)
        # optimizer_fore = optim.Adam(list(self.model_foreground.parameters()), lr = 1e-4)
        # # scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
        # # iter
        # for i in range(0, 500, 1):
        #     optimizer_back.zero_grad()
        #     optimizer_fore.zero_grad()
        #     # model pred
        #     pred_back, feature_map = self.model_background(image_batch)

        #     background_diff = torch.abs(pred_back - image_batch)
        #     for index1, i in enumerate(range(-32, 32, 8)):
        #         for index2, j in enumerate(range(-32, 32, 8)):
        #             background_diff[index1*8+index2, :, 32+j:96+j, 32+i:96+i] = 0.0
        #     # (y ln(x) + (1-y) ln(1-x)) (not useful)
        #     # background_diff = -1*(search*torch.log(pred_back + 1e-10) + (1-search)*torch.log(1-(pred_back+ 1e-10)))
        #     background_diff_loss = background_diff.mean()

        #     error_map_back = torch.abs(pred_back - image_batch)
        #     error_map_back[image_batch == 0.0] = 0.0
        #     # dx_b, dy_b = gradient(error_map_back)
        #     # error_map_back = error_map_back.mean(axis = 0)
        #     # error_map_back = error_map_back.mean(axis = 0)
        #     error_map_back = error_map_back.mean(axis = 1)

        #     threshold_map = torch.nn.functional.threshold(error_map_back, self.threshold_for_background, 0.0, inplace=False)
        #     threshold_map[threshold_map >= self.threshold_for_background] = 1.0
        #     threshold_map_mask_center= torch.zeros(threshold_map.shape)
        #     for index1, i in enumerate(range(-32, 32, 8)):
        #         for index2, j in enumerate(range(-32, 32, 8)):
        #             threshold_map_mask_center[index1*8+index2, 32+j:96+j, 32+i:96+i] = threshold_map[index1*8+index2, 32+j:96+j, 32+i:96+i]
        #     threshold_map = threshold_map_mask_center
        #     threshold_map = torch.unsqueeze(threshold_map, 1)
        #     threshold_map = threshold_map.repeat([1, 3, 1, 1])
        #     threshold_map = threshold_map.to(self.device, dtype=self.data_type)

        #     pred_fore, feature_map_fore = self.model_foreground(image_batch)

        #     reconstruction_diff = torch.abs(pred_fore - image_batch) * threshold_map.to(self.device, dtype=self.data_type)
        #     # (y ln(x) + (1-y) ln(1-x)) (not useful)
        #     # reconstruction_diff = -1*(search*torch.log(pred_fore+ 1e-10) + (1-search)*torch.log(1-(pred_fore+ 1e-10)))* threshold_map.to(self.device, dtype=self.data_type)
        #     reconstruction_loss = reconstruction_diff.mean()

        #     error_map_fore = 1.0 - torch.abs(pred_fore - image_batch)
        #     error_map_fore[image_batch == 0.0] = 0.0
        #     # dx, dy = gradient(error_map_fore)
        #     # error_map_fore = error_map_fore.mean(axis = 0)
        #     # error_map_fore = error_map_fore.mean(axis = 0)
        #     error_map_fore = error_map_fore.mean(axis = 1)
        #     # gaussian_error_diff = abs(error_map_fore[32:96, 32:96] - gaussian_map)
        #     # (y ln(x) + (1-y) ln(1-x)) (not useful)
        #     # gaussian_error_diff = -1*(gaussian_map*torch.log(error_map_fore[32:96, 32:96]) + (1-gaussian_map)*torch.log(1-error_map_fore[32:96, 32:96]))
        #     # gaussian_error_loss = gaussian_error_diff.mean()

        #     threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)
        #     # i have no idea 
        #     # threshold_map_fore[threshold_map_fore>=self.threshold_for_foreground] = 1.0

        #     threshold_map = threshold_map.mean(axis = 1)

        #     consistency_diff = torch.abs((threshold_map - threshold_map_fore)).mean()
        #     # (y ln(x) + (1-y) ln(1-x)) (not useful)
        #     # consistency_diff = -1*(threshold_map*torch.log(threshold_map_fore + 1e-10) + (1-threshold_map)*torch.log(1-(threshold_map_fore + 1e-10)))
        #     consistency_loss = consistency_diff.mean()

        #     # dx_c, dy_c = gradient(search)
        #     # dx, dy, dx_c, dy_c, dx_b, dy_b = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
        #     # dx, dy, dx_c, dy_c, dx_b, dy_b = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
        #     # dx_c, dy_c = 1.0-dx_c, 1.0-dy_c

        #     # smooth_loss = ((abs(dx_c*dx)).mean() + (abs(dy_c*dy)).mean()+ (abs(dx_c*dx_b)).mean()+ (abs(dy_c*dy_b)).mean())

        #     # grad_loss = 0.0
        #     # for index1, i in enumerate(range(-32, 32, 8)):
        #     #     for index2, j in enumerate(range(-32, 32, 8)):
        #     #         background_diff[index1*8+index2, :, 32+j:96+j, 32+i:96+i] = 0.0 
        #     #         dx_search, dy_search =  gradient(image_batch[index1*8+index2:index1*8+index2+1, :, 32+j:96+j, 32+i:96+i])
        #     #         dx_pred, dy_pred = gradient(pred[index1*8+index2:index1*8+index2+1, :, 32+j:96+j, 32+i:96+i])
        #     #         grad_loss += abs(dx_search - dx_pred).mean() + abs(dy_search - dy_pred).mean()

        #     loss = background_diff_loss + reconstruction_loss + consistency_loss# + grad_loss

        #     # if i % 100 == 0:
        #     #     print(loss)

        #     loss.backward()
        #     optimizer_back.step()
        #     optimizer_fore.step()
        #     # scheduler.step()
        # print("all finish !!!")

        # # check pred
        # with torch.no_grad():
        #     pred, feature_map = self.model_foreground(image_batch)
        # pred_pil = torchvision.transforms.ToPILImage()(pred[self.check_num].detach().cpu())
        # pred_pil.save("./pred_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # # check error
        # function.write_heat_map(error_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_error_fore_" + str(video_num) + "_")
        # function.write_heat_map(threshold_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_thres_fore_" + str(video_num) + "_")

        # # check threshold
        # function.write_heat_map(error_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_error_back_" + str(video_num) + "_")
        # function.write_heat_map(threshold_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_thres_back_" + str(video_num) + "_")


        # self.threshold_map_save = threshold_map_fore.detach()
        # self.threshold_map_back_save = threshold_map.detach()

        self.count_image += 1
    def tracker_inference(self, img, x, y, w, h, video_count, number_of_frame, video_num):
        # input data init
        data_transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        img = data_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=self.data_type)

        # set model
        self.model_foreground = self.model_foreground.to(self.device, dtype=self.data_type)
        self.model_foreground.eval()

        # Get search grid
        grid = function.get_grid(img.shape[3], img.shape[2], self.x + int(self.w/2), self.y + int(self.h/2), int(3*self.w), int(3*self.h), 192, 192)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid)
        search = search.to(self.device, dtype=self.data_type)

        # inference
        with torch.no_grad():
            pred, feature_map = self.model_foreground(search)

        # check search
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        # search_pil.save("./search_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check pred
        img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # img_pil.save("./pred_fore_img_" + str(video_num) + "_" + str(self.count_image) + ".jpg")
        
        # error map
        error_map = pred
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        # function.write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map
        threshold_map = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        # function.write_heat_map(threshold_map, self.count_image, "./threshold_map_fore_" + str(video_num) + "_")
        
        # Connected component
        # get center
        threshold_map = threshold_map.astype(np.uint8)
        self.x, self.y, self.w, self.h = function.get_obj_x_y_w_h(threshold_map, self.x, self.y, self.w, self.h, img, self.device, self.data_type, self.model_foreground)
        # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_map)
        # lblareas = stats[1:, cv2.CC_STAT_AREA]
        # print(lblareas)
        # pred_center_x, pred_center_y = centroids[np.argmax(np.array(lblareas)) + 1]
        # new_center_x = (pred_center_x*3*self.w/192) + int(self.x - self.w)
        # new_center_y = (pred_center_y*3*self.h/192) + int(self.y - self.h)
        # # get w, h
        # pred_w, pred_h = stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_WIDTH], stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_HEIGHT]
        # new_w, new_h = (pred_w*3*self.w/192), (pred_h*3*self.h/192)
        # self.w = int(new_w)
        # self.h = int(new_h)
        # # get x, y 
        # self.x = int(new_center_x) - int(self.w/2)
        # self.y = int(new_center_y) - int(self.h/2)

        # image_batch
        image_batch = function.get_image_batch_with_translate_augmentation(img, 4, self.x, self.y, self.w, 192, self.h, 192, self.data_type)
        # memory
        self.memory[self.count_image] = image_batch

        function.write_tracking_result(img.detach().cpu().numpy(), self.x, self.y, self.count_image, self.w, self.h, "./" + str(video_num) + "_")

        self.count_image+=1

        return self.x, self.y, self.w, self.h
    def tracker_update(self, img, realx, realy, realw, realh, video_count, number_of_frame, video_num):
        # input data init
        data_transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        img = data_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=self.data_type)
        # model init
        self.model_background = FCNet().to(self.device, dtype=self.data_type)
        self.model_background.train()
        self.model_foreground.train()

        self.training_set = torch.cat((self.memory[0], self.memory[self.count_image-1]), 0)
        self.training_set = self.memory[self.count_image-1]
        self.training_set = self.training_set.to(self.device, dtype=self.data_type)
        # image_batch = image_batch.to(self.device, dtype=self.data_type)
        
        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)

        # train
        for i in range(0, 1, 1):
            # opt init
            optimizer.zero_grad()
            pred, feature_map = self.model_background(self.training_set)
            background_diff = torch.abs(pred - self.training_set)
            for index1, i in enumerate(range(-64, 64, 32)):
                for index2, j in enumerate(range(-64, 64, 32)):
                    background_diff[index1*4+index2, :, 64+j:128+j, 64+i:128+i] = 0.0
                    # background_diff[index1*4+index2+16, :, 64+j:128+j, 64+i:128+i] = 0.0
            background_diff_loss = background_diff.mean()
            loss = background_diff_loss
            loss.backward()
            optimizer.step()
        print("background finish !!!")

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_background(self.training_set)
        pred_pil = torchvision.transforms.ToPILImage()(pred[self.check_num].detach().cpu())
        # pred_pil.save("./pred_img_with_background_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        error_map = torch.abs(pred - self.training_set)
        error_map = error_map.mean(axis = 1)
        # function.write_heat_map(error_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_background_" + str(video_num) + "_")

        gaussian_map = torch.from_numpy(g_kernel)
        gaussian_map = gaussian_map.to(self.device, dtype=self.data_type)
        gaussian_map = torch.unsqueeze(gaussian_map, 0)

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        for index1, i in enumerate(range(-64, 64, 32)):
            for index2, j in enumerate(range(-64, 64, 32)):
                threshold_map_mask_center[index1*4+index2, 64+j:128+j, 64+i:128+i] = gaussian_map#  *threshold_map[index1*4+index2, 64+j:128+j, 64+i:128+i]
                # threshold_map_mask_center[index1*4+index2+16, 64+j:128+j, 64+i:128+i] = gaussian_map# *threshold_map[index1*4+index2+16, 64+j:128+j, 64+i:128+i]
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        # g_filter = torch.from_numpy(g_kernel)
        # g_filter = g_filter.to(dtype=self.data_type)
        # g_filter = torch.unsqueeze(g_filter, 0)
        # g_filter = torch.unsqueeze(g_filter, 0)
        # threshold_map = F.conv2d(threshold_map, g_filter, padding=3)
        # threshold_map[threshold_map > 1.0] = 1.0
        # function.write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        # foreground
        # optimizer init
        optimizer = optim.Adam(self.model_foreground.parameters(), lr = 1e-4)
        # loss function init
        criterion_bec_loss = nn.BCELoss()
        # train
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, feature_map = self.model_foreground(self.training_set)
            bce_loss = criterion_bec_loss(pred, threshold_map.to(self.device, dtype=self.data_type))
            loss = bce_loss
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")
