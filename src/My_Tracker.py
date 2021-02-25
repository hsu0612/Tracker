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
# from gaussian import g_kernel
# check function
def write_heat_map(img, count, write_path):
    img = img*255
    img = img.astype(np.uint8)
    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(write_path + str(count) + ".jpg", im_color)
def write_batch_image(img, batch_index, write_path):
    img = img.cpu().detach().numpy()
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 2, 3)
    for i in range(0, img.shape[0], 1):
        img_temp = cv2.cvtColor(img[i].astype(np.float32), cv2.COLOR_BGR2RGB)
        cv2.imwrite(write_path + str(i + 65*batch_index) + ".jpg", img_temp*255)
def write_batch_pred(img, batch_index, write_path):
    img = img.cpu().detach().numpy()
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 2, 3)
    for i in range(0, img.shape[0], 1):
        img_temp = cv2.cvtColor(img[i].astype(np.float32), cv2.COLOR_BGR2RGB)
        cv2.imwrite(write_path + str(i + 65*batch_index) + ".jpg", img_temp*255)
def write_tracking_result(img, x, y, count, w, h, write_path):
    img = img.cpu().detach().numpy()
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 2, 3)
    img_render = img[0].copy()
    img_render = cv2.cvtColor(img_render.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.rectangle(img_render, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(write_path + str(count) + ".jpg", img_render*255)
# Function
def get_grid(w, h, x, y, crop_w, crop_h, grid_w, grid_h):
    ax = 1 / (w/2)
    bx = -1
    ay = 1 / (h/2)
    by = -1
        
    left_x = x - (crop_w/2)
    right_x = x + (crop_w/2)
    left_y = y - (crop_h/2)
    right_y = y + (crop_h/2)
        
    left_x = left_x*ax + bx
    right_x = right_x*ax + bx
    left_y = left_y*ay + by
    right_y = right_y*ay + by
        
    grid_x = torch.linspace(float(left_x), float(right_x), grid_w)
    grid_y = torch.linspace(float(left_y), float(right_y), grid_h)
        
    meshy, meshx = torch.meshgrid((grid_y, grid_x))
        
    grid = torch.stack((meshx, meshy), 2)
    # add batch dim
    grid = grid.unsqueeze(0)
    
    return grid
def score_sort(x):
    return(int(x["score"]))
def get_normalized_error_map(img):
    # Normalization
    img_max = img.max()
    img_min = img.min()
    if (img_max-img_min) == 0:
        img = np.ones(img.shape)
    else:
        img = (img - img_min)/(img_max-img_min)
        img = 1.0 - img
    return(img)
def get_next_x_y(x, y, pred_center_x, pred_center_y, w, h, old_w, old_h):
    center_x = 63.5
    center_y = 63.5
        
    bias_x = pred_center_x - center_x 
    bias_y = pred_center_y - center_y
        
    diff_w = (old_w - w)/2
    diff_h = (old_h - h)/2
        
    next_x = int(x + diff_w + bias_x*(w/128))
    next_y = int(y + diff_h + bias_y*(h/128))
            
    return int(next_x), int(next_y)
def get_next_x_y_2(w, h, x, y, crop_w, crop_h, grid_w, grid_h):
    ax = 1 / (w/2)
    bx = -1
    ay = 1 / (h/2)
    by = -1
        
    left_x = x - (crop_w/2)
    right_x = x + (crop_w/2)
    left_y = y - (crop_h/2)
    right_y = y + (crop_h/2)
        
    left_x = left_x*ax + bx
    right_x = right_x*ax + bx
    left_y = left_y*ay + by
    right_y = right_y*ay + by

    return left_x, right_x, left_y, right_y
def get_best_center_with_best_scale(scale_error_list, scale_area_list, scale_center_list, scale_index_list):
    # scale error
    scale_error_list = np.array(scale_error_list)
    scale_error_max = scale_error_list.max()
    scale_error_min = scale_error_list.min()
    # check zero divide
    if (scale_error_max - scale_error_min) == 0:
        scale_error_list = np.zeros(scale_error_list.shape)
    else:
        scale_error_list = (scale_error_list - scale_error_min) / (scale_error_max - scale_error_min)
        scale_error_list = 1.0 - scale_error_list
    # scale area       
    scale_area_list = np.array(scale_area_list)
    scale_area_max = scale_area_list.max()
    scale_area_min = scale_area_list.min()
    # checck zero divide
    if (scale_area_max - scale_area_min) == 0:
        scale_area_list = np.zeros(scale_area_list.shape)
    else:
        scale_area_list = (scale_area_list - scale_area_min) / (scale_area_max - scale_area_min)
        scale_area_list = scale_area_list
    # distance
    distance_list = []
    for i in range(0, len(scale_center_list), 1):
        distance_list.append(abs(8 - scale_center_list[i][0]) + abs(8 - scale_center_list[i][1]))
    distance_list = np.array(distance_list)
    distance_list_max = distance_list.max()
    distance_list_min = distance_list.min()
    # check zero divide
    if (distance_list_max - distance_list_min) == 0:
        distance_list = np.zeros(distance_list.shape)
    else:
        distance_list = (distance_list - distance_list_min) / (distance_list_max - distance_list_min)
        distance_list = 1.0 - distance_list
    # result
    result = scale_error_list
    result_list = []
    # rank
    for i in range(0, len(result), 1):
        result_list.append({"score" :result[i], "index": scale_index_list[i], "center": scale_center_list[i]})
    # sort  
    result_list.sort(key=score_sort)
    return result_list[-1]
def gkern(kernlen=17, std=1):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d
def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

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
        self.check_num = 20
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
        # image_batch
        image_batch = torch.zeros(8*8, 3, 128, 128)
        for index1, i in enumerate(range(-64, 64, 16)):
            for index2, j in enumerate(range(-64, 64, 16)):
                # get the cropped img
                grid = function.get_grid(img.shape[3], img.shape[2], x + (w/2) + (-1*i*w/128), y + (h/2) + (-1*j*h/128), (2*w), (2*h), 128, 128)
                grid = grid.to(dtype=self.data_type)
                search = torch.nn.functional.grid_sample(img, grid, mode="nearest")
                search = search.to(dtype=self.data_type)
                image_batch[index1*8+index2] = search
        # memory
        self.memory = torch.zeros(number_of_frame, 8*8, 3, 128, 128)
        self.memory[0] = image_batch
        image_batch = image_batch.to(self.device, dtype=self.data_type)
        # count image
        self.count_image = 0
        self.count_image += 1
        # train
        for i in range(0, 500, 1):
            # optimizer init
            optimizer.zero_grad()
            pred, feature_map = self.model_background(image_batch)
            background_diff = torch.abs(pred - image_batch)
            for index1, i in enumerate(range(-32, 32, 8)):
                for index2, j in enumerate(range(-32, 32, 8)):
                    background_diff[index1*8+index2, :, 32+j:96+j, 32+i:96+i] = 0.0
            background_diff_loss = background_diff.mean()
            loss = background_diff_loss
            loss.backward()
            optimizer.step()
        print("background finish !!!")

        # check image_batch
        # for index1, i in enumerate(range(-64, 64, 16)):
        #     for index2, j in enumerate(range(-64, 64, 16)):
        #         # get the cropped img
        #         search_pil = torchvision.transforms.ToPILImage()(image_batch[index1*8+index2])
        #         search_pil.save("./test" + str(index1*8+index2) +".jpg")

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
        # write_heat_map(error_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_background_" + str(video_num) + "_")

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        for index1, i in enumerate(range(-32, 32, 8)):
            for index2, j in enumerate(range(-32, 32, 8)):
                threshold_map_mask_center[index1*8+index2, 32+j:96+j, 32+i:96+i] = threshold_map[index1*8+index2, 32+j:96+j, 32+i:96+i]
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        g_filter = torch.from_numpy(g_kernel)
        g_filter = g_filter.to(dtype=self.data_type)
        g_filter = torch.unsqueeze(g_filter, 0)
        g_filter = torch.unsqueeze(g_filter, 0)
        threshold_map = F.conv2d(threshold_map, g_filter, padding=3)
        threshold_map[threshold_map > 1.0] = 1.0
        # write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        # check mask
        # mask = np.zeros((128, 128, 3))
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
        # threshold map init
        # threshold_map = torch.unsqueeze(threshold_map, 1)
        self.first_threshold_map = threshold_map
        # threshold_map = threshold_map.repeat([1, 3, 1, 1])
        # train
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, feature_map = self.model_foreground(image_batch)
            # reconstruction_diff = torch.abs(pred - image_batch) * threshold_map.to(self.device, dtype=self.data_type)
            bce_loss = criterion_bec_loss(pred, threshold_map.to(self.device, dtype=self.data_type))
            # (y ln(x) + (1-y) ln(1-x)) (not useful)
            # reconstruction_diff = -1*(search*torch.log(pred + 1e-10) + (1-search)*torch.log(1-pred + 1e-10))* threshold_map.to(self.device, dtype=self.data_type)
            # reconstruction_loss = reconstruction_diff.mean()

            # background_diff = 1.0 - torch.abs(pred - image_batch)
            # # (y ln(x) + (1-y) ln(1-x)) (not useful)
            # background_diff = -1*(search*torch.log(pred+1e-10) + (1-search)*torch.log(1-(pred + 1e-10)))
            # background_diff = 1.0 - background_diff
            # background_diff[:, :, 32:96, 32:96] = 0.0
            # background_diff_loss = background_diff.mean()
            # error_map_fore = 1.0 - torch.abs(pred[:, :, 32:96, 32:96] - search[:, :, 32:96, 32:96])
            # error_map_fore = error_map_fore.mean(axis = 0)
            # error_map_fore = error_map_fore.mean(axis = 0)
            # gaussian_error_loss = abs(error_map_fore - gaussian_map).mean()
            # (y ln(x) + (1-y) ln(1-x)) (not useful)
            # gaussian_error_diff = -1*(gaussian_map*torch.log(error_map_fore) + (1-gaussian_map)*torch.log(1-error_map_fore))
            # gaussian_error_loss = gaussian_error_diff.mean()
            # grad_loss = 0.0
            # for index1, i in enumerate(range(-32, 32, 8)):
            #     for index2, j in enumerate(range(-32, 32, 8)):
            #         background_diff[index1*8+index2, :, 32+j:96+j, 32+i:96+i] = 0.0 
            #         dx_search, dy_search =  gradient(image_batch[index1*8+index2:index1*8+index2+1, :, 32+j:96+j, 32+i:96+i])
            #         dx_pred, dy_pred = gradient(pred[index1*8+index2:index1*8+index2+1, :, 32+j:96+j, 32+i:96+i])
            #         grad_loss += abs(dx_search - dx_pred).mean() + abs(dy_search - dy_pred).mean()
            # loss = reconstruction_loss# + grad_loss
            loss = bce_loss
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")

        # check pred
        # with torch.no_grad():
        #     pred, feature_map = self.model_foreground(image_batch)
        # pred_pil = torchvision.transforms.ToPILImage()(pred[self.check_num].detach().cpu())
        # pred_pil.save("./pred_img_with_foreground_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # # check error map
        # error_map_fore = 1.0 - torch.abs(pred - image_batch)
        # # error_map_fore = error_map_fore.mean(axis = 0)
        # # error_map_fore = error_map_fore.mean(axis = 0)
        # error_map_fore = error_map_fore.mean(axis = 1)
        # write_heat_map(error_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_foregroud" + str(video_num) + "_")

        # # check threshold map
        # threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)
        # threshold_map_fore[threshold_map_fore!=0.0] = 1.0
        # write_heat_map(threshold_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./threshold_foregroud" + str(video_num) + "_")

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
        # write_heat_map(error_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_error_fore_" + str(video_num) + "_")
        # write_heat_map(threshold_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_thres_fore_" + str(video_num) + "_")

        # # check threshold
        # write_heat_map(error_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_error_back_" + str(video_num) + "_")
        # write_heat_map(threshold_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./final_thres_back_" + str(video_num) + "_")


        # self.threshold_map_save = threshold_map_fore.detach()
        # self.threshold_map_back_save = threshold_map.detach()

        # set model
        self.model_foreground = self.model_foreground.to(self.device, dtype=self.data_type)
        self.model_foreground.eval()

        # Get search grid
        grid = get_grid(img.shape[3], img.shape[2], self.x + int(self.w/2), self.y + int(self.h/2), int(self.w), int(self.h), 64, 64)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid)
        search = search.to(self.device, dtype=self.data_type)

        # inference
        with torch.no_grad():
            pred, feature_map = self.model_foreground(search)

        # error map
        self.error_reference = pred.mean()

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
        grid = get_grid(img.shape[3], img.shape[2], self.x + int(self.w/2), self.y + int(self.h/2), int(3*self.w), int(3*self.h), 192, 192)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid)
        search = search.to(self.device, dtype=self.data_type)

        # inference
        with torch.no_grad():
            pred, feature_map = self.model_foreground(search)

        # check search
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        search_pil.save("./search_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check pred
        # img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # img_pil.save("./pred_fore_img_" + str(video_num) + "_" + str(self.count_image) + ".jpg")
        
        # error map
        error_map = pred
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map
        threshold_map = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        write_heat_map(threshold_map, self.count_image, "./threshold_map_fore_" + str(video_num) + "_")
        
        # Connected component
        # get center
        threshold_map = threshold_map.astype(np.uint8)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_map)
        lblareas = stats[1:, cv2.CC_STAT_AREA]
        pred_center_x, pred_center_y = centroids[np.argmax(np.array(lblareas)) + 1]
        new_center_x = (pred_center_x*3*self.w/192) + int(self.x - self.w)
        new_center_y = (pred_center_y*3*self.h/192) + int(self.y - self.h)
        # pred_x, pred_y = stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_LEFT], stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_TOP]
        pred_w, pred_h = stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_WIDTH], stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_HEIGHT]
        # new_x, new_y, new_w, new_h = (pred_x*3*self.w/192), (pred_y*3*self.h/192), (pred_w*3*self.w/192), (pred_h*3*self.h/192)
        new_w, new_h = (pred_w*3*self.w/192), (pred_h*3*self.h/192)
        # new_x = int(new_x) + int(self.x - self.w)
        # new_y = int(new_y) + int(self.y - self.h)
        # self.x = int(new_center_x) - int(self.w/2)
        # self.y = int(new_center_y) - int(self.h/2)

        # # scale list
        # factor_list = np.array([1.0, 0.98, 1.02, 0.98*0.98, 1.02*1.02])
        # scale_list = np.array(np.meshgrid(factor_list, factor_list))
        # scale_list = scale_list.T.reshape(25, 2)
        # # confidence score
        # confidence_score = np.zeros(25)
        # for index, scale in enumerate(scale_list):
        #     # Get search grid
        #     grid = get_grid(img.shape[3], img.shape[2], new_center_x, new_center_y, int(self.w*scale[0]), int(self.h*scale[1]), 64, 64)
        #     grid = grid.to(dtype=self.data_type)
        #     search = torch.nn.functional.grid_sample(img, grid)
        #     search = search.to(self.device, dtype=self.data_type)
        #     # inference
        #     with torch.no_grad():
        #         pred, feature_map = self.model_foreground(search)
        #     # error map
        #     error_map = pred
        #     confidence_score[index] = error_map.detach().cpu().detach().numpy().sum()# - self.error_reference
        # # get w, h
        # abs_confidence_score = abs(confidence_score)
        # max_index = np.argmax(abs_confidence_score)
        # best_scale = scale_list[max_index]
        # new_w, new_h = (best_scale[0] * self.w), (best_scale[1] * self.h)
        self.w = int(new_w)
        self.h = int(new_h)
        # get x, y 
        self.x = int(new_center_x) - int(self.w/2)
        self.y = int(new_center_y) - int(self.h/2)
        #self.error_reference = confidence_score[max_index] + self.error_reference

        write_tracking_result(img, self.x, self.y, self.count_image, self.w, self.h, "./" + str(video_num) + "_")
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
        # self.model_foreground = Discriminator().to(self.device, dtype=self.data_type)
        self.model_foreground.train()

        # image_batch
        image_batch = torch.zeros(8*8, 3, 128, 128)
        for index1, i in enumerate(range(-64, 64, 16)):
            for index2, j in enumerate(range(-64, 64, 16)):
                # get the cropped img
                grid = function.get_grid(img.shape[3], img.shape[2],
                    self.x + (self.w/2) + (-1*i*self.w/128), self.y + (self.h/2) + (-1*j*self.h/128), (2*self.w), (2*self.h), 128, 128)
                grid = grid.to(dtype=self.data_type)
                search = torch.nn.functional.grid_sample(img, grid, mode="nearest")
                search = search.to(dtype=self.data_type)
                image_batch[index1*8+index2] = search
        # memory
        self.memory[self.count_image-1] = image_batch

        self.training_set = torch.cat((self.memory[0], self.memory[self.count_image-1]), 0)
        # self.training_set = image_batch
        self.training_set = self.training_set.to(self.device, dtype=self.data_type)
        # image_batch = image_batch.to(self.device, dtype=self.data_type)
        
        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)

        # train
        for i in range(0, 500, 1):
            # opt init
            optimizer.zero_grad()
            pred, feature_map = self.model_background(self.training_set)
            background_diff = torch.abs(pred - self.training_set)
            for index1, i in enumerate(range(-32, 32, 8)):
                for index2, j in enumerate(range(-32, 32, 8)):
                    background_diff[index1*8+index2, :, 32+j:96+j, 32+i:96+i] = 0.0
                    background_diff[index1*8+index2+64, :, 32+j:96+j, 32+i:96+i] = 0.0
            background_diff_loss = background_diff.mean()
            loss = background_diff_loss
            loss.backward()
            optimizer.step()
        print("background finish !!!")

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_background(self.training_set)

        # check error map
        error_map = torch.abs(pred - self.training_set)
        error_map = error_map.mean(axis = 1)
        write_heat_map(error_map[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_background_" + str(video_num) + "_")

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        for index1, i in enumerate(range(-32, 32, 8)):
            for index2, j in enumerate(range(-32, 32, 8)):
                threshold_map_mask_center[index1*8+index2, 32+j:96+j, 32+i:96+i] = threshold_map[index1*8+index2, 32+j:96+j, 32+i:96+i]
                threshold_map_mask_center[index1*8+index2+64, 32+j:96+j, 32+i:96+i] = threshold_map[index1*8+index2+64, 32+j:96+j, 32+i:96+i]
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        g_filter = torch.from_numpy(g_kernel)
        g_filter = g_filter.to(dtype=self.data_type)
        g_filter = torch.unsqueeze(g_filter, 0)
        g_filter = torch.unsqueeze(g_filter, 0)
        threshold_map = F.conv2d(threshold_map, g_filter, padding=3)
        threshold_map[threshold_map > 1.0] = 1.0
        write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        # foreground

        # optimizer init
        optimizer = optim.Adam(self.model_foreground.parameters(), lr = 1e-4)

        # loss function init
        criterion_bec_loss = nn.BCELoss()

        # threshold map init
        # threshold_map = threshold_map.repeat([1, 3, 1, 1])

        # train
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, feature_map = self.model_foreground(self.training_set)
            # reconstruction_diff = torch.abs(pred - image_batch) * threshold_map.to(self.device, dtype=self.data_type)
            bce_loss = criterion_bec_loss(pred, threshold_map.to(self.device, dtype=self.data_type))
            # (y ln(x) + (1-y) ln(1-x)) (not useful)
            # reconstruction_diff = -1*(search*torch.log(pred + 1e-10) + (1-search)*torch.log(1-pred + 1e-10))* threshold_map.to(self.device, dtype=self.data_type)
            # reconstruction_loss = reconstruction_diff.mean()

            # background_diff = 1.0 - torch.abs(pred - image_batch)
            # # (y ln(x) + (1-y) ln(1-x)) (not useful)
            # background_diff = -1*(search*torch.log(pred+1e-10) + (1-search)*torch.log(1-(pred + 1e-10)))
            # background_diff = 1.0 - background_diff
            # background_diff[:, :, 32:96, 32:96] = 0.0
            # background_diff_loss = background_diff.mean()
            # error_map_fore = 1.0 - torch.abs(pred[:, :, 32:96, 32:96] - search[:, :, 32:96, 32:96])
            # error_map_fore = error_map_fore.mean(axis = 0)
            # error_map_fore = error_map_fore.mean(axis = 0)
            # gaussian_error_loss = abs(error_map_fore - gaussian_map).mean()
            # (y ln(x) + (1-y) ln(1-x)) (not useful)
            # gaussian_error_diff = -1*(gaussian_map*torch.log(error_map_fore) + (1-gaussian_map)*torch.log(1-error_map_fore))
            # gaussian_error_loss = gaussian_error_diff.mean()
            # grad_loss = 0.0
            # for index1, i in enumerate(range(-32, 32, 8)):
            #     for index2, j in enumerate(range(-32, 32, 8)):
            #         background_diff[index1*8+index2, :, 32+j:96+j, 32+i:96+i] = 0.0 
            #         dx_search, dy_search =  gradient(image_batch[index1*8+index2:index1*8+index2+1, :, 32+j:96+j, 32+i:96+i])
            #         dx_pred, dy_pred = gradient(pred[index1*8+index2:index1*8+index2+1, :, 32+j:96+j, 32+i:96+i])
            #         grad_loss += abs(dx_search - dx_pred).mean() + abs(dy_search - dy_pred).mean()
            # loss = reconstruction_loss# + grad_loss
            loss = bce_loss
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")
        