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
from src.model import Net
from src.model import VAENet
from src.model import FCNet
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
    def __init__(self, device):
        # set random seed
        np.random.seed(999)
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        torch.backends.cudnn.deterministic = True
        self.device = device
        self.data_type = torch.float32
        self.threshold_for_background = 0.1
        self.threshold_for_foreground = 0.8
    def tracker_init(self, img, x, y, w, h, number_of_frame, video_num):
        # input data init
        train_transformation = transforms.Compose([
            transforms.ToTensor(),
            ])
        #img = img.filter(ImageFilter.GaussianBlur(radius=3))
        img = train_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=self.data_type)
        # set x, y, w, h 
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        # model init
        self.model_foreground = FCNet().to(self.device, dtype=self.data_type)
        self.model_foreground.train()
        self.model_background = FCNet().to(self.device, dtype=self.data_type)
        self.model_background.train()
        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)
        # get the cropped img
        grid = function.get_grid(img.shape[3], img.shape[2], x + int(w/2), y + int(h/2), int(2*w), int(2*h), 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid, mode="nearest")
        search = search.to(self.device, dtype=self.data_type)
        # count image
        self.count_image = 0
        # iter
        for i in range(0, 500, 1):
            # opt init
            optimizer.zero_grad()
            pred, feature_map = self.model_background(search)
            # loss
            background_diff = torch.abs(pred - search)
            background_diff[:, :, 32:96, 32:96] = 0.0
            background_diff_loss = background_diff.mean()
            loss = background_diff_loss
            loss.backward()
            optimizer.step()
        print("background finish !!!")
        
        # check search
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        # search_pil.save("./search" + str(self.count_image) + str(video_num) + ".jpg")
        search_np = np.array(search_pil)

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_background(search)
        # pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # pred_pil.save("./pred_img_with_background" + str(self.count_image) + str(video_num) + ".jpg")

        # check error map
        error_map = torch.abs(pred - search)
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        # write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_background_" + str(video_num))

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        threshold_map_mask_center[32:96, 32:96] = threshold_map[32:96, 32:96]
        threshold_map = threshold_map_mask_center
        # write_heat_map(threshold_map.detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num))

        # check mask
        mask = np.zeros((128, 128, 3))
        for i in range(0, 3, 1):
            mask[:, :, i] = np.where(threshold_map.detach().cpu().detach().numpy() == 1.0, search_np[:, :, i], 0.0)

        search_with_mask = Image.fromarray(mask.astype("uint8"))
        search_with_mask = train_transformation(search_with_mask)
        search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
        #search_with_mask.save("./mask_" + str(video_num) + ".jpg")

        gaussian_map = g_kernel
        gaussian_map = torch.tensor(gaussian_map).to(self.device, dtype=self.data_type)

        # foreground
        optimizer = optim.Adam(self.model_foreground.parameters(), lr = 1e-4)
        # iter
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, feature_map = self.model_foreground(search)
            # loss
            reconstruction_diff = torch.abs(pred[:, :, 32:96, 32:96] - search[:, :, 32:96, 32:96])
            reconstruction_loss = reconstruction_diff.mean()
            error_map_fore = 1.0 - torch.abs(pred[:, :, 32:96, 32:96] - search[:, :, 32:96, 32:96])
            error_map_fore = error_map_fore.mean(axis = 0)
            error_map_fore = error_map_fore.mean(axis = 0)
            gaussian_error_loss = abs(error_map_fore - gaussian_map).mean()
            loss = 0.5*reconstruction_loss + 0.5*gaussian_error_loss
            loss.backward()
            optimizer.step()

        print("foreground finish !!!")
        
        with torch.no_grad():
            pred, feature_map = self.model_foreground(search)
        
        # pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # pred_pil.save("./pred_img_with_foreground" + str(self.count_image) + str(video_num) + ".jpg")

        # check error map
        error_map_fore = 1.0 - torch.abs(pred - search)
        error_map_fore = error_map_fore.mean(axis = 0)
        error_map_fore = error_map_fore.mean(axis = 0)
        # write_heat_map(error_map_fore.detach().cpu().detach().numpy(), self.count_image, "./error_foregroud" + str(video_num))

        # check threshold map
        threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)
        threshold_map_fore[threshold_map_fore!=0.0] = 1.0
        # write_heat_map(threshold_map_fore.detach().cpu().detach().numpy(), self.count_image, "./threshold_foregroud" + str(video_num))

        # second stage
        # optimizer init
        optimizer_back = optim.Adam(list(self.model_background.parameters()), lr = 1e-4)
        optimizer_fore = optim.Adam(list(self.model_foreground.parameters()), lr = 1e-4)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
        # iter
        for i in range(0, 500, 1):
            optimizer_back.zero_grad()
            optimizer_fore.zero_grad()

            pred_back, feature_map = self.model_background(search)
            error_map_back = torch.abs(pred_back - search)
            dx_b, dy_b = gradient(error_map_back)
            error_map_back = error_map_back.mean(axis = 0)
            error_map_back = error_map_back.mean(axis = 0)

            background_diff = torch.abs(pred_back - search)
            background_diff[:, :, 32:96, 32:96] = 0.0
            background_diff_loss = background_diff.mean()

            threshold_map = torch.nn.functional.threshold(error_map_back, self.threshold_for_background, 0.0, inplace=False)
            threshold_map[threshold_map>=self.threshold_for_background] = 1.0
            threshold_map_temp= torch.zeros(threshold_map.shape)
            threshold_map_temp[32:96, 32:96] = threshold_map[32:96, 32:96]
            threshold_map = threshold_map_temp
            threshold_map = threshold_map.to(self.device, dtype=self.data_type)

            pred_fore, feature_map_fore = self.model_foreground(search)
            error_map_fore = torch.abs(pred_fore - search)
            dx, dy = gradient(error_map_fore)
            error_map_fore = torch.squeeze(error_map_fore)
            error_map_fore = error_map_fore.mean(axis = 0)
            error_map_fore = 1.0 - error_map_fore

            reconstruction_diff = torch.abs(pred_fore - search) * threshold_map.to(self.device, dtype=self.data_type)
            reconstruction_diff = torch.squeeze(reconstruction_diff)
            reconstruction_loss = reconstruction_diff.mean()

            threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)
            # i have no idea 
            # threshold_map_fore[threshold_map_fore>=self.threshold_for_foreground] = 1.0

            consistency_loss = torch.abs((threshold_map - threshold_map_fore)).mean()

            dx_c, dy_c = gradient(search)
            dx, dy, dx_c, dy_c, dx_b, dy_b = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
            dx, dy, dx_c, dy_c, dx_b, dy_b = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
            dx_c, dy_c = 1.0-dx_c, 1.0-dy_c

            smooth_loss = ((abs(dx_c*dx)).mean() + (abs(dy_c*dy)).mean()+ (abs(dx_c*dx_b)).mean()+ (abs(dy_c*dy_b)).mean())

            loss = background_diff_loss + reconstruction_loss + consistency_loss# + 0.1*smooth_loss
            if i % 100 == 0:
                print(consistency_loss)

            loss.backward()
            optimizer_back.step()
            optimizer_fore.step()
            # scheduler.step()

        print("all finish !!!")

        self.threshold_map_save = threshold_map_fore.detach()
        self.threshold_map_back_save = threshold_map.detach()

        write_heat_map(error_map_fore.detach().cpu().detach().numpy(), self.count_image, "./final_error" + str(video_num))
        write_heat_map(threshold_map_fore.detach().cpu().detach().numpy(), self.count_image, "./final_thres" + str(video_num))


        write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./final_error_back" + str(video_num))
        write_heat_map(threshold_map.detach().cpu().detach().numpy(), self.count_image, "./final_thres_back" + str(video_num))

        # with torch.no_grad():
        #     pred, feature_map = self.model_foreground(search)
        # pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # pred_pil.save("./pred" + str(self.count_image) + str(video_num) + ".jpg")

        # memory
        self.memory = torch.zeros((number_of_frame, 3, 128, 128))
        self.memory[self.count_image] = search[0]
        # count for image
        self.count_image += 1
    def tracker_inference(self, img, x, y, w, h, video_count, number_of_frame, video_num):
        # set model
        self.model_background = self.model_background.to(self.device, dtype=self.data_type)
        self.model_background.eval()
        self.model_foreground = self.model_foreground.to(self.device, dtype=self.data_type)
        self.model_foreground.eval()
        test_transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        #img = img.filter(ImageFilter.GaussianBlur(radius=3))
        img = test_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=self.data_type)
        # Get search grid
        grid = get_grid(img.shape[3], img.shape[2], self.x + int(self.w/2), self.y + int(self.h/2), int(2*self.w), int(2*self.h), 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid)
        search = search.to(self.device, dtype=self.data_type)
        with torch.no_grad():
            pred, feature_map = self.model_foreground(search)
        img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        img_pil.save("./pred_img" + str(self.count_image) + "_" + str(video_num) + ".jpg")
        # error map
        error_map = torch.abs(pred - search)
        error_map = torch.squeeze(error_map)
        error_map = 1.0 - error_map.mean(axis = 0)
        # maybe has error
        error_map = np.where(search.detach().cpu().mean(axis = 1)[0, :, :] == 0, 0.0, error_map.detach().cpu())
        write_heat_map(error_map, self.count_image, "./error_map" + str(video_num))
        threshold_map = np.where(error_map > self.threshold_for_foreground, 1.0, 0.0)
        write_heat_map(threshold_map, self.count_image, "./threshold_map" + str(video_num))
        
        # Connected component
        threshold_map = threshold_map.astype(np.uint8)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_map)
        lblareas = stats[1:, cv2.CC_STAT_AREA]
        #pred_x = centroids[np.argmax(np.array(lblareas)) + 1][0]
        #pred_y = centroids[np.argmax(np.array(lblareas)) + 1][1]
        pred_x, pred_y = stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_LEFT], stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_TOP]
        pred_w, pred_h = stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_WIDTH], stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_HEIGHT]

        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        draw = ImageDraw.Draw(search_pil)
        #draw.rectangle((pred_x-int(pred_w/2), pred_y-int(pred_h/2), pred_x+int(pred_w/2), pred_y+int(pred_h/2)), outline=(255, 0, 0))
        draw.rectangle((int(pred_x), int(pred_y), int(pred_x + pred_w), int(pred_y + pred_h)), outline=(255, 0, 0))
        search_pil.save("./tracking_result" + str(self.count_image) + "_"  + str(video_num) + ".jpg")
        
        new_x, new_y, new_w, new_h = (pred_x*2*self.w / 128), (pred_y*2*self.h / 128), (pred_w * 2*self.w / 128), (pred_h * 2*self.h / 128)
        #new_x = int(new_x - new_w/2 + self.x - self.w/2)
        #new_y = int(new_y - new_h/2 + self.y - self.h/2)
        #pred_center_x = centroids[np.argmax(np.array(lblareas)) + 1][0]
        #pred_center_y = centroids[np.argmax(np.array(lblareas)) + 1][1]
        new_x = int(new_x) + int(self.x - self.w/2)
        new_y = int(new_y) + int(self.y - self.h/2)
        # pred_center_x = int(new_x + self.x)
        # pred_center_y = int(new_y + self.y)
        new_w = int(new_w)
        new_h = int(new_h)
        
        # # scale list
        # factor_list = np.array([-0.025, 0.025, 0.0, -0.05, 0.05])
        # scale_list = np.array(np.meshgrid(factor_list, factor_list))
        # scale_list = scale_list.T.reshape(25, 2)

        # location_list = []
        # for i, scale in enumerate(scale_list):
        #     grid = get_grid(img.shape[3], img.shape[2], self.x + scale[0]*img.shape[3] + int(self.w/2), self.y+scale[1]*img.shape[2]+ int(self.h/2), int(2*self.w), int(2*self.h), 128, 128)
        #     grid = grid.to(dtype=self.data_type)
        #     search = torch.nn.functional.grid_sample(img, grid)
        #     search = search.to(self.device, dtype=self.data_type)
        #     test = torch.ones(search.shape)
        #     test = test.to(self.device, dtype=self.data_type)
        #     with torch.no_grad():
        #         pred, feature_map = self.model_background(search)
        #         pred2, feature_map2 = self.model_foreground(pred)
        #     img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        #     img_pil.save("./pred_img_" + str(i) + ".jpg")
        #     # error map
        #     error_map = torch.abs(pred - search)
        #     error_map = torch.squeeze(error_map)
        #     error_map = error_map.mean(axis = 0)
        #     feature_error = abs(feature_map - feature_map2)
        #     feature_error = torch.squeeze(feature_error)
        #     feature_error = 1.0 - feature_error.mean(axis = 0)

        #     write_heat_map(error_map.detach().cpu().numpy(), str(i), "./error_map")
        #     location_list.append(error_map[32:96, 32:96].detach().cpu().numpy().mean())
        # location_list = np.array(location_list)
        # location_list = (location_list - location_list.min()) / (location_list.max() - location_list.min())
        # location_list = location_list / location_list.sum()
        # final_x = self.x
        # final_y = self.y
        # for i, scale in enumerate(location_list):
        #     final_x += scale_list[i][0]*scale**img.shape[3]
        #     final_y += scale_list[i][1]*scale**img.shape[2]
            
        # print(self.x)
        # print(self.y)

        # self.x = int(final_x)
        # self.y = int(final_y)

        # print(self.x)
        # print(self.y)

        # scale list
        # factor_list = np.array([1.0, 0.9, 1.1, 0.9*0.9, 1.1*1.1])
        # scale_list = np.array(np.meshgrid(factor_list, factor_list))
        # scale_list = scale_list.T.reshape(25, 2)

        # error_list = []
        # for i, scale in enumerate(scale_list):
        #     grid = get_grid(img.shape[3], img.shape[2], self.x, self.y, int(2*self.w*scale[0]), int(2*self.h*scale[1]), 128, 128)
        #     grid = grid.to(dtype=self.data_type)
        #     search = torch.nn.functional.grid_sample(img, grid)
        #     search = search.to(self.device, dtype=self.data_type)
        #     with torch.no_grad():
        #         pred, feature_map = self.model_foreground(search)
        #     # error map
        #     error_map = torch.abs(pred - search)
        #     error_map = torch.squeeze(error_map)
        #     error_map = 1.0 - error_map.mean(axis = 0)
        #     #write_heat_map(error_map.detach().cpu().numpy(), str(i), "./error_map")
        #     error_list.append(error_map[32:96, 32:96].detach().cpu().numpy().mean())
        # error_list = np.array(error_list)
        # error_list = (error_list - error_list.min()) / (error_list.max() - error_list.min())
        # error_list = error_list / error_list.sum()
        # final_w = 0.0
        # final_h = 0.0
        # for i, scale in enumerate(error_list):
        #     final_w += self.w*scale_list[i][0]*scale
        #     final_h += self.h*scale_list[i][1]*scale

        # print(self.w)
        # print(self.h)

        # self.w = int(final_w)
        # self.h = int(final_h)

        # print(self.w)
        # print(self.h)

        self.x = new_x
        self.y = new_y
        self.w = new_w
        self.h = new_h

        # memory
        # Get search grid
        grid = get_grid(img.shape[3], img.shape[2], self.x + int(self.w/2), self.y + int(self.h/2), int(2*self.w), int(2*self.h), 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid)
        self.memory[self.count_image] = search[0]

        write_tracking_result(img, self.x, self.y, self.count_image, self.w, self.h, "./" + str(video_num))
        self.count_image+=1

        return self.x, self.y, self.w, self.h
    def tracker_update(self, img, realx, realy, realw, realh, video_count, number_of_frame, video_num):
        # model init
        self.model_background = FCNet().to(self.device, dtype=self.data_type)
        self.model_background.train()
        self.model_foreground = FCNet().to(self.device, dtype=self.data_type)
        self.model_foreground.train()
        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)
        self.training_set = self.memory[self.count_image-1:self.count_image]
        self.training_set = self.training_set.to(self.device, dtype=self.data_type)

        # iter
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, feature_map = self.model_background(self.training_set)
            # loss
            background_diff = torch.abs(pred - self.training_set)
            background_diff[:, :, 32:96, 32:96] = 0.0
            background_diff_loss = background_diff.mean()

            loss = background_diff_loss
            # check loss
            # if i % 100 == 0:
            #     print("back")
            #     print(smooth_loss)
            loss.backward()
            optimizer.step()

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_background(self.training_set)
        img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        img_pil.save("./pred_img_with_background" + str(self.count_image) + ".jpg")

        # check error map
        error_map = torch.abs(pred - self.training_set)
        error_map = error_map.mean(axis = 1)
        #write_heat_map(error_map[0].detach().cpu().detach().numpy(), self.count_image, "./error_background")

        # check threshold map
        threshold_map = torch.nn.functional.threshold(error_map, self.threshold_for_background, 0.0, inplace=False)
        threshold_map[threshold_map!=0.0] = 1.0

        write_heat_map(threshold_map[0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background" + str(video_num))
        threshold_map_mask_center= torch.zeros(threshold_map.shape)
        threshold_map_mask_center[:, 32:96, 32:96] = threshold_map[:, 32:96, 32:96]
        threshold_map=threshold_map_mask_center

        # check search
        # img_pil = torchvision.transforms.ToPILImage()(self.training_set[0].detach().cpu())
        # img_pil.save("./search" + str(self.count_image) + ".jpg")
        # img_pil = np.array(img_pil)

        # get mask
        # mask = np.zeros((128, 128, 3))
        # for i in range(0, 3, 1):
        #     mask[:, :, i] = np.where(threshold_map[0].detach().cpu().numpy() == 1.0, img_pil[:, :, i], 0.0)
        
        # check mask
        # cv2.imwrite("./mask.jpg", mask)

        # foreground
        optimizer = optim.Adam(self.model_foreground.parameters(), lr = 1e-4)
        gaussian_map = g_kernel
        gaussian_map = torch.tensor(gaussian_map).to(self.device, dtype=self.data_type)
        # iter
        for i in range(0, 500, 1):
            optimizer.zero_grad()
            pred, feature_map = self.model_foreground(self.training_set)
            # loss
            reconstruction_diff = torch.abs(pred[:, :, 32:96, 32:96] - self.training_set[:, :, 32:96, 32:96])
            reconstruction_loss = reconstruction_diff.mean()
            error_map_fore = 1.0 - torch.abs(pred[:, :, 32:96, 32:96] - self.training_set[:, :, 32:96, 32:96])
            error_map_fore = error_map_fore.mean(axis = 0)
            error_map_fore = error_map_fore.mean(axis = 0)
            gaussian_error_loss = abs(error_map_fore - gaussian_map).mean()
            loss = 0.5*reconstruction_loss + 0.5*gaussian_error_loss
            loss.backward()
            optimizer.step()

        print("back&fore finish !!!")

        with torch.no_grad():
            pred, feature_map = self.model_foreground(self.training_set)
        
        img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        img_pil.save("./pred_img_with_foreground" + str(self.count_image) + ".jpg")

        # check error map
        error_map = torch.abs(pred - self.training_set)
        error_map = error_map.mean(axis = 1)
        error_map = 1.0 - error_map

        write_heat_map(error_map[0].detach().cpu().numpy(), self.count_image, "./error_foregroud")

        threshold_map_fore = torch.nn.functional.threshold(error_map, self.threshold_for_foreground, 0.0, inplace=False)
        threshold_map_fore[threshold_map_fore!=0.0] = 1.0
        write_heat_map(threshold_map_fore[0].detach().cpu().numpy(), self.count_image, "./threshold_foregroud")

        # # second stage
        # # optimizer init
        # optimizer_back = optim.Adam(list(self.model_background.parameters()), lr = 1e-4)
        # optimizer_fore = optim.Adam(list(self.model_foreground.parameters()), lr = 1e-4)
        # #scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
        # # iter
        # for i in range(0, 1000, 1):
        #     optimizer_back.zero_grad()
        #     optimizer_fore.zero_grad()

        #     pred_back, feature_map_back = self.model_background(self.training_set)
        #     error_map_back = torch.abs(pred_back - self.training_set)

        #     dx_b, dy_b = gradient(error_map_back)

        #     error_map_back = error_map_back.mean(axis = 1)
        #     # loss
        #     background_diff = torch.abs(pred_back - self.training_set)
        #     background_diff[:, :, 32:96, 32:96] = 0.0
        #     background_diff_loss = background_diff.mean()

        #     threshold_map = torch.nn.functional.threshold(error_map_back, self.threshold_for_background, 0.0, inplace=False)
        #     threshold_map[threshold_map!=0.0] = 1.0
        #     threshold_map_temp= torch.zeros(threshold_map.shape)
        #     threshold_map_temp[:, 32:96, 32:96] = threshold_map[:, 32:96, 32:96]
        #     threshold_map=threshold_map_temp
        #     threshold_map = threshold_map.to(self.device, dtype=self.data_type)

        #     pred_fore, feature_map_fore = self.model_foreground(self.training_set)
        #     error_map_fore = torch.abs(pred_fore - self.training_set)
            
        #     # gaussian_kernel = 1.0 - gkern(kernlen=128, std=1.0)
        #     # gaussian_kernel2= np.zeros(gaussian_kernel.shape)
        #     # gaussian_kernel2[32:96, 32:96] = gaussian_kernel[32:96, 32:96]
        #     # gaussian_kernel = gaussian_kernel2
        #     # gaussian_kernel = torch.Tensor(gaussian_kernel).to(self.device, dtype=self.data_type)
        #     dx, dy = gradient(error_map_fore)
        #     dx_c, dy_c = gradient(self.training_set)
        #     dx, dy, dx_c, dy_c = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
        #     dx, dy, dx_c, dy_c = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
        #     dx_c, dy_c = 1.0-dx_c, 1.0-dy_c

        #     smooth_loss = ((abs(dx_c*dx)).mean() + (abs(dy_c*dy)).mean()+ (abs(dx_c*dx_b)).mean()+ (abs(dy_c*dy_b)).mean())

        #     error_map_fore = error_map_fore.mean(axis = 1)
        #     error_map_fore = 1.0 - error_map_fore

        #     # loss
        #     reconstruction_diff = torch.pow(pred_fore - self.training_set, 2) * torch.unsqueeze(threshold_map, 1).to(self.device, dtype=self.data_type)
        #     reconstruction_diff[:, :] = reconstruction_diff[:, :]

        #     reconstruction_diff = torch.squeeze(reconstruction_diff)
        #     reconstruction_loss = reconstruction_diff.mean()

        #     threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)

        #     consistency_loss = torch.abs((threshold_map - threshold_map_fore)).mean()

        #     area_loss = torch.abs((threshold_map_fore[-1].sum() - self.threshold_map_save.sum())/(128*128))
        #     area_loss2 = torch.abs((threshold_map[-1].sum() - self.threshold_map_back_save.sum())/(128*128))

        #     background_diff_fore = 1.0-torch.abs(pred_fore - self.training_set)
        #     background_diff_fore[:, :, 16:112, 16:112] = 0.0
        #     background_diff_fore_loss = background_diff_fore.mean()

        #     # write_heat_map(error_map_fore[-1].detach().cpu().detach().numpy(), self.count_image, "./final_error" + str(video_num))
        #     # write_heat_map(threshold_map_fore[-1].detach().cpu().detach().numpy(), self.count_image, "./final_thres"+ str(video_num))

        #     # write_heat_map(error_map_back[-1].detach().cpu().detach().numpy(), self.count_image, "./final_error_back" + str(video_num))
        #     # write_heat_map(threshold_map[-1].detach().cpu().detach().numpy(), self.count_image, "./final_thres_back" + str(video_num))

        #     # assert False

        #     loss = background_diff_loss + reconstruction_loss + consistency_loss + area_loss + area_loss2 + 0.1*smooth_loss
        #     # check loss
        #     if i % 100 == 0:
        #         print("loss")
        #         print(loss)
        #     loss.backward()
        #     if i % 50 == 0 and i !=0:
        #         optimizer_back.step()
        #     optimizer_fore.step()
        #     #scheduler.step()

        # self.threshold_map_save = threshold_map_fore[-1].detach()
        # self.threshold_map_back_save = threshold_map.detach()

        # write_heat_map(error_map_fore[-1].detach().cpu().detach().numpy(), self.count_image, "./final_error" + str(video_num))
        # write_heat_map(threshold_map_fore[-1].detach().cpu().detach().numpy(), self.count_image, "./final_thres"+ str(video_num))

        # write_heat_map(error_map_back[-1].detach().cpu().detach().numpy(), self.count_image, "./final_error_back" + str(video_num))
        # write_heat_map(threshold_map[-1].detach().cpu().detach().numpy(), self.count_image, "./final_thres_back" + str(video_num))


        # with torch.no_grad():
        #     pred, feature_map = self.model_foreground(self.training_set)
        # img_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # img_pil.save("./pred" + str(self.count_image) + ".jpg")
        