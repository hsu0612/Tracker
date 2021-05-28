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
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
# mine
from src.model import FCNet
from src.model import Discriminator
import utils.function as function
from utils.gaussian import g_kernel
from segmentation.grab_cut import Grabcut
# Function
class FCAE_tracker():
    def __init__(self):
        np.random.seed(999)
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        torch.backends.cudnn.deterministic = True
        self.device = 'cuda:0'
        self.data_type = torch.float32
        self.threshold_for_background = 0.05
        self.threshold_for_foreground = 0.5
        # check img num
        self.check_num = 10
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

        # image batch
        image_batch = function.get_image_batch_with_translate_augmentation(img, 4, x, y, w, 128, h, 128, self.data_type)

        grid = function.get_grid(img.shape[3], img.shape[2], x + (w/2), y + (h/2), (2*w), (2*h), 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")
        search = search.to(dtype=self.data_type)
        grid_np = grid.detach().cpu().numpy()
        grid_np = grid_np.squeeze()
        grid_np_x = grid_np[:, :, 0]
        grid_np_y = grid_np[:, :, 1]

        grabcut = Grabcut()
        mask_batch = np.zeros((128, 128, 16))
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        mask = grabcut.get_mask(np.array(search_pil))

        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                mask_batch[32+j:96+j, 32+i:96+i, index1*4+index2] = mask[32:96, 32:96]

        mask_batch = data_transformation(mask_batch)
        mask_batch = mask_batch.unsqueeze(1).to(self.device, dtype=torch.float32)

        self.prvs = cv2.cvtColor(np.array(search_pil), cv2.COLOR_BGR2GRAY)

        # count image
        self.count_image = 0

        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)

        # input for model init
        image_batch = image_batch.to(self.device, dtype=self.data_type)

        # train
        img_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        img_np = np.array(img_pil)
        img_np = img_np.astype(np.uint8)*255

        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (90 + hsv[:, :, 0]) % 180
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for iter in range(0, 1001, 1):
            noise_r = torch.normal(rgb[32:96, 32:96, 0].mean()/255, std=1.0, size=(1, 1, 64, 64)).to(self.device, dtype=torch.float32)
            noise_g = torch.normal(rgb[32:96, 32:96, 1].mean()/255, std=1.0, size=(1, 1, 64, 64)).to(self.device, dtype=torch.float32)
            noise_b = torch.normal(rgb[32:96, 32:96, 2].mean()/255, std=1.0, size=(1, 1, 64, 64)).to(self.device, dtype=torch.float32)
            # optimizer init
            optimizer.zero_grad()                                                                                                                                                                                                                                                                                                                                             
            # background diff
            img_with_noise = image_batch.clone()
            for index1, i in enumerate(range(-32, 32, 16)):
                for index2, j in enumerate(range(-32, 32, 16)):
                    img_with_noise[index1*4+index2, 0, 32+j:96+j, 32+i:96+i] = noise_r
                    img_with_noise[index1*4+index2, 1, 32+j:96+j, 32+i:96+i] = noise_g
                    img_with_noise[index1*4+index2, 2, 32+j:96+j, 32+i:96+i] = noise_b
            pred, feature_map = self.model_background(image_batch)
            background_diff = torch.abs(pred - img_with_noise)
            background_diff_loss = background_diff.mean()
            # mask rec
            pred_mask, feature_map = self.model_background(image_batch)
            mask_rec = 1.0 - torch.abs(pred_mask - image_batch)
            mask_rec = mask_rec * mask_batch
            mask_rec_loss = mask_rec.mean()

            loss = background_diff_loss + mask_rec_loss
            loss.backward()
            optimizer.step()
            if iter % 100 == 0:
                print(loss)
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
            pred, feature_map = self.model_background(search.to(self.device, dtype=self.data_type))
        pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        # pred_pil.save("./pred_img_with_background_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        error_map = torch.abs(pred - search.to(self.device, dtype=self.data_type))
        error_map = error_map.sum(axis = 1)
        error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        # function.write_heat_map(error_map[0].detach().cpu().numpy(), self.count_image, "./error_background_" + str(video_num) + "_")

        # gaussian_map
        gaussian_map = torch.from_numpy(g_kernel)
        gaussian_map = gaussian_map.to(self.device, dtype=self.data_type)
        gaussian_map = torch.unsqueeze(gaussian_map, 0)
        gaussian_map_mask_center= torch.zeros(16, 128, 128)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                gaussian_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = gaussian_map
        gaussian_map_mask_center = torch.unsqueeze(gaussian_map_mask_center, 1)

        # check threshold map
        threshold_map = np.where(error_map.detach().cpu().detach().numpy() > 0.1, 1.0, 0.0)
        threshold_map = np.where(grid_np_x > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_x < -1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y < -1.0, 0.0, threshold_map)
        threshold_map = 255*threshold_map[0].astype(np.uint8)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_map)
        lblareas = stats[1:, cv2.CC_STAT_AREA]
        try:
            mask = np.where(labels == np.argmax(np.array(lblareas))+1, 1.0, 0).astype(np.uint8)
        except:
            mask = np.zeros_like(threshold_map)
        mask_temp = np.zeros_like(mask)
        mask_temp[32:96, 32:96] = mask[32:96, 32:96]
        threshold_map_mask_center= torch.zeros(16, 128, 128)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                threshold_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = torch.Tensor(mask_temp[32:96, 32:96])
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        threshold_map[threshold_map > 1.0] = 1.0
        self.threshold_map_temp = threshold_map.clone()
        # function.write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        if (np.array(lblareas).max() < 2048):
            segments_slic = slic(search_pil, n_segments=50, compactness=10, sigma=1, start_label=1)

            superpixel_mask = np.zeros((128, 128), dtype=int)
            superpixel_mask = superpixel_mask
            superpixel_mask[32:96, 32:96] = segments_slic[32:96, 32:96].copy()

            superpixel_mask_2 = segments_slic.copy()
            superpixel_mask_2[32:96, 32:96] = 0.0

            for i in range(0, segments_slic.max() + 1, 1):
                if np.count_nonzero(superpixel_mask_2 == i) != 0:
                    superpixel_mask[superpixel_mask == i] = 0

            superpixel_mask[superpixel_mask > 0] = 1.0
            superpixel_mask = superpixel_mask.astype(np.float32)

            mask_batch = np.zeros((128, 128, 16))

            for index1, i in enumerate(range(-32, 32, 16)):
                for index2, j in enumerate(range(-32, 32, 16)):
                    mask_batch[32+j:96+j, 32+i:96+i, index1*4+index2] = superpixel_mask[32:96, 32:96]

            mask_batch = data_transformation(mask_batch)
            mask_batch = mask_batch.unsqueeze(1).to(self.device, dtype=torch.float32)
            threshold_map = mask_batch.detach().cpu().clone() + threshold_map.detach().cpu().clone()
            threshold_map[threshold_map>1.0] = 1.0

        # check mask
        mask = np.zeros((128, 128, 3))
        search_np = np.array(search_pil)
        for i in range(0, 3, 1):
            mask[:, :, i] = np.where(threshold_map[self.check_num][0].detach().cpu().detach().numpy() == 1.0, search_np[:, :, i], 0.0)
        search_with_mask = Image.fromarray(mask.astype("uint8"))
        search_with_mask = data_transformation(search_with_mask)
        search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
        # search_with_mask.save("./mask_" + str(video_num) + ".jpg")

        # discreminator

        # optimizer init
        optimizer = optim.Adam(self.model_discriminator.parameters(), lr = 1e-4)

        # loss function init
        criterion_bec_loss = nn.BCELoss()

        # train
        for i in range(0, 501, 1):
            optimizer.zero_grad()
            pred, pred_seg, feature_map = self.model_discriminator(image_batch)
            correlation_loss = criterion_bec_loss(pred, gaussian_map_mask_center.to(self.device, dtype=self.data_type))
            seg_loss = criterion_bec_loss(pred_seg, threshold_map.to(self.device, dtype=self.data_type))
            loss = seg_loss + correlation_loss
            if i %100 == 0:
                print(loss)
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")

        torch.save(self.model_discriminator, "./checkpoint/model_discriminator_save_" + str(video_num) + "_"+ str(self.count_image) + ".pt")
        # self.model_discriminator = torch.load("./checkpoint/model_discriminator_save_" + str(0) + "_"+ str(0) + ".pt")

        # check pred
        with torch.no_grad():
            pred, pred_seg, feature_map = self.model_discriminator(image_batch)
        pred_pil = torchvision.transforms.ToPILImage()(pred_seg[self.check_num].detach().cpu())
        pred_pil.save("./pred_img_with_foreground_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        error_map_fore = pred
        # error_map_fore = error_map_fore.mean(axis = 0)
        # error_map_fore = error_map_fore.mean(axis = 0)
        error_map_fore = error_map_fore.mean(axis = 1)
        function.write_heat_map(error_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./error_foregroud" + str(video_num) + "_")

        # check threshold map
        threshold_map_fore = torch.nn.functional.threshold(error_map_fore, self.threshold_for_foreground, 0.0, inplace=False)
        threshold_map_fore[threshold_map_fore!=0.0] = 1.0
        function.write_heat_map(threshold_map_fore[self.check_num].detach().cpu().detach().numpy(), self.count_image, "./threshold_foregroud" + str(video_num) + "_")

        # cv2.imwrite("./threshold_foregroud.jpg", threshold_map_fore[self.check_num].detach().cpu().detach().numpy()*255)

        self.prev_mask = threshold_map_fore[self.check_num].detach().cpu().detach().numpy()*255

        self.count_image += 1
    def tracker_inference_for_eval(self, img, video_num, flag, realx, realy, realw, realh):
        self.x = realx
        self.y = realy
        # input data init
        data_transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            ])
        img = data_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=self.data_type)

        # set model
        self.model_background = self.model_background.to(self.device, dtype=self.data_type)
        self.model_background.eval()
        self.model_discriminator = self.model_discriminator.to(self.device, dtype=self.data_type)
        self.model_discriminator.eval()

        # Get search grid
        grid = function.get_grid(img.shape[3], img.shape[2], self.x + self.w/2, self.y + self.h/2, 2*self.w, 2*self.h, 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")
        search = search.to(self.device, dtype=self.data_type)

        grid_np = grid.detach().cpu().numpy()
        grid_np = grid_np.squeeze()
        grid_np_x = grid_np[:, :, 0]
        grid_np_y = grid_np[:, :, 1]

        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        segments_slic = slic(search_pil, n_segments=50, compactness=10, sigma=1, start_label=1)

        # print(mark_boundaries(search_pil, segments_slic).dtype)
        # print(np.array(search_pil))

        cv2.imwrite("superpixel"+ str(self.count_image) + ".jpg", mark_boundaries(search_pil, segments_slic)*255)

        self.next = cv2.cvtColor(np.array(search_pil), cv2.COLOR_BGR2GRAY)
        
        hsv = np.zeros_like(search_pil)
        hsv[...,1] = 255
        flow = cv2.calcOpticalFlowFarneback(self.prvs, self.next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        # cv2.imwrite("test.jpg", bgr)

        min_x = 128
        min_y = 128
        max_x = 0
        max_y = 0

        # inference
        with torch.no_grad():
            pred, pred_seg, feature_map = self.model_discriminator(search)
            # pred_seg, feature_map = self.model_background(search)

        # check search
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        search_pil.save("./search_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check pred
        img_pil = torchvision.transforms.ToPILImage()(pred_seg[0].detach().cpu())
        img_pil.save("./pred_fore_img_" + str(video_num) + "_" + str(self.count_image) + ".jpg")
        
        # error map
        error_map = pred
        # error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        function.write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map
        threshold_map = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        threshold_map = np.where(grid_np_x > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_x < -1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y < -1.0, 0.0, threshold_map)
        function.write_heat_map(threshold_map, self.count_image, "./threshold_map_fore_" + str(video_num) + "_")

        # error map with seg
        error_map = pred_seg
        # error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        # function.write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map with seg
        threshold_map_seg = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        threshold_map_seg = np.where(grid_np_x > 1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_x < -1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_y > 1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_y < -1.0, 0.0, threshold_map_seg)
        # function.write_heat_map(threshold_map_seg, self.count_image, "./threshold_map_fore_seg" + str(video_num) + "_")

        # mask = np.zeros((128, 128, 3))
        search_np = np.array(search_pil)
        # for i in range(0, 3, 1):
        #     mask[:, :, i] = np.where(threshold_map_seg == 1.0, search_np[:, :, i], 0.0)
        # search_with_mask = Image.fromarray(mask.astype("uint8"))
        # search_with_mask = data_transformation(search_with_mask)
        # search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
        # search_with_mask.save("./mask_" + str(NUM) + ".jpg")

        final_mask = np.zeros((128, 128))
        final_result = np.zeros((128, 128, 3))
        for i in range(0, segments_slic.max() + 1, 1):
            num_ele = np.count_nonzero(segments_slic == i)
            error_map_copy = threshold_map_seg
            error_map_copy = np.where(segments_slic == i, error_map_copy, 0)
            error_map_copy = (error_map_copy.sum() / num_ele)
            final_mask = np.where(segments_slic == i, error_map_copy, final_mask)

        final_result[:, :, 0] = np.where(final_mask > 0.5, search_np[:, :, 0], 0)
        final_result[:, :, 1] = np.where(final_mask > 0.5, search_np[:, :, 1], 0)
        final_result[:, :, 2] = np.where(final_mask > 0.5, search_np[:, :, 2], 0)

        cv2.imwrite("./superpixel_color"+ str(0) +".jpg", final_mask*255)
        cv2.imwrite("./superpixel_color_result"+ str(0) +".jpg", final_result)
        # assert False

        # threshold_map_seg = final_mask
        # threshold_map_seg = np.where(grid_np_x > 1.0, 0.0, threshold_map_seg)
        # threshold_map_seg = np.where(grid_np_x < -1.0, 0.0, threshold_map_seg)
        # threshold_map_seg = np.where(grid_np_y > 1.0, 0.0, threshold_map_seg)
        # threshold_map_seg = np.where(grid_np_y < -1.0, 0.0, threshold_map_seg)

        optical_result = np.zeros_like(self.prev_mask)
        # for i in range(0, 128, 1):
        #     for j in range(0, 128, 1):
        #         if self.prev_mask[i][j] > 0:
        #             optical_result[int(i + flow[i, j, 1])][int(j + flow[i, j, 0])] = 255
        #             if min_y > int(i + flow[i, j, 1]):
        #                 min_y = int(i + flow[i, j, 1])
        #             if min_x > int(j + flow[i, j, 0]):
        #                 min_x = int(j + flow[i, j, 0])
        #             if max_y < int(i + flow[i, j, 1]):
        #                 max_y = int(i + flow[i, j, 1])
        #             if max_x < int(i + flow[i, j, 0]):
        #                 max_x = int(i + flow[i, j, 0])

        new_x, new_y = (min_x*2*self.w/128) + (self.x - 1/2*self.w), (min_y*2*self.h/128) + (self.y - 1/2*self.h)
        new_w, new_h = ((max_x - min_x)*2*self.w/128), ((max_y - min_y)*2*self.h/128)
        img_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        img_pil_d = ImageDraw.Draw(img_pil)
        img_pil_d.rectangle([min_x, min_y, max_x, max_y], outline ="red")

        img_pil_d.rectangle([32, 32, 96, 96], outline ="green")
        img_pil.save("./mask_" + str(self.count_image) + ".jpg")

        cv2.imwrite("test" + str(self.count_image) + ".jpg", optical_result)

        # self.x, self.y, self.w, self.h = new_x, new_y, new_w, new_h

        self.next = self.prvs
        self.prev_mask = optical_result

        # threshold_map_seg_temp = np.zeros_like(threshold_map_seg)
        # threshold_map_seg_temp[min_x:max_x, min_y:max_y] = threshold_map_seg[min_x:max_x, min_y:max_y]
        # threshold_map_seg = threshold_map_seg_temp
        threshold_map_seg = threshold_map_seg.astype(np.uint8)
        # function.write_heat_map(threshold_map_seg, self.count_image, "./threshold_map_fore_seg" + str(video_num) + "_")

        self.x, self.y, self.w, self.h, flag = function.get_obj_x_y_w_h(threshold_map, threshold_map_seg, self.x, self.y, self.w, self.h, img, self.device, self.data_type, self.model_discriminator, search)
        function.write_tracking_result(img.detach().cpu().numpy(), self.x, self.y, self.count_image, self.w, self.h, "./" + str(video_num) + "_")

        self.count_image+=1

        return self.x, self.y, self.w, self.h

        # cv2.imwrite("test2.jpg", optical_result)
        # assert False

        # superpixel_mask = np.zeros((128, 128), dtype=int)
        # superpixel_mask = superpixel_mask
        # superpixel_mask[32:96, 32:96] = segments_slic[32:96, 32:96].copy()

        # superpixel_mask_2 = segments_slic.copy()
        # superpixel_mask_2[32:96, 32:96] = 0.0

        # for i in range(0, segments_slic.max() + 1, 1):
        #     if np.count_nonzero(superpixel_mask_2 == i) != 0:
        #         superpixel_mask[superpixel_mask == i] = 0

        # superpixel_mask[superpixel_mask > 0] = 1.0
        # superpixel_mask = superpixel_mask.astype(np.float32)

        # mask_batch = np.zeros((128, 128, 16))

        # for index1, i in enumerate(range(-32, 32, 16)):
        #     for index2, j in enumerate(range(-32, 32, 16)):
        #         mask_batch[32+j:96+j, 32+i:96+i, index1*4+index2] = superpixel_mask[32:96, 32:96]

        # mask_batch = data_transformation(mask_batch)
        # mask_batch = mask_batch.unsqueeze(1).to(self.device, dtype=torch.float32)
        # threshold_map = mask_batch.detach().cpu().clone()
        # threshold_map[threshold_map>1.0] = 1.0

        # # check mask
        # mask = np.zeros((128, 128, 3))
        # search_np = np.array(search_pil)
        # for i in range(0, 3, 1):
        #     mask[:, :, i] = np.where(threshold_map[self.check_num][0].detach().cpu().detach().numpy() == 1.0, search_np[:, :, i], 0.0)
        # search_with_mask = Image.fromarray(mask.astype("uint8"))
        # search_with_mask = data_transformation(search_with_mask)
        # search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
        # search_with_mask.save("./mask_" + str(video_num) + ".jpg")

        # assert False

        # img_temp = img[:, :, 300:500, 100:500];    
        # img_temp_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        # img_temp_pil.save("./" + str(self.count_image) + ".jpg")

        # test = torch.nn.functional.grid_sample(self.img_temp, grid, mode="bilinear", padding_mode="zeros")
        # test = test.to(self.device, dtype=self.data_type)

        # # inference
        # with torch.no_grad():
        #     pred, pred_seg, feature_map1 = self.model_discriminator(test)

        # test = torch.nn.functional.grid_sample(img_temp, grid, mode="bilinear", padding_mode="zeros")
        # test = test.to(self.device, dtype=self.data_type)

        # # inference
        # with torch.no_grad():
        #     pred, pred_seg, feature_map2 = self.model_discriminator(test)

        # print((feature_map1 - feature_map2).mean())

        # assert False

        # inference
        with torch.no_grad():
            pred, pred_seg, feature_map = self.model_discriminator(search)
            # pred_seg, feature_map = self.model_background(search)

        # check search
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        search_pil.save("./search_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check pred
        img_pil = torchvision.transforms.ToPILImage()(pred_seg[0].detach().cpu())
        img_pil.save("./pred_fore_img_" + str(video_num) + "_" + str(self.count_image) + ".jpg")
        
        # error map
        error_map = pred
        # error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        # function.write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map
        threshold_map = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        threshold_map = np.where(grid_np_x > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_x < -1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y < -1.0, 0.0, threshold_map)
        # function.write_heat_map(threshold_map, self.count_image, "./threshold_map_fore_" + str(video_num) + "_")

        # error map with seg
        error_map = pred_seg
        # error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map = error_map.mean(axis = 0)
        error_map = error_map.mean(axis = 0)
        function.write_heat_map(error_map.detach().cpu().detach().numpy(), self.count_image, "./error_map_fore_" + str(video_num) + "_")
        
        # threshold map with seg
        threshold_map_seg = np.where(error_map.detach().cpu().detach().numpy() > 0.5, 1.0, 0.0)
        threshold_map_seg = np.where(grid_np_x > 1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_x < -1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_y > 1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_y < -1.0, 0.0, threshold_map_seg)
        function.write_heat_map(threshold_map_seg, self.count_image, "./threshold_map_fore_seg" + str(video_num) + "_")

        # mask = np.zeros((128, 128, 3))
        search_np = np.array(search_pil)
        # for i in range(0, 3, 1):
        #     mask[:, :, i] = np.where(threshold_map_seg == 1.0, search_np[:, :, i], 0.0)
        # search_with_mask = Image.fromarray(mask.astype("uint8"))
        # search_with_mask = data_transformation(search_with_mask)
        # search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
        # search_with_mask.save("./mask_" + str(NUM) + ".jpg")

        final_mask = np.zeros((128, 128))
        final_result = np.zeros((128, 128, 3))
        for i in range(0, segments_slic.max() + 1, 1):
            num_ele = np.count_nonzero(segments_slic == i)
            error_map_copy = threshold_map_seg
            error_map_copy = np.where(segments_slic == i, error_map_copy, 0)
            error_map_copy = (error_map_copy.sum() / num_ele)
            final_mask = np.where(segments_slic == i, error_map_copy, final_mask)

        final_result[:, :, 0] = np.where(final_mask > 0.5, search_np[:, :, 0], 0)
        final_result[:, :, 1] = np.where(final_mask > 0.5, search_np[:, :, 1], 0)
        final_result[:, :, 2] = np.where(final_mask > 0.5, search_np[:, :, 2], 0)

        cv2.imwrite("./superpixel_color"+ str(0) +".jpg", final_mask*255)

        threshold_map_seg = final_mask
        threshold_map_seg = np.where(grid_np_x > 1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_x < -1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_y > 1.0, 0.0, threshold_map_seg)
        threshold_map_seg = np.where(grid_np_y < -1.0, 0.0, threshold_map_seg)

        assert False
        
        # get x, y, w, h
        threshold_map = threshold_map.astype(np.uint8)
        threshold_map_seg = threshold_map_seg.astype(np.uint8)
        self.x, self.y, self.w, self.h, flag = function.get_obj_x_y_w_h(threshold_map, threshold_map_seg, self.x, self.y, self.w, self.h, img, self.device, self.data_type, self.model_discriminator, search)

        # image_batch
        image_batch = function.get_image_batch_with_translate_augmentation(img, 4, self.x, self.y, self.w, 128, self.h, 128, self.data_type)

        # if flag:
        #     # memory
        #     self.memory[self.count_image] = image_batch

        function.write_tracking_result(img.detach().cpu().numpy(), self.x, self.y, self.count_image, self.w, self.h, "./" + str(video_num) + "_")

        self.count_image+=1

        return self.x, self.y, self.w, self.h
    
    def tracker_update(self, img, x, y, w, h, number_of_frame, video_num):
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
        # self.model_discriminator = Discriminator().to(self.device, dtype=self.data_type)
        self.model_discriminator.train()
        self.model_background = FCNet().to(self.device, dtype=self.data_type)
        self.model_background.train()

        grid = function.get_grid(img.shape[3], img.shape[2], x + (w/2), y + (h/2), (2*w), (2*h), 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")
        search = search.to(dtype=self.data_type)
        grid_np = grid.detach().cpu().numpy()
        grid_np = grid_np.squeeze()
        grid_np_x = grid_np[:, :, 0]
        grid_np_y = grid_np[:, :, 1]

        # background

        # image batch
        image_batch = function.get_image_batch_with_translate_augmentation(img, 4, x, y, w, 128, h, 128, self.data_type)

        grabcut = Grabcut()
        mask_batch = np.zeros((128, 128, 16))
        search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        mask = grabcut.get_mask(np.array(search_pil))

        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                mask_batch[32+j:96+j, 32+i:96+i, index1*4+index2] = mask[32:96, 32:96]

        mask_batch = data_transformation(mask_batch)
        mask_batch = mask_batch.unsqueeze(1).to(self.device, dtype=torch.float32)

        # count image
        # self.count_image = 0

        # optimizer init
        optimizer = optim.Adam(self.model_background.parameters(), lr = 1e-4)

        # input for model init
        image_batch = image_batch.to(self.device, dtype=self.data_type)

        # train
        # print("background finish !!!")

        img_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
        img_np = np.array(img_pil)
        img_np = img_np.astype(np.uint8)*255

        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (90 + hsv[:, :, 0]) % 180
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for iter in range(0, 1001, 1):
            noise_r = torch.normal(rgb[32:96, 32:96, 0].mean()/255, std=1.0, size=(1, 1, 64, 64)).to(self.device, dtype=torch.float32)
            noise_g = torch.normal(rgb[32:96, 32:96, 1].mean()/255, std=1.0, size=(1, 1, 64, 64)).to(self.device, dtype=torch.float32)
            noise_b = torch.normal(rgb[32:96, 32:96, 2].mean()/255, std=1.0, size=(1, 1, 64, 64)).to(self.device, dtype=torch.float32)
            # optimizer init
            optimizer.zero_grad()                                                                                                                                                                                                                                                                                                                                             
            # background diff
            img_with_noise = image_batch.clone()
            for index1, i in enumerate(range(-32, 32, 16)):
                for index2, j in enumerate(range(-32, 32, 16)):
                    img_with_noise[index1*4+index2, 0, 32+j:96+j, 32+i:96+i] = noise_r
                    img_with_noise[index1*4+index2, 1, 32+j:96+j, 32+i:96+i] = noise_g
                    img_with_noise[index1*4+index2, 2, 32+j:96+j, 32+i:96+i] = noise_b
            pred, feature_map = self.model_background(image_batch)
            background_diff = torch.abs(pred - img_with_noise)
            background_diff_loss = background_diff.mean()
            # mask rec
            pred_mask, feature_map = self.model_background(image_batch)
            mask_rec = 1.0 - torch.abs(pred_mask - image_batch)
            mask_rec = mask_rec * mask_batch
            mask_rec_loss = mask_rec.mean()

            loss = background_diff_loss + mask_rec_loss
            loss.backward()
            optimizer.step()
            # optimizer.ster % 100 == 0:
            #     priep()
            # if itnt(loss)
        print("background finish !!!")

        # check image_batch
        for index1, i in enumerate(range(-64, 64, 32)):
            for index2, j in enumerate(range(-64, 64, 32)):
                # get the cropped img
                search_pil = torchvision.transforms.ToPILImage()(image_batch[index1*4+index2].detach().cpu())
                # search_pil.save("./test" + str(index1*4+index2) +".jpg")

        # check search
        search_pil = torchvision.transforms.ToPILImage()(image_batch[self.check_num].detach().cpu())
        # search_pil.save("./search_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        grid = function.get_grid(img.shape[3], img.shape[2], x + (w/2), y + (h/2), (2*w), (2*h), 128, 128)
        grid = grid.to(dtype=self.data_type)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")
        search = search.to(dtype=self.data_type)
        grid_np = grid.detach().cpu().numpy()
        grid_np = grid_np.squeeze()
        grid_np_x = grid_np[:, :, 0]
        grid_np_y = grid_np[:, :, 1]

        # check pred
        with torch.no_grad():
            pred, feature_map = self.model_background(search.to(self.device, dtype=self.data_type))
        pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
        pred_pil.save("./pred_img_with_background_" + str(video_num) + "_" + str(self.count_image) + ".jpg")

        # check error map
        error_map = torch.abs(pred - search.to(self.device, dtype=self.data_type))
        error_map = error_map.sum(axis = 1)
        error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        function.write_heat_map(error_map[0].detach().cpu().numpy(), self.count_image, "./error_background_" + str(video_num) + "_")

        # gaussian_map
        gaussian_map = torch.from_numpy(g_kernel)
        gaussian_map = gaussian_map.to(self.device, dtype=self.data_type)
        gaussian_map = torch.unsqueeze(gaussian_map, 0)
        gaussian_map_mask_center= torch.zeros(16, 128, 128)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                gaussian_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = gaussian_map
        gaussian_map_mask_center = torch.unsqueeze(gaussian_map_mask_center, 1)

        # check threshold map
        threshold_map = np.where(error_map.detach().cpu().detach().numpy() > 0.1, 1.0, 0.0)
        threshold_map = np.where(grid_np_x > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_x < -1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y > 1.0, 0.0, threshold_map)
        threshold_map = np.where(grid_np_y < -1.0, 0.0, threshold_map)
        threshold_map = 255*threshold_map[0].astype(np.uint8)
        cv2.imwrite("./test1.jpg", threshold_map)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_map)
        lblareas = stats[1:, cv2.CC_STAT_AREA]
        try:
            mask = np.where(labels == np.argmax(np.array(lblareas))+1, 1.0, 0).astype(np.uint8)
        except:
            mask = np.zeros_like(threshold_map)
        mask_temp = np.zeros_like(mask)
        mask_temp[32:96, 32:96] = mask[32:96, 32:96]
        cv2.imwrite("./test.jpg", mask_temp*255)
        # mask_temp = mask_temp.astype(np.float32)/255
        threshold_map_mask_center= torch.zeros(16, 128, 128)
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                threshold_map_mask_center[index1*4+index2, 32+j:96+j, 32+i:96+i] = torch.Tensor(mask_temp[32:96, 32:96])
        threshold_map = threshold_map_mask_center
        threshold_map = torch.unsqueeze(threshold_map, 1)
        threshold_map[threshold_map > 1.0] = 1.0
        self.threshold_map_temp = threshold_map.clone()
        function.write_heat_map(threshold_map[self.check_num][0].detach().cpu().detach().numpy(), self.count_image, "./threshold_background_" + str(video_num) + "_")

        if (np.array(lblareas).max() < 1024):
            segments_slic = slic(search_pil, n_segments=50, compactness=10, sigma=1, start_label=1)

            superpixel_mask = np.zeros((128, 128), dtype=int)
            superpixel_mask = superpixel_mask
            superpixel_mask[32:96, 32:96] = segments_slic[32:96, 32:96].copy()

            superpixel_mask_2 = segments_slic.copy()
            superpixel_mask_2[32:96, 32:96] = 0.0

            for i in range(0, segments_slic.max() + 1, 1):
                if np.count_nonzero(superpixel_mask_2 == i) != 0:
                    superpixel_mask[superpixel_mask == i] = 0

            superpixel_mask[superpixel_mask > 0] = 1.0
            superpixel_mask = superpixel_mask.astype(np.float32)

            mask_batch = np.zeros((128, 128, 16))

            for index1, i in enumerate(range(-32, 32, 16)):
                for index2, j in enumerate(range(-32, 32, 16)):
                    mask_batch[32+j:96+j, 32+i:96+i, index1*4+index2] = superpixel_mask[32:96, 32:96]

            mask_batch = data_transformation(mask_batch)
            mask_batch = mask_batch.unsqueeze(1).to("cuda:1", dtype=torch.float32)
            threshold_map = mask_batch.clone()

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
        for i in range(0, 501, 1):
            optimizer.zero_grad()
            pred, pred_seg, feature_map = self.model_discriminator(image_batch)
            correlation_loss = criterion_bec_loss(pred, gaussian_map_mask_center.to(self.device, dtype=self.data_type))
            seg_loss = criterion_bec_loss(pred_seg, threshold_map.to(self.device, dtype=self.data_type))
            loss = correlation_loss+seg_loss
            loss.backward()
            optimizer.step()
        print("foreground finish !!!")
        