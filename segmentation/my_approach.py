# lib
import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse
import torch
import torchvision
from torchvision import transforms
sys.path.append('./')
# mylib
from src.model import FCNet
import utils.function as function
from segmentation.grab_cut import Grabcut

class AE_Segmentation():
    def __init__(self):
        # seed
        np.random.seed(999)
        torch.manual_seed(999)
        torch.cuda.manual_seed_all(999)
        torch.backends.cudnn.deterministic = True
    def train(self, image_batch, img_without_augmentation, num1, num2):
        # data tarnsformation
        data_transformation = transforms.Compose([
                    transforms.ToTensor(),
                    ])
        # background model
        self.background_model = FCNet().to("cuda", dtype=torch.float32)
        optimizer = torch.optim.Adam(self.background_model.parameters(), lr = 1e-4)
        image_batch = image_batch.to("cuda", dtype=torch.float32)
        img_without_augmentation = img_without_augmentation.to("cuda", dtype=torch.float32)
        
        # pseudo ground truth 
        grabcut = Grabcut()
        mask_batch = np.zeros((128, 128, 16))
        search_pil = torchvision.transforms.ToPILImage()(img_without_augmentation[0].detach().cpu())
        mask = grabcut.get_mask(np.array(search_pil))
        for index1, i in enumerate(range(-32, 32, 16)):
            for index2, j in enumerate(range(-32, 32, 16)):
                mask_batch[32+j:96+j, 32+i:96+i, index1*4+index2] = mask[32:96, 32:96]
        mask_batch = data_transformation(mask_batch)
        mask_batch = mask_batch.unsqueeze(1).to("cuda", dtype=torch.float32)

        # get reverse color
        img_pil = torchvision.transforms.ToPILImage()(img_without_augmentation[0].detach().cpu())
        img_np = np.array(img_pil)
        img_np = img_np.astype(np.uint8)*255
        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (90 + hsv[:, :, 0]) % 180
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        for iter in range(0, 1001, 1):
            noise_r = torch.normal(rgb[32:96, 32:96, 0].mean()/255, std=1.0, size=(1, 1, 64, 64)).to("cuda", dtype=torch.float32)
            noise_g = torch.normal(rgb[32:96, 32:96, 1].mean()/255, std=1.0, size=(1, 1, 64, 64)).to("cuda", dtype=torch.float32)
            noise_b = torch.normal(rgb[32:96, 32:96, 2].mean()/255, std=1.0, size=(1, 1, 64, 64)).to("cuda", dtype=torch.float32)
            # optimizer init
            optimizer.zero_grad()                                                                                                                                                                                                                                                                                                                                             
            # background diff
            img_with_noise = image_batch.clone()
            for index1, i in enumerate(range(-32, 32, 16)):
                for index2, j in enumerate(range(-32, 32, 16)):
                    img_with_noise[index1*4+index2, 0, 32+j:96+j, 32+i:96+i] = noise_r
                    img_with_noise[index1*4+index2, 1, 32+j:96+j, 32+i:96+i] = noise_g
                    img_with_noise[index1*4+index2, 2, 32+j:96+j, 32+i:96+i] = noise_b
            pred, feature_map = self.background_model(image_batch)
            background_diff = torch.abs(pred - img_with_noise)
            background_diff_loss = background_diff.mean()
            # mask rec
            pred_mask, feature_map = self.background_model(image_batch)
            mask_rec = 1.0 - torch.abs(pred_mask - image_batch)
            mask_rec = mask_rec * mask_batch
            mask_rec_loss = mask_rec.mean()

            loss = background_diff_loss + mask_rec_loss
            loss.backward()
            optimizer.step()
            # if iter % 100 == 0:
            #     print(loss)
        torch.save(self.background_model, "./checkpoint/save_" + str(num1) + "_"+ str(num2) + ".pt'")

    def inference(self, img_without_augmentation, grid, num):
        with torch.no_grad():
            pred, feature_map = self.background_model(img_without_augmentation[:, :, :, :].to("cuda", dtype=torch.float32))

        grid_np = grid.detach().cpu().numpy()
        grid_np = grid_np.squeeze()
        grid_np_x = grid_np[:, :, 0]
        grid_np_y = grid_np[:, :, 1]

        error_map = torch.abs(pred - img_without_augmentation.to("cuda", dtype=torch.float32))
        error_map = error_map.sum(axis = 1)
        error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        # function.write_heat_map(error_map[0].detach().cpu().numpy(), 0, "./error_background_" + str(num) + "_")
        threshold_map = np.where(error_map.detach().cpu().numpy() > 0.2, 1.0, 0.0)
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
            return  np.zeros_like(threshold_map)
        mask_temp = np.zeros_like(mask)
        mask_temp[32:96, 32:96] = mask[32:96, 32:96]
        # function.write_heat_map(mask, 0, "./threshold_background_" + str(num) + "_")

        return mask*255

if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default = "D:/SegTrackv2/JPEGImages/bird_of_paradise/")
    parser.add_argument('--save_num', type=int, default = 0)
    args = parser.parse_args()
    DATA_PATH = args.data_path
    NUM = args.save_num

    # img_list
    img_list = os.listdir(DATA_PATH)
    # img_list.remove("absence.label")
    # img_list.remove("cover.label")
    # img_list.remove("cut_by_image.label")
    # img_list.remove("meta_info.ini")
    # img_list.remove("groundtruth.txt")
    # gt_list
    # gt  = open(DATA_PATH + "groundtruth.txt")
    gt = np.load("./segtrack/bbox/bird_of_paradise/bird_of_paradise.npy", allow_pickle=True)
    # total img
    img_total = torch.zeros(len(img_list), 3, 128, 128)

    # data tarnsformation
    data_transformation = transforms.Compose([
                transforms.ToTensor(),
                ])

    # get img_grid
    for index, i in enumerate(img_list):
        # get img
        img = Image.open(DATA_PATH+i)
        img = data_transformation(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(dtype=torch.float32)
        # get bbox
        bbox = gt[index]
        # x, y, w, h = bbox.split(",")
        x, y, w, h = bbox[0] , bbox[1] ,bbox[2] - bbox[0] ,bbox[3] - bbox[1]
        x, y, w, h = float(x), float(y), float(w), float(h)
        # grid
        grid = function.get_grid(img.shape[3], img.shape[2], x + w/2, y + h/2, 2*w, 2*h, 128, 128)
        grid = grid.to(dtype=torch.float32)
        search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")
        if index == 0:
            img_batch_1 = function.get_image_batch_with_translate_augmentation(img, 4, x, y, w, 128, h, 128, torch.float32)
        img_total[index] = search[0]

    img_1 = img_total[0:1, :, :, :]
    noise = torch.rand((1, 3, 128, 128)).to("cuda", dtype=torch.float32)
    My_Approach = AE_Segmentation()
    My_Approach.train(img_batch_1, img_1)
    My_Approach.inference(img_1, 0, grid)
