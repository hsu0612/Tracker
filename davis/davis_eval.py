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
sys.path.append('./')
import utils.function as function

# get data path
IMG_PATH = "D:/DAVIS/JPEGImages/480p/"
img_list = os.listdir(IMG_PATH)

video_bbox_list = np.load("./davis/bbox.npy", allow_pickle=True)
video_mask_list = np.load("./davis/mask.npy", allow_pickle=True)

# data tarnsformation
data_transformation = transforms.Compose([
            transforms.ToTensor(),
            ])

# get img_grid
for index, i in enumerate(img_list):
    # get img
    img = Image.open(IMG_PATH+i+"/00000.jpg")
    img = data_transformation(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(dtype=torch.float32)
    # get bbox
    x, y, w, h = float(video_bbox_list[index][0][0]), float(video_bbox_list[index][0][1]) \
        ,float(video_bbox_list[index][0][2]) - float(video_bbox_list[index][0][0]), float(video_bbox_list[index][0][3]) - float(video_bbox_list[index][0][1])
    # grid
    grid = function.get_grid(img.shape[3], img.shape[2], x + w/2, y + h/2, 2*w, 2*h, 128, 128)
    grid = grid.to(dtype=torch.float32)
    search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")
    search_pil = torchvision.transforms.ToPILImage()(search[0].detach().cpu())
    search_pil.save("./img" + str(index) + "_" + str(0) + ".jpg")



# mask_rgb = gray2rgb(video_mask_list[1][0])

# cv2.rectangle(mask_rgb, (int(video_bbox_list[1][0][0]), int(video_bbox_list[1][0][1])), (int(video_bbox_list[1][0][2]), int(video_bbox_list[1][0][3])), (0, 255, 0), 2)
# cv2.imwrite("./test.jpg", mask_rgb)
