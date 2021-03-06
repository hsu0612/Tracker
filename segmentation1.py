# lib
import os
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import argparse
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
# mylib
from src.model import FCNet
from src.model import FCNet_fore
import utils.function as function

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default = "D:/GOT/val/GOT-10k_Val_000001/")
parser.add_argument('--save_num', type=int, default = 0)
args = parser.parse_args()
DATA_PATH = args.data_path
NUM = args.save_num

# seed
np.random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed_all(999)
torch.backends.cudnn.deterministic = True

# img_list
img_list = os.listdir(DATA_PATH)
img_list.remove("absence.label")
img_list.remove("cover.label")
img_list.remove("cut_by_image.label")
img_list.remove("meta_info.ini")
img_list.remove("groundtruth.txt")
# gt_list
gt  = open(DATA_PATH + "groundtruth.txt")
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
    bbox = gt.readline()
    x, y, w, h = bbox.split(",")
    x, y, w, h = float(x), float(y), float(w), float(h)
    # grid
    grid = function.get_grid(img.shape[3], img.shape[2], x + w/2, y + h/2, 2*w, 2*h, 128, 128)
    grid = grid.to(dtype=torch.float32)
    search = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="border")
    if index == 0:
        img_batch_1 = function.get_image_batch_with_translate_augmentation(img, 4, x, y, w, 128, h, 128, torch.float32)
    img_total[index] = search[0]

# background model
background_model = FCNet().to("cuda", dtype=torch.float32)
optimizer = torch.optim.Adam(background_model.parameters(), lr = 1e-4)

# get img batch
img_batch = img_total[0:1, :, :, :]
img_pil = torchvision.transforms.ToPILImage()(img_batch[0].detach().cpu())
img_pil.save("./img" + str(NUM) + "_" + str(0) + ".jpg")
img_np = np.array(img_pil.convert('RGB'))

mask = np.zeros(img_np.shape[:2], np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (32,32,64,64)
cv2.grabCut(img_np,mask,rect,bgdModel,fgdModel,100,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_np_with_mask = img_np*mask2[:,:,np.newaxis]
cv2.imwrite("./grab"+ str(NUM) +".jpg", img_np_with_mask)

s = np.linspace(0, 2*np.pi, 400)
r = 64 + 64*np.sin(s)
c = 64 + 64*np.cos(s)
init = np.array([r, c]).T

img_np_gray = rgb2gray(img_np)
snake = active_contour(gaussian(img_np_gray, 3),
                       init, alpha=0.015, beta=10, gamma=0.001, coordinates='rc')

init_uint = init.astype('uint8')
snake_uint = snake.astype('uint8')

snake_result = img_np.copy()

for i in range(0, 400, 1):
    cv2.circle(snake_result, (init_uint[i, 1], init_uint[i, 0]), 3, (255, 0, 0), -1)
    cv2.circle(snake_result, (snake_uint[i, 1], snake_uint[i, 0]), 3, (0, 0, 255), -1)

cv2.imwrite("./snake"+ str(NUM) +".jpg", snake_result)

img_np_float = img_np.astype('float32')/255
temp = np.zeros((64, 64, 3))
temp = img_np_float[32:96, 32:96, :]
segments_slic = slic(temp, n_segments=50, compactness=10, sigma=1,
                     start_label=1)

super_result = mark_boundaries(temp, segments_slic)

super_result_uint8 = super_result.astype('uint8')*255

super_result_uint8 = np.where(super_result_uint8 == 0, temp*255, super_result_uint8)

cv2.imwrite("./superpixel"+ str(NUM) +".jpg", super_result_uint8)

superpixel_mask = np.zeros((128, 128), dtype=int)
superpixel_mask[32:96, 32:96] = segments_slic

mask_fore = torch.Tensor(mask2).to("cuda", dtype=torch.float32)
mask_fore = mask_fore.unsqueeze(0)
mask_fore = mask_fore.unsqueeze(0)

for iter in range(0, 1001, 1):
    img_batch = img_total[0:1, :, :, :]
    # img_batch = img_batch_1
    img_batch = img_batch.to("cuda", dtype=torch.float32)
    noise = torch.rand((1, 3, 64, 64)).to("cuda", dtype=torch.float32)
    # optimizer init
    optimizer.zero_grad()
                                                                                                                                                                                                                                                                                                                                                                                      
    # background diff
    img_with_noise = img_batch.clone()
    # for index1, i in enumerate(range(-32, 32, 16)):
    #     for index2, j in enumerate(range(-32, 32, 16)):
    #         img_with_noise[index1*4+index2, :, 32+j:96+j, 32+i:96+i] = 1.0-img_with_noise[index1*4+index2, :, 32+j:96+j, 32+i:96+i]
    pred, feature_map = background_model(img_batch)
    background_diff = torch.abs(pred - img_with_noise)
    # background_diff[:, :, 32:96, 32:96] = 0.0
    background_diff_loss = background_diff.mean()
    # smooth
    pred, feature_map = background_model(img_batch)
    error_map = torch.abs(pred - img_batch)
    dx, dy = function.gradient(error_map)
    dx_c, dy_c = function.gradient(img_batch)
    dx, dy, dx_c, dy_c = dx.mean(axis=1), dy.mean(axis=1), dx_c.mean(axis=1), dy_c.mean(axis=1)
    dx_c, dy_c = 1.0-dx_c, 1.0-dy_c
    smooth_loss = ((abs(dx*dx_c)).mean() + (abs(dy*dy_c)).mean())
    # grad
    # dx_search, dy_search =  function.gradient(img_with_noise)
    # dy_search[:, :, 32-1:96-1, 32:96] = 0.0
    # dx_search[:, :, 32:96, 32-1:96-1] = 0.0
    # dx_pred, dy_pred = function.gradient(pred)
    # dy_pred[:, :, 32-1:96-1, 32:96] = 0.0
    # dx_pred[:, :, 32:96, 32-1:96-1] = 0.0
    # grad_loss = abs(dx_search - dx_pred).mean() + abs(dy_search - dy_pred).mean()
    # mask rec
    mask_rec = 1.0 - torch.abs(pred - img_batch)
    mask_rec = mask_rec * mask_fore
    mask_rec_loss = mask_rec.mean()

    loss = background_diff_loss# + smooth_loss + mask_rec_loss
    loss.backward()
    optimizer.step()
    # scheduler.step()
    if iter % 100 == 0:
        print(loss)

with torch.no_grad():
    pred, feature_map = background_model(img_total[0:1, :, :, :].to("cuda", dtype=torch.float32))

pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
pred_pil.save("./pred_img_with_background_" + str(NUM) + "_" + str(0) + ".jpg")

# check error map
error_map = torch.abs(pred - img_total[0:1, :, :, :].to("cuda", dtype=torch.float32))
error_map[:, :, 32:96, 32:96] = (error_map[:, :, 32:96, 32:96] - error_map[:, :, 32:96, 32:96].min()) / (error_map[:, :, 32:96, 32:96].max() - error_map[:, :, 32:96, 32:96].min())
error_map = error_map.mean(axis = 1)
function.write_heat_map(error_map[0].detach().cpu().numpy(), 0, "./error_background_" + str(NUM) + "_")

# check threshold map
with torch.no_grad():
    threshold_map = torch.nn.functional.threshold(error_map, 0.2, 0.0, inplace=False)
threshold_map[threshold_map!=0.0] = 1.0
threshold_map_mask_center= torch.zeros(threshold_map.shape)
threshold_map_mask_center[0, 32:96, 32:96] = threshold_map[0, 32:96, 32:96]
threshold_map = threshold_map_mask_center
threshold_map = torch.unsqueeze(threshold_map, 1)
threshold_map[threshold_map > 1.0] = 1.0

function.write_heat_map(threshold_map[0][0].detach().cpu().detach().numpy(), 0, "./threshold_background_" + str(NUM) + "_")

# check mask
mask = np.zeros((128, 128, 3))
search_np = np.array(img_pil)
for i in range(0, 3, 1):
    mask[:, :, i] = np.where(threshold_map[0][0].detach().cpu().detach().numpy() == 1.0, search_np[:, :, i], 0.0)
search_with_mask = Image.fromarray(mask.astype("uint8"))
search_with_mask = data_transformation(search_with_mask)
search_with_mask = torchvision.transforms.ToPILImage()(search_with_mask.detach().cpu())
search_with_mask.save("./mask_" + str(NUM) + ".jpg")

final_mask = np.zeros((128, 128))
final_result = np.zeros((128, 128, 3))
for i in range(0, superpixel_mask.max() + 1, 1):
    num_ele = np.count_nonzero(superpixel_mask == i)
    error_map_copy = error_map[0].detach().cpu().numpy().copy()
    error_map_copy = np.where(superpixel_mask == i, error_map_copy, 0)
    error_map_copy = (error_map_copy.sum() / num_ele)
    final_mask = np.where(superpixel_mask == i, error_map_copy, final_mask)

final_result[:, :, 0] = np.where(final_mask > 0.2, img_np[:, :, 0], 0)
final_result[:, :, 1] = np.where(final_mask > 0.2, img_np[:, :, 1], 0)
final_result[:, :, 2] = np.where(final_mask > 0.2, img_np[:, :, 2], 0)

cv2.imwrite("./superpixel_color"+ str(NUM) +".jpg", final_result)

# foreground 
# foreground_model = FCNet().to("cuda", dtype=torch.float32)
# optimizer = optim.Adam(foreground_model.parameters(), lr = 1e-4)
# criterion_bec_loss = torch.nn.BCELoss()

# for i in range(0, 1001, 1):
#     img_batch = img_total[0:1, :, :, :]
#     img_batch = img_batch.to("cuda", dtype=torch.float32)
#     # optimizer init
#     optimizer.zero_grad()
#     pred, feature_map = foreground_model(img_batch)
#     # foreground diff
#     foreground_diff = torch.abs(pred - img_batch)
#     foreground_diff = foreground_diff * threshold_map.to("cuda", dtype=torch.float32)
#     foreground_diff_loss = foreground_diff.mean()
#     # foreground_diff_loss = criterion_bec_loss(pred, threshold_map.to("cuda", dtype=torch.float32))
#     # background diff
#     background_diff = 1.0 - torch.abs(pred - img_batch)
#     background_diff[:, :, 32:96, 32:96] = 0.0
#     bakcground_diff_loss = background_diff.mean()
#     # smooth
#     error_map = torch.abs(pred - img_batch)
#     dx_c, dy_c = function.gradient(img_batch)
#     dx, dy = function.gradient(pred)
#     dx, dy, dx_c, dy_c = dx.mean(axis=1), dy.mean(axis=1), dx_c.mean(axis=1), dy_c.mean(axis=1)
#     dx_c, dy_c = 1.0-dx_c, 1.0-dy_c
#     smooth_loss = abs(dx).mean() + abs(dy).mean()
#     # grad
#     dx_search, dy_search =  function.gradient(img_batch)
#     dx_pred, dy_pred = function.gradient(pred)
#     grad_loss = (abs(dx_search - dx_pred)*threshold_map[:, :, :, 1:].to("cuda", dtype=torch.float32)).mean() + (abs(dy_search - dy_pred)*threshold_map[:, :, 1:, :].to("cuda", dtype=torch.float32)).mean()
#     # mask rec
#     mask_rec = torch.abs(pred - img_batch)
#     mask_rec = mask_rec * mask_fore
#     mask_rec_loss = mask_rec.mean()
#     # mask_rec_loss = criterion_bec_loss(mask_rec, mask_fore.to("cuda", dtype=torch.float32))

#     loss = foreground_diff_loss# + smooth_loss + mask_rec_loss + 0.1*bakcground_diff_loss
#     loss.backward()
#     optimizer.step()
#     if i % 100 == 0:
#         print(loss)

# with torch.no_grad():
#     pred, feature_map = foreground_model(img_batch)

# pred_pil = torchvision.transforms.ToPILImage()(img_batch[0].detach().cpu())
# pred_pil.save("./img" + str(NUM) + "_" + str(0) + ".jpg")

# pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
# pred_pil.save("./pred_img_with_foreground_" + str(NUM) + "_" + str(0) + ".jpg")

# # check error map
# error_map = 1.0 - torch.abs(pred - img_batch)
# error_map = error_map.mean(axis = 1)
# function.write_heat_map(error_map[0].detach().cpu().numpy(), 0, "./error_foreground_" + str(NUM) + "_")

# # check threshold map
# with torch.no_grad():
#     threshold_map = torch.nn.functional.threshold(error_map, 0.8, 0.0, inplace=False)
# threshold_map[threshold_map!=0.0] = 1.0
# # threshold_map_mask_center= torch.zeros(threshold_map.shape)
# # threshold_map_mask_center[0, 32:96, 32:96] = threshold_map[0, 32:96, 32:96]
# # threshold_map = threshold_map_mask_center
# threshold_map = torch.unsqueeze(threshold_map, 1)
# # threshold_map[threshold_map > 1.0] = 1.0

# function.write_heat_map(threshold_map[0][0].detach().cpu().detach().numpy(), 0, "./threshold_foreground_" + str(NUM) + "_")

# cycle consistency
# optimizer = optim.Adam(list(background_model.parameters()) + list(foreground_model.parameters()), lr = 1e-4)

# for i in range(0, 1001, 1):
#     img_batch = img_total[0:1, :, :, :]
#     img_batch = img_batch.to("cuda", dtype=torch.float32)
#     # optimizer init
#     optimizer.zero_grad()

#     pred, feature_map = background_model(img_batch)
#     background_diff = torch.abs(pred - img_batch)
#     background_diff[:, :, 32:96, 32:96] = 0.0
#     b_model_background_diff_loss = background_diff.mean()

#     b_error_map = torch.abs(pred - img_batch)
#     dx, dy = function.gradient(b_error_map)
#     dx_c, dy_c = function.gradient(img_batch)
#     dx, dy, dx_c, dy_c = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
#     dx, dy, dx_c, dy_c = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
#     dx_c, dy_c = 1.0-dx_c, 1.0-dy_c
#     b_smooth_loss = ((abs(dx*dx_c)).mean() + (abs(dy*dy_c)).mean())

#     dx_search, dy_search =  function.gradient(img_batch)
#     dy_search[:, :, 32-1:96-1, 32:96] = 0.0
#     dx_search[:, :, 32:96, 32-1:96-1] = 0.0
#     dx_pred, dy_pred = function.gradient(pred)
#     dy_pred[:, :, 32-1:96-1, 32:96] = 0.0
#     dx_pred[:, :, 32:96, 32-1:96-1] = 0.0
#     b_grad_loss = abs(dx_search - dx_pred).mean() + abs(dy_search - dy_pred).mean()

#     pred, feature_map = foreground_model(img_batch)
#     foreground_diff = torch.abs(pred - img_batch)
#     foreground_diff = foreground_diff * threshold_map.to("cuda", dtype=torch.float32)
#     f_foreground_diff_loss = foreground_diff.mean()

#     background_diff = 1.0 - torch.abs(pred - img_batch)
#     background_diff[:, :, 32:96, 32:96] = 0.0
#     f_bakcground_diff_loss = background_diff.mean()

#     f_error_map = torch.abs(pred - img_batch)
#     dx, dy = function.gradient(f_error_map)
#     dx_c, dy_c = function.gradient(img_batch)
#     dx, dy, dx_c, dy_c = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
#     dx, dy, dx_c, dy_c = dx.mean(axis=0), dy.mean(axis=0), dx_c.mean(axis=0), dy_c.mean(axis=0)
#     dx_c, dy_c = 1.0-dx_c, 1.0-dy_c
#     f_smooth_loss = abs(dx).mean() + abs(dy).mean()

#     dx_search, dy_search =  function.gradient(img_batch)
#     dx_pred, dy_pred = function.gradient(pred)
#     f_grad_loss = (abs(dx_search - dx_pred)*threshold_map[0, 0, :, 1:].to("cuda", dtype=torch.float32)).mean() + (abs(dy_search - dy_pred)*threshold_map[0, 0, 1:, :].to("cuda", dtype=torch.float32)).mean()

#     # consistency loss
#     consistency_loss = torch.abs(b_error_map + f_error_map).mean()

    # loss = b_model_background_diff_loss + b_smooth_loss + b_grad_loss + f_foreground_diff_loss + f_smooth_loss + f_grad_loss + 0.01*f_bakcground_diff_loss + 0.1*consistency_loss
    
#     loss.backward()
#     optimizer.step()
#     if i % 100 == 0:
#         print(loss)

# with torch.no_grad():
#     pred, feature_map = foreground_model(img_batch)

# pred_pil = torchvision.transforms.ToPILImage()(img_batch[0].detach().cpu())
# pred_pil.save("./img" + str(0) + "_" + str(0) + ".jpg")

# pred_pil = torchvision.transforms.ToPILImage()(pred[0].detach().cpu())
# pred_pil.save("./pred_img_with_foreground_" + str(0) + "_" + str(0) + ".jpg")

# # check error map
# error_map = 1.0 - torch.abs(pred - img_batch)
# error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
# error_map = error_map.mean(axis = 1)
# function.write_heat_map(error_map[0].detach().cpu().numpy(), 0, "./error_foreground_c_" + str(NUM) + "_")
