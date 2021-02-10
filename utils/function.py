import numpy as np
import cv2
from scipy import signal
import torch
# function
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
    grid = grid.unsqueeze(0) # add batch dim
    
    return grid
# check function
def write_error_map(img, count, write_path):
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
    cv2.circle(img_render, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite(write_path + str(count) + ".jpg", img_render*255)
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
    center_x = 8
    center_y = 8
        
    bias_x = pred_center_x - center_x 
    bias_y = pred_center_y - center_y
        
    diff_w = (old_w - w)/2
    diff_h = (old_h - h)/2
        
    next_x = int(x + diff_w + bias_x*(w/64)*4)
    next_y = int(y + diff_h + bias_y*(h/64)*4)
            
    return int(next_x), int(next_y)
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



    
