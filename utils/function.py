import numpy as np
import cv2
import torch
# function
# in: float, out: torch float
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
# in torch(float), out torch(float)
def get_image_batch_with_translate_augmentation(img, batch_size, x, y, w, grid_w, h, grid_h, data_type):
    image_batch = torch.zeros(batch_size*batch_size, 3, grid_w, grid_h)
    x_stride = int(grid_w/batch_size)
    y_stride = int(grid_w/batch_size)
    x_range = int(x_stride*batch_size/2)
    y_range = int(y_stride*batch_size/2)
    for index1, i in enumerate(range(-1*x_range, x_range, x_stride)):
        for index2, j in enumerate(range(-1*y_range, y_range, y_stride)):
            # get the cropped img
            grid = get_grid(img.shape[3], img.shape[2], x + (w/2) + (-1*i*w/grid_w), y + (h/2) + (-1*j*h/grid_h), (2*w), (2*h), grid_w, grid_h)
            grid = grid.to(dtype=data_type)
            search = torch.nn.functional.grid_sample(img, grid, mode="nearest")
            search = search.to(dtype=data_type)
            image_batch[index1*batch_size+index2] = search
    return image_batch
# in : numpy(float), out: int
def get_obj_x_y_w_h(threshold_map, threshold_map_seg, x, y, w, h, img, device, data_type, model_foreground):
    
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_map)
    lblareas = stats[1:, cv2.CC_STAT_AREA]
    try:
        pred_center_x, pred_center_y = centroids[np.argmax(np.array(lblareas)) + 1]
    except:
        return x, y, w, h
    new_center_x, new_center_y = (pred_center_x*2*w/128) + (x - 1/2*w), (pred_center_y*2*h/128) + (y - 1/2*h)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_map_seg)
    lblareas = stats[1:, cv2.CC_STAT_AREA]
    try:
        pred_w, pred_h = stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_WIDTH], stats[np.argmax(np.array(lblareas)) + 1, cv2.CC_STAT_HEIGHT]
    except:
        return x, y, w, h
    new_w, new_h = (pred_w*2*w/128), (pred_h*2*h/128)
    w = int(new_w)
    h = int(new_h)
    if w < 50:
        w = 50
    if h < 50:
        h = 50

    # # scale list
    # factor_list = np.array([1.0, 0.98, 1.02, 0.98*0.98, 1.02*1.02])
    # scale_list = np.array(np.meshgrid(factor_list, factor_list))
    #scale_list = scale_list.T.reshape(25, 2)
    # confidence score
    #confidence_score = np.zeros(25)
    #for index, scale in enumerate(scale_list):
        # Get search grid
    #    grid = get_grid(img.shape[3], img.shape[2], new_center_x, new_center_y, int(w*scale[0]), int(h*scale[1]), 64, 64)
    #    grid = grid.to(dtype=data_type)
    #    search = torch.nn.functional.grid_sample(img, grid)
    #    search = search.to(device, dtype=data_type)
        # inference
    #    with torch.no_grad():
    #        pred, pred_seg, feature_map = model_foreground(search)
        # error map
    #    error_map = pred_seg
    #    confidence_score[index] = error_map.detach().cpu().numpy().sum() * scale[0] * scale[1]
    # get w, h
   # max_index = np.argmax(confidence_score)
   # best_scale = scale_list[max_index]
   # new_w, new_h = (best_scale[0] * w), (best_scale[1] * h)
   # w = int(new_w)
   # h = int(new_h)

    x = int(new_center_x) - int(w/2)
    y = int(new_center_y) - int(h/2)
    return x, y, w, h
# check function
# in: numpy(float), out: write image by opencv
def write_heat_map(img, count, write_path):
    img = img*255
    img = img.astype(np.uint8)
    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(write_path + str(count) + ".jpg", im_color)
# in: numpy(float), out: write image by opencv
def write_tracking_result(img, x, y, count, w, h, write_path):
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 2, 3)
    img_render = img[0].copy()
    img_render = cv2.cvtColor(img_render.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.rectangle(img_render, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(write_path + str(count) + ".jpg", img_render*255)
