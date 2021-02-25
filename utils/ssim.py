from skimage.measure import compare_ssim
import numpy as np
import cv2
import os
import glob
import time

def write_heat_map(img, count, write_path):
    img = img*255
    img = img.astype(np.uint8)
    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(write_path + str(count) + ".jpg", im_color)

img = cv2.imread("./pred_fore_img_0_1.jpg")
img2 = cv2.imread("./search_0_1.jpg")
img = np.array(img)
img2 = np.array(img2)
x = -10
y = -10
#s = compare_ssim(img2[32-x:96-y, 32-x:96-y, :], img[32-x:96-y, 32-x:96-y, :], multichannel=True)
#e = img2[32-x:96-y, 32-x:96-y, :] - img[32-x:96-y, 32-x:96-y, :]
error = np.zeros([64, 64])
t = time.time()
for i in range(0, 64, 1):
    for j in range(0, 64, 1):
        # print(img2[0+i:64+i, 0+j:64+j, :].shape)
        # cv2.imwrite("./t.jpg", img[0+i:64+i, 0+j:64+j, :])
        # error[i , j] = compare_ssim(img2[0+i:64+i, 0+j:64+j, :], img[0+i:64+i, 0+j:64+j, :], multichannel=True)
        error[i , j] = abs((img2[0+i:64+i, 0+j:64+j, :] - img[0+i:64+i, 0+j:64+j, :]).mean())

error = (error - error.min()) / (error.max() - error.min())
print(np.argmax(error))
write_heat_map(error, 0, "./test")
print(time.time() - t) 
print('Done!!')