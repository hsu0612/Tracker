import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

class Grabcut():
    def __init__(self):
        pass
    def get_mask(self, img):
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[32:96, 32:96] = 3
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (32,32,64,64)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        return mask2

if __name__ == '__main__':
    img = np.array(cv2.imread('./img_0_0.jpg'))
    grabcut = Grabcut()
    mask = grabcut.get_mask(img)
    img = img*mask[:,:,np.newaxis]
    plt.imshow(img),plt.colorbar(),plt.show()
