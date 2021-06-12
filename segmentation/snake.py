import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import polygon
import matplotlib.pyplot as plt

class Snake():
    def __init__(self):
        pass
    def get_mask(self, img):
        imgrgb = img.copy()
        img = rgb2gray(img)
        s = np.linspace(0, 2*np.pi, 400)
        r = 64 + 64*np.sin(s)
        c = 64 + 64*np.cos(s)
        init = np.array([r, c]).T
        snake = active_contour(gaussian(img, 3),
                            init, alpha=0.015, beta=10, gamma=0.001, coordinates='rc')
        mask = np.zeros((img.shape))
        rr, cc = polygon(snake[:,0], snake[:,1], img.shape)
        mask[rr,cc] = 255
        mask = mask.astype(np.uint8)


        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(imgrgb, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()

        return mask

if __name__ == '__main__':
    img = np.array(cv2.imread('./img_44_0.jpg'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    snake = Snake()
    mask = snake.get_mask(img)
    # cv2.imshow("test", mask)
    # cv2.waitKey(1)
