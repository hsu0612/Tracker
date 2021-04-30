import cv2
import numpy as np

img = cv2.imread("./img0_0.jpg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv[50, 46, 0])
a = np.zeros_like(hsv[:, :, 1])
a = a + 90
# a = a.astype(np.uchar)
hsv[:, :, 0] = (90 + hsv[:, :, 0]) % 180
print(hsv[50, 46, 0])
# print(hsv[0])
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# img[:, :, 0] = 0

cv2.imwrite("./im2.jpg", rgb)
