import numpy as np
import cv2 as cv
cap = cv.VideoCapture(cv.samples.findFile("./output.mp4"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
fourcc = cv.VideoWriter_fourcc(*'XVID')
out1 = cv.VideoWriter('./optical_flow1.mp4', fourcc, 20.0, (1920,  1080))
out2 = cv.VideoWriter('./optical_flow2.mp4', fourcc, 20.0, (1920,  1080))
ret, frame2 = cap.read()
list1 = []
list2 = []
while(ret):
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next
    list1.append(mag)
    list2.append(ang)
    ret, frame2 = cap.read()
list1 = np.array(list1)
list2 = np.array(list2)
print(list1.shape)
np.save("./1.npy", list1)
np.save("./2.npy", list2)
out1.release()
out2.release()