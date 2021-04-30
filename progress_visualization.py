import os
import cv2

# img_list
img_list = os.listdir("D:/GOT/val/GOT-10k_Val_000001/")
img_list.remove("absence.label")
img_list.remove("cover.label")
img_list.remove("cut_by_image.label")
img_list.remove("meta_info.ini")
img_list.remove("groundtruth.txt")
# gt_list
gt  = open("D:/GOT/val/GOT-10k_Val_000001/" + "groundtruth.txt")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280,  720))

for index, i in enumerate(img_list):
    img = cv2.imread("D:/GOT/val/GOT-10k_Val_000001/" + i)
    # get bbox
    bbox = gt.readline()
    x, y, w, h = bbox.split(",")
    x, y, w, h = float(x), float(y), float(w), float(h)
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    # cv2.imwrite("./test" + str(index)+ ".jpg", img)
    out.write(img)

