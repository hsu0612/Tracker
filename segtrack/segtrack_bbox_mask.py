import os
import numpy as np
import cv2

# get data path
ANN_PATH = "D:/SegTrackv2/GroundTruth/"
ann_root_list = os.listdir(ANN_PATH)

# video annotation list : bbox/mask
# [video_num, bbox_num]
# [video_num, mask_num]
# get video annotation : bbox/mask
for index, i in enumerate(ann_root_list):
    video_bbox_list = []
    video_mask_list = []
    ann_dir_path = ANN_PATH + i + "/"
    ann_list = os.listdir(ann_dir_path)
    print(ann_dir_path)
    bbox_list = []
    mask_list = []
    if (os.path.isdir(ann_dir_path + str(ann_list[0]))):
        for index_2, j in enumerate(ann_list):
            ann_list_2 = os.listdir(ann_dir_path + str(ann_list[index_2]))
            bbox_list = []
            mask_list = []
            for index_3, k in enumerate(ann_list_2):
                img = cv2.imread(ann_dir_path + str(ann_list[index_2]) + "/" + str(ann_list_2[index_3]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = img.copy()
                mask = mask.astype(np.uint8)
                mask_list.append(mask)
                x, y = np.where(mask)
                if len(x) == 0:
                    bbox_list.append([-1.0, -1.0, -1.0, -1.0])
                elif ann_dir_path == "D:/SegTrackv2/GroundTruth/cheetah/":
                    cv2.rectangle(mask, (int(y.min()), int(x.min())), (int(y.max()), int(x.max())), (255, 255, 255), 2)
                    cv2.imshow("test", mask)
                    cv2.waitKey(1)
                    bbox_list.append([y.min(), x.min(), y.max(), x.max()])
            video_bbox_list = np.array(bbox_list)
            np.save("./segtrack/bbox" + str(index) + "_" + str(index_2) + ".npy", video_bbox_list)
    else:
        for index_2, j in enumerate(ann_list):
            img = cv2.imread(ann_dir_path + str(ann_list[index_2]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = img.copy()
            mask = mask.astype(np.uint8)
            mask_list.append(mask)
            x, y = np.where(mask)
            bbox_list.append([y.min(), x.min(), y.max(), x.max()])
        video_bbox_list = np.array(bbox_list)
        np.save("./segtrack/bbox" + str(index) + ".npy", video_bbox_list)

# video_mask_list = np.array(video_mask_list)
# video_bbox_list = np.array(video_bbox_list)
# np.save("./segtrack/mask.npy", video_mask_list)
# np.save("./segtrack/bbox.npy", video_bbox_list)
