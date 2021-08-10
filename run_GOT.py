from __future__ import absolute_import, print_function
from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
import src.My_Tracker as My_Tracker
import sys
import os
import time
import numpy as np
import cv2
sys.path.append('./')
from eco import ECOTracker
import utils.function as function
ROOT_DIR = '/mnt/disk2-part1/GOT/'
class IdentityTracker(Tracker):
    """Example on how to define a tracker.
        To define a tracker, simply override ``init`` and ``update`` methods
            from ``Tracker`` with your own pipelines.
    """
    def __init__(self):
        super(IdentityTracker, self).__init__(
            name='IdentityTracker', # name of the tracker
            is_deterministic=True   # deterministic (True) or stochastic (False)
        )
        self.path = "/mnt/disk2-part1/GOT/val/"
        self.data_list = os.listdir(self.path)
        self.count = -1
        self.video_num = -1
    def init(self, image, box):
        """Initialize your tracking model in the first frame
        Arguments:
            image {PIL.Image} -- Image in the first frame.
            box {np.ndarray} -- Target bounding box (4x1,
                [left, top, width, height]) in the first frame.
        """
        self.flag = False
        self.count+=1
        self.video_num+=1
        # gt_path = self.path + self.data_list[self.count] + "/groundtruth.txt"
        # print(gt_path)
        # self.number_of_frame = len(os.listdir(self.path + self.data_list[self.count]))-5
        self.number_of_frame = 800
        # print(self.number_of_frame)
        # self.gt = open(gt_path)
        # rect = self.gt.readline().split(',')

        self.box = box
        self.default_box = box
        self.seg_tracker = My_Tracker.FCAE_tracker(self.number_of_frame)
        self.seg_tracker.tracker_init(image, int(box[0]), int(box[1]), int(box[2]), int(box[3]), 1000, self.video_num)
        self.eco_tracker = ECOTracker(True)
        self.eco_tracker.init(np.array(image), [int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        self.count_for_online_train = 0
        self.count_img = 0
        self.real_x = int(box[0])
        self.real_y = int(box[1])
        self.real_w = int(box[2])
        self.real_h = int(box[3])
    def update(self, image):
        """Locate target in an new frame and return the estimated bounding box.
        Arguments:
            image {PIL.Image} -- Image in a new frame.
        Returns:
            np.ndarray -- Estimated target bounding box (4x1,
                [left, top, width, height]) in ``image``.
        """
        self.count_img += 1

        bbox = self.eco_tracker.update(np.array(image), True, True)
        new_bbox = []
        new_bbox.append(bbox[0])
        new_bbox.append(bbox[1])
        new_bbox.append(float(bbox[2]) - float(bbox[0]))
        new_bbox.append(float(bbox[3]) - float(bbox[1]))

        height, width = np.array(image).shape[:2]

        score = self.eco_tracker.score
        size = tuple(self.eco_tracker.crop_size.astype(np.int32))
        score = cv2.resize(score, size)
        score -= score.min()
        score /= score.max()
        score = (score * 255).astype(np.uint8)
        score = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        pos = self.eco_tracker._pos
        pos = (int(pos[0]), int(pos[1]))
        xmin = pos[1] - size[1]//2
        xmax = pos[1] + size[1]//2 + size[1] % 2
        ymin = pos[0] - size[0] // 2
        ymax = pos[0] + size[0] // 2 + size[0] % 2
        left = abs(xmin) if xmin < 0 else 0
        xmin = 0 if xmin < 0 else xmin
        right = width - xmax
        xmax = width if right < 0 else xmax
        right = size[1] + right if right < 0 else size[1]
        top = abs(ymin) if ymin < 0 else 0
        ymin = 0 if ymin < 0 else ymin
        down = height - ymax
        ymax = height if down < 0 else ymax
        down = size[0] + down if down < 0 else size[0]
        score = score[top:down, left:right]
        score_gray = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
        gaussian = function.get_gaussian(score.shape[1], score.shape[0])

        # if self.count_img < 1919:
        self.count_for_online_train+=1
        seg_x, seg_y, seg_w, seg_h = self.seg_tracker.tracker_save_img(image, int(new_bbox[0]), int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3]))
        if self.count_for_online_train > 3 and abs(gaussian - score_gray).mean() > 140.0:                                                                                                                                            
            seg_x, seg_y, seg_w, seg_h = self.seg_tracker.tracker_update(image, int(new_bbox[0]), int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3]), 1000, self.video_num)
            self.count_for_online_train = 1

        if  self.count_for_online_train > 6:
            seg_x, seg_y, seg_w, seg_h = self.seg_tracker.tracker_update(image, int(new_bbox[0]), int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3]), 1000, self.video_num)
            self.count_for_online_train = 1

        if abs(gaussian - score_gray).mean() > 140.0:
            new_bbox[0] = seg_x
            new_bbox[1] = seg_y
            new_bbox[2] = seg_w
            new_bbox[3] = seg_h
        
        return new_bbox
if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()
    # setup experiment
    experiment = ExperimentGOT10k('/mnt/disk2-part1/GOT/', subset='val')
    # run experiments on GOT-10k
    experiment.run(tracker, visualize=False)
    # report performance on GOT-10k
    experiment.report([tracker.name])
