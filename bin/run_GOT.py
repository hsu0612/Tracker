from __future__ import absolute_import, print_function
from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
import os
import time
import sys
import numpy as np

sys.path.append('./')
from eco import ECOTracker

def sorting(x):
    return(int(x[13:]))
ROOT_DIR = '/media/hsu/data/GOT'
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
        self.path = "/media/hsu/data/GOT/val/"
        self.data_list = os.listdir(self.path)
        self.data_list.remove("list.txt")
        self.data_list.sort(key=sorting)
        self.count = -1
    def init(self, image, box):
        """Initialize your tracking model in the first frame
        Arguments:
            image {PIL.Image} -- Image in the first frame.
            box {np.ndarray} -- Target bounding box (4x1,
                [left, top, width, height]) in the first frame.
        """
        # if self.count > -1+5:
        #     assert False
        self.count+=1
        gt_path = self.path + self.data_list[self.count] + "/groundtruth.txt"
        print(gt_path)
        self.number_of_frame = len(os.listdir(self.path + self.data_list[self.count]))-5
        print(self.number_of_frame)
        self.gt = open(gt_path)
        rect = self.gt.readline().split(',')

        self.box = box
        self.default_box = box
        self.tracker = ECOTracker(True)
        self.tracker.init(np.array(image), [int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        self.count_for_online_train = 0
    def update(self, image):
        """Locate target in an new frame and return the estimated bounding box.
        Arguments:
            image {PIL.Image} -- Image in a new frame.
        Returns:
            np.ndarray -- Estimated target bounding box (4x1,
                [left, top, width, height]) in ``image``.
        """
        # rect = self.gt.readline().split(',')
        # real_x, real_y, real_w, real_h = int(float(rect[0])), int(float(rect[1])), int(float(rect[2])), int(float(rect[3]))
        bbox = self.tracker.update(np.array(image), True, True)
        score = self.tracker.score
        score -= score.min()
        score /= score.max()
        score = (score * 255).astype(np.uint8)
        print(score[150:350][150:350].mean())

        new_bbox = []
        new_bbox.append(bbox[0])
        new_bbox.append(bbox[1])
        new_bbox.append(float(bbox[2]) - float(bbox[0]))
        new_bbox.append(float(bbox[3]) - float(bbox[1]))

        return new_bbox
if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()
    # setup experiment
    experiment = ExperimentGOT10k(
        root_dir=ROOT_DIR,          # GOT-10k's root directory
        subset='val',               # 'train' | 'val' | 'test'
        result_dir='./results',       # where to store tracking results
        report_dir='./reports'        # where to store evaluation reports
    )
    # run experiments on GOT-10k
    experiment.run(tracker, visualize=False)
    # report performance on GOT-10k
    experiment.report([tracker.name])
