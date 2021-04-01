# 1. Quick Start: A Concise Example

from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
from got10k.experiments import ExperimentOTB
from got10k.experiments import ExperimentVOT

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
    
    def init(self, image, box):
        """Initialize your tracking model in the first frame
        
        Arguments:
            image {PIL.Image} -- Image in the first frame.
            box {np.ndarray} -- Target bounding box (4x1,
                [left, top, width, height]) in the first frame.
        """
        self.box = box

    def update(self, image):
        """Locate target in an new frame and return the estimated bounding box.
        
        Arguments:
            image {PIL.Image} -- Image in a new frame.
        
        Returns:
            np.ndarray -- Estimated target bounding box (4x1,
                [left, top, width, height]) in ``image``.
        """
        return self.box

if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()

    # setup experiment (validation subset)
    experiment = ExperimentGOT10k(
        root_dir="D:/GOT/",          # GOT-10k's root directory
        # experiments=('supervised')
        subset='val',               # 'train' | 'val' | 'test'
        # version=2013
        result_dir='results',       # where to store tracking results
        report_dir='reports'        # where to store evaluation reports
    )

    # run experiments on GOT-10k
    experiment.run(tracker, visualize=False)

    # report performance on GOT-10k (validation subset)
    experiment.report([tracker.name])