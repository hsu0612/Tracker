# 1. Quick Start: A Concise Example

from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k

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
    experiment = ExperimentGOT10k('/media/hsu/data/GOT', subset='val')

    # run experiments on GOT-10k
    experiment.run(tracker, visualize=False)

    # report performance on GOT-10k (validation subset)
    experiment.report([tracker.name])