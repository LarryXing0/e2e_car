from abc import ABCMeta, abstractmethod
from typing import Optional


class BaseAssinger(metaclass=ABCMeta):
    @abstractmethod
    def assign(self, pred_instances, gt_instances, gt_instances_ignore, *kwargs):
        """assign bboxes to either a ground truth boxes or a negative boxes"""
