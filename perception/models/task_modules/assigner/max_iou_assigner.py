import mmengine
import torch.nn as nn

class MaxIouAssigner:
    def __init__(self,pos_iou_thr,neg_iou_thr,min_pos_iou=.0,gt_max_assign_all=True):
        self.pos_iou_thr=pos_iou_thr
        self.neg_iou_thr=neg_iou_thr
        self.min_pos_iou=min_pos_iou
        self.gt_max_assign_all=gt_max_assign_all
