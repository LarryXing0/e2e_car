import torch
from perception.models.task_modules.assigner.sim_ota_assigner import SimOTAAssinger
import matplotlib.pyplot as plt
from mmengine.structures import InstanceData
from unittest import TestCase

def assigner_view(pred_instance, gt_instance):
    pass


class TestAssigner(TestCase):
    def test_simota(self):
        assigner = SimOTAAssinger(
            center_radius=2.5, candidate_topk=1, iou_weight=3.0, cls_weight=1.0
        )
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43], [4, 5, 6, 7]]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [4, 5, 6, 7]]),
        )
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43]]), labels=torch.LongTensor([0])
        )
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances
        )

        expected_gt_inds = torch.LongTensor([1, 0])


