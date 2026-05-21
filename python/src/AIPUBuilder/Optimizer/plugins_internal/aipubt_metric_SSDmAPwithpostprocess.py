# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.plugins.aipubt_metric_SSDmAP import SSDmAPMetric
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.plugins.aipubt_op_ssd_postprocess import SSDPostProcess
import numpy as np
import torch
import os


@register_plugin(PluginType.Metric, '1.0')
class SSDmAPWithPostProcessMetric(SSDmAPMetric):
    """
    This SSDmAPWithPostProcessMetric is used for the metric of resnet_34_ssd_onnx models in Optimizer.
    This plugin firstly do postprocess which including decodebox and nms and then computes the mAP of SSD models.
    """

    def __init__(self, anchor_path=None):
        super().__init__(80, 0)

        if anchor_path is not None:
            if not os.path.exists(anchor_path):
                OPT_ERROR(f"please check the anchor_path={anchor_path} which is not existed")
            anchor = np.load(anchor_path)
            self.db_params = {
                'anchor': anchor,
                'layer_bottom_shape': [[1, 15130, 81], [1, 15130, 4]],
                'class_num': 80,
                'feature_map': [[50, 50], [25, 25], [13, 13], [7, 7], [3, 3], [3, 3]],
                'score_threshold': 0.55,
                'image_height': 1200,
                'image_width': 1200,
                # 'variance': [10., 10., 10., 10.],
            }
            self.nms_params = {
                'image_height': 1200,
                'image_width': 1200,
            }

    def __call__(self, backbone_pred, target):
        assert len(backbone_pred) == 2, OPT_FATAL('please check the outputs number(should be 2)')

        scores = backbone_pred[0]
        boxes = backbone_pred[1]
        scores_with_softmax = torch.nn.functional.softmax(scores, dim=-1)
        out_db, out_nms = SSDPostProcess(decodebox_params=self.db_params, nms_params=self.nms_params)(
            input_tensors=[scores_with_softmax, boxes])

        pred = out_db + out_nms
        super().__call__(pred, target)

    def report(self):
        return "SSD mAP accuracy is %f" % (self.compute())
