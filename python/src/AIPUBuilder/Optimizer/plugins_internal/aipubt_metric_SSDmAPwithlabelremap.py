# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.plugins.aipubt_metric_mAP import mAPMetric, BasemAPMetric
from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from collections import defaultdict
import numpy as np


@register_plugin(PluginType.Metric, '1.0')
class SSDmAPwithLabelReMapMetric(mAPMetric):
    """
    This SSDmAPwithLabelReMapMetric is used for the metric of inception_v2_ssd models in Optimizer.
    This plugin computes the mAP of SSD models when 80 class_num in labels but 90 class num in forward output,
    this plugin will remap the 90 class num to 80 class num aligning with the labels.
    """

    def __init__(self, class_num=80, start_label=1):
        super().__init__(class_num, start_label)
        self.coco_80_label_map = self.get_label_map()

    def __call__(self, pred, target):
        assert len(pred) == 9, OPT_FATAL('please check the outputs number(should be 9 in SSDmAPwithLabelReMapMetric)')
        batch = pred[2].shape[0]

        pred_list = []
        targets_list = []
        for pd in pred:
            pred_list.append(pd.cpu().numpy())
        for b in range(batch):
            targets = {}
            for k, v in target.items():
                targets.update({k: v[b].cpu().numpy()})
            targets_list.append(targets)

        for i in range(batch):
            predict = defaultdict()
            obj_class_num = pred_list[2][i]
            label_pre_class = pred_list[4][i]
            boxes = pred_list[5][i]
            box_num_pre_class = pred_list[6][i]
            scores = pred_list[7][i]
            # keep = pred_list[8][i]

            label_id_list = []
            boxes_list = []
            score_list = []
            all_box_idx = 0
            for cid in range(int(obj_class_num[0])):
                box_num_cur_class = int(box_num_pre_class[cid])
                label_id = label_pre_class[cid]
                if label_id < self.start_label:
                    all_box_idx = all_box_idx + box_num_cur_class
                    continue

                label_id_list.extend([label_id]*box_num_cur_class)
                boxes_list.extend([boxes[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])
                score_list.extend([scores[all_box_idx+box_idx]
                                   for box_idx in range(box_num_cur_class)])

                all_box_idx = all_box_idx + box_num_cur_class
            label_id_list = [self.coco_80_label_map[ll] - self.start_label for ll in label_id_list]
            predict.update({'label_index': label_id_list})
            predict.update({'bbox': boxes_list})
            predict.update({'confidence': score_list})

            predict.update({'image_name': targets_list[i]['image_name']})
            BasemAPMetric.extract_obj_all_class(predict, self.predicts)
            BasemAPMetric.extract_obj_all_class(targets_list[i], self.targets)

    def reset(self):
        super().reset()

    def compute(self):
        self.eval_mAP(self.predicts, self.targets, 80, 0)
        return self.mAP

    def report(self):
        return f"SSD mAP accuracy is {self.compute()}"

    def get_label_map(self):
        origin_coco_labels = list(range(91))
        coco_80_missing_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
        coco_80_label_map = {}
        for l in origin_coco_labels:
            l_offset = np.sum(l > np.array(coco_80_missing_labels, np.int64))
            coco_80_label_map.update({l: l - l_offset})
        return coco_80_label_map
