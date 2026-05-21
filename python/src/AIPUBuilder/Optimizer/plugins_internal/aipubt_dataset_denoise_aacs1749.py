# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *

from AIPUBuilder.Optimizer.logger import *
from torch.utils.data import Dataset
import numpy as np


@register_plugin(PluginType.Dataset, '1.0')
class DenoiseAACS1749Dataset(Dataset):
    """
    This plugin is mainly used for the denoise model in jira aacs-1749.
    """
    # when used as calibration dataset, label_file can be omitted.

    def __init__(self, data_file, label_file=None):
        self.data = []
        self.label = None

        with open(data_file) as df:
            for line in df:
                fname = line.strip()
                if len(fname) > 0:
                    self.data.append(fname)

        if label_file is not None:
            self.label = []
            with open(label_file) as df:
                for line in df:
                    fname = line.strip()
                    if len(fname) > 0:
                        self.label.append(fname)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(self.data[idx])
        h, w, _ = img.shape
        x = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        x_uv = x[h:, 0:w]
        x_y = x[0:h, 0:w]
        x_y = np.expand_dims(x_y, axis=0).astype(np.float32)
        x_y /= 255.
        label = float("-inf")
        if self.label is not None:
            gt = cv2.imread(self.label[idx])
            label = np.concatenate([np.expand_dims(img, axis=0), np.expand_dims(gt, axis=0)], axis=0)
        sample = [[x_y], label]
        return sample
