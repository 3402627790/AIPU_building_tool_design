# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
import numpy as np
from AIPUBuilder.Optimizer.utils import cosine_distance

x = torch.randn(1, 256, 4608, 1)
w1 = torch.from_numpy(np.load('/home/zonyu01/Downloads/w1.npy'))  # [256,256,1,1]
b1 = torch.from_numpy(np.load('/home/zonyu01/Downloads/b1.npy'))  # [256]
w2 = torch.from_numpy(np.load('/home/zonyu01/Downloads/w2.npy'))  # [1,8,640,32]
# original graph
y = torch.nn.functional.conv2d(x, w1, b1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
z = torch.matmul(w2, y.reshape(1, 8, 32, 4608))  # [1,8,640,4608]
# merged graph
w = w2.matmul(w1.reshape(1, 8, 32, 256))  # [1,8,640,256]
b = w2.matmul(b1.reshape(1, 8, 32, 1))  # [1,8,640,1]
zm = w.matmul(x.reshape(1, 1, 256, 4608)) + b

err = (z - zm).abs().max()
cos = cosine_distance(z, zm)
print(f'max_abs_err = {err}, cos = {cos}')
