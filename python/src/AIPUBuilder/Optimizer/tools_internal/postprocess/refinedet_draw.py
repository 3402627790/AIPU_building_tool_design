# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import matplotlib.pyplot as plt
import numpy as np


def output_draw(data, color='r'):
    fig = plt.figure()
    plt.title("lib output draw")
    frame_num = data.shape[0]
    for i in range(frame_num):
        plt.gca().add_patch(plt.Rectangle(xy=(data[i][1], data[i][0]),
                                          width=data[i][3] - data[i][1],
                                          height=data[i][2] - data[i][0],
                                          edgecolor=color, alpha=0.3,
                                          fill=False, linewidth=2))
    plt.show()


lib_new_output = np.array([
    0x003b, 0x00ff, 0x2bdb, 0x1431, 0x4bfb, 0x33e3, 0x003b, 0x00ff,
    0x4c00, 0x1414, 0x6c22, 0x33f0, 0x0029, 0x00ff, 0x1d5e, 0x589a,
    0x4a84, 0x6f2c, 0x0027, 0x00ff, 0x49d2, 0x69ee, 0x59fc, 0x79d8,
    0x000c, 0x00ff, 0x440e, 0x3c0a, 0x6430, 0x5bc0
])

lib_old_output = np.array([
    0x003b, 0x00ff, 0x27c9, 0x1af0, 0x4bff, 0x307a, 0x003b, 0x00ff,
    0x4eff, 0x141b, 0x6d6f, 0x3487, 0x0029, 0x00ff, 0x1d20, 0x56a2,
    0x47f6, 0x6bbc, 0x0027, 0x00ff, 0x43f6, 0x68ab, 0x58f0, 0x76bd,
    0x000c, 0x00ff, 0x4574, 0x3cba, 0x69c6, 0x55f2, 0x0029, 0x0017,
    0x1a20, 0x5a53, 0x3be2, 0x6703, 0x0029, 0x0003, 0x1e79, 0x6047,
    0x3ccb, 0x6b79,
])
lib_new_output = lib_new_output.reshape([-1, 6])
lib_old_output = lib_old_output.reshape([-1, 6])

lib_new_d = lib_new_output[:, 2:] / 32767.
lib_old_d = lib_old_output[:, 2:] / 32767.

output_draw(lib_new_d, 'r')
output_draw(lib_old_d, 'b')
