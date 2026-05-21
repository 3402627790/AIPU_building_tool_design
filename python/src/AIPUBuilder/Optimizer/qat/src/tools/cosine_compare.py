# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
import torch
from AIPUBuilder.Optimizer.framework import PyGraph, QuantizeGraph
from AIPUBuilder.Optimizer.analyzer.cosine import check_nodes_similarity
import os


def main():
    bp = 'vitb16/IR'
    firt = os.path.join(bp, 'vitb16__opsapi_fp.txt')
    firb = os.path.join(bp, 'vitb16__opsapi_fp.bin')
    qirt = os.path.join(bp, 'vitb16__opsapi_qat.txt')
    qirb = os.path.join(bp, 'vitb16__opsapi_qat.bin')
    oirt = os.path.join(bp, 'vitb16_opt_o.txt')
    oirb = os.path.join(bp, 'vitb16_opt_o.bin')

    fg = PyGraph.parse(firt, firb)
    qg = PyGraph.parse(qirt, qirb)
    # og = PyGraph.parse(oirt, oirb)

    QuantizeGraph.deduce_quantization_infos(qg)
    # QuantizeGraph.deduce_quantization_infos(og)

    indata = np.load('./model_dataset/vitb16/dataset/dataset_20.npy')
    print(indata.shape)
    for i in range(indata.shape[0]):
        if i in [5]:
            # if i in [3, 5, 7]:
            # print(f"check float IR with opt IR.....")
            # check_nodes_similarity(fg, og, indata[i:i+1, ...].transpose([0, 3, 1, 2]))
            print(f"\n\ncheck float IR with qat IR.....")
            check_nodes_similarity(fg, qg, indata[i:i+1, ...].transpose([0, 3, 1, 2]))
            print(f"\n")


if __name__ == '__main__':
    main()
