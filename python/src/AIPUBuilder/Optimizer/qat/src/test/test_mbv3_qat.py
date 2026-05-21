# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from AIPUBuilder.Optimizer.plugins.aipubt_dataset_numpy import NumpyDataset
from AIPUBuilder.Optimizer.plugins.aipubt_metric_topk import TopKMetric

from AIPUBuilder.Optimizer.qat.src.plugin.aipubt_train_mbv3 import MobilenetV3TrainLoop
from AIPUBuilder.Optimizer.qat.src.qatlogger import QAT_INFO
from AIPUBuilder.Optimizer.qat.src.qatmaster import AIPUQATMaster
from AIPUBuilder.Optimizer.qat.src.config import QATConfig
from utils import gen_mbv3_config

big_data = False
# big_data = True
config = gen_mbv3_config(big_data=big_data)


def gen_model():
    # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
    # __all__ = [
    #     "MobileNetV3",
    #     "MobileNet_V3_Large_Weights",
    #     "MobileNet_V3_Small_Weights",
    #     "mobilenet_v3_large",
    #     "mobilenet_v3_small",
    # ]

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    mbv3_model = models.mobilenet_v3_small(weights='DEFAULT').to(device)
    torch.save(mbv3_model, 'mbv3_pretrained_cuda0.pt')


def test_fp_acc():
    qat_handler = AIPUQATMaster(config)
    qat_handler.prepare()
    qat_handler.evaluate_loop(qat_handler.qat_quantizer.model)


def train_fused_model():
    qat_handler = AIPUQATMaster(config)
    quantizer = qat_handler.qat_quantizer
    qat_handler.prepare()
    # QAT_INFO(f"orignal mbv3 evaluated acc:")
    # qat_handler.evaluate_loop(quantizer.model)

    QAT_INFO(f"begin to fuse the mbv3 and transform to zhouyi qnn:")
    fused_model = quantizer.fuse()
    QAT_INFO(f"fused mbv3 evaluated acc:")
    qat_handler.reset_metric()
    qat_handler.evaluate_loop(fused_model)

    QAT_INFO(f"fused mbv3 qat_train loop:")
    qat_handler.finetune(qat_handler.qat_quantizer.fused_module)
    qat_handler.export()


if __name__ == '__main__':
    # gen_model()
    # test_fp_acc()
    train_fused_model()
