# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from AIPUBuilder.Optimizer.plugins.aipubt_dataset_numpy import NumpyDataset
from AIPUBuilder.Optimizer.plugins.aipubt_metric_topk import TopKMetric

from AIPUBuilder.Optimizer.qat.src.plugin.aipubt_train_resnet50 import ResNet50TrainLoop
from AIPUBuilder.Optimizer.qat.src.qatlogger import QAT_INFO
from AIPUBuilder.Optimizer.qat.src.qatmaster import AIPUQATMaster
from utils import gen_resnet50_config

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
big_data = False
# big_data = True
config = gen_resnet50_config(big_data=big_data)

qat_handler = AIPUQATMaster(config)
quantizer = qat_handler.qat_quantizer


def test_dp():
    qat_handler.prepare()
    QAT_INFO(f"orignal resnet 50 evaluated acc:{0.7447 if big_data else 0.722}")
    qat_handler.evaluate_loop(quantizer.model)

    QAT_INFO(f"begin to fuse the resnet50 and transform to zhouyi qnn:")
    fused_model = quantizer.fuse()
    QAT_INFO(f"fused resnet 50 evaluated acc:0.7447")
    qat_handler.reset_metric()
    qat_handler.evaluate_loop(fused_model)

    QAT_INFO(f"fused resnet 50 qat_train loop:")
    qat_handler.finetune(qat_handler.qat_quantizer.fused_module)
    qat_handler.export()


def train_fused_model():
    qat_handler.prepare()
    QAT_INFO(f"orignal resnet 50 evaluated acc:{0.7447 if big_data else 0.722}")
    # qat_handler.evaluate_loop(quantizer.model)

    QAT_INFO(f"begin to fuse the resnet50 and transform to zhouyi qnn:")
    fused_model = quantizer.fuse()
    QAT_INFO(f"fused resnet 50 evaluated acc:0.7447")
    qat_handler.reset_metric()
    qat_handler.evaluate_loop(fused_model)

    QAT_INFO(f"fused resnet 50 qat_train loop:")
    qat_handler.finetune(qat_handler.qat_quantizer.fused_module)
    qat_handler.export()
    # qat_handler.set_stage(qat_handler.qat_quantizer.fused_module, stage='fp32')
    # input_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32).to('cuda:0')
    # torch.onnx.export(qat_handler.qat_quantizer.fused_module,
    #                   (input_tensor, ),
    #                   'resnet50.onnx',
    #                   input_names=['input'])


if __name__ == '__main__':
    # fuse_model()
    # evaluate_fused_model()
    train_fused_model()
    # test_dp()
