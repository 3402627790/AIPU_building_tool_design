# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from AIPUBuilder.Optimizer.plugins.aipubt_dataset_numpy import NumpyDataset
from AIPUBuilder.Optimizer.plugins.aipubt_metric_topk import TopKMetric

from AIPUBuilder.Optimizer.qat.plugin.aipubt_train_resnet50 import ResNet50TrainLoop
from AIPUBuilder.Optimizer.qat.qatlogger import QAT_INFO
from AIPUBuilder.Optimizer.qat.qatmaster import AIPUQATMaster
from utils import gen_config

config = gen_config()

qat_handler = AIPUQATMaster(config)
quantizer = qat_handler.qat_quantizer


def fuse_model():
    fused_model = quantizer.fuse()
    fused_fx_graph = fused_model.graph
    print(fused_fx_graph)


def evaluate_fused_model():
    qat_handler.prepare()
    QAT_INFO(f"orignal resnet 50 evaluated acc:")
    qat_handler.evaluate_loop(quantizer.model)

    QAT_INFO(f"begin to fuse the resnet50 and transform to zhouyi qnn:")
    fused_model = quantizer.fuse()
    QAT_INFO(f"fused resnet 50 evaluated acc:")
    qat_handler.reset_metric()
    qat_handler.evaluate_loop(fused_model)


def train_fused_model():
    qat_handler.prepare()
    QAT_INFO(f"orignal resnet 50 evaluated acc:")
    qat_handler.evaluate_loop(quantizer.model)

    QAT_INFO(f"begin to fuse the resnet50 and transform to zhouyi qnn:")
    fused_model = quantizer.fuse()
    QAT_INFO(f"fused resnet 50 evaluated acc:")
    qat_handler.reset_metric()
    qat_handler.evaluate_loop(fused_model)

    QAT_INFO(f"fused resnet 50 train loop:")
    qat_handler.train_loop(qat_handler.qat_quantizer.fused_module)


if __name__ == '__main__':
    # fuse_model()
    # evaluate_fused_model()
    train_fused_model()
