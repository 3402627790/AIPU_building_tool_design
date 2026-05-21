# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from AIPUBuilder.Optimizer.plugins.aipubt_dataset_numpy import NumpyDataset
from AIPUBuilder.Optimizer.plugins.aipubt_metric_topk import TopKMetric

from AIPUBuilder.Optimizer.qat.src.qatlogger import QAT_INFO
from AIPUBuilder.Optimizer.qat.src.qatmaster import AIPUQATMaster
from AIPUBuilder.Optimizer.qat.src.config import QATConfig
from utils import gen_vitb16_config

big_data = False
# big_data = True
config = gen_vitb16_config(big_data=big_data)


def gen_model():
    # https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    torch.save(vit_model, 'vitb16_pretrained_cuda0.pt')


def test_fp_acc():
    qat_handler = AIPUQATMaster(config)
    qat_handler.prepare()
    qat_handler.evaluate_loop(qat_handler.qat_quantizer.model)
    # acc for 1000pic: 0.817


def train_fused_model():
    qat_handler = AIPUQATMaster(config)
    quantizer = qat_handler.qat_quantizer
    qat_handler.prepare()
    # QAT_INFO(f"orignal mbv3 evaluated acc:")
    # qat_handler.evaluate_loop(quantizer.model)

    QAT_INFO(f"begin to fuse the {config.model_name} and transform to zhouyi qnn:")
    fused_model = quantizer.fuse()
    qat_handler.set_stage(qat_handler.qat_quantizer.fused_module, stage='fp32')
    input_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32).to('cuda:0')

    # torch.onnx.export(qat_handler.qat_quantizer.fused_module,
    #                   (input_tensor, ),
    #                   'vit16_before_qat.onnx',
    #                   input_names=['input'])

    QAT_INFO(f"fused {config.model_name} evaluated acc:")
    qat_handler.reset_metric()
    qat_handler.evaluate_loop(fused_model)

    QAT_INFO(f"fused {config.model_name} qat_train loop:")
    qat_handler.finetune(qat_handler.qat_quantizer.fused_module)
    qat_handler.export()

    # qat_handler.set_stage(qat_handler.qat_quantizer.fused_module, stage='fp32')
    # input_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32).to('cuda:0')
    # torch.onnx.export(qat_handler.qat_quantizer.fused_module,
    #                   (input_tensor, ),
    #                   'vit16.onnx',
    #                   input_names=['input'])
    #   dynamo=True)


if __name__ == '__main__':
    # gen_model()
    # test_fp_acc()
    train_fused_model()
