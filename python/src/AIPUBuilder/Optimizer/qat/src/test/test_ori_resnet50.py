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
from AIPUBuilder.Optimizer.qat.src.config import QATConfig
from utils import gen_config

config = gen_config(big_data=True)


def gen_model():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    resnet_model = models.resnet50(pretrained=True).to(device)
    torch.save(resnet_model, 'resnet50_pretrained_cd1.pt')


def evaluate_ori_model():
    qat_handler = AIPUQATMaster(config)
    qat_handler.prepare()
    qat_handler.evaluate_loop(qat_handler.qat_quantizer.model)


def train_ori_model():
    qat_handler = AIPUQATMaster(config)
    qat_handler.prepare()
    qat_handler.train_loop(qat_handler.qat_quantizer.model)


if __name__ == "__main__":
    train_ori_model()
    # gen_model()


# train_data = '/project/ai/scratch03/Dataset/ImageNet/preprocess/torch-1_12-resnet_v1_50/dataset_20.npy'
# train_label = '/project/ai/scratch03/Dataset/ImageNet/preprocess/torch-1_12-resnet_v1_50/label_20.npy'
# data = '/project/ai/scratch03/Dataset/ImageNet/preprocess/torch-1_12-resnet_v1_50/dataset_1000.npy'
# label = '/project/ai/scratch03/Dataset/ImageNet/preprocess/torch-1_12-resnet_v1_50/label_1000.npy'

# metric = TopKMetric()

# train_dataset = NumpyDataset(train_data, train_label)
# eval_dataset = NumpyDataset(data, label)

# train_dataloader = DataLoader(train_dataset, batch_size=1)
# eval_dataloader = DataLoader(eval_dataset, batch_size=10)

# for i, (input_data, target) in enumerate(eval_dataloader):
#     for j,d in enumerate(input_data):
#         input_data[j] = d.to(device)
#         if isinstance(target, (list, tuple)):
#             target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in target]
#         else:
#             target = target.to(device) if isinstance(target, torch.Tensor) else target
#     if i ==0:
#         print(input_data[0].shape)
#     out = resnet_model(input_data[0].permute([0, 3, 1, 2]).float())
#     if i ==0:
#         print(out.shape)
#     out = torch.nn.functional.softmax(out, -1)
#     metric([out], target)

# QAT_INFO(f"original float pre-trained model top1 metric: {metric.compute()}")


# train_loop = ResNet50TrainLoop(epochs=1)


# def eval_func(model):
#     model.eval()
#     for i, (input_data, target) in enumerate(eval_dataloader):
#         for j,d in enumerate(input_data):
#             input_data[j] = d.to(device)
#             if isinstance(target, (list, tuple)):
#                 target = [t.to(device) if isinstance(t, torch.Tensor) else t for t in target]
#             else:
#                 target = target.to(device) if isinstance(target, torch.Tensor) else target
#         # if i ==0:
#         #     print(input_data[0].shape)
#         out = model(input_data[0].permute([0, 3, 1, 2]).float())
#         # if i ==0:
#         #     print(out.shape)
#         out = torch.nn.functional.softmax(out, -1)
#         metric([out], target)
#     print(f"top1 metric: {metric.compute()}")

# trained_model=train_loop(resnet_model, train_dataloader, eval_func=eval_func)
