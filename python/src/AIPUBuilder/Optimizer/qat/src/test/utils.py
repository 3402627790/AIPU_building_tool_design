# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import os
from AIPUBuilder.Optimizer.config import CfgParser


class TmpConfig(object):
    argv = {}

    def __init__(self):
        pass

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self.argv.update({key: value})

    def __getattr__(self, key):
        if key in self.argv:
            return self.argv[key]
        else:
            return None


bp = '/project/ai/zhouyi_compass/samhan01/model_dataset/'


def gen_vitb16_config(big_data=False):
    config = TmpConfig()

    model_name = 'vitb16'

    config.ir_mode = 'quant'
    config.model_name = model_name
    config.output_dir = './'
    # quantization relative fields
    config.weight_bits = 8
    config.bias_bits = 32
    # config.bias_effective_bits = 28
    config.activation_bits = 8
    config.lut_items_in_bits = 8
    config.quantize_method_for_weight = 'per_tensor_symmetric_restricted_range'
    # config.quantize_method_for_activation = 'per_tensor_symmetric_full_range'
    config.quantize_method_for_activation = 'per_tensor_asymmetric'
    config.calibration_strategy_for_activation = 'mean'
    config.calibration_strategy_for_weight = 'extrema'
    config.running_statistic_momentum = 0.9

    config.input_model = os.path.join(bp, model_name, f'{model_name}_pretrained_cuda0.pt')
    if not big_data:
        config.train_data = os.path.join(bp, model_name, 'dataset/dataset_20.npy')
        config.train_label = os.path.join(bp, model_name, 'dataset/label_20.npy')
        config.data = os.path.join(bp, model_name, 'dataset/dataset_1000.npy')
        config.label = os.path.join(bp, model_name, 'dataset/label_1000.npy')
    else:
        config.train_data = os.path.join(bp, model_name, 'dataset/dataset_1000.npy')
        config.train_label = os.path.join(bp, model_name, 'dataset/label_1000.npy')
        config.data = os.path.join(bp, model_name, 'dataset/dataset_20000.npy')
        config.label = os.path.join(bp, model_name, 'dataset/label_20000.npy')

    config.dataset = 'numpynhwc2nchwdataset'
    config.metric = 'topkmetric'
    config.train = 'ViTB16trainLoop(0.000001,1)'

    config.train_batch_size = 50
    config.metric_batch_size = 100
    config.dataloader_workers = 0
    # config.train_shuffle = False
    config.train_shuffle = True
    config.parallel = False

    config.input_shape = [[1, 3, 224, 224]]
    config.concrete_args = None

    for k, v in config.__dict__.items():
        config.argv[k] = v
        print(k, v)

    # print(config.argv.items())
    return config


def gen_mbv3_config(big_data=False):
    config = TmpConfig()

    model_name = 'mbv3'

    # config.ir_mode = 'quant'
    config.ir_mode = 'fp'
    config.model_name = model_name
    config.output_dir = './'
    # quantization relative fields
    config.weight_bits = 8
    config.bias_bits = 32
    # config.bias_effective_bits = 28
    config.activation_bits = 8
    config.lut_items_in_bits = 8
    # config.quantize_method_for_weight = 'per_tensor_symmetric_restricted_range'
    config.quantize_method_for_weight = 'per_channel_symmetric_restricted_range'
    # config.quantize_method_for_activation = 'per_tensor_symmetric_full_range'
    config.quantize_method_for_activation = 'per_tensor_asymmetric'
    config.calibration_strategy_for_activation = 'mean'
    # config.calibration_strategy_for_activation = 'extrema'
    config.calibration_strategy_for_weight = 'extrema'
    config.running_statistic_momentum = 0.9

    # config.input_model = os.path.join(bp, 'mbv3_pretrained_cuda0.pt')
    config.input_model = os.path.join(bp, model_name, f'mbv3_large_pretrained_cuda0.pt')
    # config.input_model = '/project/ai/zhouyi_compass/samhan01/work/gitlab/optt/Compass_OpportunePostTrainingTools/AIPUBuilder/Optimizer/qat/test/resnet50_pretrained.pt'
    if not big_data:
        config.train_data = os.path.join(bp, model_name, 'dataset/dataset_20.npy')
        config.train_label = os.path.join(bp, model_name, 'dataset/label_20.npy')
        config.data = os.path.join(bp, model_name, 'dataset/dataset_1000.npy')
        config.label = os.path.join(bp, model_name, 'dataset/label_1000.npy')
    else:
        # config.train_data = os.path.join(bp, model_name, 'dataset/dataset_1000.npy')
        # config.train_label = os.path.join(bp, model_name, 'dataset/label_1000.npy')
        config.train_data = os.path.join(bp, model_name, 'dataset/dataset_3000.npy')
        config.train_label = os.path.join(bp, model_name, 'dataset/label_3000.npy')
        # config.train_data = os.path.join(bp, model_name, 'dataset/dataset_500.npy')
        # config.train_label = os.path.join(bp, model_name, 'dataset/label_500.npy')
        config.data = os.path.join(bp, model_name, 'dataset/dataset_20000.npy')
        config.label = os.path.join(bp, model_name, 'dataset/label_20000.npy')

    config.dataset = 'numpynhwc2nchwdataset'
    config.metric = 'topkmetric'
    config.train = 'MobilenetV3trainLoop(0.00001, 5)'

    config.train_batch_size = 50
    config.metric_batch_size = 100
    config.dataloader_workers = 0
    # config.train_shuffle = False
    config.train_shuffle = True
    config.parallel = False

    config.input_shape = [[1, 3, 224, 224]]
    config.concrete_args = None

    for k, v in config.__dict__.items():
        config.argv[k] = v
        print(k, v)

    # print(config.argv.items())
    return config


def gen_resnet50_config(big_data=False):
    config = TmpConfig()

    # quantization relative fields
    model_name = 'resnet50'
    config.ir_mode = 'quant'
    config.model_name = model_name
    config.output_dir = './'
    config.weight_bits = 8
    config.bias_bits = 32
    config.bias_effective_bits = 28
    config.activation_bits = 8
    config.lut_items_in_bits = 8
    config.quantize_method_for_weight = 'per_tensor_symmetric_restricted_range'
    config.quantize_method_for_activation = 'per_tensor_symmetric_full_range'
    config.calibration_strategy_for_activation = 'extrema'
    config.calibration_strategy_for_weight = 'extrema'
    config.running_statistic_momentum = 0.9

    config.input_model = os.path.join(bp, model_name, 'resnet50_pretrained.pt')
    # config.input_model = os.path.join(bp, 'resnet50_pretrained_cpu.pt')
    if not big_data:
        config.train_data = os.path.join(bp, model_name, 'dataset/dataset_20.npy')
        config.train_label = os.path.join(bp, model_name, 'dataset/label_20.npy')
        config.data = os.path.join(bp, model_name, 'dataset/dataset_1000.npy')
        config.label = os.path.join(bp, model_name, 'dataset/label_1000.npy')
    else:
        config.train_data = os.path.join(bp, model_name, 'dataset//dataset_1000.npy')
        config.train_label = os.path.join(bp, model_name, 'dataset/label_1000.npy')
        config.data = os.path.join(bp, model_name, 'dataset/dataset_20000.npy')
        config.label = os.path.join(bp, model_name, 'dataset/label_20000.npy')

    config.dataset = 'numpynhwc2nchwdataset'
    config.metric = 'topkmetric'
    config.train = 'ResNet50TrainLoop'

    config.train_batch_size = 50
    config.metric_batch_size = 100
    config.dataloader_workers = 0
    # config.train_shuffle = False
    config.train_shuffle = True
    config.parallel = False

    config.input_shape = [[1, 3, 224, 224]]
    config.concrete_args = None

    for k, v in config.__dict__.items():
        config.argv[k] = v
        # print(k, v)

    return config
