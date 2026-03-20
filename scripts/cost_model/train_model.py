import argparse
import os
import sys
import math
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import configparser
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
from ast import literal_eval
from AIPUBuilder.Parser.univ_main import main as aipu_parser_main


class BaseDataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0):
        batch_size = min(batch_size, len(dataset))
        num_workers = min([os.cpu_count(), batch_size, num_workers])
        if batch_size == 1:
            num_workers = 0
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=True,
                         sampler=None)
        object.__setattr__(self,
                           'batch_sampler',
                           BatchSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class BatchSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while self.sampler is not None:
            yield from iter(self.sampler)


class TrainDataLoader(BaseDataLoader):
    def __init__(self,
                 data_root,
                 data_split,
                 target,
                 input_size,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0):
        assert data_split in ['train', 'val']
        dataset = TrainDataset(data_root, data_split, target, input_size)
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers)


class EvalDataLoader(BaseDataLoader):
    def __init__(self,
                 case_info,
                 input_size,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0):
        dataset = EvalDataset(case_info, input_size)
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, input_size):
        self.input_size = input_size
        self.keys = []
        self.inputs = []
        self.labels = []

    def preprocess(self, d):
        op_type = d['op_type']
        params = d['params']
        in_shapes = [literal_eval(x['shape']) for x in d['inputs']]
        out_shapes = [literal_eval(x['shape']) for x in d['outputs']]
        inp, lbl = [], [int(x) for x in d['perf_data']]
        if op_type == 'Activation':
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['method']),
                math.prod(in_shapes[0]),
            ])
        elif op_type in ['Add', 'Mul', 'Sub']:
            inp.extend([
                self.get_key_id(op_type),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(in_shapes[1]),
            ])
        elif op_type == 'BatchToSpace':
            inp.extend([
                self.get_key_id(op_type),
                int(params['block_size_x']),
                int(params['block_size_y']),
                *self.to_nhwc(in_shapes[0]),
            ])
        elif op_type == 'Cast':
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(d['inputs'][0]['dtype']),
                self.get_key_id(params['to_dtype']),
                math.prod(in_shapes[0]),
            ])
        elif op_type == 'Concat':
            axis = int(params['axis'])
            for k, shape in enumerate(in_shapes):
                if len(shape) == 5:
                    in_shapes[k] = self.to_nhwc(shape)
            dims = []
            for k, dim in enumerate(in_shapes[0]):
                if k != axis:
                    dims.append(dim)
            for k, shape in enumerate(in_shapes):
                if k >= 10:
                    break
                dims.append(shape[axis])
            inp.extend([
                self.get_key_id(op_type),
                axis,
                *dims,
            ])
        elif op_type == 'Convolution':
            inp.extend([
                self.get_key_id(op_type),
                int(params['dilation_x']),
                int(params['dilation_y']),
                int(params['kernel_x']),
                int(params['kernel_y']),
                int(params['stride_x']),
                int(params['stride_y']),
                int(params['group']),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'Crop':
            inp.extend([
                self.get_key_id(op_type),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'DepthToSpace':
            inp.extend([
                self.get_key_id(op_type),
                int(params['block_size_x']),
                int(params['block_size_y']),
                *self.to_nhwc(in_shapes[0]),
            ])
        elif op_type == 'Eltwise':
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['method']),
                math.prod(in_shapes[0]),
            ])
        elif op_type == 'FullyConnected':
            inp.extend([
                self.get_key_id(op_type),
                *in_shapes[0],
                *out_shapes[0],
            ])
        elif op_type == 'Gather':
            inp.extend([
                self.get_key_id(op_type),
                *in_shapes[0],
                *in_shapes[1],
            ])
        elif op_type == 'MatMul':
            raw_in_shapes = literal_eval(params['origin_bottom_shapes'])
            inp.extend([
                self.get_key_id(op_type),
                raw_in_shapes[0][0] * raw_in_shapes[0][1],
                raw_in_shapes[0][2],
                raw_in_shapes[0][3],
                raw_in_shapes[1][3],
            ])
        elif op_type == 'Pad':
            inp.extend([
                self.get_key_id(op_type),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'Pooling':
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['method']),
                int(params['dilation_x']),
                int(params['dilation_y']),
                int(params['kernel_x']),
                int(params['kernel_y']),
                int(params['stride_x']),
                int(params['stride_y']),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'Resize':
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['method']),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'ReverseSequence':
            inp.extend([
                self.get_key_id(op_type),
                int(params['time_axis']),
                int(params['batch_axis']),
                *in_shapes[0],
            ])
        elif op_type in ['Rsqrt', 'SquaredDifference']:
            inp.extend([
                self.get_key_id(op_type),
                math.prod(in_shapes[0]),
            ])
        elif op_type == 'Softmax':
            axis = int(params['axis'])
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            dim = shape[axis]
            inp.extend([
                self.get_key_id(op_type),
                axis,
                math.prod(shape) / dim,
                dim,
            ])
        elif op_type == 'SpaceToDepth':
            inp.extend([
                self.get_key_id(op_type),
                int(params['block_size_x']),
                int(params['block_size_y']),
                *self.to_nhwc(in_shapes[0]),
            ])
        elif op_type == 'Split':
            splits = literal_eval(params['splits'])
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            dims = []
            for k, dim in enumerate(splits):
                if k >= 10:
                    break
                dims.append(dim)
            inp.extend([
                self.get_key_id(op_type),
                int(params['axis']),
                *shape,
                *dims,
            ])
        elif op_type == 'Tile':
            inp.extend([
                self.get_key_id(op_type),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'Transpose':
            perm = literal_eval(params['perm'])
            inp.extend([
                self.get_key_id(op_type),
                *in_shapes[0],
                *perm,
            ])
        elif op_type == 'LayoutConvertor':
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['from']),
                self.get_key_id(params['to']),
                *in_shapes[0],
            ])
        elif op_type == 'Normalization':
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            axis = literal_eval(params['axis'])
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['method']),
                *axis,
                *shape,
            ])
        elif op_type == 'BatchNorm':
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            inp.extend([
                self.get_key_id(op_type),
                int(params['axis']),
                *shape,
            ])
        elif op_type == 'GroupNorm':
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            inp.extend([
                self.get_key_id(op_type),
                int(params['axis']),
                int(params['group']),
                *shape,
            ])
        elif op_type == 'InstanceNorm':
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            inp.extend([
                self.get_key_id(op_type),
                *shape,
            ])
        elif op_type == 'LayerNorm':
            axis = literal_eval(params['axis'])
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            inp.extend([
                self.get_key_id(op_type),
                *axis,
                *shape,
            ])
        elif op_type == 'RMSNorm':
            axis = literal_eval(params['axis'])
            shape = in_shapes[0]
            eps = float(params['epsilon'])
            inp.extend([
                self.get_key_id(op_type),
                *axis,
                *shape,
            ])
        elif op_type == 'GridSample':
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['method']),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(in_shapes[1]),
            ])
        elif op_type == 'Slice':
            inp.extend([
                self.get_key_id(op_type),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'Reduce':
            axis = literal_eval(params['axis'])
            shape = in_shapes[0]
            if len(shape) == 5:
                shape = self.to_nhwc(shape)
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(params['method']),
                *axis,
                *shape,
            ])
        elif op_type == 'ConvTranspose':
            inp.extend([
                self.get_key_id(op_type),
                int(params['dilation_x']),
                int(params['dilation_y']),
                int(params['kernel_x']),
                int(params['kernel_y']),
                int(params['stride_x']),
                int(params['stride_y']),
                int(params['group']),
                *self.to_nhwc(in_shapes[0]),
                *self.to_nhwc(out_shapes[0]),
            ])
        elif op_type == 'BasicLSTM':
            inp.extend([
                self.get_key_id(op_type),
                int(params['cell_size']),
                *in_shapes[0],
            ])
        elif op_type == 'ChannelShuffle':
            shape = self.to_nhwc(in_shapes[0])
            inp.extend([
                self.get_key_id(op_type),
                int(params['group']),
                math.prod(shape) / shape[3],
                shape[3],
            ])
        elif op_type in ['Sine', 'Cosine']:
            inp.extend([
                self.get_key_id(op_type),
                math.prod(in_shapes[0]),
            ])
        elif op_type in ['Quantize', 'DeQuantize']:
            inp.extend([
                self.get_key_id(op_type),
                self.get_key_id(d['inputs'][0]['dtype']),
                self.get_key_id(d['outputs'][0]['dtype']),
                math.prod(in_shapes[0]),
            ])
        else:
            raise ValueError(f'Unknown op_type: {op_type}')
        return inp, lbl

    def encode(self, inp, lbl):
        tmp = []
        for num in inp:
            tmp.extend(self.split_num(num))
        if len(tmp) < self.input_size:
            for _ in range(self.input_size - len(tmp)):
                tmp.append(0)
        inp = tmp
        assert len(inp) == self.input_size
        tmp = []
        for num in lbl:
            base, exp = self.split_num(num)
            tmp.append(math.log(base + 1e-7))
            tmp.append(math.log(exp + 1e-7))
        lbl = tmp
        assert len(lbl) == 10
        return inp, lbl

    def get_key_id(self, key):
        if len(self.keys) == 0:
            self.keys.extend([
                [
                    # OpType keys
                    'Activation',
                    'Add',
                    'Mul',
                    'Sub',
                    'BatchToSpace',
                    'Cast',
                    'Concat',
                    'Convolution',
                    'Crop',
                    'DepthToSpace',
                    'Eltwise',
                    'FullyConnected',
                    'Gather',
                    'MatMul',
                    'Pad',
                    'Pooling',
                    'Resize',
                    'ReverseSequence',
                    'Rsqrt',
                    'SquaredDifference',
                    'Softmax',
                    'SpaceToDepth',
                    'Split',
                    'Tile',
                    'Transpose',
                    'LayoutConvertor',
                    'Normalization',
                    'BatchNorm',
                    'GroupNorm',
                    'InstanceNorm',
                    'LayerNorm',
                    'GridSample',
                    'Slice',
                    'Reduce',
                    'ConvTranspose',
                    'BasicLSTM',
                    'ChannelShuffle',
                    'Sine',
                    'Cosine',
                    'Quantize',
                    'DeQuantize',
                    # new add
                    'RMSNorm',
                ],
                [
                    # Method keys
                    'LEAKYRELU',
                    'PRELU',
                    'SIGMOID',
                    'SILU',
                    'TANH',
                    'HARDSWISH',
                    'MAX',
                    'AVG',
                    'ADD',
                    'SUB',
                    'MUL',
                    'MIN',
                    'BILINEAR',
                    'NEAREST',
                    'L1',
                    'L2',
                    'MEAN',
                ],
                [
                    # DType keys
                    'int8',
                    'uint8',
                    'int16',
                    'uint16',
                    'float16',
                    'int32',
                    'uint32',
                    'float32',
                ],
                [
                    # Layout keys
                    'NHWC',
                    'NCHW',
                    'NCHWC16',
                    'NCHWC32',
                ],
            ])
            for keys in self.keys:
                assert len(set(keys)) == len(keys)
        for keys in self.keys:
            if key in keys:
                return keys.index(key)
        raise ValueError(f'Unknown key: {key}')

    def to_nhwc(self, shape):
        if len(shape) == 5:
            shape = [shape[0], shape[2], shape[3], shape[1] * shape[4]]
        elif len(shape) == 3:
            shape = [shape[0], shape[1], 1, shape[2]]
        elif len(shape) == 2:
            shape = [shape[0], 1, 1, shape[1]]
        elif len(shape) == 1:
            shape = [1, 1, 1, shape[0]]
        elif len(shape) != 4:
            raise ValueError(f'Unsupported shape: {shape}')
        return shape

    def split_num(self, num):
        if num < -1.0 or num > 1.0:
            splits = '{:e}'.format(num).split('e')
        else:
            splits = [str(num), '0']
        return [float(x) for x in splits]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_np = np.array(self.inputs[index])
        label_np = np.array(self.labels[index])
        return input_np, label_np


class TrainDataset(BaseDataset):
    def __init__(self,
                 data_root,
                 data_split,
                 target,
                 input_size):
        super().__init__(input_size)
        assert os.path.exists(data_root), \
            f'Dataset not found in {data_root}'
        self.data_root = data_root
        self.data_split = data_split
        self.target = target
        self.load_dataset()

    def load_dataset(self):
        data_root = os.path.abspath(self.data_root)
        if self.target == 'X1_1204':
            data_path = f'{self.data_split}_x1.json'
        elif self.target == 'X2_1204':
            data_path = f'{self.data_split}_x2.json'
        elif self.target == 'x3P_1304':
            data_path = f'{self.data_split}_x3p_1304.json'
        inputs, labels = [], []
        if os.path.exists(data_path):
            with open(data_path, 'r') as fd:
                data_info = json.load(fd)
            if data_info:
                inputs = data_info['inputs']
                labels = data_info['labels']
        if len(inputs) == 0:
            if self.target == 'X1_1204':
                suffix = '_x1.json'
            else:
                suffix = '_x2.json'
            if self.data_split == 'val':
                train_data_path = f'train{suffix}'
                if os.path.exists(train_data_path):
                    with open(train_data_path, 'r') as fd:
                        data_info = json.load(fd)
                    if data_info:
                        inputs = data_info['inputs']
                        labels = data_info['labels']
                    os.remove(train_data_path)
            if len(inputs) == 0:
                for data_file in os.listdir(data_root):
                    if not data_file.lower().endswith(suffix):
                        continue
                    file_path = os.path.join(data_root, data_file)
                    with open(file_path, 'r') as fd:
                        case_info = json.load(fd)
                    if case_info:
                        for key, item in case_info.items():
                            inp, lbl = self.preprocess(item)
                            inputs.append(inp)
                            labels.append(lbl)
            if self.data_split == 'val':
                total_idx = np.arange(len(inputs)).tolist()
                val_idx = np.random.choice(total_idx,
                                           round(0.3 * len(inputs)),
                                           replace=False).tolist()
                inputs = [inputs[x] for x in val_idx]
                labels = [labels[x] for x in val_idx]
            else:
                with open(data_path, 'w') as fd:
                    data = {
                        'inputs' : inputs,
                        'labels' : labels,
                    }
                    json.dump(data, fd)
        for inp, lbl in zip(inputs, labels):
            inp, lbl = self.encode(inp, lbl)
            self.inputs.append(inp)
            self.labels.append(lbl)


class EvalDataset(BaseDataset):
    def __init__(self,
                 case_info,
                 input_size):
        super().__init__(input_size)
        self.names = []
        if case_info:
            for key, item in case_info.items():
                inp, lbl = self.preprocess(item)
                if len(inp) > 0:
                    inp, lbl = self.encode(inp, lbl)
                    self.inputs.append(inp)
                    self.labels.append(lbl)
                    self.names.append(key)

    def __getitem__(self, index):
        input_np = np.array(self.inputs[index])
        label_np = np.array(self.labels[index])
        name = self.names[index]
        return name, input_np, label_np

    @staticmethod
    def collate_fn(batch):
        names, inputs, labels = zip(*batch)
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)
        return names, inputs, labels


class Model(object):
    def __init__(self, model_type, **kargs):
        super().__init__()
        if model_type == 'mlp':
            model_func = MLP
        else:
            raise ValueError(f'Unknown model_type: {model_type}')
        self.model = model_func(**kargs)
        self.model_ema = None

    def apply_ema(self):
        if self.model_ema is None:
            self.model_ema = ModelEMA(self.model)

    def update_ema(self, state_dict=None, updates=0):
        if self.model_ema is not None:
            if state_dict is None:
                self.model_ema.update(self.model)
            else:
                self.model_ema.module.load_state_dict(state_dict)
                self.model_ema.updates = updates

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device):
        self.model.to(device)
        return self

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        self.model.load_state_dict(state_dict, strict=strict)

    def modules(self):
        return self.model.modules()

    def __call__(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        in_channels = kargs['in_channels']
        out_channels = kargs['out_channels']
        dropout = kargs.get('dropout', 0)
        self.network = nn.Sequential(*[
            nn.Linear(in_channels, 4 * in_channels),
            nn.SiLU(inplace=True),
            nn.Linear(4 * in_channels, 12 * in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(12 * in_channels, out_channels),
        ])

    def forward(self, x):
        return self.network(x)


class ModelEMA(object):
    # https://github.com/rwightman/pytorch-image-models
    def __init__(self, model, decay=0.9999):
        self.module = deepcopy(model).eval()
        self.updates = 0
        # Decay exponential ramp (to help early epochs)
        self.decay_lambda = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.module.parameters():
            p.requires_grad = False

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            decay = self.decay_lambda(self.updates)
            state_dict = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1.0 - decay) * state_dict[k].detach()


class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val * num
        self.cnt += num
        self.avg = self.sum / self.cnt


def evaluate(model, device, dataloader, criterion=None):
    losses = AverageMeter()
    # Switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device).float()
            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, targets)[0]
            else:
                loss = torch.zeros(1, device=device)
            losses.update(loss.item(), targets.size(0))
        print('%10.4g\n' % losses.avg)
    return losses.avg


def load_yaml(file_path):
    if not os.path.isfile(file_path):
        raise ValueError('Load %s failed!' % file_path)
    with open(file_path, 'r') as fd:
        data = yaml.load(fd, Loader=yaml.SafeLoader)
    return data


def save_yaml(file_path, data):
    with open(file_path, 'w') as fd:
        yaml.dump(data, fd, sort_keys=False)


def decay_lambda(final_ratio, total_steps, linear_decay=False):
    # Linear or cosine decay
    if linear_decay:
        lambda_fn = lambda x: final_ratio + (1.0 - final_ratio) * (1.0 - x / total_steps)
    else:
        lambda_fn = lambda x: 1.0 + (final_ratio - 1.0) * (
            (1.0 - math.cos(x * math.pi / total_steps)) / 2)
    return lambda_fn


def find_directory(root_path, base_name):
    index = 0
    while True:
        dir_path = os.path.join(root_path, f'{base_name}{index}')
        if os.path.exists(dir_path):
            if not os.path.exists(os.path.join(dir_path, 'last.pt')) and \
                    not os.path.exists(os.path.join(dir_path, 'best.pt')):
                break
        else:
            break
        index += 1
    return dir_path


def select_intersect(source, target, exclude):
    intersect = {}
    for k, v in source.items():
        if k in exclude:
            continue
        if k in target and v.shape == target[k].shape:
            intersect[k] = v
    return intersect


def load_model(model_path, map_location=None, load_ema=True):
    if not os.path.isfile(model_path):
        raise ValueError('Load %s failed' % model_path)
    ckpt = torch.load(model_path, map_location=map_location)
    if load_ema and ckpt.get('model_ema'):
        model = ckpt['model_ema']
    else:
        model = ckpt['model']
    model = model.float().eval()
    return model


def strip_optimizer(ckpt_path):
    # Strip optimizer from ckpt_path to finalize training
    device = torch.device('cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get('model_ema'):
        ckpt['model'] = ckpt['model_ema']
    ckpt['model_ema'] = None
    ckpt['updates'] = None
    ckpt['optimizer'] = None
    ckpt['epoch'] = -1
    if torch.cuda.is_available():
        ckpt['model'].half()
    for p in ckpt['model'].parameters():
        p.requires_grad = False
    torch.save(ckpt, ckpt_path)
    ckpt_size = os.path.getsize(ckpt_path) / 1e6
    print('Optimizer stripped from: %s, file size: %g MB' % (ckpt_path, ckpt_size))


def decode_target(target):
    target = torch.exp(target)
    target = target[:, 0::2] * torch.pow(10, target[:, 1::2])
    target = torch.clamp(target, min=0)
    return target


class Criterion(nn.Module):
    def __init__(self, loss_type='smoothl1', reduction='mean'):
        super().__init__()
        if loss_type == 'mse':
            self.criteria = nn.MSELoss(reduction=reduction)
        elif loss_type == 'smoothl1':
            self.criteria = nn.SmoothL1Loss(reduction=reduction, beta=1.0)
        else:
            raise ValueError(f'Unknown loss_type: {loss_type}')

    def forward(self, pred, true):
        pred_num, pred_exp = pred[:, 0::2], pred[:, 1::2]
        gt_num, gt_exp = true[:, 0::2], true[:, 1::2]
        loss_num = self.criteria(pred_num, gt_num)
        loss_exp = self.criteria(pred_exp, gt_exp)
        total_loss = self.criteria(pred, true)
        loss_items = torch.cat([loss_num.reshape(1,),
                                loss_exp.reshape(1,),
                                total_loss.reshape(1,)], dim=0).detach()
        return total_loss, loss_items


def train_network(opt):
    device = torch.device(opt.device)
    model = Model(opt.model_type, 
                  in_channels=opt.input_size, 
                  out_channels=opt.output_size, 
                  dropout=opt.dropout)
    model.to(device).train()
    if opt.weights:
        ckpt = torch.load(opt.weights, map_location=device)
        state_dict = ckpt['model'].float().state_dict()
        state_dict = select_intersect(state_dict,
                                      model.state_dict(),
                                      exclude=[])
        model.load_state_dict(state_dict, strict=False)
        print('Transferred %g/%g items from %s' % (
            len(state_dict), len(model.state_dict()), opt.weights))

    # Accumulate loss before optimizing
    if opt.nominated_batch_size <= 0:
        opt.nominated_batch_size = opt.batch_size
    accumulate = max(1, round(opt.nominated_batch_size / opt.batch_size))
    # Linear scale weight decay
    opt.weight_decay *= opt.batch_size * accumulate / opt.nominated_batch_size
    print('Scaled weight decay to %g' % opt.weight_decay)

    # Optimizer parameter groups
    weights, biases, others = [], [], []
    for m in model.modules():
        if hasattr(m, 'weight'):
            if isinstance(m.weight, nn.Parameter):
                if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                    others.append(m.weight)
                else:
                    weights.append(m.weight)
        if hasattr(m, 'bias'):
            if isinstance(m.bias, nn.Parameter):
                biases.append(m.bias)
    if opt.adam:
        optimizer = optim.Adam(biases,
                               lr=opt.initial_lr,
                               betas=(opt.momentum, 0.999))
    else:
        optimizer = optim.SGD(biases,
                              lr=opt.initial_lr,
                              momentum=opt.momentum,
                              nesterov=True)
    optimizer.add_param_group({'params': weights,
                               'weight_decay': opt.weight_decay})
    if len(others) > 0:
        optimizer.add_param_group({'params': others})
    del weights, biases, others

    if opt.cosine_lr:
        # Cosine learning rate from 1.0 to final_lr_ratio
        lr_lambda = decay_lambda(opt.final_lr_ratio, total_steps=opt.epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        if len(opt.lr_steps) == 0:
            opt.lr_steps = [round(0.33 * opt.epochs), round(0.66 * opt.epochs)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=opt.lr_steps, gamma=0.1)

    if opt.model_ema:
        model.apply_ema()
        print('Using exponential moving average')

    start_epoch, best_score = 0, float('inf')
    if opt.weights:
        if opt.resume:
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_score = ckpt['best_score']
            if ckpt.get('model_ema') and opt.model_ema:
                model.update_ema(ckpt['model_ema'].float().state_dict(),
                                 updates=ckpt['updates'])
            start_epoch = ckpt['epoch'] + 1
            assert start_epoch > 0, 'Training already finished'
            scheduler.last_epoch = start_epoch - 1
        del ckpt, state_dict

    # Create dataset loader
    trainloader = TrainDataLoader(opt.data_root,
                                  data_split='train',
                                  target=opt.target,
                                  input_size=opt.input_size,
                                  batch_size=opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.workers)
    testloader = TrainDataLoader(opt.data_root,
                                 data_split='val',
                                 target=opt.target,
                                 input_size=opt.input_size,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.workers)
    num_batchs = len(trainloader)

    # Start training network
    criterion = Criterion()
    print('Dataloader workers: %g' % trainloader.num_workers)
    print('Input size: %g' % opt.input_size)
    print('Batch size: %g' % opt.batch_size)
    print('Training %s for %g epochs' % (opt.model_type, opt.epochs))
    print('Logging results to %s\n' % opt.save_path)

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        mean_loss = torch.zeros(3, device=device)
        print(('%10s' * 6) % ('Epoch', 'Num', 'Exp', 'Loss', 'Step', 'Input'))
        pbar = tqdm(enumerate(trainloader), total=num_batchs)
        optimizer.zero_grad()

        for index, (inputs, targets) in pbar:
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device).float()
            total_steps = num_batchs * epoch + index

            # Warmup stage learning rate
            if total_steps <= opt.warmup_steps:
                x_pos = [0, opt.warmup_steps]
                y_pos = [1.0, opt.nominated_batch_size / opt.batch_size]
                accumulate = max(1, round(np.interp(total_steps, x_pos, y_pos)))
                for params in optimizer.param_groups:
                    # Learning rate rise from 0 to final_lr
                    if opt.cosine_lr:
                        final_lr = opt.initial_lr * lr_lambda(epoch)
                    else:
                        final_lr = opt.initial_lr
                    params['lr'] = np.interp(total_steps, x_pos, [0, final_lr])
                    if 'momentum' in params:
                        params['momentum'] = np.interp(
                            total_steps,
                            x_pos,
                            [opt.warmup_momentum, opt.momentum])

            outputs = model(inputs)
            loss, loss_items = criterion(outputs, targets)
            loss.backward()
            if total_steps % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                if opt.model_ema:
                    model.update_ema()

            mean_loss = (mean_loss * index + loss_items) / (index + 1)
            desc = ('%10s' + '%10.4g' * 5) % (
                '%g/%g' % (epoch + 1, opt.epochs),
                *mean_loss,
                scheduler.get_last_lr()[-1],
                inputs.shape[-1])
            pbar.set_description(desc)

        scheduler.step()
        result = 0
        if epoch == opt.epochs - 1 or not opt.notest:
            if opt.model_ema:
                test_model = model.model_ema.module
            else:
                test_model = model.model
            result = evaluate(test_model,
                              device,
                              dataloader=testloader,
                              criterion=criterion)
        txt_path = os.path.join(opt.save_path, 'results.txt')
        with open(txt_path, 'a') as fd:
            fd.write((desc + '%10.4g' + '\n') % result)
        if result < best_score:
            best_score = result

        half_model_ema, ema_updates = None, None
        if opt.model_ema:
            half_model_ema = deepcopy(model.model_ema.module).half()
            ema_updates = model.model_ema.updates
        half_model = deepcopy(model.model).half()
        saved_ckpt = {'epoch': epoch,
                      'best_score': best_score,
                      'model': half_model,
                      'model_ema': half_model_ema,
                      'updates': ema_updates,
                      'optimizer': optimizer.state_dict()}
        last_ckpt_path = os.path.join(opt.save_path, 'last.pt')
        best_ckpt_path = os.path.join(opt.save_path, 'best.pt')
        torch.save(saved_ckpt, last_ckpt_path)
        if best_score == result:
            torch.save(saved_ckpt, best_ckpt_path)
        del saved_ckpt
    for file_path in [last_ckpt_path, best_ckpt_path]:
        strip_optimizer(file_path)


def export_model(opt):
    device = torch.device('cpu')
    model = load_model(opt.weights, device)
    model.eval()
    dummy_input = torch.rand(1, opt.input_size)
    pts_path = opt.weights.replace('.pt', '.pts')
    try:
        print('Exporting TorchScript...')
        torch.jit.trace(model, dummy_input).save(pts_path)
        print(f'TorchScript saved to {pts_path}')
    except Exception as err:
        print(f'Export TorchScript failed: {err}')
    onnx_path = opt.weights.replace('.pt', '.onnx')
    try:
        print('Exporting Onnx...')
        torch.onnx.export(model,
                          dummy_input,
                          onnx_path,
                          verbose=False,
                          opset_version=12,
                          input_names=['input'],
                          output_names=['output'])
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f'Onnx saved to {onnx_path}')
    except Exception as err:
        print(f'Export Onnx failed: {err}')


def gen_parser_config(opt):
    config = configparser.ConfigParser()
    config.add_section('Common')
    common = config['Common']
    common['model_type'] = 'onnx'
    common['model_name'] = 'cost_model'
    common['detection_postprocess'] = ''
    onnx_path = opt.weights.replace('.pt', '.onnx')
    common['input_model'] = os.path.abspath(onnx_path)
    common['input'] = 'input'
    common['input_shape'] = str([1, opt.input_size])
    common['output'] = 'output'
    output_dir = os.path.dirname(opt.weights)
    common['output_dir'] = os.path.abspath(output_dir)
    cfg_path = os.path.join(output_dir, 'parser.cfg')
    with open(cfg_path, 'w') as fd:
        config.write(fd)
    return cfg_path


def get_perf_node_info(node_info):
    perf_node_info = dict()
    if node_info['op_type'] == 'SubGraph':
        if node_info['plugin']:
            perf_node_info[node_info['layer node name']] = node_info
        else:
            for node in node_info['internal_nodes']:
                for name, info in get_perf_node_info(node).items():
                    perf_node_info[name] = info
    else:
        perf_node_info[node_info['layer node name']] = node_info
    return perf_node_info


def get_case_info(graph_path):
    with open(graph_path, 'r') as fd:
        graph_info = json.load(fd)

    case_info = dict()
    tensor_info = graph_info['tensors']
    for node_info in graph_info['nodes']:
        for name, info in get_perf_node_info(node_info).items():
            items = []
            if info['op_type'] == 'SubGraph':
                for node in info['internal_nodes']:
                    items.append(node)
            else:
                items.append(info)
            cases = []
            for item in items:
                inputs, outputs = [], []
                if item['inputs'] is not None:
                    for inp in item['inputs']:
                        inputs.append(tensor_info[inp])
                for out in info['outputs']:
                    outputs.append(tensor_info[out])
                cases.append({
                    'perf_data' : [0] * 5,
                    'op_type' : item['op_type'],
                    'params' : item['params'],
                    'inputs' : inputs,
                    'outputs' : outputs,
                })
            case_info[name] = cases
    return case_info


def evaluate_graph(opt):
    graph_path = os.path.join(opt.eval_path, 'graph.json')
    case_info = get_case_info(graph_path)
    dataloader = EvalDataLoader(case_info,
                                input_size=opt.input_size,
                                batch_size=1,
                                shuffle=False,
                                num_workers=opt.workers)
    device = torch.device('cpu')
    model = load_model(opt.weights, device)
    model.eval()
    for name, inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True).float()
        targets = decode_target(targets.to(device))
        with torch.no_grad():
            outputs = model(inputs)
        outputs = decode_target(outputs)
        pred = outputs[0].tolist()
        pred[3] = min(pred[3], pred[0]) / (pred[0] + 1e-7) * 100
        pred[4] = min(pred[4], pred[0]) / (pred[0] + 1e-7) * 100
        print(name, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training options
    parser.add_argument('--data_root', type=str, default='ops_dataset',
                        help='dataset root')
    parser.add_argument('--target', type=str, default='X2_1204',
                        help='AIPU target')
    parser.add_argument('--model_type', type=str, default='mlp',
                        help='model type')
    parser.add_argument('--input_size', type=int, default=64,
                        help='input size')
    parser.add_argument('--output_size', type=int, default=10,
                        help='output size')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout ratio')
    parser.add_argument('--weights', type=str, default='',
                        help='weights path')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training')
    parser.add_argument('--epochs', type=int, default=300,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--nominated_batch_size', type=int, default=128,
                        help='accumulated batch size to optimize')
    parser.add_argument('--notest', type=bool, default=False,
                        help='only test final epoch')
    parser.add_argument('--adam', type=bool, default=False,
                        help='train with adam optimizer')
    parser.add_argument('--lr_steps', type=list, default=[],
                        help='learning rate decay steps')
    parser.add_argument('--cosine_lr', type=bool, default=False,
                        help='cosine learning rate')
    parser.add_argument('--initial_lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--final_lr_ratio', type=float, default=0.01,
                        help='final learning rate ratio')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='warmup steps')
    parser.add_argument('--warmup_momentum', type=float, default=0.8,
                        help='warmup momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--model_ema', type=bool, default=True,
                        help='model ema')
    parser.add_argument('--device', type=str, default='cpu',
                        help='torch device')
    parser.add_argument('--workers', type=int, default=4,
                        help='dataloader workers')
    # Evaluation options
    parser.add_argument('--eval_path', type=str, default='',
                        help='json path to evaluate')
    # Exportation options
    parser.add_argument('--export', action='store_true', default=False,
                        help='export model to AIPU format')
    opt = parser.parse_args()
    assert opt.target in ['X1_1204', 'X2_1204']
    if opt.export:
        print(f'Loading weights from {opt.weights}...')
        print(f'Input size: {opt.input_size}')
        export_model(opt)
        cfg_path = gen_parser_config(opt)
        sys.argv = ['aipu_parser', '-c', cfg_path]
        aipu_parser_main()
    elif opt.eval_path:
        print(f'Parsing graph from {opt.eval_path}...')
        print(f'Loading weights from {opt.weights}...')
        print(f'Input size: {opt.input_size}')
        evaluate_graph(opt)
    else:
        if opt.resume:
            save_path = Path(opt.weights).parent
            print(f'Resuming training from {save_path}...')
        else:
            save_path = find_directory('exp_output', f'{opt.target}_T')
            Path(save_path).mkdir(parents=True, exist_ok=True)
        opt_path = os.path.join(save_path, 'options.yaml')
        if opt.resume:
            backup = opt.device, opt.weights, opt.resume
            opt_params = load_yaml(opt_path)
            opt = argparse.Namespace(**opt_params)
            opt.device, opt.weights, opt.resume = backup
        else:
            save_yaml(opt_path, vars(opt))
        opt.save_path = save_path
        print(f'Training options: {vars(opt)}')
        train_network(opt)
