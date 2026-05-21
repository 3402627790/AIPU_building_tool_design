# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import sys
try:
    from AIPUBuilder.Optimizer import cfg_parser
    default_fields = cfg_parser.DEFAULT_FIELDS.keys()
except Exception:
    default_fields = (
        'dataset',
        'metric',
        'calibration_strategy_for_activation',
        'calibration_strategy_for_weight',
        'calibration_batch_size',
        'metric_batch_size',
        'weight_bits',
        'activation_bits',
        'bias_bits',
        'quantize_method_for_activation',
        'quantize_method_for_weight',
        'running_statistic_momentum',
        'hists_bins',
    )

# aqt config field:
_1to1_fields = ['graph',
                'bin',
                'model_name',
                'calibration_data',
                'data',
                'label',
                'resize_degrade_to_nearest',
                'output_dir',
                'dump',
                'dump_dir',
                ]


def _only_copy(k, v):
    return f'{k}={v}'


str2bits = {'int8': 8, 'int16': 16, 'int32': 32, 'int64': 64, 'uint8': 8, 'uint16': 16, 'uint32': 32}


def _precision_copy(k, v):
    bits = str2bits[v]
    return _only_copy(k, bits)


def _precision_all_copy(k, v):
    bits = str2bits[v]
    if bits == 8:
        msg = f'activation_bits={bits}\nweight_bits={bits}\nbias_bits=32'
    elif bits == 16:
        msg = f'activation_bits={bits}\nweight_bits={bits}\nbias_bits=48'
    else:
        msg = f'activation_bits={bits}\nweight_bits={bits}\nbias_bits='
    return msg

# aqt vs opt fields


_need_transfer_fields = {
    'quantize_batch_size': (_only_copy, ['calibration_batch_size', 'metric_batch_size']),
    'calibration_method': (_only_copy, ['calibration_strategy_for_activation']),
    'weight_scale_optimize': (_only_copy, ['calibration_strategy_for_weight=weighted_scale_param']),
    'quant_precision': (_precision_all_copy, ['activation_bits']),
    'calibration_bin_num': (_only_copy, ['histc_bins']),
    'weight_precision': (_precision_copy, ['weight_bits']),
    'bias_precision': (_precision_copy, ['bias_bits']),
    'activation_precision': (_precision_copy, ['activation_bits']),
    'graph_optimizer_concat': (_only_copy, ['unify_scales_for_concat']),
}

_need_give_msg_fields = {
    # dataset relative field
    'calibration_label': "Opt cfg does not require 'calibration_label'.",
    'label_id_offset': "Opt cfg does not support 'label_id_offset'. please compensate this offset in preparing the dataset label.",
    'input_scale': "Opt cfg does not support 'input_scale'. please use this scale in the dataset pre-process.",
    'input_mean': "Opt cfg does not support 'input_mean'. please use this mean in the dataset pre-process.",
    'preprocess_mode': "Opt cfg does not support 'preprocess_mode'. please use this mode in the dataset pre-process.",
    'reverse_rgb': "Opt cfg does not support 'reverse_rgb'. please reverse RGB in the dataset pre-process.",
    'input_data_format': "Opt cfg does not support 'input_data_format'. please transpose NCHW to NHWC in the dataset pre-process if needed",
    'embedding_lookup': "Opt cfg does not require 'embedding_lookup'. please put this lookup in the dataset.",
    'gt': "Opt cfg does not require 'gt'. please put this gt of label in the dataset.",
    'vocab': "Opt cfg does not require 'vocab'. please put this vocab in the metric.",
    'enable_record': "Opt cfg does not require 'enable_record'.",

    'test_each_layer_error': "Opt cfg does not support 'test_each_layer_error'. please check the .json file to check cos distance and quantization information",
    'input_max': "Opt cfg does not support 'input_max'. please modify the extrema_max/running_max of Input op in statistic file.",
    'input_min': "Opt cfg does not support 'input_min'. please modify the extrema_min/running_min of Input op in statistic file.",
    'ts_max_file': "Opt cfg does not support 'ts_max_file'. please use the 'statistic_file' field.",
    'ts_min_file': "Opt cfg does not support 'ts_min_file'. please use the 'statistic_file' field.",
    'dataset_name': "Opt cfg does not support 'dataset_name'. please use the 'dataset' and 'metric' field to implement this field in Opt.",

    'ops_per_channel': "Opt cfg does not support 'ops_per_channel'. Depthwiseconv defaultly uses the per-channel and you can set per-channel quantization using 'quantize_method_for_activation/weight'.",
    'data_for_test_or_quantize': "Opt cfg does not require 'data_for_test_or_quantize'.",

    # post-process
    'detection_postprocess': "Opt cfg does not require 'detection_postprocess'.",
    'anchor_generator': "Opt cfg does not require 'anchor_generator'.",
    'special_anchor_mode': "Opt cfg does not require 'special_anchor_mode'.",

    'quant_param': "Opt cfg does not require 'quant_param'.",
    'graph_optimize_lstm': "Opt cfg does not require 'graph_optimize_lstm'.",

    # 'log': "Opt cfg does not support 'log'.",
}


class TransferCfg(object):

    def __init__(self, pth):
        self.path = pth

        self.need_saved = {}
        self.need_deleted = {}

        self.opt_cfg = []
        self.pre_opt_cfg = []
        self.post_opt_cfg = []

    def __call__(self):
        paths = self.get_cfgs()
        for p in paths:
            print(f'[begin] to transfer aqt cfg:{p} to opt cfg.....')
            lines = self.read_cfg(p)
            aqt_lines = self.get_aqt_section(lines)
            self.rules(aqt_lines)
            self.write_cfg(p)
            self.need_deleted.clear()
            self.need_saved.clear()
            print('[finish].')

    def read_cfg(self, cfg_file):
        with open(cfg_file, 'r') as fi:
            lines = fi.readlines()
            lns = []
            for l in lines:
                l = l.replace(' ', '')
                lns.append(l.strip().strip('\n'))
            return lns

    def rules(self, lines):
        for i, l in enumerate(lines):
            lkv = l.split('=')
            if len(lkv) == 1:
                if lkv == ['[AutoQuantizationTool]']:
                    lkv = ['[Optimizer]']
                self.need_saved.update({i: lkv[0]})
                continue

            lk, lv = lkv[0], lkv[1]
            if lk in _1to1_fields:
                self.need_saved.update({i: l})
            elif lk in _need_transfer_fields.keys():
                tg = _need_transfer_fields[lk]
                func = tg[0]
                fields = tg[1]
                for f in fields:
                    ret = func(f, lv)
                    self.need_saved.update({f'{i}{f}': ret})
            elif lk in default_fields:
                self.need_saved.update({i: l})
            elif lk in _need_give_msg_fields.keys():
                print('\t', _need_give_msg_fields[lk])
                self.need_deleted.update({i: l})
            else:
                self.need_deleted.update({i: l})

    def get_aqt_section(self, lines):
        if '[AutoQuantizationTool]' in lines:
            pre_idx = lines.index('[AutoQuantizationTool]')
        elif '[Optimizer]' in lines:
            pre_idx = lines.index('[Optimizer]')
        else:
            pre_idx = 0

        if '[GBuilder]' in lines:
            post_idx = lines.index('[GBuilder]')
        else:
            post_idx = -1

        if post_idx == -1:
            self.opt_cfg = lines[pre_idx:]
        else:
            self.opt_cfg = lines[pre_idx:post_idx]
            self.post_opt_cfg = lines[post_idx:]
        self.pre_opt_cfg = lines[:pre_idx]
        return self.opt_cfg

    def get_cfgs(self):
        # check path
        if os.path.exists(self.path):
            if os.path.isfile(self.path):
                path = [self.path]
            elif os.path.isdir(self.path):
                flns = os.listdir(self.path)
                path = [os.path.join(self.path, cf) for cf in flns if '.cfg' in cf]
            else:
                path = []
        else:
            print(f'please check the path: {self.path} , which does not exist.')
            path = []
        return path

    def write_cfg(self, pth):
        opt_pth = pth.replace('.cfg', '_opt.cfg')
        with open(opt_pth, 'w') as fw:
            for field in self.pre_opt_cfg:
                fw.write(field+'\n')
            # msg = '# Optimizer needs these fields:\n'
            # fw.write(msg)
            for k, field in self.need_saved.items():
                fw.write(f'{field}\n')
            if self.need_deleted:
                msg = '\n# Optimizer does not need these fields:\n'
                fw.write(msg)
                for k, field in self.need_deleted.items():
                    fw.write(f"#{field}\n")
            for field in self.post_opt_cfg:
                fw.write(f'{field}\n')


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('please input the transfer config path or file.')
    pth = sys.argv[1]
    transfer_cfg = TransferCfg(pth)
    transfer_cfg()
