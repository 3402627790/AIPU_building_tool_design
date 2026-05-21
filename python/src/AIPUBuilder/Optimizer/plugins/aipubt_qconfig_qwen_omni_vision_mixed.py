# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.


from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
import numpy as np
import re

###################################################
# this config only available on op name fully_connected,fully_connected_0
# fully_connected quant fp16
# fully_connected_0 quant with w fp4 act fp8 block size 128
# if use pertoken,add fully_connected_3 to rules_for_act_perchannel, quant with act perchanel, wfp4afp8
# the other fc quant with wfp4 afp16 perchannel for weight
@register_plugin(PluginType.QConfig, '1.0')
class Qwen_omni_vision_mixed_qconfig(object):

    def __init__(self, graph, config):
        self.graph = graph
        self.config = config

    def _get_matched(self, keys, rule):
        matcheds = []
        for k in keys:
            m = re.match(rule, k)
            if m is not None:
                matcheds.append(m.group())
        return matcheds

    def __call__(self):
        # 8bits rule
        rules_for_8bits = [

        ]
        rules_for_16bits = [

        ]
        rules_for_float_op = [
            f"fully_connected",
        ]
        rules_for_w4fp8bits=[
            f"fully_connected_0",
            # f"fully_connected_2",

        ]
        rules_for_act_perchannel=[
            # f"fully_connected_3",
            # f"fully_connected",

        ]
        for n in self.graph.nodes:
            # n.attrs['trigger_float_op'] = 'float16_preferred'
            for r in rules_for_8bits:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['trigger_float_op'] = 'disable'
                        break
                else:
                    if n.name == r:
                        n.attrs['trigger_float_op'] = 'disable'
                        break

            for r in rules_for_16bits:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['q_bits_weight'] = 16
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 40
                        break
                else:
                    if n.name == r:
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['q_bits_weight'] = 16
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 40
                        break
            for r in rules_for_w4fp8bits:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['quant_type']='wfp4afp8e4m3'
                        n.attrs['q_mode_weight'] = 'per_block_symmetric_restricted_range'
                        n.attrs["q_mode_bias"] = 'per_block_symmetric_restricted_range'
                        n.attrs['q_mode_activation'] = 'per_tensor_symmetric_full_range'
                        n.attrs['q_bits_weight'] = 4
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 16
                        n.attrs['weight_block_size'] = 128
                        break
                else:
                    if n.name == r:
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['quant_type']='wfp4afp8e4m3'
                        n.attrs['q_mode_weight'] = 'per_block_symmetric_restricted_range'
                        n.attrs["q_mode_bias"] = 'per_block_symmetric_restricted_range'
                        n.attrs['q_mode_activation'] = 'per_tensor_symmetric_full_range'
                        n.attrs['q_bits_weight'] = 4
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 16
                        n.attrs['weight_block_size'] = 128
                        break
            for r in rules_for_act_perchannel:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['quant_type']='wfp4afp8e4m3'
                        n.attrs['q_mode_weight'] = 'per_channel_symmetric_restricted_range'
                        n.attrs["q_mode_bias"] = 'per_channel_symmetric_restricted_range'
                        n.attrs['q_mode_activation'] = 'per_channel_symmetric_full_range'
                        n.attrs['q_bits_weight'] = 8
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 32
                        break
                else:
                    if n.name == r:
                        n.attrs['trigger_float_op'] = 'disable'
                        n.attrs['quant_type']='wfp4afp8e4m3'
                        n.attrs['q_mode_weight'] = 'per_channel_symmetric_restricted_range'
                        n.attrs["q_mode_bias"] = 'per_channel_symmetric_restricted_range'
                        n.attrs['q_mode_activation'] = 'per_channel_symmetric_full_range'
                        n.attrs['q_bits_weight'] = 8
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 32
                        # n.attrs['weight_block_size'] = 128
                        break
            for r in rules_for_float_op:
                if '*' in r:
                    matched = self._get_matched([n.name], r)
                    if len(matched):
                        n.attrs['quant_type'] = 'fp16'
                        break
                else:
                    if n.name == r:
                        n.attrs['quant_type'] = 'fp16'
                        n.attrs['trigger_float_op'] = 'float16_preferred'
                        n.attrs["q_mode_weight"] = 'per_channel_symmetric_full_range'
                        n.attrs["q_mode_bias"] = 'per_channel_symmetric_full_range'
                        n.attrs['q_bits_weight'] = 16
                        n.attrs['q_bits_activation'] = 16
                        n.attrs['q_bits_bias'] = 40
                        break

            if n.type in [OpType.LayerNorm, OpType.Softmax]:
                n.attrs['trigger_float_op'] = 'float16_preferred'

            if n.type == OpType.Eltwise and len(n.children) == 1 and n.children[0].type == OpType.LayerNorm:
                n.attrs['trigger_float_op'] = 'float16_preferred'
