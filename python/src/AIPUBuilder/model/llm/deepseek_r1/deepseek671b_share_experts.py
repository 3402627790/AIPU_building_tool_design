import torch.nn.functional as F
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np
import torch
import os
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from safetensors import safe_open
from ...basemodel import *
from ...config import PretrainedConfig
from ...register import register_causallm
from ...qinfo import QConf
from ...utils import update_tensor_data
from .deepseek671b_common import *


class DeepseekR1Model(nn.Module):

    def __init__(self, config, weights, layer_id, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(
            config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.layer_id = layer_id
        self.shared_experts = DeepseekR1MLP(config, weights, layer_id, -1, True)

    def forward(self, moeinfer_out,
                hidden_states):

        INFO(f"\t process the share experts {self.layer_id} block.")
        hidden_states = self.shared_experts(hidden_states.reshape(
            [*list(hidden_states.shape)[1:]]))
        hidden_states = moeinfer_out + hidden_states
        return hidden_states


@register_causallm(['ds671b_share_experts'])
class DeepseekR1ShareExperts(BaseModel):
    def __init__(self, config, prefix='', only_embed=False, attn_layers=True):
        super().__init__(config)
        self.layers_num = config.num_hidden_layers
        config.update({'only_embed': only_embed})
        config.update({'attn_layers': attn_layers})
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        use_cache = self.config.get('use_cache', True) and (past_len > 0)

        first_k_dense_replace = self.config.first_k_dense_replace
        hidden_size = self.config.hidden_size
        model_name = 'share_experts'
        for layer_id in range(first_k_dense_replace, self.layers_num, 1):
            search_safe_tensors_key = 'layers.'+str(layer_id)+'.'
            weights = self._load_state_dict(search_safe_tensors_key)
            self.config.update({'weights': weights})
            self.model = DeepseekR1Model(self.config, weights, layer_id, '')
            del weights
            with Graph(model_name) as self.aipu_g:
                residual = ops.input(
                    [1, seq_len, hidden_size], Dtype.FP32, name='moe_infer_out')
                hidden_states = ops.input(
                    [1, seq_len, hidden_size], Dtype.FP32, name='embeded')
                hidden_states = self.model(
                    residual,
                    hidden_states,
                )
                if use_cache:
                    self.aipu_g.output_tensors = TensorList(
                        [t for t in self.aipu_g.output_tensors[-1:]])
                    self.aipu_g.input_tensors = TensorList(
                        [t for t in self.aipu_g.input_tensors[:]])
            if layer_id < self.layers_num-1:
                check_aipu_graph(self.aipu_g)
                serialize(self.aipu_g, *output_file(self.config.model_type, self.config.output_dir, prefix=layer_id))
            else:
                self.config.update({'ir_prefix': layer_id})
        return self.aipu_g
