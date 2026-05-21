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
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.layer_id = layer_id
        self.experts = [DeepseekR1MLP(config, weights, layer_id, i, True)
                        for i in range(config.n_routed_experts)
                        ]

    def forward(self, hidden_states, expert_id):

        INFO(f"\t build the {self.layer_id} block. {expert_id} expert")
        hidden_states = self.experts[expert_id](hidden_states.reshape([*list(hidden_states.shape)[1:]]))

        return hidden_states


@register_causallm(['ds671b_experts'])
class DeepseekR1Experts(BaseModel):
    def __init__(self, config, prefix='', only_embed=False, attn_layers=True):
        super().__init__(config)
        self.layers_num = config.num_hidden_layers
        config.update({'only_embed': only_embed})
        config.update({'attn_layers': attn_layers})
        self.aipu_g = []
        self.prefix = prefix

    # prefix='model.' with_embed=True is prev graph
    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        use_cache = self.config.get('use_cache', True) and (past_len > 0)

        first_k_dense_replace = self.config.first_k_dense_replace
        hidden_size = self.config.hidden_size
        model_name = 'experts'
        for layer_id in range(first_k_dense_replace, self.layers_num, 1):
            search_safe_tensors_key = 'layers.'+str(layer_id)+'.'
            weights = self._load_state_dict(search_safe_tensors_key)
            self.config.update({'weights': weights})
            self.model = DeepseekR1Model(self.config, weights, layer_id, '')
            del weights
            for expert_id in range(self.config.n_routed_experts):
                with Graph(model_name) as aipu_g:
                    hidden_states = ops.input([1, seq_len, hidden_size], Dtype.FP32, name='embeded')
                    hidden_states = self.model(hidden_states, expert_id)

                    if use_cache:
                        aipu_g.output_tensors = TensorList(
                            [t for t in aipu_g.output_tensors[-1:]])
                        aipu_g.input_tensors = TensorList(
                            [t for t in aipu_g.input_tensors[:]])
                    self.aipu_g.append(aipu_g)

            if layer_id < self.layers_num-1:
                for expert_id in range(self.config.n_routed_experts):
                    check_aipu_graph(self.aipu_g[expert_id])
                    serialize(self.aipu_g[expert_id], *output_file(self.config.model_type,
                              self.config.output_dir, prefix=layer_id, postfix=expert_id))
            else:
                self.config.update({'ir_prefix': layer_id})

        return self.aipu_g
