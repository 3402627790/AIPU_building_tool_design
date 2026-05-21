import numpy as np
import torch
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...config import PretrainedConfig
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data, convert_expr
from ..qwen3_dense.qwen3 import Qwen3Model


@register_update_config(['gr00t_vla_languagemodel_qwen3'])
def nvidia_vla_languagemodel_qwen3_update_config(config):
    _local_vision_config = {
        'hidden_size': 2048,
        'head_dim': 128,
        'hidden_act': "silu",
        "intermediate_size": 6144,
        "max_position_embeddings": 40960,
        "num_attention_heads": 16,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000,
        "vocab_size": 151680,
        "backbone_embedding_dim": 2048
    }

    config.update(_local_vision_config)


@register_causallm(['gr00t_vla_languagemodel_qwen3'])
class LanguageCausalLM(BaseModel):
    def __init__(self, config, prefix='backbone.model.language_model.'):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.model = Qwen3Model(config, weights, prefix)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.norm.weights, weights.get(f'{prefix}model.norm.weight'), f'normlm.norm.weight')

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim, eps=1e-6)
        update_tensor_data(self.vlln.weights,
                           weights.get(f'action_head.vlln.weight'),
                           f"action_head.vlln.weight")
        update_tensor_data(self.vlln.biases,
                           weights.get(f'action_head.vlln.bias'),
                           f"action_head.vlln.bias")

        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        with Graph('qwen') as self.aipu_g:
            if with_embed:
                hidden_states = ops.input(
                    [batch, seq_len], Dtype.FP32, name='input_ids', exprs=['batch', 'seq_len'])
            else:
                hidden_states = ops.input(
                    [batch, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded', exprs=['batch', 'seq_len', self.config.hidden_size])
            if use_cache:
                attention_mask = ops.input([batch, 1, seq_len, seq_len + past_len],
                                           Dtype.FP32, name='attention_mask', exprs=['batch', 1, 'seq_len', 'max_len'])
            else:
                attention_mask = ops.input(
                    [batch, 1, seq_len, seq_len + past_len], Dtype.FP32, name='attention_mask', exprs=['batch', 1, 'seq_len', 'max_len'])
                # attention_mask = ops.input([1, 1, 1, seq_len], Dtype.FP32, name='attention_mask')
            position_ids = ops.input(
                [batch, seq_len], Dtype.INT32, name='position_ids', exprs=['batch', 'seq_len'])

            kcache_module = nn.KVCacheModule(self.config, name='kcache')
            vcache_module = nn.KVCacheModule(self.config, name='vcache')

            hidden_states = self.model(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       past_key=kcache_module,
                                       past_value=vcache_module,
                                       use_cache=use_cache,
                                       with_embed=with_embed)
            hidden_states = self.norm(hidden_states)
            out = self.vlln(hidden_states)
            out.op.name = f'vlln'

            cur_tensors = list(self.aipu_g.output_tensors)[-1:]  # only get fc out
            history_k, history_v = kcache_module.cache_outputs, vcache_module.cache_outputs
            history_k_v_name = [hkv.name for hkv in history_k + history_v]
            output_tensors = []
            for ct in cur_tensors:
                if ct.name not in history_k_v_name:
                    output_tensors.append(ct)
            output_tensors += history_k + history_v
            self.aipu_g.output_tensors = TensorList(output_tensors)
            self.aipu_g.input_tensors = TensorList(
                [t for t in self.aipu_g.input_tensors[:3]] + kcache_module.cache_inputs + vcache_module.cache_inputs)
        return self.aipu_g
