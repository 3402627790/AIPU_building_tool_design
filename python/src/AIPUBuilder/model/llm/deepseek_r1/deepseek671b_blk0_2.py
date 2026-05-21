import torch.nn.functional as F
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import *
from ...register import register_causallm
from ...utils import update_tensor_data
from .deepseek671b_common import *


class DeepseekR1Decoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = DeepseekR1Attention(config, weights, layer_id, prefix)

        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")
        self.mlp = DeepseekR1MLP(config, weights, layer_id, 0, False)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, k, v = self.attention(hidden_states,
                                             attention_mask,
                                             position_ids,
                                             k_cache,
                                             v_cache,
                                             use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states.reshape(
            [*list(hidden_states.shape)[1:]]))
        hidden_states = residual + hidden_states

        return hidden_states, k, v


class DeepseekR1Model(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(
            config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix

        self.decoders = [DeepseekR1Decoder(config, weights, idx, prefix)  # +3 to only generate layer3
                         for idx in range(config.num_hidden_layers)]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key=None,
                past_value=None,
                use_cache=None,
                ):

        if use_cache and (past_key is None or past_value is None):
            ERROR(f"when use_kv_cache, please provide the kv inputs")
        if use_cache:
            assert len(past_key) == self.model_config.num_hidden_layers
            assert len(past_value) == self.model_config.num_hidden_layers
        if not use_cache:
            past_key = [None] * self.model_config.num_hidden_layers
            past_value = [None] * self.model_config.num_hidden_layers

        input_embeds = input_ids

        attention_mask = ops.tile(
            attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

        hidden_states = input_embeds
        history_ks, history_vs = [], []
        for i, decoder in enumerate(self.decoders):

            hidden_states, history_k, history_v = decoder(hidden_states,
                                                          attention_mask,
                                                          position_ids,
                                                          past_key[i],
                                                          past_value[i],
                                                          use_cache=use_cache,
                                                          )
            if use_cache:
                history_ks.append(history_k)
                history_vs.append(history_v)

        return hidden_states, history_ks, history_vs


@register_causallm(['ds671b_blk0_2'])
class DeepseekR1CausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False, attn_layers=True):
        super().__init__(config)
        weights = {}
        config.num_hidden_layers = config.first_k_dense_replace
        for layer_id in range(config.num_hidden_layers):
            search_safe_tensors_key = 'layers.'+str(layer_id)+'.'
            weights.update(self._load_state_dict(search_safe_tensors_key))
        config.update({'weights': weights})
        config.update({'only_embed': only_embed})
        config.update({'attn_layers': attn_layers})
        self.config.update({'ir_prefix': ''})
        self.model = DeepseekR1Model(config, weights, prefix)

        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        block_num = self.config.num_hidden_layers
        kv_lora_rank = self.config.kv_lora_rank
        qk_rope_head_dim = self.config.qk_rope_head_dim
        model_name = 'deepseek'

        seq_len = self.config.seq_len
        past_len = self.config.past_len
        use_cache = self.config.get('use_cache', True) and (past_len > 0)

        with Graph(model_name) as self.aipu_g:
            hidden_states = ops.input(
                [1, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded')

            if self.use_cache:
                attention_mask = ops.input([1, 1, seq_len, seq_len * 2],
                                           Dtype.FP32, name='attention_mask')
            else:
                attention_mask = ops.input(
                    [1, seq_len], Dtype.INT32, name='attention_mask')
            position_ids = ops.input(
                ([1, seq_len]), Dtype.INT32, name='position_ids')

            past_key, past_value = None, None
            if use_cache:
                past_key = [ops.input([1, 1, seq_len, kv_lora_rank],
                                      Dtype.FP32, name=f"past_key_value.{i}.key") for i in range(block_num)]
                past_value = [ops.input([1, 1, seq_len, qk_rope_head_dim],
                                        Dtype.FP32, name=f"past_key_value.{i}.value") for i in range(block_num)]

            hidden_states, history_k, history_v = self.model(
                hidden_states,
                attention_mask,
                position_ids,
                past_key=past_key,
                past_value=past_value,
                use_cache=use_cache,
            )

            if use_cache:
                self.aipu_g.output_tensors = TensorList(
                    [t for t in self.aipu_g.output_tensors[-1:]] + history_k + history_v)
                self.aipu_g.input_tensors = TensorList(
                    [t for t in self.aipu_g.input_tensors[:3]] + past_key + past_value)

        return self.aipu_g
