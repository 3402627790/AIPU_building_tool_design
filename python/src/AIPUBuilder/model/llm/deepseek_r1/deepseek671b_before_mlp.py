import torch.nn.functional as F
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np
import torch
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

        return residual, hidden_states, k, v


class DeepseekR1Model(nn.Module):

    def __init__(self, config, weights, start_layer, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(
            config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix

        self.decoders = DeepseekR1Decoder(config, weights, start_layer, prefix)  # +3 to only generate layer3
        self.gating_dim = config.hidden_size
        self.logits = nn.Linear(self.gating_dim, config.n_routed_experts, dtype=Dtype.FP16)
        weight = weights.get(
            f'model.layers.{start_layer}.mlp.gate.weight')
        update_tensor_data(self.logits.weights,
                           weight.to(torch.float16),
                           f"layers.{start_layer}.mlp.gate.weight",
                           dtype=np.float16)

        self.e_score_correction_bias = nn.Constant(
            np.zeros([1, config.n_routed_experts]).astype('float32'))
        weight = weights.get(
            f'model.layers.{start_layer}.mlp.gate.e_score_correction_bias').unsqueeze(0)
        update_tensor_data(self.e_score_correction_bias.weights,
                           weight,
                           f"layers.{start_layer}.mlp.gate.e_score_correction_bias")

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

        input_embeds = input_ids
        attention_mask = ops.tile(
            attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

        hidden_states = input_embeds
        history_ks, history_vs = [], []

        # residual is signal which is from atttention
        residual, hidden_states, history_k, history_v = self.decoders(hidden_states,
                                                                      attention_mask,
                                                                      position_ids,
                                                                      past_key[0],
                                                                      past_value[0],
                                                                      use_cache=use_cache,
                                                                      )

        logits = self.logits(
            hidden_states.reshape([*list(hidden_states.shape)[1:]])
        )
        scores = ops.sigmoid(logits)

        # select top-k experts
        scores_for_choice = scores + self.e_score_correction_bias()
        if use_cache:
            history_ks.append(history_k)
            history_vs.append(history_v)

        return residual, hidden_states, scores, scores_for_choice, history_ks, history_vs


@register_causallm(['ds671b_before_mlp'])
class DeepseekR1CausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False, attn_layers=True):
        super().__init__(config)

        self.layers_num = config.num_hidden_layers

        config.update({'only_embed': only_embed})
        config.update({'attn_layers': attn_layers})

        self.aipu_g = None
        self.prefix = prefix

    # prefix='model.' with_embed=True is prev graph
    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        use_cache = self.config.get('use_cache', True) and (past_len > 0)

        block_num = 1
        first_k_dense_replace = self.config.first_k_dense_replace
        kv_lora_rank = self.config.kv_lora_rank
        qk_rope_head_dim = self.config.qk_rope_head_dim
        model_name = 'deepseek'

        for layer_id in range(first_k_dense_replace, self.layers_num, 1):
            search_safe_tensors_key = 'layers.'+str(layer_id)+'.'
            weights = self._load_state_dict(search_safe_tensors_key)
            self.config.update({'weights': weights})
            self.model = DeepseekR1Model(self.config, weights, layer_id, '')
            del weights
            with Graph(model_name) as self.aipu_g:

                hidden_states = ops.input(
                    [1, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded')

                if use_cache:
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

                residual, hidden_states, scores, score_choice, history_k, history_v = self.model(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key=past_key,
                    past_value=past_value,
                    use_cache=use_cache,
                )

                if use_cache:
                    self.aipu_g.output_tensors = TensorList([residual]+[hidden_states]+[scores] +
                                                            [t for t in self.aipu_g.output_tensors[-1:]] + history_k + history_v)
                    self.aipu_g.input_tensors = TensorList(
                        [t for t in self.aipu_g.input_tensors[:3]] + past_key + past_value)
            if layer_id < self.layers_num-1:
                check_aipu_graph(self.aipu_g)
                serialize(self.aipu_g, *output_file(self.config.model_type, self.config.output_dir, prefix=layer_id))
            else:
                self.config.update({'ir_prefix': layer_id})

        return self.aipu_g
