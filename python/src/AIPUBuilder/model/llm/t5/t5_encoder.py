import numpy as np
import torch
import math
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.logger import INFO, ERROR
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


@register_causallm(['umt5_embedding'])
class UMT5Embedding(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('shared.weight')


class UMT5LinearPositionBias(nn.Module):
    def __init__(self, max_position_embeddings=512):
        super().__init__()
        context_position = torch.arange(max_position_embeddings)[:, None]
        mem_position = torch.arange(max_position_embeddings)[None, :]
        relative_position = mem_position - context_position
        relative_buckets = 0
        num_buckets = 16
        max_distance = 128
        relative_buckets += (relative_position > 0).long() * num_buckets
        relative_position = relative_position.abs()

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        log_ratio = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
        log_ratio = log_ratio * (num_buckets - max_exact)
        relative_position_if_large = max_exact + (log_ratio).long()
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        self.cached_relative_position_bias = relative_buckets

    def forward(self, position_ids):
        pos_bias = ops.constant(Tensor(self.cached_relative_position_bias))
        gather_pos = ops.gather(pos_bias, position_ids, axis=0)
        return gather_pos  # [1, seqlen, max_pos_embeddings]


class UMT5Block(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config.d_ff

        self.layer_norm_0 = nn.RMSNorm(self.hidden_size)

        self.q = nn.Linear(self.hidden_size, self.hidden_size)
        self.k = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, self.hidden_size)
        self.o = nn.Linear(self.hidden_size, self.hidden_size)

        self.bias_cache = UMT5LinearPositionBias()

        self.relative_attn_bias = nn.Embedding(config.relative_attention_num_buckets, self.num_heads)

        self.wi_0 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.wi_1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.wo = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act = nn.ACT2FN[config.dense_act_fn]
        self.layer_norm_1 = nn.RMSNorm(self.hidden_size)

        q_proj_w = weights.get(f"{prefix}block.{layer_id}.layer.0.SelfAttention.q.weight")
        update_tensor_data(self.q.weights, q_proj_w)
        k_proj_w = weights.get(f"{prefix}block.{layer_id}.layer.0.SelfAttention.k.weight")
        update_tensor_data(self.k.weights, k_proj_w)
        v_proj_w = weights.get(f"{prefix}block.{layer_id}.layer.0.SelfAttention.v.weight")
        update_tensor_data(self.v.weights, v_proj_w)
        o_proj_w = weights.get(f"{prefix}block.{layer_id}.layer.0.SelfAttention.o.weight")
        update_tensor_data(self.o.weights, o_proj_w)
        layer_norm_0_w = weights.get(f"{prefix}block.{layer_id}.layer.0.layer_norm.weight")
        update_tensor_data(self.layer_norm_0.weights, layer_norm_0_w)

        rel_attn_bias_w = weights.get(f"{prefix}block.{layer_id}.layer.0.SelfAttention.relative_attention_bias.weight")
        update_tensor_data(self.relative_attn_bias.weights, rel_attn_bias_w)

        wi_0_w = weights.get(f"{prefix}block.{layer_id}.layer.1.DenseReluDense.wi_0.weight")
        update_tensor_data(self.wi_0.weights, wi_0_w)
        wi_1_w = weights.get(f"{prefix}block.{layer_id}.layer.1.DenseReluDense.wi_1.weight")
        update_tensor_data(self.wi_1.weights, wi_1_w)
        wo_w = weights.get(f"{prefix}block.{layer_id}.layer.1.DenseReluDense.wo.weight")
        update_tensor_data(self.wo.weights, wo_w)
        layer_norm_1_w = weights.get(f"{prefix}block.{layer_id}.layer.1.layer_norm.weight")
        update_tensor_data(self.layer_norm_1.weights, layer_norm_1_w)

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape

        residual = hidden_states
        hidden_states = self.layer_norm_0(hidden_states)

        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])

        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        query_states = query_states.reshape(bs, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bs, seq_len, self.num_heads, self.head_dim)
        value_states = value_states.reshape(bs, seq_len, self.num_heads, self.head_dim)

        query_states = ops.transpose(query_states, [0, 2, 1, 3])
        key_states = ops.transpose(key_states, [0, 2, 1, 3])
        value_states = ops.transpose(value_states, [0, 2, 1, 3])

        if use_cache:
            key_states = k_cache(key_states, position_ids=position_ids, block_idx=self.layer_id,
                                 name=f"past_key_value.{self.layer_id}.key")
            key_states = key_states[0]
            value_states = v_cache(value_states, position_ids=position_ids, block_idx=self.layer_id,
                                   name=f"past_key_value.{self.layer_id}.value")
            value_states = value_states[0]

        context_len = key_states.shape[1]
        scores = self.kq_matmul(query_states, key_states)

        pos_idx = self.bias_cache(position_ids)
        pos_idx = pos_idx.reshape(seq_len, -1)
        pos_bias = self.relative_attn_bias(pos_idx)
        pos_bias = ops.transpose(pos_bias, [2, 0, 1])
        pos_bias = pos_bias.reshape(1, self.num_heads, seq_len, -1)
        pos_bias = ops.cast(pos_bias, Dtype.FP32)

        scores += pos_bias + attention_mask

        attn_weights = ops.softmax(scores, axis=-1)
        attn_output = ops.matmul(attn_weights, value_states)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = attn_output.reshape(bs * seq_len, hidden_size)
        attn_output = self.o(attn_output)

        attn_output = residual + attn_output

        residual = attn_output
        hidden_states = self.layer_norm_1(attn_output)
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.wo(hidden_states)
        hidden_states = ops.reshape(hidden_states, [bs, seq_len, hidden_size])
        hidden_states = residual + hidden_states
        return hidden_states


@register_causallm(['umt5'])
class UMT5EncoderModel(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.final_layernorm = nn.RMSNorm(self.config.d_model)
        final_layernorm_w = weights.get("encoder.final_layer_norm.weight")
        update_tensor_data(self.final_layernorm.weights, final_layernorm_w)
        self.with_embed = self.config.get('with_embed', False)
        num_hidden_layers = gen_layer_ids(config.num_layers)

        self.layers = [
            UMT5Block(config, weights, i, prefix+"encoder.") for i in num_hidden_layers
        ]
        self.prefix = prefix
        self.aipu_g = None

        self.config.num_attention_heads = self.config.num_heads
        self.config.hidden_size = self.config.d_model
        self.config.num_hidden_layers = self.config.num_layers

    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        with Graph("umt5_encoder") as self.aipu_g:
            if with_embed:
                hidden_states = ops.input(
                    [batch, seq_len], Dtype.FP32, name='input_ids', exprs=['batch', 'seq_len']
                )
            else:
                hidden_states = ops.input([batch, seq_len, self.config.d_model],
                                          Dtype.FP32,
                                          name='embeded',
                                          exprs=['batch', 'seq_len', self.config.d_model])

            attention_mask = ops.input([batch, 1, seq_len, seq_len + past_len],
                                       Dtype.FP32,
                                       name='attention_mask',
                                       exprs=['batch', 1, 'seq_len', 'max_len'])
            position_ids = ops.input([batch, seq_len],
                                     Dtype.INT32,
                                     name='position_ids',
                                     exprs=['batch', 'seq_len'])

            kcache_module = nn.KVCacheModule(self.config, name='kcache')
            vcache_module = nn.KVCacheModule(self.config, name='vcache')

            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask, position_ids,
                                      kcache_module, vcache_module, use_cache)

            last_hidden_states = self.final_layernorm(hidden_states)
            history_k, history_v = kcache_module.cache_outputs, vcache_module.cache_outputs
            output_tensors = [last_hidden_states] + history_k + history_v
            self.aipu_g.output_tensors = TensorList(output_tensors)
            self.aipu_g.input_tensors = TensorList(
                [t for t in self.aipu_g.input_tensors[:3]] + kcache_module.cache_inputs + vcache_module.cache_inputs)
        return self.aipu_g
