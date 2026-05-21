import numpy as np
import torch
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...register import register_causallm, gen_subgraph
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


class Gemma2MLP(nn.Module):
    def __init__(self, config, weights, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act = nn.ACT2FN[config.hidden_activation]

        update_tensor_data(self.gate_proj.weights, weights.get(
            f'model.layers.{layer_idx}.mlp.gate_proj.weight'), f'model.layers.{layer_idx}.mlp.gate_proj.weight')
        update_tensor_data(self.up_proj.weights, weights.get(
            f'model.layers.{layer_idx}.mlp.up_proj.weight'), f'model.layers.{layer_idx}.mlp.up_proj.weight')
        update_tensor_data(self.down_proj.weights, weights.get(
            f'model.layers.{layer_idx}.mlp.down_proj.weight'), f'model.layers.{layer_idx}.mlp.down_proj.weight')

    def forward(self, x: Tensor):
        bsz, q_len, hidden_size = x.shape
        x = ops.reshape(x, [bsz * q_len, hidden_size])
        out = self.gate_proj(x)
        out = self.act(out)
        out = out * self.up_proj(x)
        out = self.down_proj(out)
        out = ops.reshape(out, [bsz, q_len, hidden_size])
        return out


class Gemma2RMSNorm(nn.Module):
    def __init__(self, hidden_size, weights, weights_name, eps):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        norm_weights = weights.get(weights_name)['data'] + 1
        update_tensor_data(self.norm.weights, norm_weights, weights_name)

    def forward(self, x: Tensor):
        out = self.norm(x)
        return out


class Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, num_heads, dim, repeat_kv=False, max_position_embeddings=2048, base=10000, device=None, name=''):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.name = name
        self.repeat_kv = repeat_kv
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                               dtype=torch.int64).float().to(device) / self.dim))

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(torch.get_default_dtype())
        self.sin_cached = emb.sin().to(torch.get_default_dtype())

    def forward(self, pos_ids):
        bsz_symbol, seq_symbol = pos_ids.shape.symbol
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        sin_out_reshape = ops.reshape(gather_sin_out, [bsz_symbol, 1, seq_symbol, self.dim])
        cos_out_reshape = ops.reshape(gather_cos_out, [bsz_symbol, 1, seq_symbol, self.dim])
        sin_reshape_tile = sin_out_reshape
        cos_reshape_tile = cos_out_reshape
        # simulator support broadcast mul
        # if self.repeat_kv:
        #     sin_reshape_tile = ops.tile(sin_out_reshape, [1, self.num_heads, 1, 1])
        #     cos_reshape_tile = ops.tile(cos_out_reshape, [1, self.num_heads, 1, 1])
        sin_out.op.constants['weights'].name = f'rotary_emb.sin_weight'
        cos_out.op.constants['weights'].name = f'rotary_emb.cos_weight'
        sin_out.op.name = f"rotary_emb.sin"
        cos_out.op.name = f"rotary_emb.cos"
        # gather_sin_out.name = f"{gather_sin_out.name}"
        # gather_cos_out.name = f"{gather_cos_out.name}"

        return cos_reshape_tile, sin_reshape_tile


class Gemma2Attention(nn.Module):
    def __init__(self, config, weights, layer_idx):
        super().__init__()
        self.config = config
        self.layer_id = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.with_bias = config.attention_bias
        self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.weights_scaling = self.scaling / self.attn_logit_softcapping
        # self.weights_scaling = weights_scaling
        # self.attn_logit_softcapping = attn_logit_softcapping

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=False, fused_multiplier=self.weights_scaling)
        update_tensor_data(self.q_proj.weights, weights.get(
            f'model.layers.{layer_idx}.self_attn.q_proj.weight'), f'model.layers.{layer_idx}.self_attn.q_proj.weight')
        update_tensor_data(self.k_proj.weights, weights.get(
            f'model.layers.{layer_idx}.self_attn.k_proj.weight'), f'model.layers.{layer_idx}.self_attn.k_proj.weight')
        update_tensor_data(self.v_proj.weights, weights.get(
            f'model.layers.{layer_idx}.self_attn.v_proj.weight'), f'model.layers.{layer_idx}.self_attn.v_proj.weight')
        update_tensor_data(self.o_proj.weights, weights.get(
            f'model.layers.{layer_idx}.self_attn.o_proj.weight'), f'model.layers.{layer_idx}.self_attn.o_proj.weight')

        if self.with_bias:
            update_tensor_data(self.q_proj.bias, weights.get(
                f'model.layers.{layer_idx}.self_attn.q_proj.bias'), f'model.layers.{layer_idx}.self_attn.q_proj.bias')
            update_tensor_data(self.k_proj.bias, weights.get(
                f'model.layers.{layer_idx}.self_attn.k_proj.bias'), f'model.layers.{layer_idx}.self_attn.k_proj.bias')
            update_tensor_data(self.v_proj.bias, weights.get(
                f'model.layers.{layer_idx}.self_attn.v_proj.bias'), f'model.layers.{layer_idx}.self_attn.v_proj.bias')
            update_tensor_data(self.o_proj.bias, weights.get(
                f'model.layers.{layer_idx}.self_attn.o_proj.bias'), f'model.layers.{layer_idx}.self_attn.o_proj.bias')

    def forward(self,  hidden_states: Tensor, attention_mask: Tensor, position_ids: Tensor, position_emb: Tensor, k_cache, v_cache):
        bsz, q_len, hidden_size = hidden_states.shape
        bs_symbol, seq_len_symbol, _ = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symbol * seq_len_symbol, hidden_size])
        cos_emb, sin_emb = position_emb

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = ops.reshape(query_states, [bs_symbol, seq_len_symbol, self.num_heads, self.head_dim])
        key_states = ops.reshape(key_states, [bs_symbol, seq_len_symbol, self.num_key_value_heads, self.head_dim])
        value_states = ops.reshape(value_states, [bs_symbol, seq_len_symbol, self.num_key_value_heads, self.head_dim])

        query_states = ops.transpose(query_states, [0, 2, 1, 3])
        key_states = ops.transpose(key_states, [0, 2, 1, 3])
        value_states = ops.transpose(value_states, [0, 2, 1, 3])

        q_pre_half, q_post_half = ops.split(query_states, 2, -1)
        k_pre_half, k_post_half = ops.split(key_states, 2, -1)
        q_post_half = ops.negative(q_post_half)
        k_post_half = ops.negative(k_post_half)
        rotate_q = nn.Concat(axis=-1)([q_post_half, q_pre_half])
        rotate_k = nn.Concat(axis=-1)([k_post_half, k_pre_half])

        query_states = (query_states * cos_emb) + (rotate_q * sin_emb)
        key_states = (key_states * cos_emb) + (rotate_k * sin_emb)

        k_cache_out = k_cache(key_states, position_ids=position_ids, block_idx=self.layer_id)
        new_k_cache = k_cache_out[0]

        v_cache_out = v_cache(value_states, position_ids=position_ids, block_idx=self.layer_id)
        new_v_cache = v_cache_out[0]

        if self.num_key_value_groups > 1:
            batch_symbol, _, max_len_symbol, _ = new_k_cache.shape.symbol
            key_states = ops.reshape(
                new_k_cache, [batch_symbol, self.num_key_value_heads, 1, max_len_symbol, self.head_dim])
            value_states = ops.reshape(
                new_v_cache, [batch_symbol, self.num_key_value_heads, 1, max_len_symbol, self.head_dim])
            key_states = ops.tile(key_states, [1, 1, self.num_key_value_groups, 1, 1])
            value_states = ops.tile(value_states, [1, 1, self.num_key_value_groups, 1, 1])
            key_states = ops.reshape(key_states, [
                                     batch_symbol, self.num_key_value_groups * self.num_key_value_heads, max_len_symbol, self.head_dim])
            key_states = ops.transpose(key_states, [0, 1, 3, 2])
            value_states = ops.reshape(value_states, [
                                       batch_symbol, self.num_key_value_groups * self.num_key_value_heads, max_len_symbol, self.head_dim])
        else:
            key_states = ops.transpose(new_k_cache, [0, 1, 3, 2])
            value_states = new_v_cache

        attn_logit_softcapping = ops.constant(Tensor([self.attn_logit_softcapping]),
                                              [1], Dtype.FP32)
        attn_logit_softcapping.op.constants['weights'].name = f"attn_logit_softcapping"
        attn_weights = self.kq_matmul(query_states, key_states)
        # attn_weights = attn_weights * weights_scaling
        attn_weights = ops.tanh(attn_weights)
        attn_weights = attn_weights * attn_logit_softcapping

        # -inf
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.Softmax()(attn_weights)
        attn_output = nn.MatMul()(attn_weights, value_states)

        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [bs_symbol * seq_len_symbol, self.num_heads * self.head_dim])
        attn_output = self.o_proj(attn_output)
        attn_output = ops.reshape(attn_output, [bs_symbol, seq_len_symbol, hidden_size])

        return attn_output


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config, weights, layer_idx):
        super().__init__()
        self.attn = Gemma2Attention(config, weights, layer_idx)
        self.mlp = Gemma2MLP(config, weights, layer_idx)
        self.input_layernorm = Gemma2RMSNorm(
            config.hidden_size, weights, f'model.layers.{layer_idx}.input_layernorm.weight', eps=config.rms_norm_eps)
        self.is_sliding = not bool(layer_idx % 2)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, weights, f'model.layers.{layer_idx}.pre_feedforward_layernorm.weight', eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, weights, f'model.layers.{layer_idx}.post_feedforward_layernorm.weight', eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window
        self.post_attention_layernorm = Gemma2RMSNorm(
            config.hidden_size, weights, f'model.layers.{layer_idx}.post_attention_layernorm.weight', eps=config.rms_norm_eps)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, x: Tensor, attention_mask: Tensor, position_ids: Tensor, position_emb: Tensor, k_cache=None, v_cache=None):
        residual = x
        hidden_states = self.input_layernorm(x)

        hidden_states = self.attn(hidden_states, attention_mask, position_ids, position_emb, k_cache, v_cache)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class Gemma2LMHead(nn.Module):
    def __init__(self, config, weights):
        self.config = config
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.inv_final_logit_softcapping = (1 / config.final_logit_softcapping)
        self.final_logit_softcapping = config.final_logit_softcapping
        self.norm = Gemma2RMSNorm(config.hidden_size, weights, f'model.norm.weight', eps=config.rms_norm_eps)

        update_tensor_data(self.lm_head.weights, weights.get(
            f'model.embed_tokens.weight')['data'].float() * self.inv_final_logit_softcapping, f'model.embed_tokens.weight')

    @gen_subgraph(subgraph_name='lm_head')
    def forward(self, x: Tensor):
        x = self.norm(x)
        _, _, hidden_size = x.shape
        bsz_symbol, sqen_symbol, _ = x.shape.symbol
        x = ops.reshape(x, [bsz_symbol * sqen_symbol, hidden_size])

        logits = self.lm_head(x)
        logits = ops.reshape(logits, [bsz_symbol, sqen_symbol, self.vocab_size])

        # logits = logits * self.inv_final_logit_softcapping
        logits = ops.tanh(logits)
        logits = logits * self.final_logit_softcapping

        return logits


class Gemma2Model(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.with_bias = config.attention_bias

        self.embed_tokens = weights[f'model.embed_tokens.weight'].float().cpu().numpy()
        self.blocks = [Gemma2DecoderLayer(config, weights, idx) for idx in gen_layer_ids(config.num_hidden_layers)]

    def forward(self, x: Tensor, attention_mask: Tensor, position_ids: Tensor, position_emb: Tensor, past_key, past_value):
        hidden_states = x
        for i, block in enumerate(self.blocks):
            hidden_states, _, _ = block(hidden_states,
                                        attention_mask,
                                        position_ids,
                                        position_emb,
                                        past_key,
                                        past_value)
        return hidden_states


class Gemma2ForCausalLM(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.model = Gemma2Model(config, weights)
        self.vocab_size = config.vocab_size
        self.lm_head = Gemma2LMHead(config, weights)
        # self.inv_final_logit_softcapping = (1 / config.final_logit_softcapping)
        # self.final_logit_softcapping = config.final_logit_softcapping
        # self.normalizer = config.hidden_size**0.5

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor, position_ids: Tensor, position_emb: Tensor, past_key=None, past_value=None):
        hidden_states = inputs_embeds
        # hidden_states = hidden_states * normalizer
        output = self.model(hidden_states, attention_mask, position_ids, position_emb, past_key, past_value)
        logits = self.lm_head(output)

        return logits


@register_causallm(['gemma2'])
class Gemma2(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        self.config = config
        self.config.update({'weights': weights})
        self.model = Gemma2ForCausalLM(config, weights, prefix)
        self.aipu_g = None
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim
        self.rotary_emb = Gemma2RotaryEmbedding(self.num_heads,
                                                self.head_dim,
                                                repeat_kv=self.num_key_value_groups > 1,
                                                max_position_embeddings=self.max_position_embeddings,
                                                base=self.rope_theta)

    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        batch = self.config.get('batch', 1)

        with Graph('gemma2') as self.aipu_g:
            hidden_states = ops.input(
                [batch, seq_len, self.config.hidden_size], Dtype.FP32, name='input_embed', exprs=['batch', 'seq_len', self.config.hidden_size])
            attention_mask = ops.input(
                [batch, 1, seq_len, past_len + seq_len], Dtype.FP32, name='attention_mask', exprs=['batch', 1, 'seq_len', 'max_len'])
            position_ids = ops.input([batch, seq_len], Dtype.INT32,
                                     name='position_ids', exprs=['batch', 'seq_len'])
            input_tensors = [hidden_states, attention_mask, position_ids]

            kcache_module = nn.KVCacheModule(self.config, name='kcache')
            vcache_module = nn.KVCacheModule(self.config, name='vcache')

            cos_emb, sin_emb = self.rotary_emb(position_ids)
            position_emb = (cos_emb, sin_emb)

            hidden_states = self.model(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       position_emb,
                                       past_key=kcache_module,
                                       past_value=vcache_module)

            history_k, history_v = kcache_module.cache_outputs, vcache_module.cache_outputs
            output_tensors = [hidden_states] + history_k + history_v
            input_tensors += kcache_module.cache_inputs + vcache_module.cache_inputs
            self.aipu_g.input_tensors = TensorList(input_tensors)
            self.aipu_g.output_tensors = TensorList(output_tensors)
        return self.aipu_g
