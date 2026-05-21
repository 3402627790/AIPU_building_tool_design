import numpy as np
import torch
import math
import functools
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from AIPUBuilder._C._tongue import Symbol, Expr
from ...register import register_causallm, gen_subgraph
from ...utils import update_tensor_data, convert_expr
from ..llmutils import gen_layer_ids


def to_expr(eles):
    if isinstance(eles, (list, tuple)):
        return [to_expr(xx) for xx in eles]
    elif isinstance(eles, str):
        if '*' in eles:
            ele = eles.replace(' ', '').split('*')
            ele = [int(e) if not e.startswith('d') else e[1:] for e in ele]
            ele = to_expr(ele)
            return functools.reduce(lambda a, b: a*b, ele)
        else:
            return Expr(Symbol(eles))
    elif isinstance(eles, (int, float)):
        return Expr(eles)
    else:
        ERROR(f"un-supported type: {type(x)}, only support [list, tuple, str, int, float]")
        return None


@register_causallm(['gpt_oss_embedding'])
class GptOssEmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')

# Inverse dim formula to find dim based on number of rotations


def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

# Find dim range bounds based on rotations


def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class GptOssRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device='cpu',
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = (emb.cos() * _mscale)
        self.sin_cached = (emb.sin() * _mscale)

    def forward(self, pos_ids):
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        # sin_out_reshape = gather_sin_out.unsqueeze(dim=1)
        # cos_out_reshape = gather_cos_out.unsqueeze(dim=1)
        sin_out_reshape = ops.reshape(gather_sin_out, [1, 1, pos_ids.shape[1], self.sin_cached.shape[1]])
        cos_out_reshape = ops.reshape(gather_cos_out, [1, 1, pos_ids.shape[1], self.sin_cached.shape[1]])
        sin_reshape_tile = sin_out_reshape
        cos_reshape_tile = cos_out_reshape
        return cos_reshape_tile, sin_reshape_tile


class GptOssAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings if not hasattr(
            self.config, 'max_len') or self.config.max_len is None else self.config.max_len
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        if (self.head_dim * self.num_heads) != self.hidden_size:
            WARN(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                 f"{self.hidden_size} and `num_heads`: {self.num_heads}).")
        self.swap_matmul_input = config.get('swap_matmul_input', False)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.sink = nn.Constant([self.num_heads])

        q_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.q_proj.weight')
        q_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.q_proj.bias')
        update_tensor_data(self.q_proj.weights, q_proj_w, f"{prefix}layers.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q_proj.biases, q_proj_b, f"{prefix}layers.{layer_id}.self_attn.q_proj.bias")

        k_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.k_proj.weight')
        k_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.k_proj.bias')
        update_tensor_data(self.k_proj.weights, k_proj_w, f"{prefix}layers.{layer_id}.self_attn.k_proj.weight")
        update_tensor_data(self.k_proj.biases, k_proj_b, f"{prefix}layers.{layer_id}.self_attn.k_proj.bias")

        v_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.v_proj.weight')
        v_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.v_proj.bias')
        update_tensor_data(self.v_proj.weights, v_proj_w, f"{prefix}layers.{layer_id}.self_attn.v_proj.weight")
        update_tensor_data(self.v_proj.biases, v_proj_b, f"{prefix}layers.{layer_id}.self_attn.v_proj.bias")

        o_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.o_proj.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.self_attn.o_proj.weight")

        sink_w = weights.get(f'model.layers.{layer_id}.self_attn.sinks')
        update_tensor_data(self.sink.weights, sink_w, f"{prefix}layers.{layer_id}.self_attn.sinks")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kq_matmul_swap = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape.symbol
        bs_symbol, seq_len_symbol, _ = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symbol * seq_len_symbol, hidden_size])

        # bs = 1 optimization.: swap bs and num_heads to reduce LayoutCVTs in attention (for static prefill only)
        #     before this opt.: gb inserts Reshapes to swap bs and num_heads before Matmuls, which results in LayoutCVTs
        #                         before/after Reshapes
        #     after this opt.:  swap during Transpose therefore no Reshapes before Matmuls, and less LayoutCVTs
        en_bs_1_opt = 0
        if bs == 1 and not use_cache:
            en_bs_1_opt = 1
            self.qkv_transpose = nn.Transpose([2, 0, 1, 3])
            # prefill k, v cache shape is [num_key_value_heads, 1, seq_len, head_dim]
            # in pdmodellmchat, need to transpose prefill's output kv cache
            self.config.update({'prefill_batch_opt': True})

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        self.rotary_emb = GptOssRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            scaling_factor=self.config.rope_scaling["factor"],
            base=self.config.rope_theta,
            beta_fast=self.config.rope_scaling['beta_fast'],
            beta_slow=self.config.rope_scaling['beta_slow'])
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        # not repeat_kv
        if self.num_key_value_groups <= 1:
            if en_bs_1_opt == 1:
                sin_emb = ops.tile(sin_emb, [self.num_heads, 1, 1, 1])
                cos_emb = ops.tile(cos_emb, [self.num_heads, 1, 1, 1])
            else:
                sin_emb = ops.tile(sin_emb, [1, self.num_heads, 1, 1])
                cos_emb = ops.tile(cos_emb, [1, self.num_heads, 1, 1])

        self.q_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_key_value_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.q_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.kv_reshape(key_states))

        rtq_symbol0, rtq_symbol1, rtq_symbol2, _ = reshaped_trans_query.shape.symbol
        crop0 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [0, self.head_dim//2]])
        # query
        q_crop_out0 = crop0(reshaped_trans_query)
        q_crop_out1 = crop1(reshaped_trans_query)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        rtk_symbol0, rtk_symbol1, rtk_symbol2, _ = reshaped_trans_key.shape.symbol
        crop0 = nn.Crop([[0, rtk_symbol0], [0, rtk_symbol1], [0, rtk_symbol2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, rtk_symbol0], [0, rtk_symbol1], [0, rtk_symbol2], [0, self.head_dim//2]])
        # key
        k_crop_out0 = crop0(reshaped_trans_key)
        k_crop_out1 = crop1(reshaped_trans_key)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        neg_q_crop_out0.op.name = f"{self.matmul_name}.negative_0"
        neg_k_crop_out0.op.name = f"{self.matmul_name}.negative_1"

        q_cosine = reshaped_trans_query * cos_emb
        k_cosine = reshaped_trans_key * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine

        k_cache_out = k_cache(k_add_out, position_ids=position_ids, block_idx=self.layer_id,
                              name=f"past_key_value.{self.layer_id}.key")
        k_cache_out = k_cache_out[0]

        # if self.num_key_value_groups > 1 and self.config.get('split_GQA', False):
        #     pass
        if self.num_key_value_groups > 1 and en_bs_1_opt == 0:
            batch, num_key_value_heads, slen, head_dim = k_cache_out.shape.symbol
            if not self.config.get('broadcast_matmul', False):
                k_cache_out = k_cache_out.reshape([batch, num_key_value_heads, 1, slen, head_dim])
                k_cache_out = ops.tile(k_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
                k_cache_out = k_cache_out.reshape(
                    [batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])
            else:
                # [CP-23956]: insert reshape to use kq_matmul's broadcast feature to eliminate tile op for k_out, kq_matmul will broadcast k_out's second dim to num_key_value_groups
                k_cache_out = k_cache_out.reshape([batch * num_key_value_heads, 1, slen, head_dim])
                batch, _, slen, head_dim = q_add_out.shape.symbol
                q_add_out = q_add_out.reshape([batch * self.num_key_value_heads,
                                              self.num_key_value_groups, slen, head_dim])
        elif self.num_key_value_groups > 1 and en_bs_1_opt == 1:
            num_key_value_heads, batch, slen, head_dim = k_cache_out.shape.symbol
            if not self.config.get('broadcast_matmul', False):
                k_cache_out = k_cache_out.reshape([num_key_value_heads, 1, batch, slen, head_dim])
                k_cache_out = ops.tile(k_cache_out, [1, self.num_key_value_groups, 1, 1, 1])
                k_cache_out = k_cache_out.reshape(
                    [num_key_value_heads * self.num_key_value_groups, batch, slen, head_dim])
            else:
                k_cache_out = k_cache_out.reshape([batch * num_key_value_heads, 1, slen, head_dim])
                _, batch, slen, head_dim = q_add_out.shape.symbol
                q_add_out = q_add_out.reshape([batch * self.num_key_value_heads,
                                              self.num_key_value_groups, slen, head_dim])

        if self.swap_matmul_input and seq_len == 1:
            # [CP-19313]C=A*B=(B^T*A^T)^T, when B size >> A size, swap input can improve wtram reuse
            kq_matmul = nn.Transpose([0, 1, 3, 2])(self.kq_matmul_swap(k_cache_out, q_add_out))
        else:
            kq_matmul = self.kq_matmul(q_add_out, k_cache_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"

        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        sink_tile = nn.Tile([bs_symbol, 1, seq_len_symbol, 1])
        sink_crop = nn.Crop([[0, bs_symbol], [0, self.num_heads], [
                            0, seq_len_symbol], [0, k_cache_out.shape.symbol[2]]])

        attn_weights = kq_matmul + attention_mask
        sinks = ops.reshape(self.sink(), [1, self.num_heads, 1, 1])
        sinks = sink_tile(sinks)
        combined_attn_weights = ops.concat([attn_weights, sinks], -1)

        reduce_max = ops.reduce_max(combined_attn_weights, [-1], True)
        combined_attn_weights = combined_attn_weights - reduce_max
        softmax_masked = ops.softmax(combined_attn_weights)
        softmax_masked = sink_crop(softmax_masked)

        reshaped_trans_value = self.qkv_transpose(self.kv_reshape(value_states))

        v_cache_out = v_cache(reshaped_trans_value, position_ids=position_ids,
                              block_idx=self.layer_id, name=f"past_key_value.{self.layer_id}.value")
        v_cache_out = v_cache_out[0]

        if self.num_key_value_groups > 1 and en_bs_1_opt == 0:
            if not self.config.get('broadcast_matmul', False):
                batch, num_key_value_heads, slen, head_dim = v_cache_out.shape.symbol
                v_cache_out = v_cache_out.reshape([batch, num_key_value_heads, 1, slen, head_dim])
                v_cache_out = ops.tile(v_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
                v_cache_out = v_cache_out.reshape(
                    [batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])
            else:
                # [CP-23956]: insert reshape to use kqv_matmul's broadcast feature to eliminate tile op for reshaped_trans_value
                batch, num_key_value_heads, slen, head_dim = v_cache_out.shape.symbol
                v_cache_out = v_cache_out.reshape([batch * num_key_value_heads, 1, slen, head_dim])
        elif self.num_key_value_groups > 1 and en_bs_1_opt == 1:
            if not self.config.get('broadcast_matmul', False):
                num_key_value_heads, batch, slen, head_dim = v_cache_out.shape.symbol
                v_cache_out = v_cache_out.reshape([num_key_value_heads, 1, batch, slen, head_dim])
                v_cache_out = ops.tile(v_cache_out, [1, self.num_key_value_groups, 1, 1, 1])
                v_cache_out = v_cache_out.reshape(
                    [num_key_value_heads * self.num_key_value_groups, batch, slen, head_dim])
            else:
                num_key_value_heads, batch, slen, head_dim = v_cache_out.shape.symbol
                v_cache_out = v_cache_out.reshape([batch * num_key_value_heads, 1, slen, head_dim])

        kqv_out = self.kqv_matmul(softmax_masked, v_cache_out)
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs_symbol * seq_len_symbol, self.num_heads * self.head_dim])
        if en_bs_1_opt == 1:
            reshaped_trans = self.qkv_reshape(nn.Transpose([1, 2, 0, 3])(kqv_out))
        else:
            # [CP-23956]: recover batch dim
            kqv_out = kqv_out.reshape(bs_symbol, self.num_key_value_heads *
                                      self.num_key_value_groups, seq_len_symbol, self.head_dim)
            reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))

        o_t = self.o_proj(reshaped_trans)

        return o_t


class GptOssMoeMLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='', shared=False, expert_id=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)

        gate_up_proj_w = weights.get(f'model.layers.{layer_id}.mlp.experts.{expert_id}.gate_up_proj.weight')
        gate_up_proj_bias = weights.get(f'model.layers.{layer_id}.mlp.experts.{expert_id}.gate_up_proj.bias')
        down_proj_w = weights.get(f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight')
        down_proj_bias = weights.get(f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.bias')

        update_tensor_data(self.gate_up_proj.weights, gate_up_proj_w,
                           name=f'{prefix}layers.{layer_id}.mlp.experts.{expert_id}.gate_up_proj.weight')
        update_tensor_data(self.gate_up_proj.biases, gate_up_proj_bias,
                           name=f'{prefix}layers.{layer_id}.mlp.experts.{expert_id}.gate_up_proj.bias')
        update_tensor_data(self.down_proj.weights, down_proj_w,
                           name=f'{prefix}layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight')
        update_tensor_data(self.down_proj.biases, down_proj_bias,
                           name=f'{prefix}layers.{layer_id}.mlp.experts.{expert_id}.down_proj.bias')

    def forward(self, x):
        seq_len, hidden_size = x.shape.symbol
        gate_up = self.gate_up_proj(x)
        # gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate_up = ops.reshape(gate_up, [seq_len, self.intermediate_size, 2])
        gate = ops.crop(gate_up, [[0, seq_len], [0, self.intermediate_size], [0, 1]])
        up = ops.crop(gate_up, [[0, seq_len], [0, self.intermediate_size], [1, 2]])
        gate = ops.reshape(gate, [seq_len, self.intermediate_size])
        up = ops.reshape(up, [seq_len, self.intermediate_size])

        up = ops.clip(up, -7.0, 7.0)
        glu = gate * ops.sigmoid(gate * 1.702)
        gate_out = (up + 1) * glu
        out = self.down_proj(gate_out)
        return out


class GptOssMoeGate(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.topk = config.num_experts_per_tok

        self.gate = nn.Linear(config.hidden_size, config.num_local_experts)
        gate_w = weights.get(f"model.layers.{layer_id}.mlp.router.weight")
        gate_w = gate_w['data'][:config.num_local_experts]
        gate_b = weights.get(f"model.layers.{layer_id}.mlp.router.bias")
        gate_b = gate_b['data'][:config.num_local_experts]
        update_tensor_data(self.gate.weights, gate_w, name=f"{prefix}model.layers.{layer_id}.mlp.gate.weight")
        update_tensor_data(self.gate.biases, gate_b, name=f"{prefix}model.layers.{layer_id}.mlp.gate.bias")

        down_w = weights.get(f"model.layers.{layer_id}.mlp.experts.down_proj_blocks")['data']
        down_scale = weights.get(f"model.layers.{layer_id}.mlp.experts.down_proj_scales")['data']
        down_bias = weights.get(f"model.layers.{layer_id}.mlp.experts.down_proj_bias")['data']
        gate_up_w = weights.get(f"model.layers.{layer_id}.mlp.experts.gate_up_proj_blocks")['data']
        gate_up_scale = weights.get(f"model.layers.{layer_id}.mlp.experts.gate_up_proj_scales")['data']
        gate_up_bias = weights.get(f"model.layers.{layer_id}.mlp.experts.gate_up_proj_bias")['data']
        for i in range(config.num_local_experts):
            weights.set_item(f"{prefix}model.layers.{layer_id}.mlp.experts.{i}.down_proj.weight", down_w[i])
            weights.set_item(f"{prefix}model.layers.{layer_id}.mlp.experts.{i}.down_proj.scales", down_scale[i])
            weights.set_item(f"{prefix}model.layers.{layer_id}.mlp.experts.{i}.down_proj.bias", down_bias[i])
            weights.set_item(f"{prefix}model.layers.{layer_id}.mlp.experts.{i}.gate_up_proj.weight", gate_up_w[i])
            weights.set_item(f"{prefix}model.layers.{layer_id}.mlp.experts.{i}.gate_up_proj.scales", gate_up_scale[i])
            weights.set_item(f"{prefix}model.layers.{layer_id}.mlp.experts.{i}.gate_up_proj.bias", gate_up_bias[i])

        self.experts = [GptOssMoeMLP(config, weights, layer_id, prefix, False, i)
                        for i in range(config.num_local_experts)]

    def forward(self, hidden_states):
        seqlen, hidden_size = hidden_states.shape
        bsz = 1
        self.bsz = bsz
        self.seq_len = seqlen
        self.hidden_size = hidden_size
        logits = self.gate(hidden_states)
        scores = ops.softmax(logits, 1)
        topk_weight, topk_idx = ops.topk(scores, self.topk, 1, sorted=False, select_index='last')
        identity = hidden_states

        hidden_states = ops.reshape(hidden_states, [bsz*seqlen, hidden_size])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        return y

    def moe_infer(self, x, topk_idx, topk_weight):
        cnts = ops.constant(np.zeros([self.bsz * self.seq_len, len(self.experts)]).astype('float32'))
        src = ops.constant(np.ones([self.bsz * self.seq_len, len(self.experts)]).astype('float32'))
        cnts = ops.scatter_elements(cnts, topk_idx, src, axis=1)
        tokens_per_expert = ops.reduce_sum(cnts, [0], keep_dim=False)
        tokens_per_expert_idx, iter_num = ops.nonzero(tokens_per_expert)
        tokens_per_expert_idx = ops.reshape(tokens_per_expert_idx, [len(self.experts)])
        tokens_per_expert = ops.gather(tokens_per_expert, tokens_per_expert_idx, 0, 0)
        tokens_per_expert_idx = ops.cast(tokens_per_expert_idx, Dtype.INT32, ignore_scale_zp=True)
        idxs = ops.reshape(topk_idx, [self.bsz * self.seq_len * self.config.num_experts_per_tok])
        topk_val = ops.shape(idxs)
        idxs = ops.topk(idxs, topk_val, 0, largest=False, sorted=True, select_index='last')[1]
        idxs_dtype = idxs.dtype
        idxs_dtype_bits = idxs_dtype.bits
        idxs_max = 2 ** idxs_dtype_bits - 1 if not idxs_dtype.is_signed else 2 ** (idxs_dtype_bits - 1) - 1
        const_dtype = idxs_dtype.np
        if idxs_max < self.config.num_experts_per_tok:
            const_bits = math.ceil(math.log2(self.config.num_experts_per_tok) + 0.5)
            dtype_bits = math.ceil(const_bits / 8.0) * 8
            const_dtype = Dtype(f'int{dtype_bits}' if dtype_bits <= 2 else f'int32')
            idxs = ops.cast(idxs, const_dtype, True, "TRUNCATION")
        selected = ops.divmod(idxs, nn.Constant(
            np.array([self.config.num_experts_per_tok]).astype(const_dtype))(), 'TRUNC')[0]
        sorted_tokens = ops.gather(x, selected, 0, 0)
        cond_in = nn.Constant(np.array([1]).astype(np.uint8))()
        start_idx = nn.Constant(np.array([0]).astype(np.int32))()
        self.loop_subgraph = self.create_loop_subgraph(self.config, self.layer_id)
        outs = ops.loop([iter_num, cond_in, start_idx,
                         sorted_tokens, tokens_per_expert, tokens_per_expert_idx], self.loop_subgraph,
                        1, 1, True, len(self.experts))[1]
        outs = ops.reshape(outs, [len(self.experts) * self.bsz * self.seq_len *
                           self.config.num_experts_per_tok, self.hidden_size])
        outs = ops.crop(outs, [(0, self.bsz * self.seq_len * self.config.num_experts_per_tok), (0, self.hidden_size)])
        outs.op.params['ds_crops'] = to_expr(
            [(0, f"{self.bsz} * dseq_len * {self.config.num_experts_per_tok}"), (0, self.hidden_size)])

        outs = ops.reshape(outs, [self.bsz * self.seq_len * self.config.num_experts_per_tok, self.hidden_size])
        idxs = ops.reshape(idxs, [self.bsz * self.seq_len * self.config.num_experts_per_tok, 1])
        outs = ops.scatter_nd(outs, idxs, outs, 'None')
        outs = ops.reshape(outs, [self.bsz * self.seq_len, self.config.num_experts_per_tok, self.hidden_size])
        outs.op.params['ds_output_shape'] = to_expr(
            [[f'{self.bsz}*dseq_len', self.config.num_experts_per_tok, self.hidden_size]])
        outs = outs * ops.reshape(topk_weight, [self.bsz * self.seq_len, self.config.num_experts_per_tok, 1])
        outs = ops.reduce_sum(outs, [1], keep_dim=False)
        return outs

    def create_loop_subgraph(self, config, layer_id):
        def construct_if_subgraph(layer_id, expert_id):
            with Graph(f'layer_{layer_id}_mlp_expert_{expert_id}_valid_subgraph') as valid_g:
                tokens_for_this_expert = ops.input([self.bsz * self.seq_len * self.config.num_experts_per_tok, self.hidden_size],
                                                   Dtype.FP32, name=f'token_in_layer_{layer_id}_mlp_expert_{expert_id}_then')
                out = self.experts[expert_id](tokens_for_this_expert)
            valid_g.input_tensors = TensorList([tokens_for_this_expert])
            valid_g.output_tensors = TensorList([out])
            with Graph(f'layer_{layer_id}_mlp_expert_{expert_id}_invalid_subgraph') as invalid_g:
                tokens_for_this_expert = ops.empty([self.bsz * self.seq_len * self.config.num_experts_per_tok, self.hidden_size],
                                                   Dtype.FP32, exprs=[0, self.hidden_size])
            invalid_g.input_tensors = TensorList([])
            invalid_g.output_tensors = TensorList([tokens_for_this_expert])
            return valid_g, invalid_g

        with Graph(f'layer_{layer_id}_mlp_loop_subgraph') as subgraph:
            iter_num = ops.input([1], Dtype.INT32, name=f'iter_num_graph_{layer_id}')
            cond_in = ops.input([1], Dtype.UINT8, name=f'cond_in_graph_{layer_id}')
            start_idx = ops.input([1], Dtype.INT32, name=f'start_idx_graph_{layer_id}')
            sorted_tokens = ops.input([self.bsz * self.seq_len * self.config.num_experts_per_tok,
                                      self.hidden_size], Dtype.FP32, name=f'sorted_tokens_graph_{layer_id}')
            tokens_per_expert = ops.input([len(self.experts)], Dtype.FP32, name=f'tokens_per_exp_graph_{layer_id}')
            tokens_per_expert_idx = ops.input([len(self.experts)], Dtype.INT32,
                                              name=f'tokens_per_exp_graph_idx_{layer_id}')
            GROUP_NUMBER = self.config.get('loop_subgraph_split_num', 2)
            num_tokens = ops.gather(tokens_per_expert, iter_num, 0, 0)
            expert_idx = ops.gather(tokens_per_expert_idx, iter_num, 0, 0)
            end_idx = start_idx + ops.cast(num_tokens, Dtype.INT32, ignore_scale_zp=True)
            tokens_for_this_expert_slice = ops.dynamic_slice([sorted_tokens, start_idx, end_idx, nn.Constant(
                np.zeros([1]).astype(np.int32))(), nn.Constant(np.ones([1]).astype(np.int32))()])[0]
            result = []
            scan_out = []
            for group in range(GROUP_NUMBER):
                group_size = len(self.experts) // GROUP_NUMBER
                for i in range(len(self.experts) // GROUP_NUMBER):
                    valid = ops.greater(num_tokens, nn.Constant(np.array([0]).astype(np.float32))())
                    match = ops.equal(expert_idx, nn.Constant(np.array([group_size*group+i]).astype(np.float32))())
                    if_cond = valid * match
                    if_cond = ops.cast(if_cond, Dtype.UINT8, True, "TRUNCATION")
                    valid_g, invalid_g = construct_if_subgraph(layer_id, group_size*group+i)
                    y = ops.If([if_cond, tokens_for_this_expert_slice],
                               valid_g, invalid_g)[0]
                    final_out = ops.dummy([y])[0]
                    result.append(final_out)
                result = ops.switch(result)
                scan_out.append(result)
                result = []
            scan_out = ops.switch(scan_out)
            start_idx = end_idx
            cond_out = ops.less(iter_num, nn.Constant(np.array([len(self.experts) - 1]).astype(np.int32))())
            cond_out = ops.cast(cond_out, Dtype.UINT8)
            cond_in_out = ops.dummy([cond_in, cond_out])
        subgraph.input_tensors = TensorList(
            [iter_num, cond_in, start_idx, sorted_tokens, tokens_per_expert, tokens_per_expert_idx])
        subgraph.output_tensors = TensorList([cond_out, start_idx, scan_out])
        # subgraph.serialize("loop_graph.def", "loop_graph.bin")
        return subgraph


class GptOssDecoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = GptOssAttention(config, weights, layer_id, prefix)

        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = GptOssMoeGate(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            """
            when enable_subgraph=true, will do some special operations:
            1. avoiding the constant output tensor is subgraph's input tensor, will re-generate one constant in subgraph.
            2. if one tensor is multi ops input, the subgraph will insert the duplicate input tensor into subgraph input tensors,
            which will cause big change for llm runtime. in order to avoid this graph, will insert reshape op, its output tensors will
            be as multi ops input, and reshape is including in this subgraph.
            """
            hidden_states = ops.reshape(hidden_states, hidden_states.shape)
            attention_mask = ops.tile(attention_mask, [1, self.config.num_attention_heads, 1, 1])
            position_ids = ops.reshape(position_ids, position_ids.shape)

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       k_cache,
                                       v_cache,
                                       use_cache)
        hidden_states = hidden_states.reshape([self.config.get('batch', 1), -1, hidden_states.shape[1]])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.reshape([-1, hidden_states.shape[-1]])
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.reshape([self.config.get('batch', 1), -1, hidden_states.shape[1]])
        hidden_states = residual + hidden_states

        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class GptOssModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights.get('model.embed_tokens.weight')['data'].float()

        self.decoders = [GptOssDecoder(config, weights, idx, prefix) for idx in gen_layer_ids(config.num_hidden_layers)]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key=[],
                past_value=[],
                use_cache=None,
                with_embed=True,
                with_preprocess_mask=False,
                ):

        # if use_cache and (len(past_key) == 0 or len(past_value) == 0):
        #     ERROR(f"when use_kv_cache, please provide the kv inputs")

        # if not use_cache:
        #     past_key = [None] * self.model_config.num_hidden_layers
        #     past_value = [None] * self.model_config.num_hidden_layers

        # input_embeds = self.embed_tokens(input_ids)
        if with_embed:
            embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
                                       self.model_config.vocab_size, self.model_config.hidden_size])
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        else:
            input_embeds = input_ids

        # is_gen_subgraph = hasattr(self.model_config, 'enable_subgraph') and self.model_config.enable_subgraph
        # if not is_gen_subgraph:
        #     attention_mask = ops.tile(attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

        hidden_states = input_embeds
        history_ks, history_vs = [], []
        for i, decoder in enumerate(self.decoders):
            hidden_states, history_k, history_v = decoder(hidden_states,
                                                          attention_mask,
                                                          position_ids,
                                                          past_key,
                                                          past_value,
                                                          use_cache=use_cache,
                                                          )
        return hidden_states


class NormLmHead(nn.Module):
    def __init__(self, config, weights):
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.norm.weights, weights.get(f'model.norm.weight'), f'normlm.norm.weight')

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        key = 'lm_head.weight'
        if config.get('tie_word_embeddings', False):
            key = 'model.embed_tokens.weight'
        update_tensor_data(self.lm_head.weights, weights.get(key), f'normlm.lm.weight')

    @gen_subgraph(subgraph_name='lm')
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        h_shape = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [h_shape[0] * h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        out = ops.reshape(lm_head, [1, *list(lm_head.shape)])
        return out


@register_causallm(['gpt_oss'])
class GptOssCausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        INFO(f"\t current num_hidden_layers = {config.num_hidden_layers}:")
        weights = self.config.get('weights', self._load_state_dict())
        self.model = GptOssModel(config, weights, prefix)
        self.norm_and_lm_head = NormLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        with Graph('gpt-oss') as self.aipu_g:
            if with_embed:
                hidden_states = ops.input([batch, seq_len], Dtype.FP32, name='input_ids', exprs=['batch', 'seq_len'])
            else:
                hidden_states = ops.input([batch, seq_len, self.config.hidden_size], Dtype.FP32,
                                          name='embeded', exprs=['batch', 'seq_len', self.config.hidden_size])
            if self.use_cache:
                attention_mask = ops.input([batch, 1, seq_len, seq_len + past_len],
                                           Dtype.FP32, name='attention_mask')
            else:
                attention_mask = ops.input([batch, 1, seq_len, seq_len + past_len], Dtype.FP32,
                                           name='attention_mask', exprs=['batch', 1, 'seq_len', 'max_len'])
                # attention_mask = ops.input([1, 1, 1, seq_len], Dtype.FP32, name='attention_mask')
            # attention_mask = Tensor('attention_mask', TensorShape([1, seq_len]), Dtype.INT32)
            position_ids = ops.input([batch, seq_len], Dtype.INT32, name='position_ids', exprs=['batch', 'seq_len'])

            kcache_module = nn.KVCacheModule(self.config, name='kcache')
            vcache_module = nn.KVCacheModule(self.config, name='vcache')

            hidden_states = self.model(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       past_key=kcache_module,
                                       past_value=vcache_module,
                                       use_cache=use_cache,
                                       with_embed=with_embed)
            _ = self.norm_and_lm_head(hidden_states)

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
