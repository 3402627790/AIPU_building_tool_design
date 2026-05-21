import numpy as np
import torch
import os
from AIPUBuilder.core import *
from AIPUBuilder.logger import ERROR, INFO, DEBUG
from AIPUBuilder import ops
from AIPUBuilder import nn
from safetensors import safe_open
from ...utils import update_tensor_data
import math


def weight_dequant_kernel(x_ptr, s_ptr, y, M, N, block_size):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    scale = torch.repeat_interleave(s_ptr, block_size, 0)
    scale = torch.repeat_interleave(scale, block_size, 1)
    s = scale[0:M, 0:N]
    y[:] = x_ptr.float() * s
    # return y.to(torch.float16)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float16)
    weight_dequant_kernel(x, s, y, M, N, block_size)
    return y

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


def output_file(model_type, output_dir, prefix='', postfix=''):
    prefix = '' if prefix == '' else f"_{prefix}"
    postfix = '' if postfix == '' else f"_{postfix}"
    irt = f"{model_type}{prefix}{postfix}.txt"
    irb = f"{model_type}{prefix}{postfix}.bin"
    if output_dir is not None:
        irt = os.path.join(output_dir, irt)
        irb = os.path.join(output_dir, irb)
    return irt, irb


def check_aipu_graph(g):
    if not check_graph(g):
        ERROR(f"check graph = {g} fialed.")
        return False
    return True


def serialize(g: Graph, irt: str, irb: str, postfix=''):
    if len(postfix):
        for n in g.nodes:
            if n.type == OpType.SubGraphOp:
                n.params['call_subgraph'] = n.params['call_subgraph'] + postfix
                if 'txt_path' in n.attrs and len(n.attrs['txt_path']):
                    txt_path = n.attrs['txt_path']
                    pos = txt_path.rindex('.')
                    n.attrs['txt_path'] = txt_path[:pos] + postfix + txt_path[pos:]
    g.serialize_scale_zp = True
    g.serialize(irt, irb)


class DeepseekR1RotaryEmbedding(nn.Module):
    def __init__(self, config, num_heads, dim, max_position_embeddings=2048, base=10000, device=None, name=''):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.name = name
        rope_type = 'normal'
        if config.rope_scaling is not None:
            rope_type = config.rope_scaling["type"]
        self.max_seq_len_cached = max_position_embeddings
        if rope_type != 'yarn':
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                                                              dtype=torch.int64).float().to(device) / self.dim))
            t = torch.arange(self.max_seq_len_cached, device=device,
                             dtype=torch.int64).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(torch.get_default_dtype())
            self.sin_cached = emb.sin().to(torch.get_default_dtype())
        else:
            # yarn rope
            self.scaling_factor = 1
            self.beta_fast = 32
            self.beta_slow = 1
            self.mscale = 1
            self.mscale_all_dim = 1
            if config.rope_scaling is not None:
                self.scaling_factor = config.rope_scaling["factor"]
                self.beta_fast = config.rope_scaling["beta_fast"]
                self.beta_slow = config.rope_scaling["beta_slow"]

            self.original_max_position_embeddings = 4096
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
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)

            freqs = torch.outer(t, inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(torch.get_default_dtype())
            self.sin_cached = emb.sin().to(torch.get_default_dtype())

    def forward(self, pos_ids):
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        sin_out_reshape = ops.reshape(
            gather_sin_out, [1, *gather_sin_out.shape])
        cos_out_reshape = ops.reshape(
            gather_cos_out, [1, *gather_cos_out.shape])
        # sin_reshape_tile = ops.tile(sin_out_reshape, [1, self.num_heads, 1, 1])
        # cos_reshape_tile = ops.tile(cos_out_reshape, [1, self.num_heads, 1, 1])

        if self.name != '':
            sin_out.op.constants['weights'].name = f'{self.name}.rotary_emb.sin_weight'
            cos_out.op.constants['weights'].name = f'{self.name}.rotary_emb.cos_weight'
            sin_out.op.name = f"{self.name}.rotary_emb.sin_{sin_out.op.name}"
            cos_out.op.name = f"{self.name}.rotary_emb.cos_{cos_out.op.name}"
            gather_sin_out.name = f"{self.name}.{gather_sin_out.name}"
            gather_cos_out.name = f"{self.name}.{gather_cos_out.name}"

        return cos_out_reshape, sin_out_reshape


class DeepseekR1Attention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank  # 1536
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.kv_lora_rank = config.kv_lora_rank  # 512
        self.v_head_dim = config.v_head_dim  # 128
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 128
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        # self.hidden_size = config.hidden_size
        # self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling_factor = self.config.rope_scaling['factor']
        # self.max_position_embeddings = config.max_position_embeddings
        # self.rope_theta = config.rope_theta
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            dtype=Dtype.FP16
        )
        weight_scale_inv = weights.get(
            f'model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight_scale_inv')
        weight = weights.get(
            f'model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight')
        weight = weight_dequant(weight, weight_scale_inv)
        update_tensor_data(self.kv_a_proj_with_mqa.weights,
                           weight,
                           f"{prefix}layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight",
                           dtype=np.float16)

        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, dtype=Dtype.FP16
        )
        weight_scale_inv = weights.get(
            f'model.layers.{layer_id}.self_attn.q_a_proj.weight_scale_inv')
        weight = weights.get(
            f'model.layers.{layer_id}.self_attn.q_a_proj.weight')
        weight = weight_dequant(weight, weight_scale_inv)
        update_tensor_data(self.q_a_proj.weights,
                           weight,
                           f"{prefix}layers.{layer_id}.self_attn.q_a_proj.weight",
                           dtype=np.float16)

        # self.q_a_layernorm = DeepseekR1RMSNorm(config.q_lora_rank)
        self.q_a_layernorm = nn.RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        update_tensor_data(self.q_a_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.self_attn.q_a_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.self_attn.q_a_layernorm.weight")

        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, dtype=Dtype.FP16
        )
        weight_scale_inv = weights.get(
            f'model.layers.{layer_id}.self_attn.q_b_proj.weight_scale_inv')
        weight = weights.get(
            f'model.layers.{layer_id}.self_attn.q_b_proj.weight')
        weight = weight_dequant(weight, weight_scale_inv)
        update_tensor_data(self.q_b_proj.weights,
                           weight,
                           f"{prefix}layers.{layer_id}.self_attn.q_b_proj.weight",
                           dtype=np.float16)
        # self.kv_a_layernorm = DeepseekR1RMSNorm(config.kv_lora_rank)
        self.kv_a_layernorm = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        update_tensor_data(self.kv_a_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.self_attn.kv_a_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.self_attn.kv_a_layernorm.weight")
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim), dtype=Dtype.FP16
        )
        weight_scale_inv = weights.get(
            f'model.layers.{layer_id}.self_attn.kv_b_proj.weight_scale_inv')
        weight = weights.get(
            f'model.layers.{layer_id}.self_attn.kv_b_proj.weight')
        weight = weight_dequant(weight, weight_scale_inv)
        wkvb = weight.reshape(1, self.num_heads, (self.q_head_dim -
                              self.qk_rope_head_dim + self.v_head_dim), config.kv_lora_rank)
        self.wkv_q_nope = nn.Constant(
            np.zeros([1, self.num_heads, (self.q_head_dim - self.qk_rope_head_dim), config.kv_lora_rank]).astype('float16'))

        update_tensor_data(self.wkv_q_nope.weights,
                           wkvb[:, :, :(self.q_head_dim - self.qk_rope_head_dim), :],
                           f"{prefix}layers.{layer_id}.self_attn.kv_b_proj.weight",
                           dtype=np.float16)
        self.wkv_k_pe = nn.Constant(
            np.zeros([1, self.num_heads, (self.q_head_dim - self.qk_rope_head_dim), config.kv_lora_rank]).astype('float16'))

        update_tensor_data(self.wkv_k_pe.weights,
                           wkvb[:, :, (self.q_head_dim - self.qk_rope_head_dim):, :],
                           f"{prefix}layers.{layer_id}.self_attn.kv_b_proj.weight",
                           dtype=np.float16)
        update_tensor_data(self.kv_b_proj.weights,
                           weight,
                           f"{prefix}layers.{layer_id}.self_attn.kv_b_proj.weight",
                           dtype=np.float16)

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size, dtype=Dtype.FP16
        )
        weight_scale_inv = weights.get(
            f'model.layers.{layer_id}.self_attn.o_proj.weight_scale_inv')
        weight = weights.get(
            f'model.layers.{layer_id}.self_attn.o_proj.weight')
        weight = weight_dequant(weight, weight_scale_inv)
        update_tensor_data(self.o_proj.weights,
                           weight,
                           f"{prefix}layers.{layer_id}.self_attn.o_proj.weight",
                           dtype=np.float16)

        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, kv_cache=None, pe_cache=None, use_cache=False):

        bs, seq_len, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = ops.reshape(q, [bs, seq_len, self.num_heads, self.q_head_dim])
        q = self.qkv_transpose(q)

        crop_pe = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [
            self.qk_nope_head_dim, self.qk_nope_head_dim+self.qk_rope_head_dim]])
        crop_nope = nn.Crop([[0, bs], [0, self.num_heads], [
            0, seq_len], [0, self.qk_nope_head_dim]])

        q_nope = crop_nope(q)
        q_pe = crop_pe(q)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pe_crop = nn.Crop([[0, seq_len], [
            self.kv_lora_rank, self.kv_lora_rank+self.qk_rope_head_dim]])
        compressed_kv_crop = nn.Crop([[
            0, seq_len], [0, self.kv_lora_rank]])
        k_pe = k_pe_crop(compressed_kv)
        kv = compressed_kv_crop(compressed_kv)

        k_pe = ops.reshape(k_pe, [bs, seq_len, 1, self.qk_rope_head_dim])
        k_pe = self.qkv_transpose(k_pe)

        kv_seq_len = kv_cache.shape[-2]

        self.rotary_emb = DeepseekR1RotaryEmbedding(self.config, self.num_heads,
                                                    q_pe.shape[-1],
                                                    max_position_embeddings=kv_seq_len,
                                                    base=self.rope_theta,
                                                    name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        # convert complex to real,iamge to r[0,32],i[32,64], origial format is r,i,r,i, after convert, r,r,r,r,i,i,i,i
        q_pe_0 = ops.reshape(q_pe, [q_pe.shape[0], q_pe.shape[1], q_pe.shape[2], q_pe.shape[3]//2, 2])
        q_pe_0 = ops.transpose(q_pe_0, [0, 1, 2, 4, 3])
        q_pe = ops.reshape(q_pe_0, [q_pe.shape[0], q_pe.shape[1], q_pe.shape[2], q_pe.shape[3]])

        k_pe_0 = ops.reshape(k_pe, [k_pe.shape[0], k_pe.shape[1], k_pe.shape[2], k_pe.shape[3]//2, 2])
        k_pe_0 = ops.transpose(k_pe_0, [0, 1, 2, 4, 3])
        k_pe = ops.reshape(k_pe_0, [k_pe.shape[0], k_pe.shape[1], k_pe.shape[2], k_pe.shape[3]])
        # query
        q_crop0 = nn.Crop([[0, bs], [0, q_pe.shape[1]], [0, seq_len], [
            self.qk_rope_head_dim//2, self.qk_rope_head_dim]])
        q_crop1 = nn.Crop([[0, bs], [0, q_pe.shape[1]], [
            0, seq_len], [0, self.qk_rope_head_dim//2]])

        q_crop_out0 = q_crop0(q_pe)
        q_crop_out1 = q_crop1(q_pe)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        # key
        k_crop0 = nn.Crop([[0, bs], [0, k_pe.shape[1]], [0, seq_len], [
            self.qk_rope_head_dim//2, self.qk_rope_head_dim]])
        k_crop1 = nn.Crop([[0, bs], [0, k_pe.shape[1]], [
            0, seq_len], [0, self.qk_rope_head_dim//2]])

        k_crop_out0 = k_crop0(k_pe)
        k_crop_out1 = k_crop1(k_pe)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        neg_q_crop_out0.op.name = f"{self.matmul_name}.negative_0"
        neg_k_crop_out0.op.name = f"{self.matmul_name}.negative_1"

        q_cosine = q_pe * cos_emb
        k_cosine = k_pe * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb

        q_pe = q_cosine + q_r_sine
        k_pe = k_cosine + k_r_sine

        history_kv, history_pe = None, None
        wkv_q_nope = self.wkv_q_nope()

        q_nope = self.kqv_matmul(q_nope, wkv_q_nope)
        kv = self.kv_a_layernorm(kv)
        kv = ops.reshape(kv, [1, 1, kv.shape[0], kv.shape[1]])
        if use_cache:
            history_kv = nn.Concat(2)([kv_cache, kv])
            history_pe = nn.Concat(2)([pe_cache, k_pe])

            kq_matmul0 = self.kq_matmul(q_nope, history_kv)
            kq_matmul1 = self.kq_matmul(q_pe, history_pe)
            kq_matmul = kq_matmul0 + kq_matmul1

        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"

        mscale = yarn_get_mscale(self.scaling_factor, 1)
        mscale = mscale*mscale
        self.scale = nn.Constant(
            np.array(mscale/np.sqrt(np.array(self.qk_nope_head_dim+self.qk_rope_head_dim))).astype('float32').reshape([1]))
        self.scale.weights.name = f"scale_for_normalization"
        scaled_kq = kq_matmul * self.scale()

        softmax_masked = self.softmax(scaled_kq + attention_mask)

        kqv_out = self.kqv_matmul(softmax_masked, history_kv)
        wkv_k_pe = self.wkv_k_pe()

        kqv_out = self.kq_matmul(kqv_out, wkv_k_pe)

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"
        # kqv_out.name = f"{self.matmul_name}_{kqv_out.name}"

        self.qkv_reshape = nn.Reshape(
            [bs*seq_len, self.num_heads * self.v_head_dim])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.o_proj(reshaped_trans)

        return o_t, history_kv, history_pe


class DeepseekR1MLP(nn.Module):

    def __init__(self, config, weights, layer_id, expert_id, expert_flag, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if expert_flag:
            # 'model.layers.3.mlp.shared_experts.down_proj.weight'

            self.intermediate_size = config.moe_intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, dtype=Dtype.FP16)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, dtype=Dtype.FP16)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, dtype=Dtype.FP16)
            self.act_fn = nn.ACT2FN[config.hidden_act]
            if expert_id == -1:
                weight_scale_inv = weights.get(
                    f'model.layers.{layer_id}.mlp.shared_experts.down_proj.weight_scale_inv')
                weight = weights.get(
                    f'model.layers.{layer_id}.mlp.shared_experts.down_proj.weight')
                weight = weight_dequant(weight, weight_scale_inv)
                update_tensor_data(self.down_proj.weights,
                                   weight,
                                   f"{prefix}layers.{layer_id}.mlp.shared_experts.down_proj.weight",
                                   dtype=np.float16)

                weight_scale_inv = weights.get(
                    f'model.layers.{layer_id}.mlp.shared_experts.up_proj.weight_scale_inv')
                weight = weights.get(
                    f'model.layers.{layer_id}.mlp.shared_experts.up_proj.weight')
                weight = weight_dequant(weight, weight_scale_inv)
                update_tensor_data(self.up_proj.weights,
                                   weight,
                                   f"{prefix}layers.{layer_id}.mlp.shared_experts.up_proj.weight",
                                   dtype=np.float16)

                weight_scale_inv = weights.get(
                    f'model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight_scale_inv')
                weight = weights.get(
                    f'model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight')
                weight = weight_dequant(weight, weight_scale_inv)
                update_tensor_data(self.gate_proj.weights,
                                   weight,
                                   f"{prefix}layers.{layer_id}.mlp.shared_experts.gate_proj.weight",
                                   dtype=np.float16)

            else:
                weight_scale_inv = weights.get(
                    f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight_scale_inv')
                weight = weights.get(
                    f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight')
                weight = weight_dequant(weight, weight_scale_inv)
                update_tensor_data(self.down_proj.weights,
                                   weight,
                                   f"{prefix}layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight",
                                   dtype=np.float16)

                weight_scale_inv = weights.get(
                    f'model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight_scale_inv')
                weight = weights.get(
                    f'model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight')
                weight = weight_dequant(weight, weight_scale_inv)
                update_tensor_data(self.up_proj.weights,
                                   weight,
                                   f"{prefix}layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight",
                                   dtype=np.float16)

                weight_scale_inv = weights.get(
                    f'model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight_scale_inv')
                weight = weights.get(
                    f'model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight')
                weight = weight_dequant(weight, weight_scale_inv)
                update_tensor_data(self.gate_proj.weights,
                                   weight,
                                   f"{prefix}layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight",
                                   dtype=np.float16)

        else:
            self.intermediate_size = config.intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, dtype=Dtype.FP16)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, dtype=Dtype.FP16)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, dtype=Dtype.FP16)
            self.act_fn = nn.ACT2FN[config.hidden_act]

            weight_scale_inv = weights.get(
                f'model.layers.{layer_id}.mlp.down_proj.weight_scale_inv')
            weight = weights.get(
                f'model.layers.{layer_id}.mlp.down_proj.weight')
            weight = weight_dequant(weight, weight_scale_inv)
            update_tensor_data(self.down_proj.weights,
                               weight,
                               f"{prefix}layers.{layer_id}.mlp.down_proj.weight",
                               dtype=np.float16)

            weight_scale_inv = weights.get(
                f'model.layers.{layer_id}.mlp.up_proj.weight_scale_inv')
            weight = weights.get(
                f'model.layers.{layer_id}.mlp.up_proj.weight')
            weight = weight_dequant(weight, weight_scale_inv)
            update_tensor_data(self.up_proj.weights,
                               weight,
                               f"{prefix}layers.{layer_id}.mlp.up_proj.weight",
                               dtype=np.float16)

            weight_scale_inv = weights.get(
                f'model.layers.{layer_id}.mlp.gate_proj.weight_scale_inv')
            weight = weights.get(
                f'model.layers.{layer_id}.mlp.gate_proj.weight')
            weight = weight_dequant(weight, weight_scale_inv)
            update_tensor_data(self.gate_proj.weights,
                               weight,
                               f"{prefix}layers.{layer_id}.mlp.gate_proj.weight",
                               dtype=np.float16)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
