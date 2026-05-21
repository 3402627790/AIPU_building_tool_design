import numpy as np
import os
import math
import torch
from typing import Any, Dict, Optional, Tuple, Union
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data, convert_expr
from ..llmutils import gen_layer_ids

ACT2CLS = {
    "swish": torch.nn.SiLU,
    "silu": torch.nn.SiLU,
    "mish": torch.nn.Mish,
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
}


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  # torch.float32, torch.float64 (flux)
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


class WanRotaryPosEmbed(nn.Module):

    def __init__(self,
                 attention_head_dim: int,
                 patch_size: Tuple[int, int, int],
                 max_seq_len: int,
                 theta: float = 10000.0,):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

    def forward(self, hidden_states):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos
        freqs_sin = self.freqs_sin

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        op_freqs_cos = nn.Constant(freqs_cos.float().cpu().numpy())()
        op_freqs_sin = nn.Constant(freqs_sin.float().cpu().numpy())()
        return op_freqs_cos, op_freqs_sin


class TimestepEmbedding(torch.nn.Module):
    def __init__(
        self,
        weights,
        prefix: str,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_1.weight.data = weights.get(f"{prefix}linear_1.weight")['data']
        self.linear_1.bias.data = weights.get(f"{prefix}linear_1.bias")['data']

        self.cond_proj = None

        self.act = ACT2CLS[act_fn]()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)
        self.linear_2.weight.data = weights.get(f"{prefix}linear_2.weight")['data']
        self.linear_2.bias.data = weights.get(f"{prefix}linear_2.bias")['data']

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = ACT2CLS[post_act_fn]()

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, weight, prefix, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        linear_1_w = weight.get(f"{prefix}linear_1.weight")
        update_tensor_data(self.linear_1.weights, linear_1_w, f'{prefix}linear_1.weight')
        linear_1_b = weight.get(f"{prefix}linear_1.bias")
        update_tensor_data(self.linear_1.biases, linear_1_b, f'{prefix}linear_1.bias')
        self.act_1 = nn.ACT2FN[act_fn]
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features)
        linear_2_w = weight.get(f"{prefix}linear_2.weight")
        update_tensor_data(self.linear_2.weights, linear_2_w, f'{prefix}linear_2.weight')
        linear_2_b = weight.get(f"{prefix}linear_2.bias")
        update_tensor_data(self.linear_2.biases, linear_2_b, f'{prefix}linear_2.bias')

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class WanTimeEmbedding(torch.nn.Module):
    def __init__(self,
                 weight,
                 dim: int,
                 time_freq_dim: int,
                 image_embed_dim: Optional[int] = None,
                 pos_embed_seq_len: Optional[int] = None,):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            weight, prefix='condition_embedder.time_embedder.', in_channels=time_freq_dim, time_embed_dim=dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.reshape(1, timestep_seq_len, -1)

        temb = self.time_embedder(timestep)
        return temb


class WanTimeProj(nn.Module):

    def __init__(self,
                 weight,
                 dim: int,
                 time_proj_dim: int,
                 ):
        super().__init__()
        self.act_fn = nn.ACT2FN["silu"]
        self.time_proj = nn.Linear(dim, time_proj_dim)
        time_proj_w = weight.get("condition_embedder.time_proj.weight")
        update_tensor_data(self.time_proj.weights, time_proj_w, "condition_embedder.time_proj.weight")
        time_proj_b = weight.get("condition_embedder.time_proj.bias")
        update_tensor_data(self.time_proj.biases, time_proj_b, "condition_embedder.time_proj.bias")

    def forward(self, temb):
        return self.time_proj(self.act_fn(temb))


class WanTextEmbedding(nn.Module):

    def __init__(self,
                 weight,
                 dim: int,
                 text_embed_dim: int):
        super().__init__()
        self.text_embedder = PixArtAlphaTextProjection(
            weight, 'condition_embedder.text_embedder.', text_embed_dim, dim, act_fn="gelu_pytorch_tanh")

    def forward(self, encoder_hidden_states):
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return encoder_hidden_states


class WanAttention(nn.Module):

    def __init__(
        self,
        weight,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        is_cross_attention=None,
        prefix: str = '',
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = nn.Linear(dim, self.inner_dim)
        q_w = weight.get(f"{prefix}to_q.weight")
        update_tensor_data(self.to_q.weights, q_w, f'{prefix}to_q.weight')
        q_b = weight.get(f"{prefix}to_q.bias")
        update_tensor_data(self.to_q.biases, q_b, f'{prefix}to_q.bias')
        self.to_k = nn.Linear(dim, self.kv_inner_dim)
        k_w = weight.get(f"{prefix}to_k.weight")
        update_tensor_data(self.to_k.weights, k_w, f'{prefix}to_k.weight')
        k_b = weight.get(f"{prefix}to_k.bias")
        update_tensor_data(self.to_k.biases, k_b, f'{prefix}to_k.bias')
        self.to_v = nn.Linear(dim, self.kv_inner_dim)
        v_w = weight.get(f"{prefix}to_v.weight")
        update_tensor_data(self.to_v.weights, v_w, f'{prefix}to_v.weight')
        v_b = weight.get(f"{prefix}to_v.bias")
        update_tensor_data(self.to_v.biases, v_b, f'{prefix}to_v.bias')
        self.to_out = nn.Linear(self.inner_dim, dim)
        to_out_w = weight.get(f"{prefix}to_out.0.weight")
        update_tensor_data(self.to_out.weights, to_out_w, f'{prefix}to_out.0.weight')
        to_out_b = weight.get(f"{prefix}to_out.0.bias")
        update_tensor_data(self.to_out.biases, to_out_b, f'{prefix}to_out.0.bias')

        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps)
        norm_q_w = weight.get(f"{prefix}norm_q.weight")
        update_tensor_data(self.norm_q.weights, norm_q_w, f'{prefix}norm_q.weight')
        norm_k_w = weight.get(f"{prefix}norm_k.weight")
        update_tensor_data(self.norm_k.weights, norm_k_w, f'{prefix}norm_k.weight')

        self.is_cross_attention = cross_attention_dim_head is not None

    def apply_rotary_emb(self, hidden_states, freqs_cos, freqs_sin):
        bsz, _, seqlen, _ = hidden_states.shape
        hidden_states = ops.transpose(hidden_states, [0, 2, 1, 3])
        hidden_states = ops.reshape(hidden_states, [bsz, seqlen, self.heads, self.dim_head//2, 2])
        # freqs_cos: [1,seqlen, 1, inner_dim]
        x1 = ops.crop(hidden_states, [[0, bsz], [0, seqlen], [0, self.heads], [0, self.dim_head], [0, 1]])
        x2 = ops.crop(hidden_states, [[0, bsz], [0, seqlen], [0, self.heads], [0, self.dim_head], [1, 2]])
        x1 = ops.reshape(x1, [bsz, seqlen, self.heads, self.dim_head//2])
        x2 = ops.reshape(x2, [bsz, seqlen, self.heads, self.dim_head//2])
        cos = ops.reshape(freqs_cos, [1, seqlen, 1, self.dim_head//2, 2])
        cos = ops.crop(cos, [[0, 1], [0, seqlen], [0, 1], [0, self.dim_head//2], [0, 1]])
        sin = ops.reshape(freqs_sin, [1, seqlen, 1, self.dim_head//2, 2])
        sin = ops.crop(sin, [[0, 1], [0, seqlen], [0, 1], [0, self.dim_head//2], [1, 2]])
        cos = ops.reshape(cos, [1, seqlen, 1, self.dim_head//2])
        sin = ops.reshape(sin, [1, seqlen, 1, self.dim_head//2])
        half_1 = x1 * cos - x2 * sin
        half_2 = x1 * sin + x2 * cos
        half_1 = ops.reshape(half_1, [bsz, seqlen, self.heads, self.dim_head//2, 1])
        half_2 = ops.reshape(half_2, [bsz, seqlen, self.heads, self.dim_head//2, 1])
        out = ops.concat([half_1, half_2], axis=4)
        out = ops.reshape(out, [bsz, seqlen, self.heads, self.dim_head])
        out = ops.transpose(out, [0, 2, 1, 3])
        return out

    def forward(self, hidden_states, encoder_hidden_states, rotary_emb_cos, rotary_emb_sin):
        bsz, seqlen, hidden_size = hidden_states.shape
        if not self.is_cross_attention:
            encoder_hidden_states = hidden_states
        _, textlen, _ = encoder_hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bsz * seqlen, self.inner_dim])
        encoder_hidden_states = ops.reshape(encoder_hidden_states, [bsz * textlen, self.inner_dim])
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = ops.reshape(query, [bsz, seqlen, self.heads, self.dim_head])
        query = ops.transpose(query, [0, 2, 1, 3])
        key = ops.reshape(key, [bsz, textlen, self.heads, self.kv_inner_dim // self.heads])
        key = ops.transpose(key, [0, 2, 1, 3])
        value = ops.reshape(value, [bsz, textlen, self.heads, self.kv_inner_dim // self.heads])
        value = ops.transpose(value, [0, 2, 1, 3])

        if not self.is_cross_attention:
            query = self.apply_rotary_emb(query, rotary_emb_cos, rotary_emb_sin)
            key = self.apply_rotary_emb(key, rotary_emb_cos, rotary_emb_sin)

        attn_weight = ops.matmul(query, key, adj_b=True, fused_multiplier=self.dim_head ** -0.5)
        attn_score = ops.softmax(attn_weight, axis=-1)
        attn_output = ops.matmul(attn_score, value)

        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [bsz * seqlen, self.inner_dim])
        attn_output = self.to_out(attn_output)
        attn_output = ops.reshape(attn_output, [bsz, seqlen, hidden_size])
        return attn_output


class FeedForward(nn.Module):
    def __init__(
        self,
        weight,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        prefix=''
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.act_fn = nn.ACT2FN[activation_fn]
        self.linear1 = nn.Linear(dim, inner_dim)
        linear1_w = weight.get(f"{prefix}net.0.proj.weight")
        update_tensor_data(self.linear1.weights, linear1_w, f'{prefix}net.0.proj.weight')
        linear1_b = weight.get(f"{prefix}net.0.proj.bias")
        update_tensor_data(self.linear1.biases, linear1_b, f'{prefix}net.0.proj.bias')
        self.linear2 = nn.Linear(inner_dim, dim_out)
        linear2_w = weight.get(f"{prefix}net.2.weight")
        update_tensor_data(self.linear2.weights, linear2_w, f'{prefix}net.2.weight')
        linear2_b = weight.get(f"{prefix}net.2.bias")
        update_tensor_data(self.linear2.biases, linear2_b, f'{prefix}net.2.bias')

    def forward(self, hidden_states):
        bsz, seqlen, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bsz * seqlen, hidden_size])
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = ops.reshape(hidden_states, [bsz, seqlen, hidden_size])
        return hidden_states


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        weight,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        prefix='',
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn1 = WanAttention(weight,
                                  dim=dim,
                                  heads=num_heads,
                                  dim_head=dim//num_heads,
                                  eps=eps,
                                  cross_attention_dim_head=None,
                                  prefix=f'{prefix}attn1.'
                                  )

        self.attn2 = WanAttention(weight,
                                  dim=dim,
                                  heads=num_heads,
                                  dim_head=dim//num_heads,
                                  eps=eps,
                                  added_kv_proj_dim=added_kv_proj_dim,
                                  cross_attention_dim_head=dim // num_heads,
                                  prefix=f'{prefix}attn2.'
                                  )
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        norm2_w = weight.get(f"{prefix}norm2.weight")
        update_tensor_data(self.norm2.weights,
                           norm2_w,
                           f"{prefix}norm2.weight")
        norm2_b = weight.get(f"{prefix}norm2.bias")
        update_tensor_data(self.norm2.biases,
                           norm2_b,
                           f"{prefix}norm2.bias")

        self.ffn = FeedForward(weight, dim, inner_dim=ffn_dim, activation_fn='gelu', prefix=f'{prefix}ffn.')
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.scale_shift_table = nn.Constant([1, 6, dim])
        lut_w = weight.get(f"{prefix}scale_shift_table")
        update_tensor_data(self.scale_shift_table.weights,
                           lut_w,
                           f"{prefix}scale_shift_table")

    def forward(self,
                hidden_states,
                encoder_hidden_states,
                temb,
                rotary_emb_cos,
                rotary_emb_sin):
        bsz, seqlen, hidden_size = hidden_states.shape
        factor = ops.reshape(self.scale_shift_table(), [1, 1, 6, hidden_size])
        factor = factor + temb

        (shift_msa, scale_msa, gate_msa,
         c_shift_msa, c_scale_msa, c_gate_msa) = [ops.reshape(t, [1, seqlen, hidden_size]) for t in ops.split(factor, 6, axis=2)]

        norm_hidden_states = self.norm1(hidden_states) * \
            (ops.constant(np.array(1.0).astype(np.float32)) + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states, None, rotary_emb_cos, rotary_emb_sin)
        hidden_states = hidden_states + attn_output * gate_msa

        norm_hidden_states = self.norm2(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm3(hidden_states) * (1.0 + c_scale_msa) + c_shift_msa
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + ff_output * c_gate_msa
        return hidden_states


@register_causallm(['wantransformer3d_time_embedding'])
class WanTransformer3DTimeEmbedding(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        from diffusers.schedulers import UniPCMultistepScheduler
        self.scheduler = UniPCMultistepScheduler.from_pretrained(config.get("scheduler_path", None))
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.hidden_size = self.head_dim * self.num_heads
        self.ffn_dim = config.ffn_dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.num_layers = config.num_layers
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.rope_max_seq_len = config.rope_max_seq_len
        weights = self._load_state_dict()

        self.time_embedder = WanTimeEmbedding(
            weights,
            dim=self.hidden_size,
            time_freq_dim=self.freq_dim,
            image_embed_dim=None,
            pos_embed_seq_len=None,
        )

    def forward(self):
        num_inference_steps = self.config.get('num_inference_steps', 50)
        p_num_frames = self.config.get('num_frames', 21) // self.patch_size[0]
        p_height = self.config.get('height', 88) // self.patch_size[1]
        p_width = self.config.get('width', 160) // self.patch_size[2]
        latent_shape = [1, self.out_channels, p_num_frames, p_height, p_width]
        ts_seq_len = p_num_frames // self.patch_size[0] * p_height // self.patch_size[1] * p_width // self.patch_size[2]

        with torch.no_grad():
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
            first_frame_mask = torch.ones(
                1, 1, p_num_frames, p_height, p_width).float()
            first_frame_mask[:, :, 0] = 0
            temb_l = []
            for t in timesteps:
                temp_ts = (first_frame_mask[0, 0, ::, ::2, ::2] * t).flatten()
                temb = self.time_embedder(temp_ts, ts_seq_len)
                temb = temb.reshape([1, ts_seq_len * self.hidden_size])
                temb_l.append(temb.float().cpu().numpy())

        with Graph("wantransformer3d_time_embedding") as self.g:
            input_idx = ops.input([1, 1], Dtype.INT32, name='input_idx')
            temb = ops.concat([ops.constant(t) for t in temb_l], 0)
            output = ops.gather(temb, input_idx, 0)
            output = ops.reshape(output, [1, ts_seq_len, self.hidden_size])
            self.g.input_tensors = TensorList([input_idx])
            self.g.output_tensors = TensorList([output])
        return self.g


@register_causallm(['wantransformer3d'])
class WanTransformer3D(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.hidden_size = self.head_dim * self.num_heads
        self.ffn_dim = config.ffn_dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.num_layers = config.num_layers
        self.out_channels = config.out_channels
        self.patch_size = config.patch_size
        self.rope_max_seq_len = config.rope_max_seq_len

        weights = self._load_state_dict()

        self.rope = WanRotaryPosEmbed(
            attention_head_dim=self.head_dim,
            patch_size=self.patch_size,
            max_seq_len=self.rope_max_seq_len,
        )
        self.patch_embedding = nn.Conv3d(self.out_channels, self.hidden_size,
                                         kernels=self.patch_size[::-1], strides=self.patch_size, padding=[0, 0, 0, 0, 0, 0])
        patch_embedding_w = weights.get('patch_embedding.weight')
        patch_embedding_w['data'] = np.transpose(patch_embedding_w['data'], [0, 3, 4, 2, 1])
        update_tensor_data(self.patch_embedding.weights, patch_embedding_w)
        patch_embedding_b = weights.get('patch_embedding.bias')
        update_tensor_data(self.patch_embedding.biases, patch_embedding_b)

        self.condition_embedder = WanTextEmbedding(
            weights,
            dim=self.hidden_size,
            text_embed_dim=self.text_dim
        )

        self.time_proj = WanTimeProj(
            weights,
            dim=self.hidden_size,
            time_proj_dim=self.hidden_size * 6,
        )

        self.blocks = [
            WanTransformerBlock(
                weights,
                self.hidden_size, self.ffn_dim, self.num_heads,
                prefix=f'{prefix}blocks.{i}.',
            )
            for i in gen_layer_ids(self.num_layers)
        ]

        self.norm_out = nn.LayerNorm(self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.out_channels * math.prod(self.patch_size))
        proj_out_w = weights.get('proj_out.weight')
        update_tensor_data(self.proj_out.weights, proj_out_w)
        proj_out_b = weights.get('proj_out.bias')
        update_tensor_data(self.proj_out.biases, proj_out_b)
        self.scale_shift_table = nn.Constant([1, 2, self.hidden_size])
        lut = weights.get('scale_shift_table')
        update_tensor_data(self.scale_shift_table.weights, lut)

    def forward(self):
        num_inference_steps = self.config.get('num_inference_steps', 50)
        num_frames = self.config.get('num_frames', 21)
        height = self.config.get('height', 44)
        width = self.config.get('width', 80)
        p_num_frames = self.config.get('num_frames', 21) // self.patch_size[0]
        p_height = self.config.get('height', 44) // self.patch_size[1]
        p_width = self.config.get('width', 80) // self.patch_size[2]
        latent_shape = [1, self.out_channels, num_frames, height, width]
        ts_seq_len = p_num_frames * p_height * p_width
        text_seq_len = self.config.get('emb_len', 512)

        with Graph("wantransformer3d") as self.g:
            latent = ops.input(latent_shape, dtype=Dtype.FP32, name='latent')
            time_emb = ops.input([1, ts_seq_len, self.hidden_size], dtype=Dtype.FP32, name='time_emb')
            encoder_hidden_states = ops.input([1, text_seq_len, self.text_dim],
                                              dtype=Dtype.FP32, name='encoder_hidden_states')

            hidden_states = latent
            rotary_emb = self.rope(hidden_states)

            hidden_states = ops.transpose(hidden_states, [0, 2, 3, 4, 1])
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = ops.reshape(hidden_states, [1, ts_seq_len, self.hidden_size])

            time_emb = ops.reshape(time_emb, [ts_seq_len, self.hidden_size])
            encoder_hidden_states = ops.reshape(encoder_hidden_states, [text_seq_len, self.text_dim])
            timestep_proj = self.time_proj(time_emb)
            encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

            timestep_proj = ops.reshape(timestep_proj, [1, ts_seq_len, 6, self.hidden_size])
            encoder_hidden_states = ops.reshape(encoder_hidden_states, [1, text_seq_len, self.hidden_size])

            for i, block in enumerate(self.blocks):
                # hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
                with Graph(f"wantransformer3d_block_{i}") as vg:
                    hidden_states_sg = ops.input([1, ts_seq_len, self.hidden_size], Dtype.FP32, name=f"block_{i}_input")
                    encoder_hidden_states_sg = ops.input(
                        [1, text_seq_len, self.hidden_size], Dtype.FP32, name="encoder_hidden_states")
                    temb_sg = ops.input([1, ts_seq_len, 6, self.hidden_size], Dtype.FP32, name=f"block_{i}_temb")
                    rotary_emb_cos_sg = ops.input([1, ts_seq_len, 1, self.head_dim], Dtype.FP32, name="rotary_emb_cos")
                    rotary_emb_sin_sg = ops.input([1, ts_seq_len, 1, self.head_dim], Dtype.FP32, name="rotary_emb_sin")
                    hidden_states_sgo = block(hidden_states_sg, encoder_hidden_states_sg,
                                              temb_sg, rotary_emb_cos_sg, rotary_emb_sin_sg)
                    vg.output_tensors = TensorList([hidden_states_sgo])
                with Graph(f"wantransformer3d_block_{i}_empty") as ig:
                    empty = ops.empty([1, ts_seq_len, self.hidden_size], Dtype.FP32)
                    ig.output_tensors = TensorList([empty])
                cond = ops.constant(np.array([1], dtype=np.uint8))
                result = ops.If([cond, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb[0], rotary_emb[1]],
                                vg, ig)[0]
                hidden_states = result

            factor = ops.reshape(self.scale_shift_table(), (1, 1, 2, self.hidden_size))
            factor = factor + ops.reshape(time_emb, [1, ts_seq_len, 1, self.hidden_size])
            shift, scale = [ops.reshape(t, [1, ts_seq_len, self.hidden_size]) for t in ops.split(factor, 2, axis=2)]
            hidden_states = self.norm_out(hidden_states) * \
                (ops.constant(np.array(1.0).astype(np.float32)) + scale) + shift
            hidden_states = ops.reshape(hidden_states, [ts_seq_len, self.hidden_size])
            hidden_states = self.proj_out(hidden_states)
            hidden_states = ops.reshape(hidden_states, [1, ts_seq_len, self.out_channels * math.prod(self.patch_size)])
            pt, ph, pw = self.patch_size
            hidden_states = ops.reshape(hidden_states, [1, p_num_frames,
                                        p_height, p_width, pt, ph, pw, self.out_channels])
            hidden_states = ops.transpose(hidden_states, [0, 7, 1, 4, 2, 5, 3, 6])
            output = ops.reshape(hidden_states, [1, self.out_channels, num_frames, height, width])

            self.g.output_tensors = TensorList([output])

        return self.g


@register_update_config(['wantransformer3d'])
def zimagetransformer_update_config(config):
    model_safetensors_index_json = 'diffusion_pytorch_model.safetensors.index.json'
    model_path = config.model_path
    index_json_path = os.path.join(model_path, model_safetensors_index_json)
    if not os.path.exists(index_json_path):
        WARN(
            f"zimage_transformer's {model_safetensors_index_json} does not exists, will use default filename = model.safetensors.index.json")
    else:
        config.update({'model_safetensors_index_json': model_safetensors_index_json})
