from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np
import torch
import os
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.core import Graph, Node, Dtype, Tensor, Expr
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder.logger import INFO, DEBUG, WARN, ERROR
from safetensors import safe_open
from ...basemodel import *
from ...config import PretrainedConfig
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data, convert_expr


def _parse_str_to_int(str_value):
    if isinstance(str_value, str):
        return [int(x.strip()) for x in str_value.replace(' ', '').split(',')]
    else:
        return str_value


def _get_vision_grid_thw(config):
    dimg_h, dimg_w, dimg_num = 336, 448, 1

    img_h = config.get('image_height', dimg_h)
    img_w = config.get('image_width', dimg_w)
    img_num = config.get('image_num', dimg_num)
    img_t = config.get('spatial_merge_size', 1)//config.get('temporal_patch_size', 1)
    patch_size = config.get('patch_size', 14)
    img_h = _parse_str_to_int(img_h)
    img_w = _parse_str_to_int(img_w)
    if isinstance(img_h, list) and isinstance(img_w, list):
        if len(img_h) != len(img_w):
            ERROR(f"the number of image_height(={img_h}) and image_width(={img_w}) should be equal")
        grid_thw = [[img_t, int(round(img_h[i]/patch_size)), int(round(img_w[i]/patch_size))]
                    for i in range(len(img_h))]
    elif isinstance(img_h, list) or isinstance(img_w, list):
        ERROR(f"image_height(={img_h}) and image_width(={img_w}) should be both list or not")
        grid_thw = []
    else:
        # [1, 26, 112] * 9
        grid_thw = [[img_t, int(round(img_h/patch_size)), int(round(img_w/patch_size))]] * img_num
    return torch.tensor(grid_thw)


_get_grid_thw = _get_vision_grid_thw


@register_update_config(['qwen2.5thinker_vision'])
def qwen2_5_omni_vision_update_config(config):
    model_type = config.get('model_type', {})
    thinker_config = config.get('thinker_config', {})
    if len(thinker_config) == 0:
        return
    vision_config = thinker_config.get('vision_config', {})
    if len(vision_config) == 0:
        return
    text_config = thinker_config.get('text_config', {})
    _local_text_config = {
        'vocab_size': text_config.get('vocab_size', ),
        'rms_norm_eps': text_config.get('rms_norm_eps', 1.0e-6),
        'rope_theta': text_config.get('rope_theta', 1000000.0),
        'max_position_embeddings': text_config.get('max_position_embeddings', 32768),
    }
    config.update(vision_config)
    config.update(_local_text_config)
    config.pop('vision_config')
    config.update({'model_type': model_type})
    config.update({'num_hidden_layers': config.get('depth')})
    config.update({'num_attention_heads': config.get('num_heads')})


@register_causallm(['qwen2_5_omni_vision_embedding'])
class Qwen2_5omniVisionEmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights[f'thinker.model.embed_tokens.weight'].float()


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.embed_dim
        kernel_size = [self.patch_size, self.patch_size, self.temporal_patch_size]
        strides = [self.temporal_patch_size, self.patch_size, self.patch_size]
        out_channels = config.hidden_size

        # self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernels=kernel_size, strides=strides)
        conv_weight = weights.get(f'thinker.visual.patch_embed.proj.weight')
        # update_tensor_data(self.proj.weights,
        #                    conv3d_w,
        #                    f"patch_embed.proj.weight")

        fc_in_channels = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        self.proj = nn.Linear(fc_in_channels, out_channels)
        data = conv_weight['data'].reshape([out_channels, fc_in_channels])
        conv_weight['data'] = data.contiguous()
        update_tensor_data(self.proj.weights, conv_weight, 'thinker.visual.patch_embed.proj.weight')

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # target_dtype = self.proj.weight.dtype
        # hidden_states = ops.reshape(hidden_states,
        #     [-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size])

        hidden_states = self.proj(hidden_states)

        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, config, theta=10000.0, name=''):
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.base = theta
        head_dim = config.hidden_size // config.num_heads
        dim = head_dim // 2

        # [1, 26, 112] * 9
        grid_thw = _get_grid_thw(config)
        pos_ids, max_grid_size = self.rot_pos_emb(grid_thw)
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        t = torch.arange(max_grid_size, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        self.rotary_pos_emb = freqs[pos_ids].flatten(1)

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        return pos_ids, max_grid_size

    def forward(self, seq_len, window_index):
        rotary_pos_emb = self.rotary_pos_emb.reshape(
            [seq_len//self.spatial_merge_unit, self.spatial_merge_unit, self.rotary_pos_emb.shape[-1]])
        rotary_pos_emb = rotary_pos_emb.to(window_index.device)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_cached = emb.cos().to(torch.get_default_dtype())
        sin_cached = emb.sin().to(torch.get_default_dtype())

        sin_out = ops.constant(Tensor(sin_cached.unsqueeze(0).unsqueeze(1).cpu().numpy()))
        cos_out = ops.constant(Tensor(cos_cached.unsqueeze(0).unsqueeze(1).cpu().numpy()))
        return cos_out, sin_out


class Qwen2_5ThinkerVisionMLP(nn.Module):
    def __init__(self, config, weights, layer_id, prefix='thinker.visual.blocks'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size)

        self.act_fn = nn.ACT2FN[config.get('hidden_act', 'silu')]

        '''
        vision: prefix='thinker.visual.blocks'

        '''
        gate_proj_w = weights.get(f'{prefix}.{layer_id}.mlp.gate_proj.weight')
        gate_proj_b = weights.get(f'{prefix}.{layer_id}.mlp.gate_proj.bias')
        up_proj_w = weights.get(f'{prefix}.{layer_id}.mlp.up_proj.weight')
        up_proj_b = weights.get(f'{prefix}.{layer_id}.mlp.up_proj.bias')
        down_proj_w = weights.get(f'{prefix}.{layer_id}.mlp.down_proj.weight')
        down_proj_b = weights.get(f'{prefix}.{layer_id}.mlp.down_proj.bias')
        update_tensor_data(self.gate_proj.weights, gate_proj_w, name=f'{prefix}.{layer_id}.mlp.gate_proj.weight')
        update_tensor_data(self.gate_proj.biases, gate_proj_b, name=f'{prefix}.{layer_id}.mlp.gate_proj.bias')
        update_tensor_data(self.up_proj.weights, up_proj_w, name=f'{prefix}.{layer_id}.mlp.up_proj.weight')
        update_tensor_data(self.up_proj.biases, up_proj_b, name=f'{prefix}.{layer_id}.mlp.up_proj.bias')
        update_tensor_data(self.down_proj.weights, down_proj_w, name=f'{prefix}.{layer_id}.mlp.down_proj.weight')
        update_tensor_data(self.down_proj.biases, down_proj_b, name=f'{prefix}.{layer_id}.mlp.down_proj.bias')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5ThinkerVisionMerger(nn.Module):

    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.ln_q = nn.RMSNorm(config.embed_dim, eps=1e-6)
        update_tensor_data(self.ln_q.weights,
                           weights.get(f'thinker.visual.merger.ln_q.weight'),
                           f"{prefix}thinker.visual.merger.ln_q.weight")
        self.hidden_size = config.embed_dim * (config.spatial_merge_size**2)

        self.mlp0 = nn.Linear(self.hidden_size, self.hidden_size)
        self.mlp2 = nn.Linear(self.hidden_size, config.out_hidden_size)

        self.act_fn = nn.ACT2FN['gelu']

        gate_proj_w = weights.get(f'thinker.visual.merger.mlp.0.weight')
        o_proj_w = weights.get(f'thinker.visual.merger.mlp.2.weight')
        gate_proj_bias = weights.get(f'thinker.visual.merger.mlp.0.bias')
        o_proj_bias = weights.get(f'thinker.visual.merger.mlp.2.bias')

        update_tensor_data(self.mlp0.weights, gate_proj_w, name=f'{prefix}thinker.visual.merger.mlp.0.weight')
        update_tensor_data(self.mlp2.weights, o_proj_w, name=f'{prefix}thinker.visual.merger.mlp.2.weight')
        update_tensor_data(self.mlp0.biases, gate_proj_bias, name=f'{prefix}thinker.visual.merger.mlp.0.bias')
        update_tensor_data(self.mlp2.biases, o_proj_bias, name=f'{prefix}thinker.visual.merger.mlp.2.bias')

    def forward(self, x):
        y = self.ln_q(x).reshape([-1, self.hidden_size])
        hidden_states = self.mlp2(self.act_fn(self.mlp0(y)))
        return hidden_states


class Qwen2_5ThinkerVisionAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size//self.num_heads  # config.thinker_config['text_config']['head_dim']
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings if not hasattr(
            self.config, 'max_len') or self.config.max_len is None else self.config.max_len
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = 0
        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
        #           f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        q_w = weights.get(f'{prefix}.{layer_id}.attn.q.weight')
        q_b = weights.get(f'{prefix}.{layer_id}.attn.q.bias')
        update_tensor_data(self.q.weights, q_w, f"{prefix}layers.{layer_id}.attn.q.weight")
        update_tensor_data(self.q.biases, q_b, f"{prefix}layers.{layer_id}.attn.q.bias")

        k_w = weights.get(f'{prefix}.{layer_id}.attn.k.weight')
        k_b = weights.get(f'{prefix}.{layer_id}.attn.k.bias')
        update_tensor_data(self.k.weights, k_w, f"{prefix}layers.{layer_id}.attn.k.weight")
        update_tensor_data(self.k.biases, k_b, f"{prefix}layers.{layer_id}.attn.k.bias")

        v_w = weights.get(f'{prefix}.{layer_id}.attn.v.weight')
        v_b = weights.get(f'{prefix}.{layer_id}.attn.v.bias')
        update_tensor_data(self.v.weights, v_w, f"{prefix}layers.{layer_id}.attn.v.weight")
        update_tensor_data(self.v.biases, v_b, f"{prefix}layers.{layer_id}.attn.v.bias")

        o_proj_w = weights.get(f'{prefix}.{layer_id}.attn.proj.weight')
        o_proj_b = weights.get(f'{prefix}.{layer_id}.attn.proj.bias')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.attn.o_proj.weight")
        update_tensor_data(self.o_proj.biases, o_proj_b, f"{prefix}layers.{layer_id}.attn.o_proj.bias")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        # self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.attn"

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, cos_emb, sin_emb):
        _, hidden_size = hidden_states.shape
        seq_len, _ = hidden_states.shape.symbol
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        seq_len_symb, hd_size_symb = q.shape.symbol
        q = q.reshape([1, seq_len_symb, self.num_heads, self.head_dim])
        k = k.reshape([1, seq_len_symb, self.num_heads, self.head_dim])
        v = v.reshape([1, seq_len_symb, self.num_heads, self.head_dim])

        q = q.transpose([0, 2, 1, 3])  #lift transpose
        k = k.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])

        # cos_emb.shape = [1, seq_len, head_dim]
        # cos_emb, sin_emb = cos_emb.unsqueeze(0), sin_emb.unsqueeze(0)  # [1, seq_len, 1, head_dim]

        q_symb0, q_symb1, q_symb2, q_symb3 = q.shape.symbol
        q = q.reshape([q_symb1*q_symb0, 1, q_symb2, q_symb3])  #reshape for matmul in advance 
        k = k.reshape([q_symb1*q_symb0, 1, q_symb2, q_symb3])
        v = v.reshape([q_symb1*q_symb0, 1, q_symb2, q_symb3])

        crop0 = nn.Crop([[0, q_symb1*q_symb0], [0, 1], [0, q_symb2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, q_symb1*q_symb0], [0, 1], [0, q_symb2], [0, self.head_dim//2]])

        # query
        q_crop_out0 = crop0(q)
        q_crop_out1 = crop1(q)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        k_symb0, k_symb1, k_symb2, _ = k.shape.symbol
        crop0 = nn.Crop([[0, k_symb0], [0, k_symb1], [0, k_symb2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, k_symb0], [0, k_symb1], [0, k_symb2], [0, self.head_dim//2]])
        # key
        k_crop_out0 = crop0(k)
        k_crop_out1 = crop1(k)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        q_cosine = q * cos_emb
        k_cosine = k * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine  # k_add_out.shape=[seq_len, num_heads, head_dim]

        kq_matmul = self.kq_matmul(q_add_out, k_add_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        # attention_mask = attention_mask.tile(kq_matmul.shape[1], 1, 1, 1)
        softmax_masked = self.softmax(kq_matmul + attention_mask)  # softmax_mask.shape=[1, num_heads, seqlen, seq_len]
        kqv_out = self.kqv_matmul(softmax_masked, v)  # v.shape=[1, num_heads, seq_len, head_dim]
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        kqv_out = kqv_out.transpose([1,2,0,3])  
        kqv_out = kqv_out.reshape([seq_len_symb, hd_size_symb])
        o_t = self.o_proj(kqv_out)

        return o_t


class Qwen2_5ThinkerVisionBlock(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps
        self.norm1 = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        update_tensor_data(self.norm1.weights,
                           weights.get(f'{prefix}.{layer_id}.norm1.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = Qwen2_5ThinkerVisionAttention(config, weights, layer_id, prefix)

        self.norm2 = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        update_tensor_data(self.norm2.weights,
                           weights.get(f'{prefix}.{layer_id}.norm2.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = Qwen2_5ThinkerVisionMLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, cos_emb, sin_emb):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       cos_emb,
                                       sin_emb
                                       )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2_5ThinkerVisionModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.window_size = config.window_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.patch_size = config.patch_size
        self.vocab_size = config.vocab_size
        self.prefix = prefix

        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(config)
        self.patch_emb = Qwen2_5_VisionPatchEmbed(config, weights)
        self.blocks = [Qwen2_5ThinkerVisionBlock(config, weights, idx, prefix)
                       for idx in range(config.num_hidden_layers)]
        self.merger = Qwen2_5ThinkerVisionMerger(config, weights)

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, seq_len, patch_embed_size, position_ids=None, cu_seqlens=None, window_index=None, cu_window_seqlens=None, grid_thw=None):
        """
        Args:
            hidden_states (`Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
        """
        if grid_thw is None:
            grid_thw = _get_grid_thw(self.config)
        if window_index is None or cu_window_seqlens is None:
            window_index_data, cu_window_seqlens = self.get_window_index(grid_thw)
            window_index = ops.constant(Tensor(window_index_data.cpu().numpy().astype(np.int32)))
            # cu_window_seqlens =ops.constant(Tensor(cu_window_seqlens.cpu().numpy().astype(np.int32)))
            cu_window_seqlens = torch.tensor(cu_window_seqlens).to(window_index_data.device)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        image = ops.input([seq_len, patch_embed_size], Dtype.FP32,
                          name='inputs_images', exprs=['seq_len', patch_embed_size])

        x = self.patch_emb(image)  # .reshape([seq_len_symb, self.config.hidden_size])

        seq_len, _ = x.shape
        seq_len_symb, hd_size_symb = x.shape.symbol
        hidden_states = x.reshape([seq_len_symb.floordiv(Expr(self.spatial_merge_unit)),
                                  self.spatial_merge_unit, hd_size_symb])
        gathered_hs = ops.gather(hidden_states, window_index, axis=0)

        gathered_hs.op.params['ds_output_shape'] = [list(hidden_states.shape.symbol)]
        hidden_states = gathered_hs.reshape(seq_len_symb, hd_size_symb)

        cos_emb, sin_emb = self.rotary_pos_emb(seq_len, window_index_data)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        full_attention_mask = torch.full(
            [1, seq_len, seq_len], -10000, device=hidden_states.betensor.device, dtype=hidden_states.betensor.dtype)

        for i in range(1, len(cu_seqlens)):
            full_attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0
        window_attention_mask = torch.full(
            [1, seq_len, seq_len], -10000, device=hidden_states.betensor.device, dtype=hidden_states.betensor.dtype)
        for i in range(1, len(cu_window_seqlens)):
            window_attention_mask[..., cu_window_seqlens[i - 1]: cu_window_seqlens[i],
                                  cu_window_seqlens[i - 1]: cu_window_seqlens[i]] = 0

        full_attn_mask = ops.constant(Tensor(full_attention_mask.unsqueeze(0).cpu().numpy()))
        window_attn_mask = ops.constant(Tensor(window_attention_mask.unsqueeze(0).cpu().numpy()))

        for blk_idx, blk in enumerate(self.blocks):
            INFO(f"\t build the {blk_idx} vision Decoder block.")
            attn_mask = full_attn_mask if blk_idx in self.fullatt_block_indexes else window_attn_mask
            hidden_states = blk(hidden_states, attn_mask, cos_emb, sin_emb)
        hidden_states = self.merger(hidden_states)
        # _, reverse_indices = ops.sort(window_index)
        # hidden_states = hidden_states[reverse_indices, :]
        return hidden_states, window_index


@register_causallm(['qwen2.5thinker_vision'])
class Qwen2_5OmniThinkerVision(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.visual = Qwen2_5ThinkerVisionModel(self.config, weights, 'thinker.visual.blocks')
        # self.merger = Qwen2_5ThinkerVisionMerger(config, weights)
        self.aipu_g_visual = None

    def _get_seq_len(self):
        # [1, 26, 112] * 9
        grid_thw = _get_grid_thw(self.config)
        seq_len = grid_thw.prod(-1).sum().item()
        return seq_len

    def forward(self):
        seq_len = self._get_seq_len()
        patch_size = self.config.patch_size
        local_hidden_size = patch_size * patch_size * 3 * 2
        output_tensors = []
        with Graph('qwen-vit') as self.aipu_g_visual:
            hidden_states, windows_index = self.visual(seq_len, local_hidden_size)
            # hidden_states = self.merger(hidden_states)
        output_tensors.append(hidden_states)
        output_tensors.append(windows_index)

        self.aipu_g_visual.output_tensors = TensorList(output_tensors)
        return self.aipu_g_visual
