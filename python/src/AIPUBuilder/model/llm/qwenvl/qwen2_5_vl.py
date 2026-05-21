from typing import Callable
import math
import torch
import numpy as np
from AIPUBuilder.core import *
from AIPUBuilder.core import Graph, Node, Dtype, Tensor, Expr
from AIPUBuilder.logger import INFO, ERROR
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder.logger import WARN
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


def _parse_str_to_int(str_value):
    if isinstance(str_value, str):
        return [int(x.strip()) for x in str_value.replace(' ', '').split(',')]
    else:
        return str_value


def _get_vision_grid_thw(config):
    dimg_h, dimg_w, dimg_num = 364, 448, 1

    img_h = config.get('image_height', dimg_h)
    img_w = config.get('image_width', dimg_w)
    img_num = config.get('image_num', dimg_num)
    img_t = config.get('image_temporal', 1)
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


@register_update_config(['qwen2_5_vl_vision'])
def qwen2_5_vl_vision_update_config(config):
    vision_config = config.get('vision_config', {})
    if len(vision_config) == 0:
        return
    _local_vision_config = {
        'depth': vision_config.get('depth', config.get('depth', 32)),
        'hidden_size': vision_config.get('hidden_size', 3584),
        'intermediate_size': vision_config.get('intermediate_size', 3420),
        'hidden_act': vision_config.get('hidden_act', 'silu'),
        'num_heads': vision_config.get('num_heads', 16),
        'in_channels': vision_config.get('in_chans', 3),
        'patch_size': vision_config.get('patch_size', 14),
        'spatial_merge_size': vision_config.get('spatial_merge_size', 2),
        'temporal_patch_size': vision_config.get('temporal_patch_size', 2),
        'tokens_per_second': vision_config.get('tokens_per_second', 4),
        'window_size': vision_config.get('window_size', 112),
        'out_hidden_size': vision_config.get('out_hidden_size', 3584),
        'fullatt_block_indexes': vision_config.get('fullatt_block_indexes', [7, 15, 23, 31]),
        'hf_quantization_config': vision_config.get('quantization_config', None),
        'image_height': config.get('image_height', 364),
        'image_width': config.get('image_width', 448),
        'image_temporal': config.get('image_temporal', 1),
        'image_num': config.get('image_num', 1),
    }

    config.update(_local_vision_config)
    config.pop('vision_config')

    config.update({'num_hidden_layers': config.get('depth')})
    config.update({'num_attention_heads': config.get('num_heads')})


class QwenVLVisionRotaryEmbedding(nn.Module):
    def __init__(self, config, theta=10000.0, name=''):
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.base = theta
        head_dim = config.hidden_size // config.num_heads
        dim = head_dim // 2

        # [1, 26, 112] * 9
        grid_thw = _get_grid_thw(config)
        # pos_ids.shape = [9*832, 2]
        pos_ids, max_grid_size = self.rot_pos_emb(grid_thw)

        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

        t = torch.arange(max_grid_size, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # print(f"freqs.shape={freqs.shape}")  # [112, 20]
        self.rotary_pos_emb = freqs[pos_ids].flatten(1)
        # print(f"rotary_pos_emb.shape = {self.rotary_pos_emb.shape}")  # [832, 2*20]

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
        # rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        # rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        # return rotary_pos_emb

    def forward(self, seq_len, window_index):
        rotary_pos_emb = self.rotary_pos_emb.reshape(
            [seq_len//self.spatial_merge_unit, self.spatial_merge_unit, self.rotary_pos_emb.shape[-1]])
        rotary_pos_emb = rotary_pos_emb.to(window_index.device)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape([seq_len, -1])
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos_cached = emb.cos().to(torch.get_default_dtype())
        sin_cached = emb.sin().to(torch.get_default_dtype())

        sin_out = ops.constant(Tensor(sin_cached.unsqueeze(0).cpu().numpy()))
        cos_out = ops.constant(Tensor(cos_cached.unsqueeze(0).cpu().numpy()))
        return cos_out, sin_out


class VisionPatchMerger(nn.Module):

    def __init__(self, config, weights):
        self.config = config
        dim = config.out_hidden_size
        context_dim = config.hidden_size
        spatial_merge_size = config.spatial_merge_size

        self.hidden_size = context_dim * (spatial_merge_size ** 2)
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
        ln_q_w_name = 'visual.merger.ln_q.weight'
        update_tensor_data(self.ln_q.weights, weights.get(ln_q_w_name), ln_q_w_name)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.gelu_fn = nn.ACT2FN['gelu']
        self.fc2 = nn.Linear(self.hidden_size, dim)

        fc1_w_name = 'visual.merger.mlp.0.weight'
        fc1_b_name = 'visual.merger.mlp.0.bias'
        update_tensor_data(self.fc1.weights, weights.get(fc1_w_name), fc1_w_name)
        update_tensor_data(self.fc1.biases, weights.get(fc1_b_name), fc1_b_name)

        fc1_w_name = 'visual.merger.mlp.2.weight'
        fc1_b_name = 'visual.merger.mlp.2.bias'
        update_tensor_data(self.fc2.weights, weights.get(fc1_w_name), fc1_w_name)
        update_tensor_data(self.fc2.biases, weights.get(fc1_b_name), fc1_b_name)

    def forward(self, x):
        y = self.ln_q(x).reshape([-1, self.hidden_size])
        y = self.fc2(self.gelu_fn(self.fc1(y)))
        return y


class QwenVLMLP(nn.Module):
    def __init__(self, config, layer_id, weights, prefix='model.layers'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, self.hidden_size)

        self.act_fn = nn.ACT2FN[config.get('hidden_act', 'silu')]

        '''
        vision: prefix='visual.blocks'
        llm: prefix='model.layers'
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


class VisionAttention(nn.Module):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, layer_id, weights, prefix=''):
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = dim // self.num_heads

        qkv_w_name = f'visual.blocks.{layer_id}.attn.qkv.weight'
        qkv_b_name = f'visual.blocks.{layer_id}.attn.qkv.bias'
        if config.get('qkv_fc', False):
            self.qkv = nn.Linear(dim, dim * 3)
            update_tensor_data(self.qkv.weights, weights.get(qkv_w_name),
                               f"{prefix}visual.blocks.{layer_id}.attn.qkv.weight")
            update_tensor_data(self.qkv.biases, weights.get(qkv_b_name),
                               f"{prefix}visual.blocks.{layer_id}.attn.qkv.bias")
        else:
            self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim)
            self.k_proj = nn.Linear(dim, self.num_heads * self.head_dim)
            self.v_proj = nn.Linear(dim, self.num_heads * self.head_dim)
            w = weights[qkv_w_name]
            qw, kw, vw = w.split([dim, dim, dim], dim=0)
            update_tensor_data(self.q_proj.weights, qw, f"{prefix}visual.blocks.{layer_id}.attn.q.weight")
            update_tensor_data(self.k_proj.weights, kw, f"{prefix}visual.blocks.{layer_id}.attn.k.weight")
            update_tensor_data(self.v_proj.weights, vw, f"{prefix}visual.blocks.{layer_id}.attn.v.weight")
            b = weights[qkv_b_name]
            qb, kb, vb = b.split([dim, dim, dim], dim=0)
            update_tensor_data(self.q_proj.biases, qb, f"{prefix}visual.blocks.{layer_id}.attn.q.bias")
            update_tensor_data(self.k_proj.biases, kb, f"{prefix}visual.blocks.{layer_id}.attn.k.bias")
            update_tensor_data(self.v_proj.biases, vb, f"{prefix}visual.blocks.{layer_id}.attn.v.bias")

        self.proj = nn.Linear(dim, dim)
        proj_w_name = f'visual.blocks.{layer_id}.attn.proj.weight'
        update_tensor_data(self.proj.weights, weights.get(proj_w_name),
                           f"{prefix}visual.blocks.{layer_id}.attn.proj.weight")
        proj_b_name = f'visual.blocks.{layer_id}.attn.proj.bias'
        update_tensor_data(self.proj.biases, weights.get(proj_b_name),
                           f"{prefix}visual.blocks.{layer_id}.attn.proj.bias")

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, query, attention_mask, cos_emb, sin_emb):
        _, hidden_size = query.shape
        seq_len, _ = query.shape.symbol
        # qkv = self.qkv(query).reshape([seq_len, 3, self.num_heads, -1]).transpose([1, 0, 2, 3])
        # q, k, v = ops.split(qkv, 3, axis=0) #q.shape=[seq_len, num_heads, head_dim]
        # qkv = self.qkv(query).reshape([seq_len, 3, self.num_heads, -1])
        # q, k, v = ops.split(qkv, 3, axis=1) #q.shape=[seq_len, 1, num_heads, head_dim]

        if self.config.get('qkv_fc', False):
            qkv = self.qkv(query).reshape([seq_len, 3, self.num_heads, -1]).transpose([1, 0, 2, 3])
            q, k, v = ops.split(qkv, 3, axis=0)  # q.shape=[1, seq_len, num_heads, head_dim]
        else:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        seq_len_symb, hd_size_symb = q.shape.symbol
        q = q.reshape([1, seq_len_symb, self.num_heads, self.head_dim])
        k = k.reshape([1, seq_len_symb, self.num_heads, self.head_dim])
        v = v.reshape([1, seq_len_symb, self.num_heads, self.head_dim])
        # cos_emb.shape = [1, seq_len, head_dim]
        cos_emb, sin_emb = cos_emb.unsqueeze(2), sin_emb.unsqueeze(2)  # [1, seq_len, 1, head_dim]

        q_symb0, q_symb1, q_symb2, _ = q.shape.symbol
        crop0 = nn.Crop([[0, q_symb0], [0, q_symb1], [0, q_symb2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, q_symb0], [0, q_symb1], [0, q_symb2], [0, self.head_dim//2]])

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

        q_add_out = q_add_out.transpose([0, 2, 1, 3])
        k_add_out = k_add_out.transpose([0, 2, 1, 3])
        v = v.transpose([0, 2, 1, 3])

        kq_matmul = self.kq_matmul(q_add_out, k_add_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        attention_mask = attention_mask.unsqueeze(1).tile(1, kq_matmul.shape[1], 1, 1)
        softmax_masked = self.softmax(kq_matmul + attention_mask)  # softmax_mask.shape=[1, num_heads, seqlen, seq_len]
        kqv_out = self.kqv_matmul(softmax_masked, v)  # v.shape=[1, num_heads, seq_len, head_dim]
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        kqv_out = kqv_out.transpose([0, 2, 1, 3])  # [1, seq_len, num_heads, head_dim]
        kqv_out = kqv_out.reshape([seq_len_symb, hd_size_symb])
        o_t = self.proj(kqv_out)

        return o_t


class VisionBlock(nn.Module):
    def __init__(self, config, layer_id, weights, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.weights = weights

        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        update_tensor_data(self.norm1.weights,
                           self.weights.get(f'visual.blocks.{layer_id}.norm1.weight'),
                           f"{prefix}blocks.{layer_id}.norm1.weight")
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-6)
        update_tensor_data(self.norm2.weights,
                           self.weights.get(f'visual.blocks.{layer_id}.norm2.weight'),
                           f"{prefix}blocks.{layer_id}.norm2.weight")

        self.attn = VisionAttention(config, layer_id, weights, prefix=prefix)
        self.mlp = QwenVLMLP(config, layer_id, weights, prefix='visual.blocks')

    def forward(self, hidden_states, attention_mask, cos_emb, sin_emb):
        residual = hidden_states
        norm1 = self.norm1(hidden_states)
        attn = self.attn(norm1, attention_mask, cos_emb, sin_emb)
        residual_attn = residual + attn
        mlp_norm = self.mlp(self.norm2(residual_attn))
        out = residual_attn + mlp_norm
        return out


class VisionTransformer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.fullatt_block_indexes = self.config.fullatt_block_indexes
        self.spatial_merge_size = self.config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.window_size = config.window_size
        self.patch_size = self.config.patch_size
        self.head_dim = self.config.hidden_size // self.config.num_heads

        self.temporal_patch_size = self.config.temporal_patch_size
        self.in_channels = self.config.in_channels
        out_channels = self.config.hidden_size

        conv_weight_name = 'visual.patch_embed.proj.weight'
        self.conv_weight = weights.get(conv_weight_name)
        self.using_3d = self.config.get('using_3d', True)
        if self.using_3d:
            kernel_size = [self.patch_size, self.patch_size, self.temporal_patch_size]
            strides = [self.temporal_patch_size, self.patch_size, self.patch_size]

            self.conv = nn.Conv3d(in_channels=self.in_channels,
                                  out_channels=out_channels,
                                  kernels=kernel_size,
                                  strides=strides)
            self.conv_weight['data'] = self.conv_weight['data'].permute(0, 3, 4, 2, 1).contiguous()
            update_tensor_data(self.conv.weights, self.conv_weight, conv_weight_name)
        else:
            fc_in_channels = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
            self.fc = nn.Linear(fc_in_channels, out_channels)
            data = self.conv_weight['data'].reshape([out_channels, fc_in_channels])
            self.conv_weight['data'] = data.contiguous()
            update_tensor_data(self.fc.weights, self.conv_weight, conv_weight_name)

        layer_nums = config.get('depth')
        self.blocks = [VisionBlock(config, layer_id, weights) for layer_id in gen_layer_ids(layer_nums)]

        self.merger = VisionPatchMerger(config, weights)

        # self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.rope = QwenVLVisionRotaryEmbedding(self.config)

    def get_window_index(self, grid_thw=None):
        config = self.config
        if grid_thw is None:
            grid_thw = _get_grid_thw(config)

        window_index = []
        cu_window_seqlens = [0]
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
        if window_index is None or cu_window_seqlens is None:
            window_index_data, cu_window_seqlens = self.get_window_index()
            window_index = ops.constant(Tensor(window_index_data.cpu().numpy().astype(np.int32)))
            # cu_window_seqlens =ops.constant(Tensor(cu_window_seqlens.cpu().numpy().astype(np.int32)))
            cu_window_seqlens = torch.tensor(cu_window_seqlens).to(window_index_data.device)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        if grid_thw is None:
            grid_thw = _get_grid_thw(self.config)

        skip_patch_embed = self.config.get('skip_patch_embed', False)
        skip_merger = self.config.get('skip_merger', False)
        if not skip_patch_embed:
            image = ops.input([seq_len, patch_embed_size], Dtype.FP32,
                              name='inputs_images', exprs=['seq_len', patch_embed_size])

            if self.using_3d:
                seq_len_symb, _ = image.shape.symbol
                hidden_states = image.reshape([seq_len_symb, self.temporal_patch_size, self.patch_size,
                                              self.patch_size, self.in_channels])
                x = self.conv(hidden_states).reshape([seq_len_symb, self.config.hidden_size])
            else:
                hidden_states = image
                x = self.fc(hidden_states)
        else:
            x = ops.input([seq_len, self.config.hidden_size], Dtype.FP32,
                          name='patch_embed', exprs=['seq_len', self.config.hidden_size])
        seq_len, _ = x.shape
        seq_len_symb, hd_size_symb = x.shape.symbol
        hidden_states = x.reshape([seq_len_symb.floordiv(Expr(self.spatial_merge_unit)),
                                  self.spatial_merge_unit, hd_size_symb])
        gathered_hs = ops.gather(hidden_states, window_index, axis=0)
        # CP-26887: set gather's ds_output_shape manully to avoid gather using static shape
        gathered_hs.op.params['ds_output_shape'] = [list(hidden_states.shape.symbol)]
        hidden_states = gathered_hs.reshape(seq_len_symb, hd_size_symb)

        cos_emb, sin_emb = self.rope(seq_len, window_index_data)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        # cu_seqlens = ops.constant(Tensor(cu_seqlens.cpu().numpy().astype(np.int32)))

        full_attention_mask = torch.full(
            [1, seq_len, seq_len], torch.finfo(hidden_states.betensor.dtype).min, device=hidden_states.betensor.device, dtype=hidden_states.betensor.dtype)

        for i in range(1, len(cu_seqlens)):
            full_attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0
        window_attention_mask = torch.full(
            [1, seq_len, seq_len], torch.finfo(hidden_states.betensor.dtype).min, device=hidden_states.betensor.device, dtype=hidden_states.betensor.dtype)
        for i in range(1, len(cu_window_seqlens)):
            window_attention_mask[..., cu_window_seqlens[i - 1]: cu_window_seqlens[i],
                                  cu_window_seqlens[i - 1]: cu_window_seqlens[i]] = 0

        full_attn_mask = ops.constant(Tensor(full_attention_mask.cpu().numpy()))
        window_attn_mask = ops.constant(Tensor(window_attention_mask.cpu().numpy()))

        for blk_idx, blk in enumerate(self.blocks):
            INFO(f"\t build the {blk_idx} vision Decoder block.")
            attn_mask = full_attn_mask if blk_idx in self.fullatt_block_indexes else window_attn_mask
            hidden_states = blk(hidden_states, attn_mask, cos_emb, sin_emb)
        if not skip_merger:
            hidden_states = self.merger(hidden_states)
        return hidden_states


@register_causallm(['qwen2_5_vl_vision'])
class Qwen2_5_VLVision(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.visual = VisionTransformer(self.config, weights)
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

        with Graph('qwen-vit') as self.aipu_g_visual:
            self.visual(seq_len, local_hidden_size)
        return self.aipu_g_visual


class QwenVLMultiModalRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None, name=''):
        self.config = config
        self.name = name

        if config.get('rope_scaling', None) is not None and 'type' in config.rope_scaling:
            if config.rope_scaling['type'] != 'default':
                WARN(f"Unsupported rope scaling type: {config.rope_scaling['type']}, now only support default scaling")

        base = config.get('rope_theta', 1000000.0)
        partial_rotary_factor = config.get('partial_rotary_factor', 1.0)

        dim = config.get('head_dim', config.hidden_size // config.num_attention_heads)
        dim = int(dim * partial_rotary_factor)
        max_position_embeddings = config.seq_len + config.past_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        inv_freq_expand = inv_freq.reshape([1, 1, -1, 1]).expand(3, 1, -1, 1)  # [3, bs, head_dim//2, 1]
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.int64).type_as(inv_freq)
        t = t.reshape(1, 1, 1, -1).expand(3, 1, 1, -1)  # [3, bs, 1, max_seq_len]
        freqs = inv_freq_expand.float() @ t.float()
        freqs = freqs.transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.attention_scaling = config.get('attention_scaling', 1.0)
        self.cos_cached = (emb.cos()*self.attention_scaling).to(torch.get_default_dtype())
        self.sin_cached = (emb.sin()*self.attention_scaling).to(torch.get_default_dtype())

        # mrope_section = config.rope_scaling['mrope_section']
        # mrope_section = mrope_section * 2
        # unsqueeze_dim = 1
        # self.cos_cached = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
        # self.sin_cached = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    def forward(self, pos_ids):
        mrope_section = self.config.rope_scaling['mrope_section']
        mrope_section = mrope_section * 2
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        # pos_ids = pos_ids.unsqueeze(2)
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=2, batch_dims=2)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=2, batch_dims=2)
        sin_split = ops.split(gather_sin_out, mrope_section, axis=-1)
        cos_split = ops.split(gather_cos_out, mrope_section, axis=-1)
        sl = []
        for i, sin_s in enumerate(sin_split):
            s0, _, _, _ = sin_s.shape
            _, s1, s2, s3 = sin_s.shape.symbol
            sins = ops.crop(sin_s, crops=[[i % 3, i % 3+1], [0, s1], [0, s2], [0, s3]])
            sl.append(sins)
        sin_concat = ops.concat(sl, axis=-1).transpose([0, 2, 1, 3])
        cl = []
        for i, cos_s in enumerate(cos_split):
            s0, _, _, _ = cos_s.shape
            _, s1, s2, s3 = cos_s.shape.symbol
            coss = ops.crop(cos_s, crops=[[i % 3, i % 3+1], [0, s1], [0, s2], [0, s3]])
            cl.append(coss)
        cos_concat = ops.concat(cl, axis=-1).transpose([0, 2, 1, 3])
        return cos_concat, sin_concat


class QwenVLAttention(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.get('num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.get('head_dim', self.hidden_size // self.num_heads)
        self.max_position_embeddings = config.seq_len + config.past_len

        if (self.head_dim * self.num_heads) != self.hidden_size:
            WARN(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                 f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

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

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])
        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=-1)
        self.mrope_section = self.config.get('rope_scaling').get('mrope_section', [])
        if len(self.mrope_section) == 0:
            WARN(
                f"please check the mrope section in rope_scaling ={self.config.get('rope_scaling', None)} in cfg file, ")

    def forward(self, hidden_states, attention_mask, position_ids, position_emb, k_cache=None, v_cache=None, use_cache=False):
        bs_symb, seq_len_symb, hidden_size_symb = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symb * seq_len_symb, hidden_size_symb])

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        self.q_reshape = nn.Reshape([bs_symb, seq_len_symb, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs_symb, seq_len_symb, self.num_key_value_heads, self.head_dim])
        query = self.q_reshape(query)
        key = self.kv_reshape(key)
        value = self.kv_reshape(value)

        # query
        rtq_symbol0, rtq_symbol1, rtq_symbol2, _ = query.shape.symbol
        crop0 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [0, self.head_dim//2]])

        q_crop_out0 = crop0(query)
        q_crop_out1 = crop1(query)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        # key
        rtk_symbol0, rtk_symbol1, rtk_symbol2, _ = key.shape.symbol
        crop0 = nn.Crop([[0, rtk_symbol0], [0, rtk_symbol1], [0, rtk_symbol2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, rtk_symbol0], [0, rtk_symbol1], [0, rtk_symbol2], [0, self.head_dim//2]])

        k_crop_out0 = crop0(key)
        k_crop_out1 = crop1(key)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        cos_emb, sin_emb = position_emb
        q_cosine = query * cos_emb
        k_cosine = key * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb
        query = q_cosine + q_r_sine
        key = k_cosine + k_r_sine

        query = self.qkv_transpose(query)
        key = self.qkv_transpose(key)
        value = self.qkv_transpose(value)

        k_cache_out = k_cache(key, position_ids=position_ids, block_idx=self.layer_id,
                              name=f"past_key_value.{self.layer_id}.key")
        v_cache_out = v_cache(value, position_ids=position_ids,
                              block_idx=self.layer_id, name=f"past_key_value.{self.layer_id}.value")
        value = v_cache_out[0]
        key = k_cache_out[0]

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = key.shape.symbol
            key_reshape = key.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            key_tile = ops.tile(key_reshape, [1, 1, self.num_key_value_groups, 1, 1])
            key = key_tile.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        if False and use_cache:
            # [CP-19313]C=A*B=(B^T*A^T)^T, when B size >> A size, swap input can improve wtram reuse
            kq_matmul = nn.Transpose([0, 1, 3, 2])(self.kq_matmul(key, query))
        else:
            kq_matmul = self.kq_matmul(query, key)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        softmax_masked = self.softmax(kq_matmul + attention_mask)

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = value.shape.symbol
            value0 = value.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            value1 = ops.tile(value0, [1, 1, self.num_key_value_groups, 1, 1])  # [16,1, seq, 128]
            value = value1.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        attn_output = self.kqv_matmul(softmax_masked, value)
        attn_output.op.name = f"{self.matmul_name}_matmul_kqv"

        # attn_output = nn.Transpose([1, 2, 0, 3])(attn_output)
        attn_output = nn.Transpose([0, 2, 1, 3])(attn_output)
        context_layer = ops.reshape(attn_output, [bs_symb * seq_len_symb, self.hidden_size])
        attn_output = self.o_proj(context_layer)

        return attn_output


class QwenVLBlock(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_rmsnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.input_rmsnorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = QwenVLAttention(config, weights, layer_id, prefix)

        self.post_attn_rmsnorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_attn_rmsnorm.weights,
                           weights.get(f'model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = QwenVLMLP(config, layer_id, weights, prefix='model.layers')

    def forward(self, hidden_states, attention_mask, position_ids, position_emb, k_cache=None, v_cache=None, use_cache=False):

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        bs_symb, seq_len_symb, hidden_size_symb = hidden_states.shape.symbol
        layernorm_output = self.input_rmsnorm(hidden_states)
        attn_output = self.attention(
            layernorm_output,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_emb=position_emb,
            k_cache=k_cache,
            v_cache=v_cache,
            use_cache=use_cache,
        )
        residual = ops.reshape(residual, [bs_symb * seq_len_symb, hidden_size_symb])
        layernorm_input = attn_output + residual
        residual = layernorm_input
        layernorm_output = self.post_attn_rmsnorm(layernorm_input)
        if len(layernorm_output.shape) > 2:
            layernorm_output = layernorm_output.reshape(bs_symb * seq_len_symb, hidden_size_symb)
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output
        hidden_states = hidden_states.reshape([bs_symb, seq_len_symb, hidden_size_symb])

        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class QwenVLModel(nn.Module):
    def __init__(self, config, weights, prefix):
        # config.num_hidden_layers = 2
        self.prefix = prefix
        self.model_config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.seq_length = config.get('seq_length', config.seq_len)

        # self.rotary_emb = QwenVLRotaryEmbedding(dim, config)

        self.h = [QwenVLBlock(config, weights, idx, prefix) for idx in gen_layer_ids(config.num_hidden_layers)]

    def forward(self, inputs_embeds, attention_mask, position_ids, position_emb, past_key=None, past_value=None, use_cache=True):
        if use_cache and (past_key is None or past_value is None):
            ERROR(f"when use_kv_cache, please provide the kv inputs")

        hidden_states = inputs_embeds
        for i, block in enumerate(self.h):
            hidden_states, _, _ = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_emb=position_emb,
                k_cache=past_key,
                v_cache=past_value,
                use_cache=use_cache
            )

        return hidden_states


class QwenVLLmHead(nn.Module):
    def __init__(self, config, weights):
        # gen_subgraph register function depends on config to get 'enable_subgraph' flag from main()
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


@register_causallm(['qwen2_5_vl_llm'])
class Qwen2_5_VLCausalLVM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self.config.get("weights", self._load_state_dict())
        self.transformer = QwenVLModel(config, weights, prefix)
        self.lm_head = QwenVLLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")

        self.prefix = prefix

        self.rotary_emb = QwenVLMultiModalRotaryEmbedding(self.config, name=f'mrope')

    def forward(self):
        seq_len = self.config.get('seq_len')
        past_len = self.config.get('past_len')
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        with Graph('qwen2_5_vl') as self.aipu_g:
            # envn if we can get the fixed_seq_len, but still use the seq_len, which can be setted by user
            # fixed_seq_len = 25 + 1872
            hidden_states = ops.input([batch, seq_len, self.config.hidden_size],
                                      Dtype.FP32,
                                      name='inputs_embeds',
                                      exprs=['batch', 'seq_len', self.config.hidden_size])
            attention_mask = ops.input([batch, 1, seq_len, past_len + seq_len],
                                       Dtype.FP32,
                                       name='attention_mask',
                                       exprs=['batch', 1, 'seq_len', 'max_len'])
            position_ids = ops.input([3, batch, seq_len],
                                     Dtype.INT32,
                                     name='position_ids',
                                     exprs=[3, 'batch', 'seq_len'])

            cos_emb, sin_emb = self.rotary_emb(position_ids)
            position_emb = (cos_emb, sin_emb)

            kcache_module = nn.KVCacheModule(self.config, name='kcache')
            vcache_module = nn.KVCacheModule(self.config, name='vcache')

            hidden_states = self.transformer(hidden_states,
                                             attention_mask,
                                             position_ids,
                                             position_emb,
                                             past_key=kcache_module,
                                             past_value=vcache_module,
                                             use_cache=use_cache,)
            _ = self.lm_head(hidden_states)

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


@register_causallm(['qwen2_5_vl_llm_embedding'])
class qwen25vlllmembcausallm(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')
