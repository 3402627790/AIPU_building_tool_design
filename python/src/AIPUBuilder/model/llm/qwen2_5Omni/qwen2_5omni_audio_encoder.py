from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder.core import Graph, Node, Dtype, Tensor, Expr
from AIPUBuilder import ops
from AIPUBuilder import nn
from safetensors import safe_open
from ...basemodel import *
from ...config import PretrainedConfig
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data,convert_expr
def _parse_str_to_int(str_value):
    if isinstance(str_value, str):
        return [int(x.strip()) for x in str_value.replace(' ', '').split(',')]
    else:
        return str_value


def _get_vision_grid_thw(config):
    dimg_h, dimg_w, dimg_num = 336,448,1

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


@register_update_config(['qwen2.5thinker_audio'])
def qwen2_5_omni_audio_update_config(config):
    model_type = config.get('model_type', {})
    thinker_config = config.get('thinker_config', {})
    if len(thinker_config) == 0:
        return
    audio_config = thinker_config.get('audio_config', {})
    if len(audio_config) == 0:
        return
    text_config = thinker_config.get('text_config', {})
    _local_text_config = {
        'vocab_size': text_config.get('vocab_size', ),
        'rms_norm_eps': text_config.get('rms_norm_eps', 1.0e-6),
        'rope_theta': text_config.get('rope_theta', 1000000.0),
        'max_position_embeddings': text_config.get('max_position_embeddings', 32768),
    }
    config.update(audio_config)
    config.update(_local_text_config)
    config.pop('vision_config')
    config.update({'model_type': model_type})
    config.update({'num_hidden_layers': config.get('depth')})
    config.update({'num_attention_heads': config.get('num_heads')})


class Qwen2_5_AudioPatchEmbed(nn.Module):
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

class Qwen2_5_AudioRotaryEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()

        # self.dim = dim
        # self.num_heads = num_heads
        # self.max_position_embeddings = max_position_embeddings
        # self.base = base
        # self.name = name
        # self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
        #                        dtype=torch.int64).float().to(device) / self.dim))

        # self.max_seq_len_cached = max_position_embeddings
        # t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        # freqs = torch.outer(t, self.inv_freq)
        # # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        # self.cos_cached = emb.cos().to(torch.get_default_dtype())
        # self.sin_cached = emb.sin().to(torch.get_default_dtype())

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        t = torch.arange(length,  dtype=torch.int64).type_as(inv_timescales)
        scaled_time = torch.outer(t, inv_timescales)
        # scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.emb = (torch.cat((scaled_time.sin(), scaled_time.cos()), dim=-1).unsqueeze(0).unsqueeze(0))



    def forward(self, seqlen):
        # sin_out = ops.constant(Tensor(self.sin_cached))
        # cos_out = ops.constant(Tensor(self.cos_cached))
        # gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        # gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        # sin_new_shape = list(gather_sin_out.shape.symbol)[:1] + [1] + list(gather_sin_out.shape.symbol)[1:]
        # cos_new_shape = list(gather_cos_out.shape.symbol)[:1] + [1] + list(gather_cos_out.shape.symbol)[1:]
        # sin_out_reshape = gather_sin_out.reshape(sin_new_shape)
        # cos_out_reshape = gather_cos_out.reshape(cos_new_shape)

        # if self.name != '':
        #     sin_out.op.constants['weights'].name = f'{self.name}.rotary_emb.sin_weight'
        #     cos_out.op.constants['weights'].name = f'{self.name}.rotary_emb.cos_weight'
        #     sin_out.op.name = f"{self.name}.rotary_emb.sin_{sin_out.op.name}"
        #     cos_out.op.name = f"{self.name}.rotary_emb.cos_{cos_out.op.name}"
        #     gather_sin_out.name = f"{self.name}.{gather_sin_out.name}"
        #     gather_cos_out.name = f"{self.name}.{gather_cos_out.name}"
        emb = ops.constant(Tensor(self.emb))
        _,_,_,channel = emb.shape
        crop0 = nn.Crop([[0, 1], [0, 1], [0, seqlen], [0, channel]])
        return crop0(emb)


class Qwen2_5ThinkerAudioMLP(nn.Module):
    def __init__(self, config, weights, layer_id, prefix='thinker.audio_tower.layers'):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        intermediate_size = config.encoder_ffn_dim

        self.gate_proj = nn.Linear(self.hidden_size, intermediate_size)

        self.down_proj = nn.Linear(intermediate_size, self.hidden_size)

        self.act_fn = nn.ACT2FN['gelu']

        '''
        audio: prefix='thinker.audio_tower'

        '''
        gate_proj_w = weights.get(f'{prefix}.{layer_id}.fc1.weight')
        gate_proj_b = weights.get(f'{prefix}.{layer_id}.fc1.bias')

        down_proj_w = weights.get(f'{prefix}.{layer_id}.fc2.weight')
        down_proj_b = weights.get(f'{prefix}.{layer_id}.fc2.bias')
        update_tensor_data(self.gate_proj.weights, gate_proj_w, name=f'{prefix}.{layer_id}.fc1.weight')
        update_tensor_data(self.gate_proj.biases, gate_proj_b, name=f'{prefix}.{layer_id}.fc1.bias')

        update_tensor_data(self.down_proj.weights, down_proj_w, name=f'{prefix}.{layer_id}.fc2.weight')
        update_tensor_data(self.down_proj.biases, down_proj_b, name=f'{prefix}.{layer_id}.fc2.bias')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)))



class Qwen2_5ThinkerAudioMerger(nn.Module):

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

class Qwen2_5ThinkerAudioAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.hidden_size//self.num_heads #config.thinker_config['text_config']['head_dim']
        self.num_key_value_heads = self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # self.max_position_embeddings = config.max_position_embeddings if not hasattr(
        #     self.config, 'max_len') or self.config.max_len is None else self.config.max_len
        # self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = 0
        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
        #           f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        q_w = weights.get(f'{prefix}.{layer_id}.self_attn.q_proj.weight')
        q_b = weights.get(f'{prefix}.{layer_id}.self_attn.q_proj.bias')
        update_tensor_data(self.q.weights, q_w, f"{prefix}.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q.biases, q_b, f"{prefix}.{layer_id}.self_attn.q_proj.bias")

        k_w = weights.get(f'{prefix}.{layer_id}.self_attn.k_proj.weight')
        # k_b = weights.get(f'{prefix}.{layer_id}.self_attn.k_proj.bias')
        update_tensor_data(self.k.weights, k_w, f"{prefix}.{layer_id}.self_attn.k_proj.weight")
        # update_tensor_data(self.k.biases, k_b, f"{prefix}.{layer_id}.self_attn.k_proj.bias")

        v_w = weights.get(f'{prefix}.{layer_id}.self_attn.v_proj.weight')
        v_b = weights.get(f'{prefix}.{layer_id}.self_attn.v_proj.bias')
        update_tensor_data(self.v.weights, v_w, f"{prefix}.{layer_id}.self_attn.v_proj.weight")
        update_tensor_data(self.v.biases, v_b, f"{prefix}.{layer_id}.self_attn.v_proj.bias")

        o_proj_w = weights.get(f'{prefix}.{layer_id}.self_attn.out_proj.weight')
        o_proj_b = weights.get(f'{prefix}.{layer_id}.self_attn.out_proj.bias')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}.{layer_id}.attn.o_proj.weight")
        update_tensor_data(self.o_proj.biases, o_proj_b, f"{prefix}.{layer_id}.attn.o_proj.bias")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        # self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.attn"

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)


        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask):
        _, hidden_size = hidden_states.shape
        seq_len, _ = hidden_states.shape.symbol
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        seq_len_symb, hd_size_symb = q.shape.symbol
        query = ops.reshape(q,[1, seq_len_symb, self.num_heads, self.head_dim])
        key = ops.reshape(k,[1, seq_len_symb, self.num_heads, self.head_dim])
        value = ops.reshape(v,[1, seq_len_symb, self.num_heads, self.head_dim])
        q = self.qkv_transpose(query)
        k = self.qkv_transpose(key)
        v = self.qkv_transpose(value)

        kq_matmul = self.kq_matmul(q, k)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        # attention_mask = attention_mask.unsqueeze(1).tile(1, kq_matmul.shape[1], 1, 1)

        softmax_masked = self.softmax(kq_matmul + attention_mask)  # softmax_mask.shape=[1, num_heads, seqlen, seq_len]
        kqv_out = self.kqv_matmul(softmax_masked, v)  # v.shape=[1, num_heads, seq_len, head_dim]
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        kqv_out = kqv_out.transpose([0, 2, 1, 3])  # [1, seq_len, num_heads, head_dim]
        kqv_out = kqv_out.reshape([seq_len_symb, hd_size_symb])
        o_t = self.o_proj(kqv_out)

        return o_t
class Qwen2_5ThinkerAudioBlock(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.d_model
        rms_norm_eps = config.thinker_config['text_config']['rms_norm_eps']
        self.norm1 = nn.LayerNorm(hidden_size, eps=rms_norm_eps)
        update_tensor_data(self.norm1.weights,
                           weights.get(f'{prefix}.{layer_id}.self_attn_layer_norm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")
        update_tensor_data(self.norm1.biases,
                           weights.get(f'{prefix}.{layer_id}.self_attn_layer_norm.bias'),
                           f"{prefix}layers.{layer_id}.input_layernorm.bias")
        self.attention = Qwen2_5ThinkerAudioAttention(config, weights, layer_id, prefix)

        self.norm2 = nn.LayerNorm(hidden_size, eps=rms_norm_eps)
        update_tensor_data(self.norm2.weights,
                           weights.get(f'{prefix}.{layer_id}.final_layer_norm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")
        update_tensor_data(self.norm2.biases,
                           weights.get(f'{prefix}.{layer_id}.final_layer_norm.bias'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.bias")
        self.mlp = Qwen2_5ThinkerAudioMLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states,
                                             attention_mask

                                             )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen2_5ThinkerAudioModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        # self.fullatt_block_indexes = config.fullatt_block_indexes
        # self.spatial_merge_size = config.spatial_merge_size
        # self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        # self.window_size = config.window_size
        # self.temporal_patch_size = config.temporal_patch_size
        # self.in_channels = config.in_channels
        # self.patch_size = config.patch_size
        # self.vocab_size = config.vocab_size
        self.prefix = prefix
        #from transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv2d(self.num_mel_bins, embed_dim, kernels=[1,3], padding=[0,0,1,1])
        weights_conv = weights.get(f'thinker.audio_tower.conv1.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        update_tensor_data(self.conv1.weights,
                           weights_conv,
                           f"thinker.audio_tower.conv1.bias")
        update_tensor_data(self.conv1.biases,
                           weights.get(f'thinker.audio_tower.conv1.bias'),
                           f"thinker.audio_tower.conv1.bias")
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, kernels=[1,3], strides=[1,2], padding=[0,0,1,1])
        weights_conv = weights.get(f'thinker.audio_tower.conv2.weight')['data'].unsqueeze(2).permute(0,2,3,1)
        update_tensor_data(self.conv2.weights,
                           weights_conv,
                           f"thinker.audio_tower.conv2.bias")
        update_tensor_data(self.conv2.biases,
                           weights.get(f'thinker.audio_tower.conv2.bias'),
                           f"thinker.audio_tower.conv2.bias")
        self.positional_embedding = Qwen2_5_AudioRotaryEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        update_tensor_data(self.audio_bos_eos_token.weights,
                           weights.get(f'thinker.audio_tower.audio_bos_eos_token.weight'),
                           f"thinker.audio_tower.audio_bos_eos_token.weight")
        # self.layers = nn.ModuleList([Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        update_tensor_data(self.ln_post.weights,
                           weights.get(f'thinker.audio_tower.ln_post.weight'),
                           f"thinker.audio_tower.ln_post.weight")
        update_tensor_data(self.ln_post.biases,
                           weights.get(f'thinker.audio_tower.ln_post.bias'),
                           f"thinker.audio_tower.ln_post.bias")
        self.avg_pooler = nn.AvgPool2d(kernels=[2], strides=[2])
        self.proj = nn.Linear(config.d_model, config.output_dim)
        update_tensor_data(self.proj.weights,
                           weights.get(f'thinker.audio_tower.proj.weight'),
                           f"thinker.audio_tower.proj.weight")
        update_tensor_data(self.proj.biases,
                           weights.get(f'thinker.audio_tower.proj.bias'),
                           f"thinker.audio_tower.proj.bias")
        self.gradient_checkpointing = False
        self.activation = nn.ACT2FN['gelu']
        # self.rotary_pos_emb = Qwen2_5_AudioRotaryEmbedding(config)
        # self.patch_emb = Qwen2_5_AudioPatchEmbed(config, weights)
        self.blocks = [Qwen2_5ThinkerAudioBlock(config, weights, idx, prefix) for idx in range(config.encoder_layers)]
    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def forward(self, feature_lens,aftercnn_lens, window_index=None, cu_window_seqlens=None, grid_thw=None):
        """
        Args:
            hidden_states (`Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.
        """
        # if grid_thw is None:
        #     grid_thw = _get_grid_thw(self.config)
        # if window_index is None or cu_window_seqlens is None:
        #     window_index_data, cu_window_seqlens = self.get_window_index(grid_thw)
        #     window_index = ops.constant(Tensor(window_index_data.cpu().numpy().astype(np.int32)))
        #     # cu_window_seqlens =ops.constant(Tensor(cu_window_seqlens.cpu().numpy().astype(np.int32)))
        #     cu_window_seqlens = torch.tensor(cu_window_seqlens).to(window_index_data.device)
        # cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        input_features = ops.input([chunk_num, 128, self.n_window * 2], Dtype.FP32,
                            name='inputs_audio', exprs=['seq_len', 'mel_bin','frame_lens'])

        attention_mask = ops.input([1, 1,chunk_num*100, chunk_num*100], Dtype.FP32,
                            name='attention_mask', exprs=['batch', 1,'seq_len*100','seq_len*100'])

        seqlen,melbin,n_win = input_features.shape
        input_features = ops.reshape(input_features,[seqlen,melbin,1,n_win])
        input_features = ops.transpose(input_features,[0,2,3,1])
        padded_embed = self.activation(self.conv1(input_features))
        padded_embed = self.activation(self.conv2(padded_embed))

        sinupos = self.positional_embedding(padded_embed.shape[2])
        padded_embed = padded_embed + sinupos
        seqlen,h,l,c = padded_embed.shape.symbol
        hidden_states = ops.reshape(padded_embed,[seqlen*l,c])
        cu_seqlens = torch.arange(0,chunk_num*100+100,100).int()

        # attention_mask = torch.full(
        #     [1, 1,chunk_num*100, chunk_num*100],
        #     -10000,

        # )
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        # full_attn_mask = ops.constant(Tensor(attention_mask.cpu().numpy()))
        for blk_idx, blk in enumerate(self.blocks):
            INFO(f"\t build the {blk_idx} audio encoder block.")

            hidden_states = blk(hidden_states, attention_mask)
        seq_len,channel = hidden_states.shape.symbol

        each_audio_states = ops.reshape(hidden_states,[1,seq_len,channel])

        each_audio_states = self.avg_pooler(each_audio_states)
        bs,seq_len,channel = each_audio_states.shape.symbol

        each_audio_states = self.ln_post(each_audio_states)

        each_audio_states = ops.reshape(each_audio_states,[seq_len,channel])
        hidden_states = self.proj(each_audio_states)
        # token_audio_list.append(each_audio_states)
        # hidden_states = torch.cat(token_audio_list, dim=0)
        return hidden_states


@register_causallm(['qwen2.5thinker_audio'])
class Qwen2_5OmniThinkerAudio(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.audio = Qwen2_5ThinkerAudioModel(self.config, weights,'thinker.audio_tower.layers')
        # self.merger = Qwen2_5ThinkerVisionMerger(config, weights)
        self.aipu_g_visual = None

    def _get_seq_len(self):
        # [1, 26, 112] * 9
        grid_thw = _get_grid_thw(self.config)
        seq_len = grid_thw.prod(-1).sum().item()
        return seq_len

    def forward(self, *args, **kwargs):
        feature_lens = torch.tensor(3600)#self._get_seq_len()


        output_tensors = []
        aftercnn_lens = feature_lens//2
        with Graph('qwen-audio') as self.aipu_g_visual:
            hidden_states = self.audio(feature_lens,aftercnn_lens)
            # hidden_states = self.merger(hidden_states)
        output_tensors.append(hidden_states)
        # output_tensors.append(windows_index)

        self.aipu_g_visual.output_tensors = TensorList(output_tensors)
        return self.aipu_g_visual
