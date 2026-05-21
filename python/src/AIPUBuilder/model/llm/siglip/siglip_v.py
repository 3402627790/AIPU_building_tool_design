from typing import Tuple, Optional
import numpy as np
import torch
import math
import os
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...config import PretrainedConfig
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids
from ...register import register_causallm, register_update_config


@register_update_config('siglip_vision')
def siglip_vision_update_config(config):
    clip_vision_config = SIGLIP_VIT_LARGE_PATCH16_384_CONFIG
    config.num_key_value_heads = clip_vision_config.num_attention_heads
    config.use_cache = config.get('use_cache', False)


class SIGLIPVisionConfig(PretrainedConfig):
    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="gelu_new",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        layer_idx=-2,
        org_img_size=[1344, 1008]
    ):

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.layer_idx = layer_idx
        self.org_img_size = org_img_size


SIGLIP_VIT_LARGE_PATCH16_384_CONFIG = SIGLIPVisionConfig(
    attention_dropout=0.0,
    hidden_act="gelu_new",
    hidden_size=1152,
    image_size=384,
    initializer_factor=1.0,
    initializer_range=0.02,
    intermediate_size=4304,
    layer_norm_eps=1e-05,
    num_attention_heads=16,
    num_channels=3,
    num_hidden_layers=27,
    patch_size=14,
    projection_dim=1152,
    layer_idx=-2,
    org_img_size=[384, 384]
)


class SIGLIPVisionEmbeddings(nn.Module):
    def __init__(self, clip_config, weights, prefix=''):
        super().__init__()

        self.embed_dim = clip_config.hidden_size
        self.image_size = clip_config.image_size
        self.patch_size = clip_config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=clip_config.num_channels,
            out_channels=self.embed_dim,
            kernels=[clip_config.patch_size, clip_config.patch_size],
            strides=[clip_config.patch_size, clip_config.patch_size],
            name="patch_embedding_conv2d"
        )

        self.patch_embedding.weights.betensor = weights[f'{prefix}vision_model.embeddings.patch_embedding.weight'].float().permute(
            0, 2, 3, 1).contiguous()
        self.patch_embedding.weights.sync_betensor()
        self.patch_embedding.biases.betensor = weights[f'{prefix}vision_model.embeddings.patch_embedding.bias'].float(
        ).contiguous()
        self.patch_embedding.biases.sync_betensor()
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        # self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        emb_shape = [1, self.num_positions, self.embed_dim]
        self.position_embedding = nn.Constant(np.zeros((1, self.num_positions, self.embed_dim)).astype(np.float32))
        emb_w = weights[f'{prefix}vision_model.embeddings.position_embedding.weight'].float()
        emb_w = emb_w.reshape(emb_shape)
        self.position_embedding.weights.betensor = emb_w
        self.position_embedding.weights.sync_betensor()

        self.patch_emb_tranpose = nn.Transpose([0, 3, 1, 2])

    def forward(self, pixel_values: Tensor) -> Tensor:

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds.name = self.patch_embedding.name
        patch_embeds_permute = self.patch_emb_tranpose(patch_embeds)  # shape = [*, width, grid, grid]
        emb_shape = patch_embeds_permute.shape
        patch_embeds_permute = ops.reshape(
            patch_embeds_permute, [emb_shape[0], emb_shape[1], emb_shape[2]*emb_shape[3]])
        embeddings = ops.transpose(patch_embeds_permute, [0, 2, 1])

        embeddings = embeddings + self.position_embedding()
        return embeddings


class SIGLIPAttention(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()

        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        update_tensor_data(self.q_proj.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.q_proj.weight'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q_proj.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.q_proj.bias'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.q_proj.bias")

        update_tensor_data(self.k_proj.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.k_proj.weight'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.k_proj.weight")
        update_tensor_data(self.k_proj.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.k_proj.bias'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.k_proj.bias")

        update_tensor_data(self.v_proj.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.v_proj.weight'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.v_proj.weight")
        update_tensor_data(self.v_proj.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.v_proj.bias'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.v_proj.bias")

        update_tensor_data(self.o_proj.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.out_proj.weight'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.out_proj.weight")
        update_tensor_data(self.o_proj.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.self_attn.out_proj.bias'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn.out_proj.bias")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype(np.float32)
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}vision_model.encoder.layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask: Optional[Tensor] = None):
        bs, seq_len, hidden_size = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        self.qk_reshape = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.qk_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.qk_reshape(key_states))

        kq_matmul = self.kq_matmul(reshaped_trans_query, reshaped_trans_key)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"

        softmax_masked = self.softmax(kq_matmul)

        reshaped_trans_value = self.qkv_transpose(self.qk_reshape(value_states))

        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs*seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        attn_output = self.o_proj(reshaped_trans)
        attn_output = ops.reshape(attn_output, [bs, seq_len, self.hidden_size])
        return attn_output


class SIGLIPMLP(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        update_tensor_data(self.gate_proj.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc1.weight'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc1.weight")
        update_tensor_data(self.gate_proj.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc1.bias'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc1.bias")

        update_tensor_data(self.down_proj.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc2.weight'),
                           f'{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc2.weight')
        update_tensor_data(self.down_proj.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc2.bias'),
                           f'{prefix}vision_model.encoder.layers.{layer_id}.mlp.fc2.bias')

    def forward(self, x):
        bs, seq_symbol, dim_symbol = x_shape = x.shape.symbol
        hidden_states = ops.reshape(x, [bs * seq_symbol, dim_symbol])
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        mlp_out = self.down_proj(hidden_states)

        return ops.reshape(mlp_out, [bs, seq_symbol, dim_symbol])


class SIGLIPEncoderLayer(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        update_tensor_data(self.layer_norm1.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.layer_norm1.weight'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.layer_norm1.weight")
        update_tensor_data(self.layer_norm1.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.layer_norm1.bias'),
                           f'{prefix}vision_model.encoder.layers.{layer_id}.layer_norm1.bias')

        self.self_attn = SIGLIPAttention(config, weights, layer_id, prefix=prefix)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        update_tensor_data(self.layer_norm2.weights,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.layer_norm2.weight'),
                           f"{prefix}vision_model.encoder.layers.{layer_id}.layer_norm2.weight")
        update_tensor_data(self.layer_norm2.biases,
                           weights.get(f'{prefix}vision_model.encoder.layers.{layer_id}.layer_norm2.bias'),
                           f'{prefix}vision_model.encoder.layers.{layer_id}.layer_norm2.bias')
        self.mlp = SIGLIPMLP(config, weights, layer_id, prefix=prefix)

    def forward(self, idx, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class SIGLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SIGLIPEncoderLayer`].

    Args:
        config: SIGLIPConfig
    """

    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layers = [SIGLIPEncoderLayer(config, weights, idx, prefix)
                       for idx in gen_layer_ids(config.num_hidden_layers)]

    def forward(self, inputs_embeds, attention_mask: Optional[Tensor] = None):
        encoder_states = ()
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                idx,
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        # encoder_states = encoder_states + (hidden_states,)
        return hidden_states
        # return [hidden_states, encoder_states]


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = config.intermediate_size
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        update_tensor_data(self.o_proj.weights,
                           weights.get(f'{prefix}vision_model.head.attention.out_proj.weight'),
                           f"{prefix}vision_model.head.attention.out_proj.weight")
        update_tensor_data(self.o_proj.biases,
                           weights.get(f'{prefix}vision_model.head.attention.out_proj.bias'),
                           f'{prefix}vision_model.head.attention.out_proj.bias')

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)

        qkv_proj_w = weights.get(f'{prefix}vision_model.head.attention.in_proj_weight')['data'].float()
        self.q_proj.weights.betensor = qkv_proj_w[0:self.hidden_size, ...]
        self.q_proj.weights.sync_betensor()

        self.k_proj.weights.betensor = qkv_proj_w[self.hidden_size:2*self.hidden_size, ...]
        self.k_proj.weights.sync_betensor()

        self.v_proj.weights.betensor = qkv_proj_w[2*self.hidden_size:3*self.hidden_size, ...]
        self.v_proj.weights.sync_betensor()

        qkv_proj_b = weights.get(f'{prefix}vision_model.head.attention.in_proj_bias')['data'].float()
        self.q_proj.biases.betensor = qkv_proj_b[0:self.hidden_size]
        self.q_proj.biases.sync_betensor()

        self.k_proj.biases.betensor = qkv_proj_b[self.hidden_size:2*self.hidden_size]
        self.k_proj.biases.sync_betensor()

        self.v_proj.biases.betensor = qkv_proj_b[2*self.hidden_size:3*self.hidden_size]
        self.v_proj.biases.sync_betensor()

        self.q_proj.weights.name = f"{prefix}vison_head.self_attn.q_proj.weight"
        self.q_proj.biases.name = f"{prefix}vison_head..self_attn.q_proj.bias"
        self.k_proj.weights.name = f'{prefix}vison_head..self_attn.k_proj.weight'
        self.k_proj.biases.name = f'{prefix}vison_head..self_attn.k_proj.bias'
        self.v_proj.weights.name = f'{prefix}vison_head..self_attn.v_proj.weight'
        self.v_proj.biases.name = f'{prefix}vison_head..self_attn.v_proj.bias'
        self.o_proj.weights.name = f'{prefix}vison_head..self_attn.o_proj.weight'

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim).astype(np.float32))
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}vison_head.self_attn"

        self.softmax = nn.Softmax(axis=3)

        # MLP
        self.img_prj0 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.img_prj1 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        update_tensor_data(self.img_prj0.weights,
                           weights.get(f'{prefix}vision_model.head.mlp.fc1.weight'),
                           f"{prefix}vision_model.head.mlp.fc1.weight")
        update_tensor_data(self.img_prj0.biases,
                           weights.get(f'{prefix}vision_model.head.mlp.fc1.bias'),
                           f'{prefix}vision_model.head.mlp.fc1.bias')

        update_tensor_data(self.img_prj1.weights,
                           weights.get(f'{prefix}vision_model.head.mlp.fc2.weight'),
                           f"{prefix}vision_model.head.mlp.fc2.weight")
        update_tensor_data(self.img_prj1.biases,
                           weights.get(f'{prefix}vision_model.head.mlp.fc2.bias'),
                           f'{prefix}vision_model.head.mlp.fc2.bias')

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        update_tensor_data(self.layer_norm.weights,
                           weights.get(f'{prefix}vision_model.head.layernorm.weight'),
                           f"{prefix}vision_model.head.layernorm.weight")
        update_tensor_data(self.layer_norm.biases,
                           weights.get(f'{prefix}vision_model.head.layernorm.bias'),
                           f'{prefix}vision_model.head.layernorm.bias')

    def forward(self, q, k):
        bs, seq_len, hidden_size = k.shape
        hidden_states = ops.reshape(k, [bs * seq_len, hidden_size])
        q = ops.reshape(q, [bs * 1, hidden_size])
        query_states = self.q_proj(q)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        self.k_reshape = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim])
        self.q_reshape = nn.Reshape([bs, 1, self.num_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.q_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.k_reshape(key_states))

        kq_matmul = self.kq_matmul(reshaped_trans_query, reshaped_trans_key)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"

        softmax_masked = self.softmax(kq_matmul)

        reshaped_trans_value = self.qkv_transpose(self.k_reshape(value_states))

        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs*1, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        residual = self.o_proj(reshaped_trans)
        residual = ops.reshape(residual, [bs, 1, self.hidden_size])
        hidden_state = self.layer_norm(residual)
        hidden_state = ops.reshape(hidden_state, [-1, hidden_state.shape[2]])

        hidden_state = self.img_prj0(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        mlp_out = self.img_prj1(hidden_state) + ops.reshape(residual, [-1, residual.shape[2]])
        return mlp_out


class SIGLIPVisionTransformer(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.layer_idx = config.layer_idx
        self.h = config.org_img_size[0]//config.image_size
        self.w = config.org_img_size[1]//config.image_size
        self.B = self.h*self.w
        self.embeddings = SIGLIPVisionEmbeddings(config, weights, prefix)

        self.encoder = SIGLIPEncoder(config, weights)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        update_tensor_data(self.post_layernorm.weights,
                           weights.get(f'{prefix}vision_model.post_layernorm.weight'),
                           f"{prefix}vision_model.post_layernorm.weight")
        update_tensor_data(self.post_layernorm.biases,
                           weights.get(f'{prefix}vision_model.post_layernorm.bias'),
                           f'{prefix}vision_model.post_layernorm.bias')

        self.head = SiglipMultiheadAttentionPoolingHead(config, weights, prefix)
        self.probe = nn.Constant(np.zeros((1, 1, embed_dim)).astype(np.float32))
        self.probe.weights.betensor = weights.get(f'{prefix}vision_model.head.probe')['data'].float()
        self.probe.weights.sync_betensor()

    def forward(
        self,
        pixel_values: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,

    ):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        last_hidden_state = self.head(self.probe(), last_hidden_state)

        return last_hidden_state


@register_causallm(['siglip_vision'])
class SigLipCausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})

        self.aipu_g = None
        self.prefix = prefix

        self.clip_vision_config = SIGLIP_VIT_LARGE_PATCH16_384_CONFIG
        self.vision_model = SIGLIPVisionTransformer(self.clip_vision_config, weights, prefix)

        config.num_key_value_heads = self.clip_vision_config.num_attention_heads
        config.use_cache = config.get('use_cache', False) and config.get('enable_ds', False)

    def forward(self):
        with Graph('siglip_vision') as self.aipu_g:
            inputs = ops.input([self.config.seq_len, 384, 384, 3], Dtype.FP32, name='input_ids')
            hidden_states = self.vision_model(inputs)
        return self.aipu_g
