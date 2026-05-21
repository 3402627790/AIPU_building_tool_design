from typing import Optional, Tuple
import numpy as np
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...config import PretrainedConfig
from ...register import register_causallm
from ..llmutils import gen_layer_ids


class VPMEncoderConfig(PretrainedConfig):
    model_type = "clip_vpm"

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


VPM_ENCODER_LARGE_PATCH16_448_CONFIG = VPMEncoderConfig(
    attention_dropout=0.0,
    hidden_act="gelu_new",
    hidden_size=1152,
    image_size=448,
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
    org_img_size=[448, 448]
)


class VPMEmbeddings(nn.Module):
    def __init__(self, clip_config, weights):
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

        self.patch_embedding.weights.betensor = weights[f'vpm.embeddings.patch_embedding.weight'].float().permute(
            0, 2, 3, 1).contiguous()
        self.patch_embedding.weights.sync_betensor()
        self.patch_embedding.biases.betensor = weights[f'vpm.embeddings.patch_embedding.bias'].float().contiguous()
        self.patch_embedding.biases.sync_betensor()
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        max_im_h, max_im_w = clip_config.org_img_size
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / 70.0, 1.0, 1 / 70.0)
        position_ids = torch.full(
            size=(
                1,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / max_nb_patches_h)
        fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / max_nb_patches_w)

        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

        pos_ids = (bucket_coords_h[:, None] * 70 + bucket_coords_w).flatten()
        position_ids[0][:] = pos_ids

        self.position_embedding = nn.Constant(torch.randn((self.num_positions, self.embed_dim)).float())

        self.position_embedding.weights.betensor = (weights[f'vpm.embeddings.position_embedding.weight'].float())[
            position_ids[0]].clone().detach()
        self.position_embedding.weights.sync_betensor()

        self.patch_emb_tranpose = nn.Transpose([0, 3, 1, 2])

    def forward(self, pixel_values: Tensor) -> Tensor:

        pixel_values = ops.transpose(pixel_values, [0, 2, 3, 1])

        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds.name = self.patch_embedding.name
        patch_embeds_permute = self.patch_emb_tranpose(patch_embeds)  # shape = [*, width, grid, grid]
        emb_shape = patch_embeds_permute.shape
        patch_embeds_permute = ops.reshape(
            patch_embeds_permute, [emb_shape[0], emb_shape[1], emb_shape[2]*emb_shape[3]])
        embeddings = ops.transpose(patch_embeds_permute, [0, 2, 1])

        embeddings = embeddings + self.position_embedding()
        return embeddings


class VPMAttention(nn.Module):
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

        self.q_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.q_proj.weight'].float()
        self.q_proj.weights.sync_betensor()
        self.q_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.q_proj.bias'].float()
        self.q_proj.biases.sync_betensor()

        self.k_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.k_proj.weight'].float()
        self.k_proj.weights.sync_betensor()
        self.k_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.k_proj.bias'].float()
        self.k_proj.biases.sync_betensor()

        self.v_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.v_proj.weight'].float()
        self.v_proj.weights.sync_betensor()
        self.v_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.v_proj.bias'].float()
        self.v_proj.biases.sync_betensor()

        self.o_proj.weights.remove_betensor()
        self.o_proj.biases.remove_betensor()
        self.o_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.out_proj.weight'].float()
        self.o_proj.weights.sync_betensor()
        self.o_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.out_proj.bias'].float()
        self.o_proj.biases.sync_betensor()

        self.q_proj.weights.name = f"{prefix}vpm.encoder.layers.{layer_id}.self_attn.q_proj.weight"
        self.q_proj.biases.name = f"{prefix}vpm.encoder.layers.{layer_id}.self_attn.q_proj.bias"
        self.k_proj.weights.name = f'{prefix}vpm.encoder.layers.{layer_id}.self_attn.k_proj.weight'
        self.k_proj.biases.name = f'{prefix}vpm.encoder.layers.{layer_id}.self_attn.k_proj.bias'
        self.v_proj.weights.name = f'{prefix}vpm.encoder.layers.{layer_id}.self_attn.v_proj.weight'
        self.v_proj.biases.name = f'{prefix}vpm.encoder.layers.{layer_id}.self_attn.v_proj.bias'
        self.o_proj.weights.name = f'{prefix}vpm.encoder.layers.{layer_id}.self_attn.o_proj.weight'
        self.o_proj.biases.name = f'{prefix}vpm.encoder.layers.{layer_id}.self_attn.o_proj.bias'

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}vpm.encoder.layers.{layer_id}.self_attn"
        self.softmax = nn.Softmax(axis=3)

    def forward(self, sqrt_k_scale, hidden_states, attention_mask: Optional[Tensor] = None):
        bs, seq_len, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        self.qk_reshape = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.qk_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.qk_reshape(key_states))

        scaled_q = reshaped_trans_query*sqrt_k_scale

        kq_matmul = self.kq_matmul(scaled_q, reshaped_trans_key)
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


class VPMMlp(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        self.gate_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.mlp.fc1.weight'].float()
        self.gate_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.mlp.fc1.bias'].float()

        self.down_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.mlp.fc2.weight'].float()
        self.down_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.mlp.fc2.bias'].float()

        self.gate_proj.weights.sync_betensor()
        self.down_proj.weights.sync_betensor()
        self.gate_proj.biases.sync_betensor()
        self.down_proj.biases.sync_betensor()

        self.gate_proj.weights.name = f'{prefix}vpm.encoder.layers.{layer_id}.mlp.fc1.weight'
        self.down_proj.weights.name = f'{prefix}vpm.encoder.layers.{layer_id}.mlp.fc2.weight'
        self.gate_proj.biases.name = f'{prefix}vpm.encoder.layers.{layer_id}.mlp.fc1.biases'
        self.down_proj.biases.name = f'{prefix}vpm.encoder.layers.{layer_id}.mlp.fc2.biases'

    def forward(self, x):
        x_shape = x.shape
        hidden_states = ops.reshape(x, [-1, x_shape[2]])
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        mlp_out = self.down_proj(hidden_states)

        return ops.reshape(mlp_out, x_shape)


class VPMEncoderLayer(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.layer_norm1.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.layer_norm1.weight'].float()
        self.layer_norm1.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.layer_norm1.bias'].float()
        self.layer_norm1.weights.sync_betensor()
        self.layer_norm1.biases.sync_betensor()

        self.self_attn = VPMAttention(config, weights, layer_id, prefix='')
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.layer_norm2.weight'].float()
        self.layer_norm2.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.layer_norm2.bias'].float()
        self.layer_norm2.weights.sync_betensor()
        self.layer_norm2.biases.sync_betensor()
        self.mlp = VPMMlp(config, weights, layer_id, prefix='')

    def forward(
        self,
        sqrt_k_scale,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None
    ) -> Tuple[torch.FloatTensor]:
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
            sqrt_k_scale,
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


class VPMEncoder(nn.Module):
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

        self.layers = [VPMEncoderLayer(config, weights, idx, prefix) for idx in gen_layer_ids(config.num_hidden_layers)]

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[Tensor] = None,

    ):
        encoder_states = ()
        hidden_states = inputs_embeds
        sqrt_k_scale = nn.Constant(np.array(1./np.sqrt(np.array(self.head_dim))).astype('float32').reshape([1]))()
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                sqrt_k_scale,
                hidden_states,
                attention_mask,

            )
            hidden_states = layer_outputs[0]
        return hidden_states


class Resampler(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.intermediate_size = 1152
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj.weights.betensor = weights[f'resampler.attn.out_proj.weight'].float()
        self.o_proj.biases.betensor = weights[f'resampler.attn.out_proj.bias'].float()
        self.o_proj.weights.sync_betensor()
        self.o_proj.biases.sync_betensor()

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)

        qkv_proj_w = weights[f'resampler.attn.in_proj_weight'].float()
        self.q_proj.weights.betensor = qkv_proj_w[0:self.hidden_size, ...]
        self.q_proj.weights.sync_betensor()

        self.k_proj.weights.betensor = qkv_proj_w[self.hidden_size:2*self.hidden_size, ...]
        self.k_proj.weights.sync_betensor()

        self.v_proj.weights.betensor = qkv_proj_w[2*self.hidden_size:3*self.hidden_size, ...]
        self.v_proj.weights.sync_betensor()

        qkv_proj_b = weights[f'resampler.attn.in_proj_bias'].float()
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

        scale = 1. / np.sqrt(self.head_dim).astype(np.float32)
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}vison_head.self_attn"

        self.softmax = nn.Softmax(axis=3)

        self.kv_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.kv_proj.weights.betensor = weights[f'resampler.kv_proj.weight'].float()
        # self.kv_proj.biases.betensor = weights[f'resampler.kv_proj.bias'].float()
        self.kv_proj.weights.sync_betensor()
        # self.kv_proj.biases.sync_betensor()

        self.ln_kv = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.ln_kv.weights.betensor = weights[f'resampler.ln_kv.weight'].float()
        self.ln_kv.biases.betensor = weights[f'resampler.ln_kv.bias'].float()
        self.ln_kv.weights.sync_betensor()
        self.ln_kv.biases.sync_betensor()
        self.ln_q = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.ln_q.weights.betensor = weights[f'resampler.ln_q.weight'].float()
        self.ln_q.biases.betensor = weights[f'resampler.ln_q.bias'].float()
        self.ln_q.weights.sync_betensor()
        self.ln_q.biases.sync_betensor()
        self.ln_post = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.ln_post.weights.betensor = weights[f'resampler.ln_post.weight'].float()
        self.ln_post.biases.betensor = weights[f'resampler.ln_post.bias'].float()
        self.ln_post.weights.sync_betensor()
        self.ln_post.biases.sync_betensor()

        self.query = nn.Constant(torch.randn((64, self.hidden_size)).float())
        self.query.weights.betensor = weights[f'resampler.query'].float()
        self.query.weights.sync_betensor()

        self.proj = nn.Constant(torch.randn((self.hidden_size, self.hidden_size)).float())
        self.proj.weights.betensor = weights[f'resampler.proj'].float()
        self.proj.weights.sync_betensor()

    def forward(self, q, k):
        bs, seq_len, hidden_size = k.shape
        hidden_states = ops.reshape(k, [bs * seq_len, hidden_size])
        q = ops.reshape(q, [bs * seq_len, hidden_size])
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

        self.qkv_reshape = nn.Reshape([bs*seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        residual = self.o_proj(reshaped_trans)
        residual = ops.reshape(residual, [bs, seq_len, self.hidden_size])
        hidden_state = self.layer_norm(residual)
        hidden_state = ops.reshape(hidden_state, [-1, hidden_state.shape[2]])

        hidden_state = self.img_prj0(hidden_state)
        hidden_state = self.act_fn(hidden_state)
        mlp_out = self.img_prj1(hidden_state) + ops.reshape(residual, [-1, residual.shape[2]])
        return mlp_out


class VPMEncoderTransformer(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = VPMEmbeddings(config, weights)

        self.encoder = VPMEncoder(config, weights)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.post_layernorm.weights.betensor = weights[f'vpm.post_layernorm.weight'].float()
        self.post_layernorm.biases.betensor = weights[f'vpm.post_layernorm.bias'].float()
        self.post_layernorm.weights.sync_betensor()
        self.post_layernorm.biases.sync_betensor()

        # self.probe = nn.Constant(torch.randn((1,1,embed_dim)).float())
        # self.probe.weights.betensor = weights[f'resampler.query'].float()
        # self.probe.weights.sync_betensor()

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
        # last_hidden_state = self.head(self.probe(),last_hidden_state)
        return last_hidden_state


@register_causallm(['vpm_encoder'])
class VPMEncoderCausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})

        self.aipu_g = None
        self.prefix = prefix

        self.clip_vision_config = VPM_ENCODER_LARGE_PATCH16_448_CONFIG
        self.vpm = VPMEncoderTransformer(self.clip_vision_config, weights, prefix)
        # self.head = Resampler(config,weights)

    def forward(self):
        seq_len = self.config.seq_len
        with Graph('vpm_encoder') as self.aipu_g:
            inputs = ops.input([seq_len, 3, 14, 14336], Dtype.FP32, name='input_ids')
            _ = self.vpm(inputs)
        return self.aipu_g
