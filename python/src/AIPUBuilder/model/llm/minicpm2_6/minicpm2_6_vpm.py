from typing import Tuple, Optional
import numpy as np
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...config import PretrainedConfig
from ...register import register_causallm


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
            position_ids].clone().detach()
        self.position_embedding.weights.sync_betensor()

        self.patch_emb_tranpose = nn.Transpose([0, 3, 1, 2])

    def forward(self, pixel_values: Tensor) -> Tensor:

        # pixel_values = ops.transpose(pixel_values, [0, 2, 3, 1])
        # changed model to inputs NHWC, so related tranpose op can be removed
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds.name = self.patch_embedding.name
        # patch_embeds_permute = self.patch_emb_tranpose(patch_embeds)  # shape = [*, width, grid, grid]
        # emb_shape = patch_embeds_permute.shape
        # patch_embeds_permute = ops.reshape(
        #     patch_embeds_permute, [emb_shape[0], emb_shape[1], emb_shape[2]*emb_shape[3]])
        # embeddings = ops.transpose(patch_embeds_permute, [0, 2, 1])
        emb_shape = patch_embeds.shape.symbol
        embeddings = ops.reshape(
            patch_embeds, [emb_shape[0]*emb_shape[1], emb_shape[2], emb_shape[3]])

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

        sqrt_d = np.sqrt(1./np.sqrt(self.head_dim))
        self.q_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.q_proj.weight'].float()*sqrt_d
        self.q_proj.weights.sync_betensor()
        self.q_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.q_proj.bias'].float()*sqrt_d
        self.q_proj.biases.sync_betensor()

        self.k_proj.weights.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.k_proj.weight'].float()*sqrt_d
        self.k_proj.weights.sync_betensor()
        self.k_proj.biases.betensor = weights[f'vpm.encoder.layers.{layer_id}.self_attn.k_proj.bias'].float()*sqrt_d
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

        self.qkv_transpose = nn.Transpose([2, 0, 1, 3])
        self.matmul_transpose = nn.Transpose([1, 2, 0, 3])
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}vpm.encoder.layers.{layer_id}.self_attn"
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

        scaled_q = reshaped_trans_query

        kq_matmul = self.kq_matmul(scaled_q, reshaped_trans_key)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"

        softmax_masked = self.softmax(kq_matmul)

        reshaped_trans_value = self.qkv_transpose(self.qk_reshape(value_states))

        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs*seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.matmul_transpose(kqv_out))
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
        bs, seq_len, hds = x.shape.symbol
        hidden_states = ops.reshape(x, [bs * seq_len, hds])
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        mlp_out = self.down_proj(hidden_states)

        return ops.reshape(mlp_out, [bs, seq_len, hds])


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

        self.layers = [VPMEncoderLayer(config, weights, idx, prefix) for idx in range(config.num_hidden_layers)]

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[Tensor] = None,

    ):
        encoder_states = ()
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,

            )
            hidden_states = layer_outputs[0]
        return hidden_states


class Resampler(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()

        def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
            """
            embed_dim: output dimension for each position
            pos: a list of positions to be encoded: size (H, W)
            out: (H, W, D)
            """
            assert embed_dim % 2 == 0
            omega = np.arange(embed_dim // 2, dtype=np.float32)
            omega /= embed_dim / 2.
            omega = 1. / 10000 ** omega  # (D/2,)

            out = np.einsum('hw,d->hwd', pos, omega)  # (H, W, D/2), outer product

            emb_sin = np.sin(out)  # (H, W, D/2)
            emb_cos = np.cos(out)  # (H, W, D/2)

            emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
            return emb

        def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
            assert embed_dim % 2 == 0

            # use half of dimensions to encode grid_h
            emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
            emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

            emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
            return emb

        def get_2d_sincos_pos_embed(embed_dim, image_size):
            """
            image_size: image_size or (image_height, image_width)
            return:
            pos_embed: [image_height, image_width, embed_dim]
            """
            if isinstance(image_size, int):
                grid_h_size, grid_w_size = image_size, image_size
            else:
                grid_h_size, grid_w_size = image_size[0], image_size[1]

            grid_h = np.arange(grid_h_size, dtype=np.float32)
            grid_w = np.arange(grid_w_size, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)  # here w goes first
            grid = np.stack(grid, axis=0)

            pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
            return pos_embed

        self.pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(config.hidden_size, (70, 70))).float()

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

        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)

        qkv_proj_w = weights[f'resampler.attn.in_proj_weight'].float()
        self.k_proj.weights.betensor = qkv_proj_w[self.hidden_size:2*self.hidden_size, ...]
        self.k_proj.weights.sync_betensor()

        self.v_proj.weights.betensor = qkv_proj_w[2*self.hidden_size:3*self.hidden_size, ...]
        self.v_proj.weights.sync_betensor()

        qkv_proj_b = weights[f'resampler.attn.in_proj_bias'].float()
        self.k_proj.biases.betensor = qkv_proj_b[self.hidden_size:2*self.hidden_size]
        self.k_proj.biases.sync_betensor()

        self.v_proj.biases.betensor = qkv_proj_b[2*self.hidden_size:3*self.hidden_size]
        self.v_proj.biases.sync_betensor()

        self.k_proj.weights.name = f'{prefix}vison_head..self_attn.k_proj.weight'
        self.k_proj.biases.name = f'{prefix}vison_head..self_attn.k_proj.bias'
        self.v_proj.weights.name = f'{prefix}vison_head..self_attn.v_proj.weight'
        self.v_proj.biases.name = f'{prefix}vison_head..self_attn.v_proj.bias'
        self.o_proj.weights.name = f'{prefix}vison_head..self_attn.o_proj.weight'

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=False)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}vison_head.self_attn"

        self.softmax = nn.Softmax(axis=3)

        self.kv_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.kv_proj.weights.betensor = weights[f'resampler.kv_proj.weight'].float()
        # self.kv_proj.biases.betensor = weights[f'resampler.kv_proj.bias'].float()
        self.kv_proj.weights.sync_betensor()
        # self.kv_proj.biases.sync_betensor()

        self.ln_kv = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.ln_kv.weights.betensor = weights[f'resampler.ln_kv.weight'].float()
        self.ln_kv.biases.betensor = weights[f'resampler.ln_kv.bias'].float()
        self.ln_kv.weights.sync_betensor()
        self.ln_kv.biases.sync_betensor()

        self.ln_post = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.ln_post.weights.betensor = weights[f'resampler.ln_post.weight'].float()
        self.ln_post.biases.betensor = weights[f'resampler.ln_post.bias'].float()
        self.ln_post.weights.sync_betensor()
        self.ln_post.biases.sync_betensor()

        self.ln_q = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        ln_q_w = weights[f'resampler.ln_q.weight'].float()
        ln_q_b = weights[f'resampler.ln_q.bias'].float()

        self.query = nn.Constant(torch.randn((1, self.num_heads, 64, self.head_dim)).float())
        query = weights[f'resampler.query'].float()
        q = torch.nn.functional.layer_norm(query, [self.hidden_size], weight=ln_q_w, bias=ln_q_b, eps=1e-06)
        qq = torch.nn.functional.linear(q, qkv_proj_w[0:self.hidden_size, ...],
                                        qkv_proj_b[0:self.hidden_size, ...],)/math.sqrt(self.head_dim)
        self.query.weights.betensor = qq.reshape(1, 64, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        self.query.weights.sync_betensor()

        self.x_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.x_proj.weights.betensor = weights[f'resampler.proj'].float().T
        self.x_proj.weights.sync_betensor()

        pos_embed = []
        tgt_h, tgt_w = 32, 32
        self.pos_embed_cnst = nn.Constant(torch.randn((1, tgt_h*tgt_w, self.num_heads*self.head_dim)).float())

        pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)))  # patches * D
        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BL
        self.pos_embed_cnst.weights.betensor = pos_embed.reshape(1, tgt_h*tgt_w, -1)
        self.pos_embed_cnst.weights.sync_betensor()

    def forward(self, x):
        bs, seq_len, t = x.shape.symbol
        hidden_states = ops.reshape(x, [bs * seq_len, t])

        kv_states = self.kv_proj(hidden_states)
        kv_states = ops.reshape(kv_states, [bs, seq_len, kv_states.shape[-1]])
        kv_states = self.ln_kv(kv_states)

        k_add_pos = kv_states+self.pos_embed_cnst()

        bs, seq_len, t = k_add_pos.shape.symbol

        key_states = self.k_proj(ops.reshape(k_add_pos, [bs*seq_len, t]))

        bs, seq_len, t = kv_states.shape.symbol
        value_states = self.v_proj(ops.reshape(kv_states, [bs*seq_len, t]))

        self.k_reshape = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim])

        reshaped_trans_k = ops.transpose(self.k_reshape(key_states), [0, 2, 3, 1])
        query_cst = self.query()
        if not self.config.get('broadcast_matmul', False):
            if query_cst.shape[0] == 1 and reshaped_trans_k.shape[0] != 1:
                query_cst = ops.tile(query_cst, [reshaped_trans_k.shape[0], 1, 1, 1])
        query_cst.shape.layout = DataLayout.NHWC
        kq_matmul = self.kq_matmul(query_cst, reshaped_trans_k)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        softmax_masked = self.softmax(kq_matmul)

        reshaped_trans_value = self.qkv_transpose(self.k_reshape(value_states))

        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs*64, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        residual = self.o_proj(reshaped_trans)
        residual = ops.reshape(residual, [bs, 64, self.hidden_size])
        hidden_state = self.ln_post(residual)

        hd_symb = hidden_state.shape.symbol
        hidden_state = ops.reshape(hidden_state, [hd_symb[0] * hd_symb[1], hd_symb[2]])

        hidden_state = self.x_proj(hidden_state)
        hidden_state = ops.reshape(hidden_state, [bs, 64, self.hidden_size])
        return hidden_state


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
        return last_hidden_state


@register_causallm(['minicpm2_6_vpm'])
class VPMEncoderCausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})

        self.aipu_g = None
        self.prefix = prefix

        self.clip_vision_config = VPM_ENCODER_LARGE_PATCH16_448_CONFIG
        self.vpm = VPMEncoderTransformer(self.clip_vision_config, weights, prefix)
        self.head = Resampler(config, weights)

    def forward(self):
        seq_len = self.config.get('seq_len', self.config.get('max_position_embeddings', 5))
        with Graph('vpm_resampler') as self.aipu_g:
            inputs = ops.input([seq_len, 14, 14336, 3], Dtype.FP32, name='input_ids', exprs=['seq_len', 14, 14336, 3])
            last_hidden_state = self.vpm(inputs)
            hidden_states = self.head(last_hidden_state)
        return self.aipu_g
