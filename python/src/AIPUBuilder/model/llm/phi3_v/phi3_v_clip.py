from typing import Tuple, Optional
import numpy as np
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder.logger import INFO, ERROR, WARN, DEBUG
from ...basemodel import *
from ...config import PretrainedConfig
from ...register import register_causallm
from ..llmutils import gen_layer_ids


class CLIPVisionConfig(PretrainedConfig):
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
        rms_norm_eps=1e-5,
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
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.layer_idx = layer_idx
        self.org_img_size = org_img_size


CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
    attention_dropout=0.0,
    hidden_act="quick_gelu",
    hidden_size=1024,
    image_size=336,
    initializer_factor=1.0,
    initializer_range=0.02,
    intermediate_size=4096,
    rms_norm_eps=1e-05,
    num_attention_heads=16,
    num_channels=3,
    num_hidden_layers=23,
    patch_size=14,
    projection_dim=768,
    layer_idx=-2,
    org_img_size=[336, 336]
)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, clip_config, weights):
        super().__init__()

        self.embed_dim = clip_config.hidden_size
        self.image_size = clip_config.image_size
        self.patch_size = clip_config.patch_size

        self.class_embedding = nn.Constant(torch.randn(self.embed_dim).float())
        self.class_embedding.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.embeddings.class_embedding'].float(
        )
        self.class_embedding.weights.sync_betensor()
        self.patch_embedding = nn.Conv2d(
            in_channels=clip_config.num_channels,
            out_channels=self.embed_dim,
            kernels=[clip_config.patch_size, clip_config.patch_size],
            strides=[clip_config.patch_size, clip_config.patch_size],
            name="patch_embedding_conv2d"
        )

        self.patch_embedding.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.embeddings.patch_embedding.weight'].float(
        ).permute(0, 2, 3, 1).contiguous()
        self.patch_embedding.weights.sync_betensor()
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        # self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # self.position_embedding.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.embeddings.position_embedding.weight'].float()
        # self.position_embedding.weights.sync_betensor()
        self.position_embedding = nn.Constant(torch.randn((self.num_positions, self.embed_dim)).float())
        self.position_embedding.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.embeddings.position_embedding.weight'].float(
        )
        self.position_embedding.weights.sync_betensor()

        # self.position_ids = nn.Constant((torch.arange(self.num_positions).expand((1, -1)).int())) #here!!!!!!!!!!!!! have issue
        # self.position_ids.weights.sync_betensor()
        self.patch_emb_tranpose = nn.Transpose([0, 3, 1, 2])

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds.name = self.patch_embedding.name
        patch_embeds = self.patch_emb_tranpose(patch_embeds)  # shape = [*, width, grid, grid]
        emb_shape = patch_embeds.shape.symbol
        patch_embeds = ops.reshape(patch_embeds, [emb_shape[0], emb_shape[1], emb_shape[2] * emb_shape[3]])
        patch_embeds = ops.transpose(patch_embeds, [0, 2, 1])
        class_embeding = self.class_embedding()
        reshape_class_embedding = ops.reshape(class_embeding, [1, 1, class_embeding.shape.symbol[0]])
        class_embeds = ops.tile(reshape_class_embedding, [emb_shape[0], 1, 1])
        embeddings = ops.concat([class_embeds, patch_embeds], axis=1)  # torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding()
        return embeddings


class CLIPAttention(nn.Module):
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

        self.q_proj.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.q_proj.weight'].float(
        )
        self.q_proj.weights.sync_betensor()
        self.q_proj.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.q_proj.bias'].float(
        )
        self.q_proj.biases.sync_betensor()

        self.k_proj.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.k_proj.weight'].float(
        )
        self.k_proj.weights.sync_betensor()
        self.k_proj.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.k_proj.bias'].float(
        )
        self.k_proj.biases.sync_betensor()

        self.v_proj.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.v_proj.weight'].float(
        )
        self.v_proj.weights.sync_betensor()
        self.v_proj.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.v_proj.bias'].float(
        )
        self.v_proj.biases.sync_betensor()

        self.o_proj.weights.remove_betensor()
        self.o_proj.biases.remove_betensor()
        self.o_proj.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.out_proj.weight'].float(
        )
        self.o_proj.weights.sync_betensor()
        self.o_proj.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.out_proj.bias'].float(
        )
        self.o_proj.biases.sync_betensor()

        self.q_proj.weights.name = f"{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.q_proj.weight"
        self.q_proj.biases.name = f"{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.q_proj.bias"
        self.k_proj.weights.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.k_proj.weight'
        self.k_proj.biases.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.k_proj.bias'
        self.v_proj.weights.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.v_proj.weight'
        self.v_proj.biases.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.v_proj.bias'
        self.o_proj.weights.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.o_proj.weight'
        self.o_proj.biases.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn.o_proj.bias'

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.self_attn"

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

        self.qkv_reshape = nn.Reshape([bs * seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        attn_output = self.o_proj(reshaped_trans)
        attn_output = ops.reshape(attn_output, [bs, seq_len, self.hidden_size])
        return attn_output


class CLIPMLP(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        self.gate_proj.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc1.weight'].float(
        )
        self.gate_proj.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc1.bias'].float(
        )

        self.down_proj.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc2.weight'].float(
        )
        self.down_proj.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc2.bias'].float(
        )

        self.gate_proj.weights.sync_betensor()
        self.down_proj.weights.sync_betensor()
        self.gate_proj.biases.sync_betensor()
        self.down_proj.biases.sync_betensor()

        self.gate_proj.weights.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc1.weight'
        self.down_proj.weights.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc2.weight'
        self.gate_proj.biases.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc1.biases'
        self.down_proj.biases.name = f'{prefix}model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.mlp.fc2.biases'

    def forward(self, x):
        bs, seq_len, hds = x.shape.symbol
        hidden_states = ops.reshape(x, [bs * seq_len, hds])
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        mlp_out = self.down_proj(hidden_states)

        return ops.reshape(mlp_out, x.shape.symbol)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config, weights, layer_id, prefix='')
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.rms_norm_eps)
        self.layer_norm1.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.layer_norm1.weight'].float(
        )
        self.layer_norm1.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.layer_norm1.bias'].float(
        )
        self.layer_norm1.weights.sync_betensor()
        self.layer_norm1.biases.sync_betensor()
        self.mlp = CLIPMLP(config, weights, layer_id, prefix='')
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.rms_norm_eps)
        self.layer_norm2.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.layer_norm2.weight'].float(
        )
        self.layer_norm2.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.encoder.layers.{layer_id}.layer_norm2.bias'].float(
        )
        self.layer_norm2.weights.sync_betensor()
        self.layer_norm2.biases.sync_betensor()

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


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.layers = [CLIPEncoderLayer(config, weights, idx, prefix)
                       for idx in gen_layer_ids(config.num_hidden_layers)]
        # self.layers = nn.ModuleList([CLIPEncoderLayer(config,weights,idx,prefix) for idx in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask: Optional[Tensor] = None):
        encoder_states = ()
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
            )
            hidden_states = layer_outputs[0]

        # encoder_states = encoder_states + (hidden_states,)
        return hidden_states
        # return [hidden_states, encoder_states]


class ImageMLP(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size*3
        intermediate_size = config.intermediate_size
        self.img_prj0 = nn.Linear(intermediate_size, hidden_size)
        self.img_prj1 = nn.Linear(hidden_size, hidden_size)
        self.act_fn = nn.ACT2FN['gelu']

        self.img_prj0.weights.betensor = weights[f'model.vision_embed_tokens.img_projection.0.weight'].float()
        self.img_prj0.biases.betensor = weights[f'model.vision_embed_tokens.img_projection.0.bias'].float()

        self.img_prj1.weights.betensor = weights[f'model.vision_embed_tokens.img_projection.2.weight'].float()
        self.img_prj1.biases.betensor = weights[f'model.vision_embed_tokens.img_projection.2.bias'].float()

        self.img_prj0.weights.sync_betensor()
        self.img_prj1.weights.sync_betensor()
        self.img_prj0.biases.sync_betensor()
        self.img_prj1.biases.sync_betensor()

        self.img_prj0.weights.name = f'{prefix}model.vision_embed_tokens.img_projection.0.mlp.fc1.weight'
        self.img_prj1.weights.name = f'{prefix}model.vision_embed_tokens.img_projection.1.mlp.fc2.weight'
        self.img_prj0.biases.name = f'{prefix}model.vision_embed_tokens.img_projection.0.mlp.fc1.biases'
        self.img_prj1.biases.name = f'{prefix}model.vision_embed_tokens.img_projection.1.mlp.fc2.biases'

    def forward(self, x):
        x_shape = x.shape.symbol
        hidden_states = ops.reshape(x, [x_shape[0] * x_shape[1], x_shape[2]])
        hidden_states = self.img_prj0(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        mlp_out = self.img_prj1(hidden_states)

        return mlp_out


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.layer_idx = config.layer_idx
        self.h = config.org_img_size[0]//config.image_size
        self.w = config.org_img_size[1]//config.image_size
        self.B = self.h*self.w
        self.embeddings = CLIPVisionEmbeddings(config, weights)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.rms_norm_eps)

        self.pre_layrnorm.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.pre_layrnorm.weight'].float(
        )
        self.pre_layrnorm.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.pre_layrnorm.bias'].float(
        )
        self.pre_layrnorm.weights.sync_betensor()
        self.pre_layrnorm.biases.sync_betensor()
        self.encoder = CLIPEncoder(config, weights)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.rms_norm_eps)
        self.post_layernorm.weights.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.post_layernorm.weight'].float(
        )
        self.post_layernorm.biases.betensor = weights[f'model.vision_embed_tokens.img_processor.vision_model.post_layernorm.bias'].float(
        )
        self.post_layernorm.weights.sync_betensor()
        self.post_layernorm.biases.sync_betensor()

        # get glb_img and sub_img
        self.glb_GN = nn.Constant(np.zeros((1, 1, 4096)).astype('float32'))
        self.glb_GN.weights.betensor = weights[f'model.vision_embed_tokens.glb_GN'].float()
        self.glb_GN.weights.sync_betensor()
        self.glb_GN.weights.name = f"glb_GN_cnst"

        self.sub_GN = nn.Constant(np.zeros((1, 1, 1, 4096)).astype('float32'))
        self.sub_GN.weights.betensor = weights[f'model.vision_embed_tokens.sub_GN'].float()
        self.sub_GN.weights.sync_betensor()
        self.sub_GN.weights.name = f"sub_GN_cnst"
        self.img_projection = ImageMLP(config, weights, prefix='')

    def forward(self, pixel_values: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )

        # last_hidden_state = encoder_outputs[1][self.layer_idx]
        glb_crop0 = nn.Crop([[0, 1], [1, last_hidden_state.shape.symbol[1]], [0, last_hidden_state.shape.symbol[-1]]])
        sub_crop0 = nn.Crop([[1, self.B+1], [1, last_hidden_state.shape.symbol[1]],
                            [0, last_hidden_state.shape.symbol[-1]]])
        image_dim_out = last_hidden_state.shape.symbol[-1]
        glb_img = glb_crop0(last_hidden_state)
        sub_img = sub_crop0(last_hidden_state)
        glb_img = ops.reshape(glb_img, [glb_img.shape.symbol[0], 24, 24, image_dim_out])
        glb_img = ops.space_to_depth(glb_img, 2)
        glb_GN = self.glb_GN()
        glb_GN_reshape = ops.reshape(glb_GN, [1, 1, 1, glb_GN.shape[-1]])
        temp_glb_GN = ops.tile(glb_GN_reshape, [1, 12, 1, 1])
        glb_img = ops.concat([glb_img, temp_glb_GN], axis=2)
        glb_img = ops.reshape(glb_img, [1, glb_img.shape.symbol[1] * glb_img.shape.symbol[2], temp_glb_GN.shape[-1]])

        sub_img = ops.reshape(sub_img, [sub_img.shape.symbol[0], 24, 24, image_dim_out])
        sub_img = ops.space_to_depth(sub_img, 2)
        sub_img = ops.reshape(sub_img, [self.h, self.w, 12, 12, image_dim_out*4])

        sub_img = ops.transpose(sub_img, [0, 2, 1, 3, 4])
        sub_img = ops.reshape(sub_img, [1, 12*self.h, 12*self.w, image_dim_out*4])
        temp_sub_GN = ops.tile(self.sub_GN(), [1, 12*self.h, 1, 1])
        sub_img = ops.concat([sub_img, temp_sub_GN], axis=2)
        sub_img = ops.reshape(sub_img, [1, sub_img.shape.symbol[1] * sub_img.shape.symbol[2], temp_sub_GN.shape[-1]])

        output_imgs = ops.concat([sub_img, glb_GN, glb_img], axis=1)
        img_feature_proj = self.img_projection(output_imgs)
        return img_feature_proj


@register_causallm(['phi3_v_clip'])
class Phi3ClipCausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})

        self.aipu_g = None
        self.prefix = prefix

        self.clip_vision_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
        self.vision_model = CLIPVisionTransformer(self.clip_vision_config, weights, prefix)

    def forward(self):
        seq_len = self.config.seq_len
        with Graph('phi3_clip') as self.aipu_g:
            hidden_states = ops.input([seq_len, 336, 336, 3],
                                      Dtype.FP32,
                                      name='input_ids',)
            #   exprs=['seq_len', 336, 336, 3])
            hidden_states = self.vision_model(hidden_states)

        return self.aipu_g
