from typing import Tuple, Optional
import numpy as np
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...utils import update_tensor_data
from ...basemodel import BaseModel
from ...register import register_causallm, register_update_config


@register_update_config('siglip_text')
def siglip_text_update_config(config):
    config.num_key_value_heads = config.num_attention_heads
    config.use_cache = config.get('use_cache', False)


class SIGLIPTextEmbeddings(nn.Module):
    def __init__(self, clip_config, weights, prefix=''):
        super().__init__()

        self.embed_dim = clip_config.hidden_size
        self.max_position_embeddings = clip_config.max_position_embeddings
        self.token_embedding = nn.Embedding(clip_config.vocab_size, self.embed_dim)
        update_tensor_data(self.token_embedding.weights,
                           weights.get(f'{prefix}text_model.embeddings.token_embedding.weight'),
                           f"{prefix}text_model.embeddings.token_embedding.weight")

        # self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.position_embedding = nn.Constant(torch.randn((self.max_position_embeddings, self.embed_dim)).float())
        update_tensor_data(self.position_embedding.weights,
                           weights.get(f'{prefix}text_model.embeddings.position_embedding.weight'),
                           f"{prefix}text_model.embeddings.position_embedding.weight")

    def forward(self, input_ids: Tensor) -> Tensor:

        inputs_embeds = self.token_embedding(input_ids)
        embeddings = inputs_embeds + self.position_embedding()
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
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.q_proj.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q_proj.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.q_proj.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.q_proj.bias")

        update_tensor_data(self.k_proj.weights,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.k_proj.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.k_proj.weight")
        update_tensor_data(self.k_proj.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.k_proj.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.k_proj.bias")

        update_tensor_data(self.v_proj.weights,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.v_proj.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.v_proj.weight")
        update_tensor_data(self.v_proj.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.v_proj.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.v_proj.bias")

        update_tensor_data(self.o_proj.weights,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.out_proj.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.out_proj.weight")
        update_tensor_data(self.o_proj.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.self_attn.out_proj.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.self_attn.out_proj.bias")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}text_model.encoder.layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask: Optional[Tensor] = None):
        bs, seq_len, hidden_size = hidden_states.shape
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
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.mlp.fc1.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.mlp.fc1.weight")
        update_tensor_data(self.gate_proj.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.mlp.fc1.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.mlp.fc1.bias")

        update_tensor_data(self.down_proj.weights,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.mlp.fc2.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.mlp.fc2.weight")
        update_tensor_data(self.down_proj.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.mlp.fc2.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.mlp.fc2.bias")

    def forward(self, x):
        x_shape = x.shape
        hidden_states = ops.reshape(x, [-1, x_shape[2]])
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        mlp_out = self.down_proj(hidden_states)

        return ops.reshape(mlp_out, x_shape)


class SIGLIPEncoderLayer(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.get('layer_norm_eps', 0.00001))
        update_tensor_data(self.layer_norm1.weights,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.layer_norm1.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.layer_norm1.weight")
        update_tensor_data(self.layer_norm1.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.layer_norm1.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.layer_norm1.bias")

        self.self_attn = SIGLIPAttention(config, weights, layer_id, prefix=prefix)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.get('layer_norm_eps', 0.00001))
        update_tensor_data(self.layer_norm2.weights,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.layer_norm2.weight'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.layer_norm2.weight")
        update_tensor_data(self.layer_norm2.biases,
                           weights.get(f'{prefix}text_model.encoder.layers.{layer_id}.layer_norm2.bias'),
                           f"{prefix}text_model.encoder.layers.{layer_id}.layer_norm2.bias")

        self.mlp = SIGLIPMLP(config, weights, layer_id, prefix=prefix)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[torch.FloatTensor]:
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
        self.layers = [SIGLIPEncoderLayer(config, weights, idx, prefix) for idx in range(config.num_hidden_layers)]

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


class SIGLIPTextTransformer(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SIGLIPTextEmbeddings(config, weights, prefix)

        self.encoder = SIGLIPEncoder(config, weights, prefix)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.get('layer_norm_eps', 0.00001))
        update_tensor_data(self.final_layer_norm.weights,
                           weights.get(f'{prefix}text_model.final_layer_norm.weight'),
                           f"{prefix}text_model.final_layer_norm.weight")
        update_tensor_data(self.final_layer_norm.biases,
                           weights.get(f'{prefix}text_model.final_layer_norm.bias'),
                           f"{prefix}text_model.final_layer_norm.bias")

        self.head = nn.Linear(embed_dim, embed_dim)
        update_tensor_data(self.head.weights,
                           weights.get(f'{prefix}text_model.head.weight'),
                           f"{prefix}text_model.head.weight")
        update_tensor_data(self.head.biases,
                           weights.get(f'{prefix}text_model.head.bias'),
                           f"{prefix}text_model.head.bias")

    def forward(self, pixel_values: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        bs, max_pos, embed_dim = last_hidden_state.shape
        self.crop0 = nn.Crop([[0, bs], [max_pos-1, max_pos], [0, embed_dim]])
        hidden_state_crop = self.crop0(last_hidden_state)
        hidden_state_reshape = ops.reshape(hidden_state_crop, [-1, hidden_state_crop.shape[-1]])
        last_hidden_state = self.head(hidden_state_reshape)

        return last_hidden_state


@register_causallm(['siglip_text'])
class SigLipCausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})

        self.aipu_g = None
        self.prefix = prefix

        self.text_model = SIGLIPTextTransformer(self.config, weights, prefix)

        config.num_key_value_heads = self.config.num_attention_heads
        config.use_cache = config.get('use_cache', False) and config.get('enable_ds', False)

    def forward(self):
        with Graph('siglip_text') as self.aipu_g:
            inputs = ops.input(
                [self.config.seq_len, self.config.max_position_embeddings],
                Dtype.INT32,
                name='input_ids')
            _ = self.text_model(inputs)
        return self.aipu_g
