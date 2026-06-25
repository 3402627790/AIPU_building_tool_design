from typing import Tuple, Optional
import numpy as np
import torch
from AIPUBuilder.core import *  # noqa
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, register_update_config
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


@register_update_config(['bge_m3'])
def bgem3_update_config(config):
    if not hasattr(config, 'num_key_value_heads') and hasattr(config, 'num_attention_heads'):
        config.num_key_value_heads = config.num_attention_heads


@register_causallm(['bge_m3_embedding'])
class BgeM3EmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.model = BgeM3Embeddings(config, self.config.get('weights'))

    def forward(self, seq_len=None, use_cache=None):
        seq_len = self.config.max_position_embeddings if seq_len is None else seq_len
        self.config.seq_len = seq_len
        model_name = 'bge_m3_embedding'

        with Graph(model_name) as self.aipu_g:

            hidden_states = ops.input([1, seq_len], Dtype.INT32, name='input_ids')
            position_ids = ops.input([1, seq_len], Dtype.INT32, name='position_ids')
            hidden_states = self.model(hidden_states, position_ids)

        return self.aipu_g


class BgeM3Embeddings(nn.Module):
    def __init__(self, clip_config, weights):
        super().__init__()

        self.embed_dim = clip_config.hidden_size
        self.max_position_embeddings = clip_config.max_position_embeddings
        self.token_embedding = nn.Embedding(clip_config.vocab_size, self.embed_dim)
        self.token_embedding.weights.betensor = weights.get(f'embeddings.word_embeddings.weight')['data'].float()
        self.token_embedding.weights.sync_betensor()

        self.position_embedding = nn.Embedding(self.max_position_embeddings, self.embed_dim)

        # self.position_embedding = nn.Constant(torch.randn((self.max_position_embeddings, self.embed_dim)).float())
        self.position_embedding.weights.betensor = weights.get(f'embeddings.position_embeddings.weight')['data'].float()
        self.position_embedding.weights.sync_betensor()

        self.token_type_embedding = nn.Constant(torch.randn((1, self.embed_dim)).float())
        self.token_type_embedding.weights.betensor = weights.get(
            f'embeddings.token_type_embeddings.weight')['data'].float()
        self.token_type_embedding.weights.sync_betensor()

        self.emb_layer_norm = nn.LayerNorm(self.embed_dim, eps=clip_config.layer_norm_eps)
        norm_w = weights.get(f'embeddings.LayerNorm.weight')
        norm_b = weights.get(f'embeddings.LayerNorm.bias')
        update_tensor_data(self.emb_layer_norm.weights, norm_w)
        update_tensor_data(self.emb_layer_norm.biases, norm_b)

    def forward(self, input_ids: Tensor, postion_ids: Tensor) -> Tensor:

        inputs_embeds = self.token_embedding(input_ids)
        embeddings = inputs_embeds + self.token_type_embedding()
        embeddings = embeddings + self.position_embedding(postion_ids)
        embeddings = self.emb_layer_norm(embeddings)
        return embeddings


class BgeM3Attention(nn.Module):
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

        self.q_proj.weights.betensor = weights[f'encoder.layer.{layer_id}.attention.self.query.weight'].float()
        self.q_proj.weights.sync_betensor()
        self.q_proj.biases.betensor = weights[f'encoder.layer.{layer_id}.attention.self.query.bias'].float()
        self.q_proj.biases.sync_betensor()

        self.k_proj.weights.betensor = weights[f'encoder.layer.{layer_id}.attention.self.key.weight'].float()
        self.k_proj.weights.sync_betensor()
        self.k_proj.biases.betensor = weights[f'encoder.layer.{layer_id}.attention.self.key.bias'].float()
        self.k_proj.biases.sync_betensor()

        self.v_proj.weights.betensor = weights[f'encoder.layer.{layer_id}.attention.self.value.weight'].float()
        self.v_proj.weights.sync_betensor()
        self.v_proj.biases.betensor = weights[f'encoder.layer.{layer_id}.attention.self.value.bias'].float()
        self.v_proj.biases.sync_betensor()

        self.o_proj.weights.remove_betensor()
        self.o_proj.biases.remove_betensor()
        self.o_proj.weights.betensor = weights[f'encoder.layer.{layer_id}.attention.output.dense.weight'].float()
        self.o_proj.weights.sync_betensor()
        self.o_proj.biases.betensor = weights[f'encoder.layer.{layer_id}.attention.output.dense.bias'].float()
        self.o_proj.biases.sync_betensor()

        self.q_proj.weights.name = f"{prefix}encoder.layer.{layer_id}.self_attn.q_proj.weight"
        self.q_proj.biases.name = f"{prefix}encoder.layer.{layer_id}.self_attn.q_proj.bias"
        self.k_proj.weights.name = f'{prefix}encoder.layer.{layer_id}.self_attn.k_proj.weight'
        self.k_proj.biases.name = f'{prefix}encoder.layer.{layer_id}.self_attn.k_proj.bias'
        self.v_proj.weights.name = f'{prefix}encoder.layer.{layer_id}.self_attn.v_proj.weight'
        self.v_proj.biases.name = f'{prefix}encoder.layer.{layer_id}.self_attn.v_proj.bias'
        self.o_proj.weights.name = f'{prefix}encoder.layer.{layer_id}.self_attn.o_proj.weight'
        self.o_proj.biases.name = f'{prefix}encoder.layer.{layer_id}.self_attn.o_proj.bias'

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}encoder.layer.{layer_id}.self_attn"

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

        softmax_masked = self.softmax(kq_matmul + attention_mask)

        reshaped_trans_value = self.qkv_transpose(self.qk_reshape(value_states))

        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs*seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        attn_output = self.o_proj(reshaped_trans)
        attn_output = ops.reshape(attn_output, [bs, seq_len, self.hidden_size])
        return attn_output


class BgeM3MLP(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        self.gate_proj.weights.betensor = weights[f'encoder.layer.{layer_id}.intermediate.dense.weight'].float()
        self.gate_proj.biases.betensor = weights[f'encoder.layer.{layer_id}.intermediate.dense.bias'].float()

        self.down_proj.weights.betensor = weights[f'encoder.layer.{layer_id}.output.dense.weight'].float()
        self.down_proj.biases.betensor = weights[f'encoder.layer.{layer_id}.output.dense.bias'].float()

        self.gate_proj.weights.sync_betensor()
        self.down_proj.weights.sync_betensor()
        self.gate_proj.biases.sync_betensor()
        self.down_proj.biases.sync_betensor()

        self.gate_proj.weights.name = f'{prefix}encoder.layer.{layer_id}.mlp.fc1.weight'
        self.down_proj.weights.name = f'{prefix}encoder.layer.{layer_id}.mlp.fc2.weight'
        self.gate_proj.biases.name = f'{prefix}encoder.layer.{layer_id}.mlp.fc1.biases'
        self.down_proj.biases.name = f'{prefix}encoder.layer.{layer_id}.mlp.fc2.biases'

    def forward(self, x):
        x_shape = x.shape
        hidden_states = ops.reshape(x, [-1, x_shape[2]])
        hidden_states = self.gate_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        mlp_out = self.down_proj(hidden_states)

        return ops.reshape(mlp_out, x_shape)


class BgeM3EncoderLayer(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm1.weights.betensor = weights[f'encoder.layer.{layer_id}.attention.output.LayerNorm.weight'].float(
        )
        self.layer_norm1.biases.betensor = weights[f'encoder.layer.{layer_id}.attention.output.LayerNorm.bias'].float()
        self.layer_norm1.weights.sync_betensor()
        self.layer_norm1.biases.sync_betensor()

        self.self_attn = BgeM3Attention(config, weights, layer_id, prefix='')
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.layer_norm2.weights.betensor = weights[f'encoder.layer.{layer_id}.output.LayerNorm.weight'].float()
        self.layer_norm2.biases.betensor = weights[f'encoder.layer.{layer_id}.output.LayerNorm.bias'].float()
        self.layer_norm2.weights.sync_betensor()
        self.layer_norm2.biases.sync_betensor()
        self.mlp = BgeM3MLP(config, weights, layer_id, prefix='')

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

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        residual = hidden_states

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        outputs = (hidden_states,)

        return outputs


class BgeM3Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`BgeM3EncoderLayer`].

    Args:
        config: BgeM3Config
    """

    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layers = [BgeM3EncoderLayer(config, weights, idx, prefix)
                       for idx in gen_layer_ids(config.num_hidden_layers)]

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

        # encoder_states = encoder_states + (hidden_states,)
        return hidden_states


class BgeM3Transformer(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.encoder = BgeM3Encoder(config, weights)
        self.pooler = nn.Linear(embed_dim, embed_dim)
        self.pooler.weights.betensor = weights['pooler.dense.weight'].float()
        self.pooler.biases.betensor = weights['pooler.dense.bias'].float()
        self.pooler.weights.sync_betensor()
        self.pooler.biases.sync_betensor()

    def forward(
        self,
        hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        return last_hidden_state


@register_causallm(['bge_m3'])
class BgeM3CausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})

        self.aipu_g = None
        self.prefix = prefix
        self.model = BgeM3Transformer(self.config, weights, prefix)

    def forward(self):
        seq_len = self.config.get('seq_len')
        batch = self.config.get('batch', 1)
        model_name = 'bge_m3'
        with Graph(model_name) as self.aipu_g:
            hidden_states = ops.input(
                [batch, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded')

            attention_mask = ops.input([batch, 1, seq_len, seq_len],
                                       Dtype.FP32, name='attention_mask')

            hidden_states = self.model(hidden_states, attention_mask)

        return self.aipu_g
