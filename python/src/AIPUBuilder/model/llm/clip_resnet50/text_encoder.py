import numpy as np
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...utils import update_tensor_data
from ...register import register_causallm, register_update_config
from ..llmutils import gen_layer_ids


class MultiheadAttention(nn.Module):
    def __init__(self, config, weights, prefix='', layer_id=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        inproj_weight = weights.get(f'{prefix}{layer_id}.attn.in_proj_weight')['data']
        inproj_bias = weights.get(f'{prefix}{layer_id}.attn.in_proj_bias')['data']

        outproj_weight = weights.get(f'{prefix}{layer_id}.attn.out_proj.weight')['data']
        outproj_bias = weights.get(f'{prefix}{layer_id}.attn.out_proj.bias')['data']

        update_tensor_data(self.q_proj.weights, inproj_weight[:self.hidden_size, :],
                           f"{prefix}{layer_id}.attn.q_weight")
        update_tensor_data(self.q_proj.biases, inproj_bias[:self.hidden_size],
                           f"{prefix}{layer_id}.attn.q_bias")

        update_tensor_data(self.k_proj.weights, inproj_weight[self.hidden_size:self.hidden_size*2, :],
                           f"{prefix}{layer_id}.attn.k_weight")
        update_tensor_data(self.k_proj.biases, inproj_bias[self.hidden_size:self.hidden_size*2],
                           f"{prefix}{layer_id}.attn.k_bias")

        update_tensor_data(self.v_proj.weights, inproj_weight[self.hidden_size*2:self.hidden_size*3, :],
                           f"{prefix}{layer_id}.attn.v_weight")
        update_tensor_data(self.v_proj.biases, inproj_bias[self.hidden_size*2:self.hidden_size*3],
                           f"{prefix}{layer_id}.attn.v_bias")

        update_tensor_data(self.o_proj.weights, outproj_weight, f"{prefix}{layer_id}.attn.o_weight")
        update_tensor_data(self.o_proj.biases, outproj_bias, f"{prefix}{layer_id}.attn.o_bias")

        scale = 1. / np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=False, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul(trans_a=False, trans_b=False)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, hidden_states, attn_mask):
        bs_sym, seq_len_sym, _ = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_sym * seq_len_sym, self.hidden_size])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = ops.reshape(query_states, [bs_sym, seq_len_sym, self.num_attention_heads, self.head_dim])
        key_states = ops.reshape(key_states, [bs_sym, seq_len_sym, self.num_attention_heads, self.head_dim])
        value_states = ops.reshape(value_states, [bs_sym, seq_len_sym, self.num_attention_heads, self.head_dim])

        query_states = ops.transpose(query_states, [0, 2, 1, 3])  # [N, head, seq, dim]
        key_states = ops.transpose(key_states, [0, 2, 3, 1])  # [N, head, dim, seq]
        value_states = ops.transpose(value_states, [0, 2, 1, 3])  # [N, head, seq, dim]

        kq_matmul = self.kq_matmul(query_states, key_states)
        kq_matmul = kq_matmul + attn_mask
        kq_softmax = self.softmax(kq_matmul)
        qkv = self.kqv_matmul(kq_softmax, value_states)

        qkv = ops.transpose(qkv, [0, 2, 1, 3])  # [N, seq, head, dim]
        qkv = ops.reshape(qkv, [bs_sym * seq_len_sym, self.hidden_size])
        qkv = self.o_proj(qkv)
        qkv = ops.reshape(qkv, [bs_sym, seq_len_sym, self.hidden_size])

        return qkv


class MLP(nn.Module):
    def __init__(self, config, weights, prefix='', layer_id=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.c_fc = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.c_proj = nn.Linear(self.hidden_size * 4, self.hidden_size)
        # config.act_fn is quick_gelu, but quick_gelu generates mul and sigmoid operators
        # For performance, we will use “gelu_pytorch_tanh” instead of “quick_gelu” currently
        self.act = nn.ACT2FN["gelu_pytorch_tanh"]

        update_tensor_data(self.c_fc.weights, weights.get(f"{prefix}{layer_id}.mlp.c_fc.weight"),
                           f"{prefix}{layer_id}.mlp.c_fc.weight")
        update_tensor_data(self.c_fc.biases, weights.get(f"{prefix}{layer_id}.mlp.c_fc.bias"),
                           f"{prefix}{layer_id}.mlp.c_fc.bias")

        update_tensor_data(self.c_proj.weights, weights.get(f"{prefix}{layer_id}.mlp.c_proj.weight"),
                           f"{prefix}{layer_id}.mlp.c_proj.weight")
        update_tensor_data(self.c_proj.biases, weights.get(f"{prefix}{layer_id}.mlp.c_proj.bias"),
                           f"{prefix}{layer_id}.mlp.c_proj.bias")

    def forward(self, x):
        bs_sym, seq_len_sym, _ = x.shape.symbol
        x = ops.reshape(x, [bs_sym * seq_len_sym, self.hidden_size])
        fc1 = self.c_fc(x)
        act = self.act(fc1)
        out = self.c_proj(act)
        out = ops.reshape(out, [bs_sym, seq_len_sym, self.hidden_size])
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, config, weights, prefix='', layer_id=0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.attn = MultiheadAttention(config, weights, prefix=prefix, layer_id=layer_id)
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        update_tensor_data(self.ln_1.weights, weights.get(
            f"{prefix}{layer_id}.ln_1.weight"), f"{prefix}{layer_id}.ln_1.weight")
        update_tensor_data(self.ln_1.biases, weights.get(
            f'{prefix}{layer_id}.ln_1.bias'), f'{prefix}{layer_id}.ln_1.bias')

        self.mlp = MLP(config, weights, prefix=prefix, layer_id=layer_id)

        self.ln_2 = nn.LayerNorm(self.hidden_size)
        update_tensor_data(self.ln_2.weights, weights.get(f"{prefix}{layer_id}.ln_2.weight"),
                           f"{prefix}{layer_id}.ln_2.weight")
        update_tensor_data(self.ln_2.biases, weights.get(f'{prefix}{layer_id}.ln_2.bias'),
                           f'{prefix}{layer_id}.ln_2.bias')

    def forward(self, x, attn_mask):
        ln1_out = self.ln_1(x)
        attn_out = self.attn(ln1_out, attn_mask)
        x = x + attn_out

        ln2_out = self.ln_2(x)
        mlp_out = self.mlp(ln2_out)
        x = x + mlp_out

        return x


class Transformer(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.layers = []

        for idx in gen_layer_ids(config.num_hidden_layers):
            self.layers.append(ResidualAttentionBlock(config, weights, prefix=f'{prefix}resblocks.', layer_id=idx))

    def forward(self, x, attn_mask):
        for idx in range(self.num_hidden_layers):
            x = self.layers[idx](x, attn_mask)
        return x


class TextEncoder(nn.Module):
    def __init__(self, config, weights, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.token_embedding_weight = weights.get("token_embedding.weight")['data']
        self.positional_embedding_weight = weights.get("positional_embedding")['data']
        self.text_projection = weights.get("text_projection")['data']
        self.transformer = Transformer(config, weights, prefix=f'{prefix}transformer.')
        self.ln_final = nn.LayerNorm(self.hidden_size)

        update_tensor_data(self.ln_final.weights, weights.get(f"ln_final.weight"), f"ln_final.weight")
        update_tensor_data(self.ln_final.biases, weights.get(f'ln_final.bias'), f'ln_final.bias')

        self.matmul = nn.MatMul()

    def forward(self, token, attention_mask, offset):
        bs_sym, seq_len_sym = token.shape.symbol
        token_embedding = ops.constant(Tensor(self.token_embedding_weight))
        token_embedding.op.constants['weights'].name = f"token_embedding.weight"
        token_embedding.op.name = f"token_embedding.weight"
        token_embedding_out = ops.gather(token_embedding, token, 0, 0)
        token_embedding_out.name = f"token_embedding.out"

        positional_embedding = ops.constant(Tensor(self.positional_embedding_weight))
        positional_embedding.op.constants['weights'].name = f"positional_embedding.weight"
        positional_embedding.op.name = f"positional_embedding.weight"
        positional_embedding_out = ops.crop(positional_embedding, crops=[[0, seq_len_sym], [0, self.hidden_size]])

        x = token_embedding_out + positional_embedding_out
        x = self.transformer(x, attention_mask)
        x = self.ln_final(x)

        x = ops.gather(x, offset, 1, 0)
        text_projection = ops.constant(Tensor(self.text_projection))
        text_projection.op.constants['weights'].name = f"text_projection.weight"
        text_projection.op.name = f"text_projection.weight"

        dim0_sym, dim1_sym, dim2_sym, _ = x.shape.symbol
        x = ops.reshape(x, [dim0_sym * dim1_sym * dim2_sym, self.hidden_size])
        out = self.matmul(x, text_projection)

        return out


@register_update_config(['text_encoder'])
def ClipText_update_config(config):
    _local_vision_config = {
        "num_attention_heads": config.model_cfg['text_cfg']["heads"],
        "hidden_size": config.model_cfg['text_cfg']["width"],
        "num_hidden_layers": config.model_cfg['text_cfg']["layers"],
        "max_position_embeddings": config.model_cfg['text_cfg']["context_length"],
        "act_fn": "quick_gelu",
    }
    config.update(_local_vision_config)


@register_causallm(['text_encoder'])
class clip_text(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.config = config
        self.aipu_g = None

        self.text_encoder = TextEncoder(config, weights, prefix)
        self.max_position_embeddings = config.max_position_embeddings

    def forward(self):
        seq_len = self.config.seq_len
        batch = self.config.get('batch', 1)
        with Graph('clip_text') as self.aipu_g:
            input_id = ops.input([batch, seq_len], Dtype.INT32, name='token', exprs=['batch', 'seq_len'])
            attention_mask = ops.input([batch, 1, seq_len, seq_len],
                                       Dtype.FP32,
                                       name='attention_mask',
                                       exprs=['batch', 1, 'seq_len', 'seq_len'])
            offset = ops.input([batch, 1], Dtype.INT32, name='offset', exprs=['batch', 1])
            output = self.text_encoder(input_id, attention_mask, offset)
            self.aipu_g.output_tensors = TensorList([output])
        return self.aipu_g
