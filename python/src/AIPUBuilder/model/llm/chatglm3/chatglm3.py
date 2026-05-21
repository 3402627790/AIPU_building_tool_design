import numpy as np
import torch
from AIPUBuilder.logger import INFO
from AIPUBuilder.core import *  # noqa
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


@register_update_config('chatglm')
def chatglm3_update_config(config):
    config.vocab_size = config.padded_vocab_size
    config.intermediate_size = config.ffn_hidden_size
    config.num_hidden_layers = config.num_layers
    config.num_key_value_heads = config.multi_query_group_num


@register_causallm(['chatglm_embedding', 'chatglm3_embedding'])
class ChatGlmEmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('transformer.embedding.word_embeddings.weight')


class Chatglm3RotaryEmbedding(nn.Module):
    def __init__(self, num_heads, num_key_value_heads, dim, max_position_embeddings=2048, base=10000, device=None, name=''):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_key_value_heads = num_key_value_heads
        self.base = base
        self.name = name
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                               dtype=torch.int64).float().to(device) / self.dim))

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        self.cos_cached = emb.cos().to(torch.get_default_dtype())
        self.sin_cached = emb.sin().to(torch.get_default_dtype())

    def forward(self, pos_ids):
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        bs_symbol, seq_len_symbol = pos_ids.shape.symbol
        sin_out_reshape = ops.reshape(gather_sin_out, [bs_symbol, 1, seq_len_symbol, self.num_heads, 1])
        cos_out_reshape = ops.reshape(gather_cos_out, [bs_symbol, 1, seq_len_symbol, self.num_heads, 1])
        sin_reshape_tile_q = ops.tile(sin_out_reshape, [1, self.num_heads, 1, 1, 2])
        cos_reshape_tile_q = ops.tile(cos_out_reshape, [1, self.num_heads, 1, 1, 2])
        sin_reshape_tile_k = ops.tile(sin_out_reshape, [1, self.num_key_value_heads, 1, 1, 2])
        cos_reshape_tile_k = ops.tile(cos_out_reshape, [1, self.num_key_value_heads, 1, 1, 2])

        if self.name != '':
            sin_out.op.constants['weights'].name = f'{self.name}.rotary_emb.sin_weight'
            cos_out.op.constants['weights'].name = f'{self.name}.rotary_emb.cos_weight'
            sin_out.op.name = f"{self.name}.rotary_emb.sin_{sin_out.op.name}"
            cos_out.op.name = f"{self.name}.rotary_emb.cos_{cos_out.op.name}"
            gather_sin_out.name = f"{self.name}.{gather_sin_out.name}"
            gather_cos_out.name = f"{self.name}.{gather_cos_out.name}"

        return cos_reshape_tile_q, sin_reshape_tile_q, cos_reshape_tile_k, sin_reshape_tile_k


class Chatglm3MLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.ffn_hidden_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN["silu"]

        mlp_in_w = weights[f'transformer.encoder.layers.{layer_id}.mlp.dense_h_to_4h.weight']
        gate_proj_w, up_proj_w = torch.chunk(mlp_in_w, 2, dim=0)
        down_proj_w = weights[f'transformer.encoder.layers.{layer_id}.mlp.dense_4h_to_h.weight']

        update_tensor_data(self.gate_proj.weights, gate_proj_w, name=f'{prefix}layers.{layer_id}.mlp.gate_proj.weight')
        update_tensor_data(self.up_proj.weights, up_proj_w, name=f'{prefix}layers.{layer_id}.mlp.up_proj.weight')
        update_tensor_data(self.down_proj.weights, down_proj_w, name=f'{prefix}layers.{layer_id}.mlp.down_proj.weight')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Chatglm3Attention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.multi_query_group_num
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.seq_length
        self.rope_theta = 10000
        self.is_causal = True
        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        qkv_w_name = f'transformer.encoder.layers.{layer_id}.self_attention.query_key_value.weight'
        qkv_w = weights[qkv_w_name]
        q_proj_w = qkv_w[:self.q_proj.out_features]
        k_proj_w = qkv_w[self.q_proj.out_features:self.q_proj.out_features + self.k_proj.out_features]
        v_proj_w = qkv_w[self.q_proj.out_features + self.k_proj.out_features:]

        qkv_b_name = f'transformer.encoder.layers.{layer_id}.self_attention.query_key_value.bias'
        qkv_b = weights[qkv_b_name]
        q_proj_b, k_proj_b, v_proj_b = None, None, None
        if len(qkv_b):
            q_proj_b = qkv_b[:self.q_proj.out_features]
            k_proj_b = qkv_b[self.q_proj.out_features:self.q_proj.out_features + self.k_proj.out_features]
            v_proj_b = qkv_b[self.q_proj.out_features + self.k_proj.out_features:]

        update_tensor_data(self.q_proj.weights, q_proj_w, f"{prefix}layers.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q_proj.biases, q_proj_b, f"{prefix}layers.{layer_id}.self_attn.q_proj.bias")

        update_tensor_data(self.k_proj.weights, k_proj_w, f"{prefix}layers.{layer_id}.self_attn.k_proj.weight")
        update_tensor_data(self.k_proj.biases, k_proj_b, f"{prefix}layers.{layer_id}.self_attn.k_proj.bias")

        update_tensor_data(self.v_proj.weights, v_proj_w, f"{prefix}layers.{layer_id}.self_attn.v_proj.weight")
        update_tensor_data(self.v_proj.biases, v_proj_b, f"{prefix}layers.{layer_id}.self_attn.v_proj.bias")

        o_proj_w = weights.get(f'transformer.encoder.layers.{layer_id}.self_attention.dense.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.self_attn.o_proj.weight")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"
        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape
        bs_symbol, seq_len_symbol, _ = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symbol * seq_len_symbol, hidden_size])
        self.rotary_emb = Chatglm3RotaryEmbedding(self.num_heads,
                                                  self.num_key_value_heads,
                                                  self.head_dim // 2,
                                                  max_position_embeddings=seq_len,
                                                  base=self.rope_theta,
                                                  name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        # crop0 = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [0, self.head_dim//4], [1, 2]])
        # crop1 = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [0, self.head_dim//4], [0, 1]])
        # crop2 = nn.Crop([[0, bs], [0, self.num_key_value_heads], [0, seq_len], [0, self.head_dim//4], [1, 2]])
        # crop3 = nn.Crop([[0, bs], [0, self.num_key_value_heads], [0, seq_len], [0, self.head_dim//4], [0, 1]])

        crop0 = nn.Crop([[0, None], [0, None], [0, None], [0, self.head_dim//4], [1, 2]])
        crop1 = nn.Crop([[0, None], [0, None], [0, None], [0, self.head_dim//4], [0, 1]])

        crop2 = nn.Crop([[0, None], [0, None], [0, self.num_key_value_heads * self.head_dim//4], [1, 2]])
        crop3 = nn.Crop([[0, None], [0, None], [0, self.num_key_value_heads * self.head_dim//4], [0, 1]])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        cos_emb_q, sin_emb_q, cos_emb_k, sin_emb_k = self.rotary_emb(position_ids)

        self.q_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_key_value_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.q_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.kv_reshape(key_states))

        q_crop = nn.Crop([[0, None], [0, None], [0, None], [0, self.head_dim // 2]])
        q_crop_pass = nn.Crop([[0, None], [0, None], [0, None], [self.head_dim // 2, self.head_dim]])
        k_crop = nn.Crop([[0, None], [0, None], [0, None], [0, self.head_dim // 2]])
        k_crop_pass = nn.Crop([[0, None], [0, None], [0, None], [self.head_dim // 2, self.head_dim]])
        # q_crop_reshape = nn.Reshape([bs, self.num_heads, seq_len, self.head_dim // 4, 2])
        # k_crop_reshape = nn.Reshape([bs, self.num_key_value_heads, seq_len, self.head_dim // 4, 2])

        q_crop_reshape = nn.Reshape([bs_symbol, self.num_heads, seq_len_symbol, self.head_dim // 4, 2])
        k_crop_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_key_value_heads * self.head_dim // 4, 2])

        # query
        query_reshaped = q_crop(reshaped_trans_query)
        # query_reshaped_t = nn.Transpose([0, 2, 1, 3])(query_reshaped)
        # query_clip = q_crop_reshape(query_reshaped_t)
        query_clip = q_crop_reshape(query_reshaped)
        query_pass = q_crop_pass(reshaped_trans_query)
        q_crop_out0 = crop0(query_clip)
        q_crop_out1 = crop1(query_clip)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])
        # rope_query = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim // 4, 2])(rope_query)
        # rope_query = nn.Transpose([0, 2, 1, 3, 4])(rope_query)

        # key
        key_reshaped = k_crop(reshaped_trans_key)
        key_reshaped_t = nn.Transpose([0, 2, 1, 3])(key_reshaped)
        key_clip = k_crop_reshape(key_reshaped_t)
        key_pass = k_crop_pass(reshaped_trans_key)
        k_crop_out0 = crop2(key_clip)
        k_crop_out1 = crop3(key_clip)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])
        rope_key = nn.Reshape([bs_symbol, seq_len_symbol, self.num_key_value_heads, self.head_dim // 4, 2])(rope_key)
        rope_key = nn.Transpose([0, 2, 1, 3, 4])(rope_key)

        neg_q_crop_out0.op.name = f"{self.matmul_name}.negative_0"
        neg_k_crop_out0.op.name = f"{self.matmul_name}.negative_1"

        q_cosine = nn.Reshape([bs_symbol, self.num_heads, seq_len_symbol,
                              self.head_dim // 4, 2])(query_reshaped) * cos_emb_q
        k_cosine = nn.Reshape([bs_symbol, self.num_key_value_heads, seq_len_symbol,
                              self.head_dim // 4, 2])(key_reshaped) * cos_emb_k
        q_r_sine = rope_query * sin_emb_q
        k_r_sine = rope_key * sin_emb_k

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine

        q_add_out = nn.Reshape([bs_symbol, self.num_heads, seq_len_symbol, self.head_dim // 2])(q_add_out)
        k_add_out = nn.Reshape([bs_symbol, self.num_key_value_heads, seq_len_symbol, self.head_dim // 2])(k_add_out)

        q_out = nn.Concat(-1)([q_add_out, query_pass])
        k_out = nn.Concat(-1)([k_add_out, key_pass])

        k_cache_out = k_cache(k_out, position_ids=position_ids, block_idx=self.layer_id,
                              name=f"past_key_value.{self.layer_id}.key")
        k_cache_out = k_cache_out[0]

        if self.num_key_value_groups > 1:
            if not self.config.get('broadcast_matmul', False):
                batch, _, slen, _ = k_cache_out.shape.symbol
                k_cache_out = k_cache_out.reshape([batch, self.num_key_value_heads, 1, slen, self.head_dim])
                k_cache_out = ops.tile(k_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
                k_cache_out = k_cache_out.reshape(
                    [batch, self.num_key_value_heads * self.num_key_value_groups, slen, self.head_dim])
            else:
                # broadcast ds matmul is not supported by gb when use_cache=true
                # [CP-23956]: insert reshape to use kq_matmul's broadcast feature to eliminate tile op for k_out, kq_matmul will broadcast k_out's second dim to num_key_value_groups
                batch, _, slen, _ = k_cache_out.shape.symbol
                k_cache_out = k_cache_out.reshape([batch * self.num_key_value_heads, 1, slen, self.head_dim])
                b, _, slen, _ = q_out.shape.symbol
                q_out = q_out.reshape([b * self.num_key_value_heads, self.num_key_value_groups, slen, self.head_dim])

        if seq_len == 1:
            # [CP-19313]C=A*B=(B^T*A^T)^T, when B size >> A size, swap input can improve wtram reuse
            kq_matmul = nn.Transpose([0, 1, 3, 2])(self.kq_matmul(k_cache_out, q_out))
        else:
            kq_matmul = self.kq_matmul(q_out, k_cache_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        softmax_masked = self.softmax(kq_matmul + attention_mask)

        if seq_len == 1:
            reshaped_trans_value = nn.Reshape([bs_symbol, self.num_key_value_heads,
                                              seq_len_symbol, self.head_dim])(value_states)
        else:
            reshaped_trans_value = self.qkv_transpose(self.kv_reshape(value_states))

        v_cache_out = v_cache(reshaped_trans_value, position_ids=position_ids,
                              block_idx=self.layer_id, name=f"past_key_value.{self.layer_id}.value")
        v_cache_out = v_cache_out[0]

        if self.num_key_value_groups > 1:
            if not self.config.get('broadcast_matmul', False):
                batch, _, slen, _ = v_cache_out.shape.symbol
                v_cache_out = v_cache_out.reshape([batch, self.num_key_value_heads, 1, slen, self.head_dim])
                v_cache_out = ops.tile(v_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
                v_cache_out = v_cache_out.reshape(
                    [batch, self.num_key_value_heads * self.num_key_value_groups, slen, self.head_dim])
            else:
                # [CP-23956]: insert reshape to use kqv_matmul's broadcast feature to eliminate tile op for reshaped_trans_value
                batch, num_key_value_heads, slen, head_dim = v_cache_out.shape.symbol
                v_cache_out = v_cache_out.reshape([bs_symbol * num_key_value_heads, 1, slen, head_dim])

        kqv_out = self.kqv_matmul(softmax_masked, v_cache_out)
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        # [CP-23956]: recover batch dim
        kqv_out = kqv_out.reshape(bs_symbol, self.num_key_value_heads *
                                  self.num_key_value_groups, seq_len_symbol, self.head_dim)

        self.qkv_reshape = nn.Reshape([bs_symbol * seq_len_symbol, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.o_proj(reshaped_trans)

        return o_t


class Chatglm3Decoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'transformer.encoder.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = Chatglm3Attention(config, weights, layer_id, prefix)

        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(
                               f'transformer.encoder.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = Chatglm3MLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name="attnblk")
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            """
            when enable_subgraph=true, will do some special operations:
            1. avoiding the constant output tensor is subgraph's input tensor, will re-generate one constant in subgraph.
            2. if one tensor is multi ops input, the subgraph will insert the duplicate input tensor into subgraph input tensors,
            which will cause big change for llm runtime. in order to avoid this graph, will insert reshape op, its output tensors will
            be as multi ops input, and reshape is including in this subgraph.
            """
            hidden_states = ops.reshape(hidden_states, hidden_states.shape.symbol)
            attention_mask = ops.tile(attention_mask, [1, self.config.num_attention_heads, 1, 1])
            position_ids = ops.reshape(position_ids, position_ids.shape.symbol)

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       k_cache,
                                       v_cache,
                                       use_cache)
        # hidden_states = residual + hidden_states
        # eltwise faster than add
        bs_symbol, seq_len_symbol, _ = residual.shape.symbol
        hidden_states = hidden_states.reshape([bs_symbol, seq_len_symbol, hidden_states.shape[1]])
        hidden_states = ops.elementwise_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = hidden_states.reshape([bs_symbol * seq_len_symbol, hidden_states.shape[-1]])
        hidden_states = self.mlp(hidden_states)
        # hidden_states = residual + hidden_states
        hidden_states = hidden_states.reshape([bs_symbol, seq_len_symbol, hidden_states.shape[1]])
        hidden_states = ops.elementwise_add(residual, hidden_states)

        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class NormLmHead(nn.Module):
    def __init__(self, config, weights):
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        update_tensor_data(self.norm.weights, weights.get(
            f'transformer.encoder.final_layernorm.weight'), f'normlm.norm.weight')

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        key = 'transformer.output_layer.weight'
        if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
            key = 'transformer.embedding.word_embeddings.weight'
        update_tensor_data(self.lm_head.weights, weights.get(key), f'normlm.lm.weight')

    @gen_subgraph(subgraph_name='lm')
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        h_shape = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [h_shape[0] * h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        out = ops.reshape(lm_head, [h_shape[0], h_shape[1], lm_head.shape[-1]])
        return out


class Chatglm3Model(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.padded_vocab_size
        self.prefix = prefix
        # config.num_layers = 1
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights[f'transformer.embedding.word_embeddings.weight'].float()

        self.decoders = [Chatglm3Decoder(config, weights, idx, prefix) for idx in gen_layer_ids(config.num_layers)]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key=None,
                past_value=None,
                use_cache=None,
                with_embed=True,
                with_preprocess_mask=False,
                ):

        if past_key is None or past_value is None:
            ERROR(f"please provide the correct kv cache input in llama2.")

        # input_embeds = self.embed_tokens(input_ids)
        if with_embed:
            embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
                                       self.model_config.padded_vocab_size, self.model_config.hidden_size])
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        else:
            input_embeds = input_ids

        hidden_states = input_embeds
        for i, decoder in enumerate(self.decoders):
            hidden_states, _, _ = decoder(hidden_states,
                                          attention_mask,
                                          position_ids,
                                          past_key,
                                          past_value,
                                          use_cache=use_cache)
        return hidden_states


def convert_glm_tensor_to_qwen(src, tgt):
    tgt['model.norm.weight'] = src['transformer.encoder.final_layernorm.weight']
    tgt['lm_head.weight'] = src['transformer.output_layer.weight']
    tgt['model.embed_tokens.weight'] = src['transformer.embedding.word_embeddings.weight']

    for i in range(28):
        # prenorm
        pre_norm = src[f'transformer.encoder.layers.{i}.input_layernorm.weight']
        tgt[f'model.layers.{i}.input_layernorm.weight'] = pre_norm
        # postnorm
        post_norm = src[f'transformer.encoder.layers.{i}.post_attention_layernorm.weight']
        tgt[f'model.layers.{i}.post_attention_layernorm.weight'] = post_norm
        # qkv
        qkv_w = src[f'transformer.encoder.layers.{i}.self_attention.query_key_value.weight'].clone()
        qkv_b = src[f'transformer.encoder.layers.{i}.self_attention.query_key_value.bias'].clone()
        q_w = qkv_w[:4096]
        k_w = qkv_w[4096:4096+256]
        v_w = qkv_w[-256:]
        q_b = qkv_b[:4096]
        k_b = qkv_b[4096:4096+256]
        v_b = qkv_b[-256:]
        tgt[f'model.layers.{i}.self_attn.q_proj.bias'] = q_b
        tgt[f'model.layers.{i}.self_attn.q_proj.weight'] = q_w
        tgt[f'model.layers.{i}.self_attn.k_proj.bias'] = k_b
        tgt[f'model.layers.{i}.self_attn.k_proj.weight'] = k_w
        tgt[f'model.layers.{i}.self_attn.v_proj.bias'] = v_b
        tgt[f'model.layers.{i}.self_attn.v_proj.weight'] = v_w
        # o
        o_w = src[f'transformer.encoder.layers.{i}.self_attention.dense.weight']
        tgt[f'model.layers.{i}.self_attn.o_proj.weight'] = o_w
        # mlp
        mlp_in_w = src[f'transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight'].clone()
        gate_w, up_w = torch.chunk(mlp_in_w, 2, dim=0)
        tgt[f'model.layers.{i}.mlp.up_proj.weight'] = up_w
        tgt[f'model.layers.{i}.mlp.gate_proj.weight'] = gate_w
        down_w = src[f'transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight']
        tgt[f'model.layers.{i}.mlp.down_proj.weight'] = down_w


@register_causallm(['chatglm3', 'chatglm'])
class Chatglm3CausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()

        config.vocab_size = config.padded_vocab_size
        config.intermediate_size = config.ffn_hidden_size
        config.num_hidden_layers = config.num_layers
        config.num_key_value_heads = config.multi_query_group_num

        self.model = Chatglm3Model(config, weights, prefix)
        self.norm_and_lm_head = NormLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.get('seq_len')
        past_len = self.config.get('past_len')
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        with Graph('qwen') as self.aipu_g:
            if with_embed:
                hidden_states = ops.input(
                    [batch, seq_len], Dtype.FP32, name='input_ids', exprs=['batch', 'seq_len'])
            else:
                hidden_states = ops.input([batch, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded', exprs=[
                                          'batch', 'seq_len', self.config.hidden_size])

            attention_mask = ops.input(
                [batch, 1, seq_len, seq_len + past_len], Dtype.FP32, name='attention_mask',
                exprs=['batch', 1, 'seq_len', 'max_len'])
            position_ids = ops.input([batch, seq_len], Dtype.INT32, name='position_ids',
                                     exprs=['batch', 'seq_len'])

            kcache_module = nn.KVCacheModule(self.config, name='kcache')
            vcache_module = nn.KVCacheModule(self.config, name='vcache')

            hidden_states = self.model(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       past_key=kcache_module,
                                       past_value=vcache_module,
                                       use_cache=use_cache,
                                       with_embed=with_embed)
            _ = self.norm_and_lm_head(hidden_states)

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
