from typing import Tuple, Optional
import numpy as np
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO, WARN
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


@register_causallm(['phi3_v_embedding'])
class Phi3EmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')


class Phi3RotaryEmbedding(nn.Module):
    def __init__(self, dim, seq_len, config, rope_scaling=None, device=None, name=''):
        super().__init__()
        self.dim = dim
        self.num_heads = config.num_attention_heads
        self.max_position_embeddings = seq_len
        self.base = config.rope_theta
        self.name = name

        if rope_scaling is None:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                                                              dtype=torch.int64).float().to(device) / self.dim))

            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos().to(torch.get_default_dtype())
            self.sin_cached = emb.sin().to(torch.get_default_dtype())
        else:
            scaling_type = config.rope_scaling["type"]
            self.short_factor = torch.tensor(config.rope_scaling["short_factor"], dtype=torch.float32, device=device)
            self.long_factor = torch.tensor(config.rope_scaling["long_factor"], dtype=torch.float32, device=device)
            if scaling_type == "su":
                scale = config.max_position_embeddings / config.original_max_position_embeddings
                if scale <= 1.0:
                    self.scaling_factor = 1.0
                else:
                    self.scaling_factor = math.sqrt(
                        1 + math.log(scale) / math.log(config.original_max_position_embeddings))

                self.inv_freq = 1.0 / (self.short_factor*self.base ** (torch.arange(0, self.dim, 2,
                                                                                    dtype=torch.int64).float().to(device) / self.dim))

                self.max_seq_len_cached = seq_len
                t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
                freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                # Different from paper, but it uses a different permutation in order to obtain the same calculation
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos_cached = self.scaling_factor*emb.cos().to(torch.get_default_dtype())
                self.sin_cached = self.scaling_factor*emb.sin().to(torch.get_default_dtype())

            elif scaling_type == "yarn":

                scale = config.max_position_embeddings / config.original_max_position_embeddings
                if scale <= 1.0:
                    self.scaling_factor = 1.0
                else:
                    self.scaling_factor = 0.1 * math.log(scale) + 1.0

                self.inv_freq = 1.0 / (self.short_factor*self.base ** (torch.arange(0, self.dim, 2,
                                                                                    dtype=torch.int64).float().to(device) / self.dim))

                self.max_seq_len_cached = seq_len
                t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
                freqs = torch.einsum("i,j->ij", t, self.inv_freq)
                # Different from paper, but it uses a different permutation in order to obtain the same calculation
                emb = torch.cat((freqs, freqs), dim=-1)
                self.cos_cached = self.scaling_factor*emb.cos().to(torch.get_default_dtype())
                self.sin_cached = self.scaling_factor*emb.sin().to(torch.get_default_dtype())

            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(self, pos_ids):
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        sin_new_shape = list(gather_sin_out.shape.symbol)[:1] + [1] + list(gather_sin_out.shape.symbol)[1:]
        cos_new_shape = list(gather_cos_out.shape.symbol)[:1] + [1] + list(gather_cos_out.shape.symbol)[1:]
        sin_out_reshape = gather_sin_out.reshape(sin_new_shape)
        cos_out_reshape = gather_cos_out.reshape(cos_new_shape)
        sin_reshape_tile = ops.tile(sin_out_reshape, [1, self.num_heads, 1, 1])
        cos_reshape_tile = ops.tile(cos_out_reshape, [1, self.num_heads, 1, 1])

        if self.name != '':
            sin_out.op.constants['weights'].name = f'{self.name}.rotary_emb.sin_weight'
            cos_out.op.constants['weights'].name = f'{self.name}.rotary_emb.cos_weight'
            sin_out.op.name = f"{self.name}.rotary_emb.sin_{sin_out.op.name}"
            cos_out.op.name = f"{self.name}.rotary_emb.cos_{cos_out.op.name}"
            gather_sin_out.name = f"{self.name}.{gather_sin_out.name}"
            gather_cos_out.name = f"{self.name}.{gather_cos_out.name}"

        return cos_reshape_tile, sin_reshape_tile


class Phi3MLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]
        gate_up_proj = weights[f'model.layers.{layer_id}.mlp.gate_up_proj.weight'].float()
        # weights[f'model.layers.{layer_id}.mlp.gate_up_proj.weight'].float()
        self.gate_proj.weights.betensor = gate_up_proj[0:self.intermediate_size, ...]
        # weights[f'model.layers.{layer_id}.mlp.up_proj.weight'].float()
        self.up_proj.weights.betensor = gate_up_proj[self.intermediate_size:2*self.intermediate_size, ...]
        self.down_proj.weights.betensor = weights[f'model.layers.{layer_id}.mlp.down_proj.weight'].float()

        self.gate_proj.weights.sync_betensor()
        self.up_proj.weights.sync_betensor()
        self.down_proj.weights.sync_betensor()

        self.gate_proj.weights.name = f'{prefix}layers.{layer_id}.mlp.gate_proj.weight'
        self.up_proj.weights.name = f'{prefix}layers.{layer_id}.mlp.up_proj.weight'
        self.down_proj.weights.name = f'{prefix}layers.{layer_id}.mlp.down_proj.weight'

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Phi3Attention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        qkv_proj_w = weights[f'model.layers.{layer_id}.self_attn.qkv_proj.weight'].float()
        # weights[f'model.layers.{layer_id}.self_attn.q_proj.weight'].float()
        self.q_proj.weights.betensor = qkv_proj_w[0:self.hidden_size, ...]
        self.q_proj.weights.sync_betensor()

        # weights[f'model.layers.{layer_id}.self_attn.k_proj.weight'].float()
        self.k_proj.weights.betensor = qkv_proj_w[self.hidden_size:2*self.hidden_size, ...]
        self.k_proj.weights.sync_betensor()

        # weights[f'model.layers.{layer_id}.self_attn.v_proj.weight'].float()
        self.v_proj.weights.betensor = qkv_proj_w[2*self.hidden_size:3*self.hidden_size, ...]
        self.v_proj.weights.sync_betensor()

        self.o_proj.weights.remove_betensor()
        self.o_proj.weights.betensor = weights[f'model.layers.{layer_id}.self_attn.o_proj.weight'].float()
        self.o_proj.weights.sync_betensor()

        self.q_proj.weights.name = f"{prefix}layers.{layer_id}.self_attn.q_proj.weight"
        self.q_proj.biases.name = f"{prefix}layers.{layer_id}.self_attn.q_proj.bias"
        self.k_proj.weights.name = f'{prefix}layers.{layer_id}.self_attn.k_proj.weight'
        self.k_proj.biases.name = f'{prefix}layers.{layer_id}.self_attn.k_proj.bias'
        self.v_proj.weights.name = f'{prefix}layers.{layer_id}.self_attn.v_proj.weight'
        self.v_proj.biases.name = f'{prefix}layers.{layer_id}.self_attn.v_proj.bias'
        self.o_proj.weights.name = f'{prefix}layers.{layer_id}.self_attn.o_proj.weight'

        # self.rotary_emb = Llama2RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape
        bs_symb, seq_len_symb, _ = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symb * seq_len_symb, hidden_size])
        crop0 = nn.Crop([[0, bs_symb], [0, self.num_heads], [0, seq_len_symb], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, bs_symb], [0, self.num_heads], [0, seq_len_symb], [0, self.head_dim//2]])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        self.rotary_emb = Phi3RotaryEmbedding(self.head_dim,
                                              seq_len,
                                              self.config,
                                              self.config.rope_scaling,
                                              name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        self.q_reshape = nn.Reshape([bs_symb, seq_len_symb, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs_symb, seq_len_symb, self.num_key_value_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.q_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.kv_reshape(key_states))

        # query
        q_crop_out0 = crop0(reshaped_trans_query)
        q_crop_out1 = crop1(reshaped_trans_query)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        # key
        k_crop_out0 = crop0(reshaped_trans_key)
        k_crop_out1 = crop1(reshaped_trans_key)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        neg_q_crop_out0.op.name = f"{self.matmul_name}.negative_0"
        neg_k_crop_out0.op.name = f"{self.matmul_name}.negative_1"

        q_cosine = reshaped_trans_query * cos_emb
        k_cosine = reshaped_trans_key * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine

        k_cache_out = k_cache(k_add_out, position_ids=position_ids, block_idx=self.layer_id,
                              name=f"past_key_value.{self.layer_id}.key")
        k_cache_out = k_cache_out[0]

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = k_cache_out.shape.symbol
            k_cache_out = ops.reshape(k_cache_out, [batch, num_key_value_heads, 1, slen, head_dim])
            k_cache_out = ops.tile(k_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
            k_cache_out = k_cache_out.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        if seq_len == 1:
            # [CP-19313]C=A*B=(B^T*A^T)^T, when B size >> A size, swap input can improve wtram reuse
            kq_matmul = nn.Transpose([0, 1, 3, 2])(self.kq_matmul(k_cache_out, q_add_out))
        else:
            kq_matmul = self.kq_matmul(q_add_out, k_cache_out)

        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        softmax_masked = self.softmax(kq_matmul + attention_mask)

        if seq_len == 1:
            reshaped_trans_value = nn.Reshape([bs_symb, self.num_key_value_heads,
                                              seq_len_symb, self.head_dim])(value_states)
        else:
            reshaped_trans_value = self.qkv_transpose(
                self.kv_reshape(value_states))

        v_cache_out = v_cache(reshaped_trans_value, position_ids=position_ids,
                              block_idx=self.layer_id, name=f"past_key_value.{self.layer_id}.value")
        v_cache_out = v_cache_out[0]

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = v_cache_out.shape.symbol
            v_cache_out = v_cache_out.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            v_cache_out = ops.tile(v_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
            v_cache_out = v_cache_out.reshape(
                [batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        kqv_out = self.kqv_matmul(softmax_masked, v_cache_out)
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs_symb * seq_len_symb, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.o_proj(reshaped_trans)
        return o_t


class Phi3Decoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm.weights.betensor = weights[f'model.layers.{layer_id}.input_layernorm.weight'].float()
        self.input_layernorm.weights.name = f"{prefix}layers.{layer_id}.input_layernorm.weight"
        self.input_layernorm.weights.sync_betensor()

        self.attention = Phi3Attention(config, weights, layer_id, prefix)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm.weights.betensor = weights[
            f'model.layers.{layer_id}.post_attention_layernorm.weight'].float()
        self.post_attention_layernorm.weights.sync_betensor()
        self.post_attention_layernorm.weights.name = f"{prefix}layers.{layer_id}.post_attention_layernorm.weight"
        self.mlp = Phi3MLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            hidden_states = ops.reshape(hidden_states, hidden_states.shape.symbol)
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
        bs_symbol, seq_len_symbol, _ = residual.shape.symbol
        hidden_states = hidden_states.reshape([bs_symbol, seq_len_symbol, hidden_states.shape[1]])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        bs_symbol, seq_len_symbol, hidden_size_symb = hidden_states.shape.symbol
        hidden_states = self.mlp(hidden_states.reshape([bs_symbol * seq_len_symbol, hidden_size_symb]))
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.reshape([bs_symbol, seq_len_symbol, hidden_states.shape[-1]])
        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class Phi3Embedding(nn.Module):
    def __init__(self, config, weights, prefix):
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        # self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights[f'model.embed_tokens.weight'].float()

    @gen_subgraph(subgraph_name='embedding')
    def forward(self, input_ids):
        embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
            self.config.vocab_size, self.config.hidden_size])
        embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
        input_embeds = ops.gather(embed_const, input_ids, 0, 0)

        return input_embeds


class Phi3Model(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        # self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        # self.vocab_size = config.vocab_size
        self.prefix = prefix
        # self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)

        # self.embed_weights = weights[f'model.embed_tokens.weight'].float()
        self.embedding = Phi3Embedding(config, weights, prefix)
        self.decoders = [Phi3Decoder(config, weights, idx, prefix) for idx in gen_layer_ids(config.num_hidden_layers)]

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
            WARN(f"when use_kv_cache, please provide the kv inputs")

        is_gen_subgraph = True and hasattr(self.model_config, 'enable_subgraph') and self.model_config.enable_subgraph
        # input_embeds = self.embed_tokens(input_ids)
        if is_gen_subgraph:
            # embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
            #                            self.model_config.vocab_size, self.model_config.hidden_size])
            # embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            # input_embeds = ops.gather(embed_const, input_ids, 0, 0)
            input_embeds = self.embedding(input_ids)
        else:
            input_embeds = input_ids
        # attention_mask = ops.cast(attention_mask, Dtype.FP32, ignore_scale_zp=True)
        attention_mask = ops.tile(
            attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

        hidden_states = input_embeds
        for i, decoder in enumerate(self.decoders):
            hidden_states, _, _ = decoder(hidden_states,
                                          attention_mask,
                                          position_ids,
                                          past_key,
                                          past_value,
                                          use_cache=use_cache)
        return hidden_states


class NormLmHead(nn.Module):
    def __init__(self, config, weights):
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.norm.weights, weights.get(f'model.norm.weight'), f'normlm.norm.weight')
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        key = 'lm_head.weight'
        if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
            key = 'model.embed_tokens.weight'
        update_tensor_data(self.lm_head.weights, weights.get(key), f'normlm.lm.weight')

    @gen_subgraph(subgraph_name='lm')
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        h_shape = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [h_shape[0] * h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        out = ops.reshape(lm_head, [h_shape[0], h_shape[1], lm_head.shape[-1]])
        return out


@register_causallm(['phi3_v'])
class Phi3CausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.aipu_g = None
        self.prefix = prefix
        self.model = Phi3Model(config, weights, prefix)
        # self.norm_and_lm_head = NormLmHead(config, weights)
        self.norm_and_lm_head = NormLmHead(config, weights) if config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")

    def forward(self):
        seq_len = self.config.get('seq_len')
        past_len = self.config.get('past_len')
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        model_name = 'phi3'

        with Graph(model_name) as self.aipu_g:
            if is_gen_subgraph:
                hidden_states = ops.input([batch, seq_len], Dtype.FP32, name='input_ids',
                                          exprs=['batch', 'seq_len'])
            else:
                hidden_states = ops.input([batch, seq_len, self.config.hidden_size],
                                          Dtype.FP32,
                                          name='embeded',
                                          exprs=['batch', 'seq_len', self.config.hidden_size])

            attention_mask = ops.input([batch, 1, seq_len, seq_len + past_len],
                                       Dtype.FP32,
                                       name='attention_mask',
                                       exprs=['batch', 1, 'seq_len', 'max_len'])
            position_ids = ops.input([batch, seq_len],
                                     Dtype.INT32,
                                     name='position_ids', exprs=['batch', 'seq_len'])

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
