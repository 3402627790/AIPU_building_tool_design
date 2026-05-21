import numpy as np
import torch
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.logger import INFO, ERROR
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


@register_update_config(['llama2', 'llama2_chn', 'llama3'])
def llama_update_config(config):
    if not hasattr(config, 'num_key_value_heads') and hasattr(config, 'num_attention_heads'):
        config.num_key_value_heads = config.num_attention_heads


@register_causallm(['llama3_embedding', 'llama2_embedding', 'llama2_chn_embedding'])
class LLama2EmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')


class Llama2RotaryEmbedding(nn.Module):
    def __init__(self, num_heads, dim, max_position_embeddings=2048, base=10000, rope_scaling=None, device=None, name=''):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.name = name
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                               dtype=torch.int64).float().to(device) / self.dim))
        if rope_scaling is not None and rope_scaling.get("rope_type", None) == "llama3":
            # refer to https://github.com/meta-llama/llama-models/blob/f45cdfd624b98b6655540f7101d8d9cb432e631c/models/llama3_1/reference_impl/model.py#L45
            factor = rope_scaling.get("factor", 8.0)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor

            wavelen = 2 * torch.pi / self.inv_freq
            # wavelen < high_freq_wavelen: do nothing
            # wavelen > low_freq_wavelen: divide by factor
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / factor, self.inv_freq)
            # otherwise: interpolate between the two, using a smooth factor
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            self.inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(torch.get_default_dtype())
        self.sin_cached = emb.sin().to(torch.get_default_dtype())

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


class Llama2MLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        gate_proj_w = weights.get(f'model.layers.{layer_id}.mlp.gate_proj.weight')
        up_proj_w = weights.get(f'model.layers.{layer_id}.mlp.up_proj.weight')
        down_proj_w = weights.get(f'model.layers.{layer_id}.mlp.down_proj.weight')
        update_tensor_data(self.gate_proj.weights, gate_proj_w, name=f'{prefix}layers.{layer_id}.mlp.gate_proj.weight')
        update_tensor_data(self.up_proj.weights, up_proj_w, name=f'{prefix}layers.{layer_id}.mlp.up_proj.weight')
        update_tensor_data(self.down_proj.weights, down_proj_w, name=f'{prefix}layers.{layer_id}.mlp.down_proj.weight')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Llama2Attention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.get('num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.get('rope_theta', 10000.)
        self.max_position_embeddings = config.max_position_embeddings if not hasattr(
            self.config, 'max_len') or self.config.max_len is None else self.config.max_len
        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        q_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.q_proj.weight')
        q_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.q_proj.bias')
        update_tensor_data(self.q_proj.weights, q_proj_w, f"{prefix}layers.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q_proj.biases, q_proj_b, f"{prefix}layers.{layer_id}.self_attn.q_proj.bias")
        k_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.k_proj.weight')
        k_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.k_proj.bias')
        update_tensor_data(self.k_proj.weights, k_proj_w, f"{prefix}layers.{layer_id}.self_attn.k_proj.weight")
        update_tensor_data(self.k_proj.biases, k_proj_b, f"{prefix}layers.{layer_id}.self_attn.k_proj.bias")

        v_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.v_proj.weight')
        v_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.v_proj.bias')
        update_tensor_data(self.v_proj.weights, v_proj_w, f"{prefix}layers.{layer_id}.self_attn.v_proj.weight")
        update_tensor_data(self.v_proj.biases, v_proj_b, f"{prefix}layers.{layer_id}.self_attn.v_proj.bias")

        o_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.o_proj.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.self_attn.o_proj.weight")

        # self.rotary_emb = Llama2RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kq_matmul_notransb = nn.MatMul(trans_a=False, trans_b=False, fused_multiplier=scale)
        self.kq_matmul_swap = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.kqv_matmul_transpose = nn.MatMul(trans_a=False, trans_b=True)

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False, cos_emb=None, sin_emb=None):
        bs, seq_len, hidden_size = hidden_states.shape
        bs_symbol, seq_len_symbol, _ = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symbol * seq_len_symbol, hidden_size])
        crop0 = nn.Crop([[0, bs_symbol], [0, self.num_heads], [0, seq_len_symbol], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, bs_symbol], [0, self.num_heads], [0, seq_len_symbol], [0, self.head_dim//2]])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if cos_emb is None or sin_emb is None:
            self.rotary_emb = Llama2RotaryEmbedding(self.num_heads,
                                                    self.head_dim,
                                                    max_position_embeddings=self.config.seq_len+self.config.past_len,
                                                    base=self.rope_theta,
                                                    rope_scaling=self.config.rope_scaling,
                                                    name=f'{self.prefix}layers.{self.layer_id}.self_attn')
            cos_emb, sin_emb = self.rotary_emb(position_ids)
            if self.num_key_value_groups > 1:
                self.rotary_emb_k = Llama2RotaryEmbedding(self.num_key_value_heads,
                                                          self.head_dim,
                                                          max_position_embeddings=self.config.seq_len+self.config.past_len,
                                                          base=self.config.rope_theta,
                                                          rope_scaling=self.config.rope_scaling,
                                                          name=f'{self.prefix}layers.{self.layer_id}.k.self_attn')
                cos_emb_k, sin_emb_k = self.rotary_emb_k(position_ids)
            else:
                cos_emb_k, sin_emb_k = cos_emb, sin_emb
        else:
            # cos_emb and sim_emb has already builded.
            if self.num_key_value_groups > 1:
                # cos_emb and sin_emb is list
                cos_emb, cos_emb_k = cos_emb
                sin_emb, sin_emb_k = sin_emb
            else:
                cos_emb_k, sin_emb_k = cos_emb, sin_emb

        self.q_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_key_value_heads, self.head_dim])
        if seq_len == 1:
            reshaped_trans_query = nn.Reshape([bs_symbol, self.num_heads, seq_len_symbol, self.head_dim])(query_states)
            reshaped_trans_key = nn.Reshape([bs_symbol, self.num_key_value_heads,
                                            seq_len_symbol, self.head_dim])(key_states)
        else:
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
        q_r_sine = rope_query * sin_emb

        k_cosine = reshaped_trans_key * cos_emb_k
        k_r_sine = rope_key * sin_emb_k

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine

        k_cache_out = k_cache(k_add_out, position_ids=position_ids, block_idx=self.layer_id,
                              name=f"past_key_value.{self.layer_id}.key")
        k_cache_out = k_cache_out[0]

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = k_cache_out.shape.symbol
            k_cache_out = k_cache_out.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            k_cache_out = ops.tile(k_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
            k_cache_out = k_cache_out.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        if seq_len == 1:
            # [CP-19313]C=A*B=(B^T*A^T)^T, when B size >> A size, swap input can improve wtram reuse
            kq_matmul = nn.Transpose([0, 1, 3, 2])(self.kq_matmul_swap(k_cache_out, q_add_out))
        else:
            if self.config.get('kvcache_optimization', False):
                # cacheupdate will defaultly transpose the data then give to matmul(which trans_b=true), so if cacheupdate enable,
                # matmul should be trans_b=true
                kq_matmul = self.kq_matmul(q_add_out, k_cache_out)
            else:
                # [CP-25169]disbale transb can improve matmul's perf in prefill section
                kq_matmul = self.kq_matmul_notransb(q_add_out, nn.Transpose([0, 1, 3, 2])(k_cache_out))

        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        softmax_masked = self.softmax(kq_matmul + attention_mask)

        if seq_len == 1 and self.config.get('vcache_axis', 2) == 3:
            # [CP-24713] cacheupdate support c axis concat
            reshaped_trans_value = nn.Reshape([bs_symbol, self.num_key_value_heads,
                                               self.head_dim, seq_len_symbol])(value_states)
        elif seq_len == 1:
            reshaped_trans_value = nn.Reshape([bs_symbol, self.num_key_value_heads,
                                              seq_len_symbol, self.head_dim])(value_states)
        else:
            reshaped_trans_value = self.qkv_transpose(self.kv_reshape(value_states))

        vcache_axis = self.config.get('vcache_axis', 2)
        v_cache_out = v_cache(reshaped_trans_value,
                              position_ids=position_ids,
                              block_idx=self.layer_id,
                              name=f"past_key_value.{self.layer_id}.value",
                              axis=vcache_axis)
        v_cache_out = v_cache_out[0]

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = v_cache_out.shape.symbol
            v_cache_out = v_cache_out.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            v_cache_out = ops.tile(v_cache_out, [1, 1, self.num_key_value_groups, 1, 1])
            v_cache_out = v_cache_out.reshape(
                [batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        if seq_len == 1 and self.config.get("vcache_axis", 2) == 3:
            # [CP-24662] swap matmul for better performance
            kqv_out = self.kqv_matmul_transpose(v_cache_out, softmax_masked)
        else:
            kqv_out = self.qkv_transpose(self.kqv_matmul(softmax_masked, v_cache_out))

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"
        self.qkv_reshape = nn.Reshape([bs_symbol * seq_len_symbol, self.hidden_size])
        reshaped_trans = self.qkv_reshape(kqv_out)
        o_t = self.o_proj(reshaped_trans)

        return o_t


class Llama2Decoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = Llama2Attention(config, weights, layer_id, prefix)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = Llama2MLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name="attnblk")
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False, cos_emb=None, sin_emb=None):
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
            # attention_mask = ops.tile(attention_mask, [1, self.config.num_attention_heads, 1, 1])
            position_ids = ops.reshape(position_ids, position_ids.shape.symbol)

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       k_cache,
                                       v_cache,
                                       use_cache,
                                       cos_emb,
                                       sin_emb)
        bs_symbol, seq_len_symbol, _ = residual.shape.symbol
        hidden_states = hidden_states.reshape([bs_symbol, seq_len_symbol, hidden_states.shape[1]])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.reshape([bs_symbol * seq_len_symbol, hidden_states.shape[-1]])
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.reshape([bs_symbol, seq_len_symbol, hidden_states.shape[1]])
        hidden_states = residual + hidden_states
        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class NormLmHead(nn.Module):
    def __init__(self, config, weights):
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.norm.weights, weights.get(f'model.norm.weight'), f'normlm.norm.weight')

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        key = 'lm_head.weight'
        if config.get('tie_word_embeddings', False):
            key = 'model.embed_tokens.weight'
        w = weights.get(key)
        if len(w) == 0 and key == 'lm_head.weight':
            key = 'model.embed_tokens.weight'
            w = weights.get(key)
        update_tensor_data(self.lm_head.weights, w, f'normlm.lm.weight')

    @gen_subgraph(subgraph_name="lm")
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        h_shape = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [h_shape[0] * h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        out = ops.reshape(lm_head, [1, *list(lm_head.shape.symbol)])
        return out


class Llama2Model(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights.get(f'model.embed_tokens.weight')['data']
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.get('num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.get('rope_theta', 10000.)

        # self.decoders = [Llama2Decoder(config, weights, 0)]
        num_hidden_layers = gen_layer_ids(config.num_hidden_layers)
        self.decoders = [Llama2Decoder(config, weights, idx, prefix) for idx in num_hidden_layers]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key=None,
                past_value=None,
                use_cache=None,
                with_embed=True,
                with_preprocess_mask=False,
                cos_emb=None,
                sin_emb=None):

        if past_key is None or past_value is None:
            ERROR(f"please provide the correct kv cache input in llama2.")
        # input_embeds = self.embed_tokens(input_ids)
        if with_embed:
            embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
                                       self.config.vocab_size, self.config.hidden_size])
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        else:
            input_embeds = input_ids

        is_gen_subgraph = self.config.get('enable_subgraph', False)
        if not is_gen_subgraph:
            # attention_mask = ops.tile(attention_mask, [1, self.config.num_attention_heads, 1, 1])
            self.rotary_emb = Llama2RotaryEmbedding(self.num_heads,
                                                    self.head_dim,
                                                    max_position_embeddings=self.config.seq_len+self.config.past_len,
                                                    base=self.rope_theta,
                                                    rope_scaling=self.config.rope_scaling,
                                                    name=f'{self.prefix}layers.self_attn')
            cos_emb, sin_emb = self.rotary_emb(position_ids)
            if self.num_key_value_groups > 1:
                self.rotary_emb_k = Llama2RotaryEmbedding(self.num_key_value_heads,
                                                          self.head_dim,
                                                          max_position_embeddings=self.config.seq_len+self.config.past_len,
                                                          base=self.config.rope_theta,
                                                          rope_scaling=self.config.rope_scaling,
                                                          name=f'{self.prefix}layers.k.self_attn')
                cos_emb_k, sin_emb_k = self.rotary_emb_k(position_ids)
                cos_emb = [cos_emb, cos_emb_k]
                sin_emb = [sin_emb, sin_emb_k]

        hidden_states = input_embeds
        for i, decoder in enumerate(self.decoders):
            hidden_states, _, _ = decoder(hidden_states,
                                          attention_mask,
                                          position_ids,
                                          past_key,
                                          past_value,
                                          use_cache=use_cache,
                                          cos_emb=cos_emb,
                                          sin_emb=sin_emb)
        return hidden_states


def _set_default_values(config):
    _defv = {
        'norm_epsilon': 0.00001,
        'num_key_value_heads': 'num_attention_heads',
        'rope_theta': 10000.0,
        'rope_scaling': None,
    }

    for k, v in _defv.items():
        if not hasattr(config, k):
            if not isinstance(v, str):
                config.update({k: v})
            else:
                config.update({k: config.get(v)})


@register_causallm(['llama2', 'llama3', 'llama', 'llama2_chn'])
class Llama2CausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        _set_default_values(config)
        super().__init__(config)
        weights = self.config.get('weights', self._load_state_dict())
        self.model = Llama2Model(self.config, weights, prefix)
        self.norm_and_lm_head = NormLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")
        self.with_embed = self.config.get('with_embed', False)
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.get('seq_len')
        past_len = self.config.get('past_len')
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        with Graph('llama2') as self.aipu_g:
            if with_embed:
                hidden_states = ops.input(
                    [batch, seq_len], Dtype.FP32, name='input_ids', exprs=['batch', 'seq_len'])
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
                                     name='position_ids',
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
