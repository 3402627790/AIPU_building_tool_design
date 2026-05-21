import numpy as np
import torch
import os
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from safetensors import safe_open
from ..basemodel import *
from ..config import PretrainedConfig
from ..register import register_causallm, gen_subgraph
from ..qinfo import QConf
from ..utils import update_tensor_data


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
            scale_factor = rope_scaling.get("factor", 8)
            low_freq_factor = rope_scaling.get("low_freq_factor", 1)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4)
            old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            new_freqs = []
            for freq in self.inv_freq:
                wavelen = 2 * torch.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / scale_factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (old_context_len / wavelen - low_freq_factor) / (
                        high_freq_factor - low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=self.inv_freq.dtype, device=self.inv_freq.device)

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
        sin_out_reshape = ops.reshape(gather_sin_out, [1, *gather_sin_out.shape])
        cos_out_reshape = ops.reshape(gather_cos_out, [1, *gather_cos_out.shape])
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
        self.max_position_embeddings = config.max_position_embeddings if not hasattr(
            self.config, 'max_len') or self.config.max_len is None else self.config.max_len
        self.rope_theta = config.rope_theta
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

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        # self.scale = nn.Constant(np.array(1./np.sqrt(np.array(self.head_dim))).astype('float32').reshape([1]))
        # self.scale.weights.name = f"scale_for_normalization"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, scale=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])
        crop0 = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [0, self.head_dim//2]])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        self.rotary_emb = Llama2RotaryEmbedding(self.num_heads,
                                                self.head_dim,
                                                max_position_embeddings=self.max_position_embeddings,
                                                base=self.rope_theta,
                                                rope_scaling=self.config.rope_scaling,
                                                name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        self.q_reshape = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs, seq_len, self.num_key_value_heads, self.head_dim])
        if seq_len == 1:
            reshaped_trans_query = nn.Reshape([bs, self.num_heads, seq_len, self.head_dim])(query_states)
            reshaped_trans_key = nn.Reshape([bs, self.num_heads, seq_len, self.head_dim])(key_states)
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

        if self.num_key_value_groups > 1:
            self.rotary_emb_k = Llama2RotaryEmbedding(self.num_key_value_heads,
                                                      self.head_dim,
                                                      max_position_embeddings=seq_len,
                                                      base=self.config.rope_theta,
                                                      rope_scaling=self.config.rope_scaling,
                                                      name=f'{self.prefix}layers.{self.layer_id}.k.self_attn')
            cos_emb_k, sin_emb_k = self.rotary_emb_k(position_ids)
        else:
            cos_emb_k = cos_emb
            sin_emb_k = sin_emb

        k_cosine = reshaped_trans_key * cos_emb_k
        k_r_sine = rope_key * sin_emb_k

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine

        history_k, history_v = None, None
        if use_cache:
            if self.config.is_gen_subgraph_and_kv_cacheupdate:
                history_k = ops.cacheupdate(k_cache, position_ids, k_add_out, self.layer_id)[0]
            else:
                history_k = nn.Concat(2)([k_cache, k_add_out])
            k_add_out = history_k
        else:
            history_k = k_add_out

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = list(k_add_out.shape)
            k_add_out = k_add_out.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            k_add_out = ops.tile(k_add_out, [1, 1, self.num_key_value_groups, 1, 1])
            k_add_out = k_add_out.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        kq_matmul = self.kq_matmul(q_add_out, k_add_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        if scale is None:
            scale = nn.Constant(np.array(1./np.sqrt(np.array(self.head_dim))).astype('float32').reshape([1]))()
        scaled_kq = kq_matmul * scale
        softmax_masked = self.softmax(scaled_kq + attention_mask)

        if seq_len == 1:
            reshaped_trans_value = nn.Reshape([bs, self.num_heads, seq_len, self.head_dim])(value_states)
        else:
            reshaped_trans_value = self.qkv_transpose(self.kv_reshape(value_states))
        if use_cache:
            if self.config.is_gen_subgraph_and_kv_cacheupdate:
                history_v = ops.cacheupdate(v_cache, position_ids, reshaped_trans_value, self.layer_id)[0]
            else:
                history_v = nn.Concat(2)([v_cache, reshaped_trans_value])
            reshaped_trans_value = history_v
        else:
            history_v = reshaped_trans_value

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = list(reshaped_trans_value.shape)
            reshaped_trans_value = reshaped_trans_value.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            reshaped_trans_value = ops.tile(reshaped_trans_value, [1, 1, self.num_key_value_groups, 1, 1])
            reshaped_trans_value = reshaped_trans_value.reshape(
                [batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs*seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.o_proj(reshaped_trans)

        return o_t, history_k, history_v


class Llama2Decoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = Llama2Attention(config, weights, layer_id, prefix)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, config.norm_epsilon)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = Llama2MLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name="attnblk")
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, scale=None, use_cache=False):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        cur_scale = scale
        if is_gen_subgraph and scale is not None and isinstance(scale, Tensor):
            """
            when enable_subgraph=true, will do some special operations:
            1. avoiding the constant output tensor is subgraph's input tensor, will re-generate one constant in subgraph.
            2. if one tensor is multi ops input, the subgraph will insert the duplicate input tensor into subgraph input tensors,
            which will cause big change for llm runtime. in order to avoid this graph, will insert reshape op, its output tensors will
            be as multi ops input, and reshape is including in this subgraph.
            """
            cur_scale = ops.constant(scale.op.constants['weights'].betensor.cpu().numpy())
            hidden_states = ops.reshape(hidden_states, hidden_states.shape)
            attention_mask = ops.tile(attention_mask, [1, self.config.num_attention_heads, 1, 1])
            position_ids = ops.reshape(position_ids, position_ids.shape)

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, k, v = self.attention(hidden_states,
                                             attention_mask,
                                             position_ids,
                                             k_cache,
                                             v_cache,
                                             cur_scale,
                                             use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states.reshape([*list(hidden_states.shape)[1:]]))
        hidden_states = residual + hidden_states

        return hidden_states, k, v


class NormLmHead(nn.Module):
    def __init__(self, config, weights):
        self.config = config
        self.norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        update_tensor_data(self.norm.weights, weights.get(f'model.norm.weight'), f'normlm.norm.weight')

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        update_tensor_data(self.lm_head.weights, weights.get(f"lm_head.weight"), f"normlm.lm.weight")

    @gen_subgraph(subgraph_name="lm")
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        h_shape = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [h_shape[0] * h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        out = ops.reshape(lm_head, [1, *list(lm_head.shape)])
        return out


class Llama2Model(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights[f'model.embed_tokens.weight'].float()

        # self.decoders = [Llama2Decoder(config, weights, 0)]
        self.decoders = [Llama2Decoder(config, weights, idx, prefix) for idx in range(config.num_hidden_layers)]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                scale=None,
                past_key=None,
                past_value=None,
                use_cache=None,
                with_embed=True,
                with_preprocess_mask=False,
                ):

        if use_cache and (past_key is None or past_value is None):
            ERROR(f"when use_kv_cache, please provide the kv inputs")
        if not use_cache and not self.model_config.is_gen_subgraph_and_kv_cacheupdate:
            past_key = [None] * self.model_config.num_hidden_layers
            past_value = [None] * self.model_config.num_hidden_layers

        # input_embeds = self.embed_tokens(input_ids)
        if with_embed:
            embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
                                       self.model_config.vocab_size, self.model_config.hidden_size])
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        else:
            input_embeds = input_ids

        if not self.model_config.is_gen_subgraph:
            attention_mask = ops.tile(attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

        hidden_states = input_embeds
        is_gen_subgraph_and_kv_cacheupdate = self.model_config.is_gen_subgraph_and_kv_cacheupdate
        history_ks, history_vs = [], []
        for i, decoder in enumerate(self.decoders):
            hidden_states, history_k, history_v = decoder(hidden_states,
                                                          attention_mask,
                                                          position_ids,
                                                          past_key[i] if not is_gen_subgraph_and_kv_cacheupdate else past_key,
                                                          past_value[i] if not is_gen_subgraph_and_kv_cacheupdate else past_value,
                                                          scale,
                                                          use_cache=use_cache,
                                                          )
            history_ks.append(history_k)
            history_vs.append(history_v)
        return hidden_states, history_ks, history_vs


def _set_default_values(config):
    _defv = {
        'norm_epsilon': 0.000001,
        'num_key_value_heads': 'num_attention_heads',
        'rope_theta': 1000000.0,
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
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.model = Llama2Model(config, weights, prefix)
        self.norm_and_lm_head = NormLmHead(config, weights)

        self.aipu_g = None
        self.prefix = prefix

    def forward(self, seq_len=None, use_cache=None, with_embed=False, past_len=None):  # prefix='model.' with_embed=True is prev graph
        seq_len = self.config.max_position_embeddings if seq_len is None else seq_len
        past_len = self.config.max_len - seq_len

        self.config.seq_len = seq_len
        self.config.past_len = past_len
        self.with_embed = with_embed
        block_num = self.config.num_hidden_layers
        head_num = self.config.num_key_value_heads
        hidden_size = self.config.hidden_size
        num_attention_heads = self.config.num_attention_heads
        self.use_cache = use_cache if use_cache is not None else self.config.use_cache
        with Graph('llama2') as self.aipu_g:
            if self.with_embed:
                hidden_states = ops.input([1, seq_len], Dtype.FP32, name='input_ids')
            else:
                hidden_states = ops.input(
                    [1, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded')

            if self.use_cache:
                attention_mask = ops.input([1, 1, seq_len, seq_len + past_len], Dtype.FP32, name='attention_mask')
            else:
                attention_mask = ops.input([1, 1, 1, seq_len], Dtype.FP32, name='attention_mask')
            # attention_mask = Tensor('attention_mask', [1, seq_len]), Dtype.INT32)
            position_ids = ops.input([1, seq_len], Dtype.INT32, name='position_ids')

            past_key, past_value = None, None
            if self.use_cache:
                if not self.config.is_gen_subgraph_and_kv_cacheupdate:
                    past_key = [ops.input([1, head_num, past_len, hidden_size//num_attention_heads],
                                          Dtype.FP32, name=f"past_key_value.{i}.key") for i in range(block_num)]
                    past_value = [ops.input([1, head_num, past_len, hidden_size//num_attention_heads],
                                            Dtype.FP32, name=f"past_key_value.{i}.value") for i in range(block_num)]
                else:
                    k_shape = [block_num, 1, seq_len + past_len, hidden_size]
                    past_key = ops.input(k_shape, Dtype.FP32, name='k_cache')
                    past_value = ops.input(k_shape, Dtype.FP32, name='v_cache')

            scale = nn.Constant(np.array(1./np.sqrt(np.array(hidden_size // num_attention_heads))
                                         ).astype('float32').reshape([1]))()
            scale.op.constants['weights'].name = f"scale_for_normalization"

            hidden_states, history_k, history_v = self.model(hidden_states,
                                                             attention_mask,
                                                             position_ids,
                                                             scale=scale,
                                                             past_key=past_key,
                                                             past_value=past_value,
                                                             use_cache=self.use_cache,
                                                             with_embed=self.with_embed)
            _ = self.norm_and_lm_head(hidden_states)

            cur_tensors = list(self.aipu_g.output_tensors)
            history_k_v_name = [hkv.name for hkv in history_k + history_v]
            output_tensors = []
            for ct in cur_tensors:
                if ct.name not in history_k_v_name:
                    output_tensors.append(ct)
            output_tensors += history_k + history_v
            if not isinstance(past_key, list):
                past_key = [past_key]
            if not isinstance(past_value, list):
                past_value = [past_value]
            self.aipu_g.output_tensors = TensorList(output_tensors)
            if self.use_cache:
                self.aipu_g.input_tensors = TensorList(
                    [t for t in self.aipu_g.input_tensors[:3]] + past_key + past_value)
        return self.aipu_g
