import numpy as np
import torch
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


@register_update_config('baichuan2')
def baichuan2_update_config(config):
    if config.get('num_key_value_heads', None) is None:
        num_key_value_heads = config.num_key_value_heads if hasattr(
            config, 'num_key_value_heads') else config.num_attention_heads
        config.update({'num_key_value_heads': num_key_value_heads})


@register_causallm(['baichuan2_embedding'])
class BaiChuan2EmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')


class Baichuan2RotaryEmbedding(nn.Module):
    def __init__(self, num_heads, dim, max_position_embeddings=2048, base=10000, device=None, name=''):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.name = name
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2,
                               dtype=torch.int64).float().to(device) / self.dim))

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
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


class Baichuan2MLP(nn.Module):

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


class Baichuan2Attention(nn.Module):

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
        self.rope_theta = config.rope_theta if hasattr(config, 'rope_theta') else 1000000.0
        self.is_causal = True
        self.attention_dropout = 0
        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        qkv_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.W_pack.weight')['data'].float()
        q_proj_w = qkv_proj_w[0: self.hidden_size, ...]
        update_tensor_data(self.q_proj.weights, q_proj_w, f"{prefix}layers.{layer_id}.self_attn.q_proj.weight")

        k_proj_w = qkv_proj_w[self.hidden_size: self.hidden_size*2, ...]
        update_tensor_data(self.k_proj.weights, k_proj_w, f"{prefix}layers.{layer_id}.self_attn.k_proj.weight")

        v_proj_w = qkv_proj_w[self.hidden_size*2: self.hidden_size*3, ...]
        update_tensor_data(self.v_proj.weights, v_proj_w, f"{prefix}layers.{layer_id}.self_attn.v_proj.weight")

        o_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.o_proj.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.self_attn.o_proj.weight")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])
        crop0 = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [
                        self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, bs], [0, self.num_heads], [
                        0, seq_len], [0, self.head_dim//2]])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        self.rotary_emb = Baichuan2RotaryEmbedding(self.num_heads,
                                                   self.head_dim,
                                                   max_position_embeddings=attention_mask.shape[-1],
                                                   base=self.rope_theta,
                                                   name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        self.q_reshape = nn.Reshape(
            [bs, seq_len, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape(
            [bs, seq_len, self.num_key_value_heads, self.head_dim])
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

        history_k, history_v = None, None
        if use_cache:
            if k_cache is None:
                history_k = k_add_out
            else:
                history_k = nn.Concat(2)([k_cache, k_add_out])
            k_values = ops.reshape(history_k, [
                                   1, self.num_key_value_heads, 1, history_k.shape[-2], history_k.shape[-1]])
            if self.num_key_value_groups > 1:
                k_values = ops.tile(
                    k_values, [1, 1, self.num_key_value_groups, 1, 1])
            k_values = ops.reshape(
                k_values, [1, self.num_heads, history_k.shape[-2], history_k.shape[-1]])
            kq_matmul = self.kq_matmul(q_add_out, k_values)
        else:
            # q_add_out = self.bn_q(q_add_out)
            # q_add_out.op.name = q_add_out.op.constants['weights'].name + 'q'
            # k_add_out = self.bn_k(k_add_out)
            # k_add_out.op.name = k_add_out.op.constants['weights'].name + 'k'
            history_k = k_add_out
            kq_matmul = self.kq_matmul(q_add_out, k_add_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"

        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        softmax_masked = self.softmax(kq_matmul + attention_mask)

        reshaped_trans_value = self.qkv_transpose(
            self.kv_reshape(value_states))
        if use_cache:
            if v_cache is None:
                history_v = reshaped_trans_value
            else:
                history_v = nn.Concat(2)([v_cache, reshaped_trans_value])
            v_values = ops.reshape(history_v, [
                                   1, self.num_key_value_heads, 1, history_v.shape[-2], history_v.shape[-1]])
            if self.num_key_value_groups > 1:
                v_values = ops.tile(
                    v_values, [1, 1, self.num_key_value_groups, 1, 1])
            v_values = ops.reshape(
                v_values, [1, self.num_heads, history_v.shape[-2], history_v.shape[-1]])
            kqv_out = self.kqv_matmul(softmax_masked, v_values)
        else:
            history_v = reshaped_trans_value
            kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)

        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"
        # kqv_out.name = f"{self.matmul_name}_{kqv_out.name}"

        self.qkv_reshape = nn.Reshape([bs*seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.o_proj(reshaped_trans)

        return o_t, history_k, history_v


class Baichuan2Decoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = Baichuan2Attention(config, weights, layer_id, prefix)

        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = Baichuan2MLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            hidden_states = ops.reshape(hidden_states, hidden_states.shape)
            position_ids = ops.reshape(position_ids, position_ids.shape)
        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, k, v = self.attention(hidden_states,
                                             attention_mask,
                                             position_ids,
                                             k_cache,
                                             v_cache,
                                             use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states.reshape([*list(hidden_states.shape)[1:]]))
        hidden_states = residual + hidden_states

        return hidden_states, k, v


class Baichuan2Embedding(nn.Module):
    def __init__(self, config, weights, prefix):
        self.config = config
        self.prefix = prefix
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights[f'model.embed_tokens.weight'].float()

    @gen_subgraph(subgraph_name='embedding')
    def forward(self, input_ids):
        embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
            self.config.vocab_size, self.config.hidden_size])
        embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
        input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        return input_embeds


class Baichuan2Model(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.config = config
        self.prefix = prefix
        self.embedding = Baichuan2Embedding(config, weights, prefix)
        self.decoders = [Baichuan2Decoder(config, weights, idx, prefix)
                         for idx in gen_layer_ids(config.num_hidden_layers)]

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

        if use_cache and (past_key is None or past_value is None):
            WARN(f"when use_kv_cache, please provide the kv inputs")

        if not use_cache:
            past_key = [None] * self.model_config.num_hidden_layers
            past_value = [None] * self.model_config.num_hidden_layers

        # input_embeds = self.embed_tokens(input_ids)
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            input_embeds = self.embedding(input_ids)
        else:
            input_embeds = input_ids
        attention_mask = ops.tile(attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

        hidden_states = input_embeds
        history_ks, history_vs = [], []
        for i, decoder in enumerate(self.decoders):
            hidden_states, history_k, history_v = decoder(hidden_states,
                                                          attention_mask,
                                                          position_ids,
                                                          past_key[i],
                                                          past_value[i],
                                                          use_cache=use_cache,
                                                          )
            history_ks.append(history_k)
            history_vs.append(history_v)
        return hidden_states, history_ks, history_vs


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
        h_shape = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [h_shape[0] * h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        out = ops.reshape(lm_head, [1, *list(lm_head.shape)])
        return out


@register_causallm(['baichuan2'])
class Baichuan2CausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self.config.get("weights", self._load_state_dict())
        num_key_value_heads = config.num_key_value_heads if hasattr(
            config, 'num_key_value_heads') else config.num_attention_heads
        config.update({'num_key_value_heads': num_key_value_heads})
        self.model = Baichuan2Model(config, weights, prefix)
        self.norm_and_lm_head = NormLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        block_num = len(gen_layer_ids(self.config.get('num_hidden_layers')))
        head_num = self.config.num_attention_heads
        kv_head_num = self.config.num_key_value_heads
        hidden_size = self.config.hidden_size
        model_name = 'baichuan'

        with Graph(model_name) as self.aipu_g:
            if is_gen_subgraph:
                hidden_states = ops.input([1, seq_len], Dtype.FP32, name='input_ids')
            else:
                hidden_states = ops.input([1, seq_len, self.config.hidden_size],
                                          Dtype.FP32, name='embeded')

            if use_cache and past_len != 0:
                attention_mask = ops.input([1, 1, seq_len, seq_len + past_len],
                                           Dtype.FP32, name='attention_mask')
            else:
                attention_mask = ops.input(
                    [1, 1, seq_len, seq_len], Dtype.FP32, name='attention_mask')
            position_ids = ops.input(([1, seq_len]), Dtype.INT32, name='position_ids')

            past_key, past_value = [], []
            if use_cache and past_len != 0:
                past_key = [ops.input([1, kv_head_num, past_len, hidden_size//head_num],
                                      Dtype.FP32, name=f"past_key_value.{i}.key") for i in range(block_num)]
                past_value = [ops.input([1, kv_head_num, past_len, hidden_size//head_num],
                                        Dtype.FP32, name=f"past_key_value.{i}.value") for i in range(block_num)]

            hidden_states, history_k, history_v = self.model(hidden_states,
                                                             attention_mask,
                                                             position_ids,
                                                             past_key=past_key,
                                                             past_value=past_value,
                                                             use_cache=use_cache,
                                                             with_embed=with_embed)
            _ = self.norm_and_lm_head(hidden_states)

            cur_tensors = list(self.aipu_g.output_tensors)
            history_k_v_name = [hkv.name for hkv in history_k + history_v]
            output_tensors = []
            for ct in cur_tensors:
                if ct.name not in history_k_v_name:
                    output_tensors.append(ct)
            output_tensors += history_k + history_v
            self.aipu_g.output_tensors = TensorList(output_tensors)
            self.aipu_g.input_tensors = TensorList([t for t in self.aipu_g.input_tensors[:3]] + past_key + past_value)

        return self.aipu_g
