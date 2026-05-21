from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np
import torch
import os
from AIPUBuilder.core import *
from AIPUBuilder.core import Graph, Node, Dtype, Tensor, Expr
from AIPUBuilder import ops
from AIPUBuilder import nn
from safetensors import safe_open
from ...basemodel import *
from ...config import PretrainedConfig
from ...register import register_causallm, gen_subgraph, register_update_config
from ...qinfo import QConf
from ...utils import update_tensor_data, convert_expr
# self.embed_tokens = nn.Embedding(self.vocab_size, config.thinker_config['text_config']['hidden_size'], self.padding_idx)
#         self.embed_weights = weights[f'thinker.model.embed_tokens.weight'].float()


@register_causallm(['qwen2.5thinker_embedding', ])
class Qwen2_5ThinkerEmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.embed_weights = config.weights.get('thinker.model.embed_tokens.weight')
        self.vocab_size = config.vocab_size  # config.thinker_config['text_config']['vocab_size']
        self.aipu_g = None
        self.prefix = prefix
        self.hidden_size = config.hidden_size  # config.thinker_config['text_config']['hidden_size']


@register_update_config(['qwen2.5thinker'])
def qwen2_5_omni_text_update_config(config):
    model_type = config.get('model_type', {})
    thinker_config = config.get('thinker_config', {})
    if len(thinker_config) == 0:
        return
    text_config = thinker_config.get('text_config', {})
    if len(text_config) == 0:
        return
    vision_config = thinker_config.get('vision_config', {})
    # _local_text_config = {
    #     'vocab_size': text_config.get('vocab_size', ),
    #     'hidden_size': text_config.get('hidden_size', 152064),
    #     'intermediate_size': text_config.get('intermediate_size', 18944),
    #     'hidden_act': text_config.get('hidden_act', 'silu'),
    #     'num_heads': text_config.get('num_attention_heads', 28),
    #     'num_hidden_layers': text_config.get('num_hidden_layers', 28),
    #     'num_key_value_heads': text_config.get('num_key_value_heads', 28),
    # }

    config.update(text_config)
    config.update({'vision_config': vision_config})
    config.pop('text_config')
    config.update({'model_type': model_type})
    config.update({'image_token_id': 151655})
    config.update({'video_token_id': 151656})
    config.update({'vision_start_token_id': 151652})
    config.update({'vision_end_token_id': 151653})
    # config.update({'num_hidden_layers': config.get('depth')})
    # config.update({'num_attention_heads': config.get('num_heads')})


class Qwen2_5ThinkerRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, repeat_kv=False, max_position_embeddings=2048, base=10000, device=None, name=''):
        super().__init__()

        self.config = config
        self.name = name

        if config.get('rope_scaling', None) is not None and 'type' in config.rope_scaling:
            if config.rope_scaling['type'] != 'default':
                WARN(f"Unsupported rope scaling type: {config.rope_scaling['type']}, now only support default scaling")

        base = config.get('rope_theta', 1000000.0)
        partial_rotary_factor = config.get('partial_rotary_factor', 1.0)

        dim = config.get('head_dim', config.hidden_size // config.num_attention_heads)
        dim = int(dim * partial_rotary_factor)
        max_position_embeddings = config.seq_len + config.past_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        inv_freq_expand = inv_freq.reshape([1, 1, -1, 1]).expand(3, 1, -1, 1)  # [3, bs, head_dim//2, 1]
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.int64).type_as(inv_freq)
        t = t.reshape(1, 1, 1, -1).expand(3, 1, 1, -1)  # [3, bs, 1, max_seq_len]
        freqs = inv_freq_expand.float() @ t.float()
        freqs = freqs.transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.attention_scaling = config.get('attention_scaling', 1.0)
        self.cos_cached = (emb.cos()*self.attention_scaling).to(torch.get_default_dtype())
        self.sin_cached = (emb.sin()*self.attention_scaling).to(torch.get_default_dtype())

    def forward(self, pos_ids):
        mrope_section = self.config.rope_scaling['mrope_section']
        mrope_section = mrope_section * 2
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        # pos_ids = pos_ids.unsqueeze(2)
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=2, batch_dims=2)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=2, batch_dims=2)
        sin_split = ops.split(gather_sin_out, mrope_section, axis=-1)
        cos_split = ops.split(gather_cos_out, mrope_section, axis=-1)
        sl = []
        for i, sin_s in enumerate(sin_split):
            s0, _, _, _ = sin_s.shape
            _, s1, s2, s3 = sin_s.shape.symbol
            sins = ops.crop(sin_s, crops=[[i % 3, i % 3+1], [0, s1], [0, s2], [0, s3]])
            sl.append(sins)
        sin_concat = ops.concat(sl, axis=-1).transpose([0, 2, 1, 3])
        cl = []
        for i, cos_s in enumerate(cos_split):
            s0, _, _, _ = cos_s.shape
            _, s1, s2, s3 = cos_s.shape.symbol
            coss = ops.crop(cos_s, crops=[[i % 3, i % 3+1], [0, s1], [0, s2], [0, s3]])
            cl.append(coss)
        cos_concat = ops.concat(cl, axis=-1).transpose([0, 2, 1, 3])
        return cos_concat, sin_concat


class Qwen2_5ThinkerMLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        gate_proj_w = weights.get(f'thinker.model.layers.{layer_id}.mlp.gate_proj.weight')
        up_proj_w = weights.get(f'thinker.model.layers.{layer_id}.mlp.up_proj.weight')
        down_proj_w = weights.get(f'thinker.model.layers.{layer_id}.mlp.down_proj.weight')
        update_tensor_data(self.gate_proj.weights, gate_proj_w, name=f'{prefix}layers.{layer_id}.mlp.gate_proj.weight')
        update_tensor_data(self.up_proj.weights, up_proj_w, name=f'{prefix}layers.{layer_id}.mlp.up_proj.weight')
        update_tensor_data(self.down_proj.weights, down_proj_w, name=f'{prefix}layers.{layer_id}.mlp.down_proj.weight')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5ThinkerAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size//self.num_heads  # config.thinker_config['text_config']['head_dim']
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings if not hasattr(
            self.config, 'max_len') or self.config.max_len is None else self.config.max_len
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
        #           f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        q_proj_w = weights.get(f'thinker.model.layers.{layer_id}.self_attn.q_proj.weight')
        q_proj_b = weights.get(f'thinker.model.layers.{layer_id}.self_attn.q_proj.bias')
        update_tensor_data(self.q_proj.weights, q_proj_w, f"{prefix}layers.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q_proj.biases, q_proj_b, f"{prefix}layers.{layer_id}.self_attn.q_proj.bias")

        k_proj_w = weights.get(f'thinker.model.layers.{layer_id}.self_attn.k_proj.weight')
        k_proj_b = weights.get(f'thinker.model.layers.{layer_id}.self_attn.k_proj.bias')
        update_tensor_data(self.k_proj.weights, k_proj_w, f"{prefix}layers.{layer_id}.self_attn.k_proj.weight")
        update_tensor_data(self.k_proj.biases, k_proj_b, f"{prefix}layers.{layer_id}.self_attn.k_proj.bias")

        v_proj_w = weights.get(f'thinker.model.layers.{layer_id}.self_attn.v_proj.weight')
        v_proj_b = weights.get(f'thinker.model.layers.{layer_id}.self_attn.v_proj.bias')
        update_tensor_data(self.v_proj.weights, v_proj_w, f"{prefix}layers.{layer_id}.self_attn.v_proj.weight")
        update_tensor_data(self.v_proj.biases, v_proj_b, f"{prefix}layers.{layer_id}.self_attn.v_proj.bias")

        o_proj_w = weights.get(f'thinker.model.layers.{layer_id}.self_attn.o_proj.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.self_attn.o_proj.weight")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scales = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scales)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"
        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):

        self.rotary_emb = Qwen2_5ThinkerRotaryEmbedding(self.config,
                                                        self.head_dim,
                                                        repeat_kv=self.num_key_value_groups > 1,
                                                        max_position_embeddings=self.max_position_embeddings,
                                                        base=self.rope_theta,
                                                        name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        bs_symb, seq_len_symb, hidden_size_symb = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symb * seq_len_symb, hidden_size_symb])

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        self.q_reshape = nn.Reshape([bs_symb, seq_len_symb, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs_symb, seq_len_symb, self.num_key_value_heads, self.head_dim])
        query = self.q_reshape(query)
        key = self.kv_reshape(key)
        value = self.kv_reshape(value)

        # query
        rtq_symbol0, rtq_symbol1, rtq_symbol2, _ = query.shape.symbol
        crop0 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [0, self.head_dim//2]])

        q_crop_out0 = crop0(query)
        q_crop_out1 = crop1(query)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        # key
        rtk_symbol0, rtk_symbol1, rtk_symbol2, _ = key.shape.symbol
        crop0 = nn.Crop([[0, rtk_symbol0], [0, rtk_symbol1], [0, rtk_symbol2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, rtk_symbol0], [0, rtk_symbol1], [0, rtk_symbol2], [0, self.head_dim//2]])

        k_crop_out0 = crop0(key)
        k_crop_out1 = crop1(key)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        # cos_emb, sin_emb = position_emb
        q_cosine = query * cos_emb
        k_cosine = key * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb
        query = q_cosine + q_r_sine
        key = k_cosine + k_r_sine

        query = self.qkv_transpose(query)
        key = self.qkv_transpose(key)
        value = self.qkv_transpose(value)

        k_cache_out = k_cache(key, position_ids=position_ids, block_idx=self.layer_id,
                              name=f"past_key_value.{self.layer_id}.key")
        v_cache_out = v_cache(value, position_ids=position_ids,
                              block_idx=self.layer_id, name=f"past_key_value.{self.layer_id}.value")
        value = v_cache_out[0]
        key = k_cache_out[0]

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = key.shape.symbol
            key_reshape = key.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            key_tile = ops.tile(key_reshape, [1, 1, self.num_key_value_groups, 1, 1])
            key = key_tile.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        if False and use_cache:
            # [CP-19313]C=A*B=(B^T*A^T)^T, when B size >> A size, swap input can improve wtram reuse
            kq_matmul = nn.Transpose([0, 1, 3, 2])(self.kq_matmul(key, query))
        else:
            kq_matmul = self.kq_matmul(query, key)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        softmax_masked = self.softmax(kq_matmul + attention_mask)

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = value.shape.symbol
            value0 = value.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            value1 = ops.tile(value0, [1, 1, self.num_key_value_groups, 1, 1])  # [16,1, seq, 128]
            value = value1.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        attn_output = self.kqv_matmul(softmax_masked, value)
        attn_output.op.name = f"{self.matmul_name}_matmul_kqv"

        # attn_output = nn.Transpose([1, 2, 0, 3])(attn_output)
        attn_output = nn.Transpose([0, 2, 1, 3])(attn_output)
        context_layer = ops.reshape(attn_output, [bs_symb * seq_len_symb, self.hidden_size])
        attn_output = self.o_proj(context_layer)

        return attn_output


class Qwen2_5ThinkerDecoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'thinker.model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = Qwen2_5ThinkerAttention(config, weights, layer_id, prefix)

        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'thinker.model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = Qwen2_5ThinkerMLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        # is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        # cur_scale = scale
        # if is_gen_subgraph and scale is not None and isinstance(scale, Tensor):
        #     """
        #     when enable_subgraph=true, will do some special operations:
        #     1. avoiding the constant output tensor is subgraph's input tensor, will re-generate one constant in subgraph.
        #     2. if one tensor is multi ops input, the subgraph will insert the duplicate input tensor into subgraph input tensors,
        #     which will cause big change for llm runtime. in order to avoid this graph, will insert reshape op, its output tensors will
        #     be as multi ops input, and reshape is including in this subgraph.
        #     """
        #     cur_scale = ops.constant(scale.op.constants['weights'].betensor.cpu().numpy())
        #     hidden_states = ops.reshape(hidden_states, hidden_states.shape)
        #     attention_mask = ops.tile(attention_mask, [1, self.config.num_attention_heads, 1, 1])
        #     position_ids = ops.reshape(position_ids, position_ids.shape)

        # INFO(f"\t build the {self.layer_id} Decoder block.")
        # residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states)
        # hidden_states, k, v = self.attention(hidden_states,
        #                                      attention_mask,
        #                                      position_ids,
        #                                      cur_scale,
        #                                      k_cache,
        #                                      v_cache,
        #                                      use_cache)
        # hidden_states = residual + hidden_states

        # residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        # hidden_states = self.mlp(hidden_states.reshape([*list(hidden_states.shape)[1:]]))
        # hidden_states = residual + hidden_states

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        bs_symb, seq_len_symb, hidden_size_symb = hidden_states.shape.symbol
        layernorm_output = self.input_layernorm(hidden_states)
        attn_output = self.attention(layernorm_output,
                                     attention_mask,
                                     position_ids,
                                     k_cache,
                                     v_cache,
                                     use_cache)
        residual = ops.reshape(residual, [bs_symb * seq_len_symb, hidden_size_symb])
        layernorm_input = attn_output + residual
        residual = layernorm_input
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        if len(layernorm_output.shape) > 2:
            layernorm_output = layernorm_output.reshape(bs_symb * seq_len_symb, hidden_size_symb)
        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output
        hidden_states = hidden_states.reshape([bs_symb, seq_len_symb, hidden_size_symb])

        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs

        # return hidden_states, k, v


class Qwen2_5ThinkerModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.thinker_config['text_config']['vocab_size']
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.thinker_config['text_config']['hidden_size'], self.padding_idx)
        self.embed_weights = weights[f'thinker.model.embed_tokens.weight'].float()

        # self.decoders = [Qwen2_5ThinkerDecoder(config, weights, 0)]
        self.blocks = [Qwen2_5ThinkerDecoder(config, weights, idx, prefix)
                       for idx in range(config.thinker_config['text_config']['num_hidden_layers'])]

    def forward(self,
                input_embeds=None,
                attention_mask=None,
                position_ids=None,
                past_key=None,
                past_value=None,
                use_cache=None,

                ):

        hidden_states = input_embeds
        history_ks, history_vs = [], []
        for i, blk in enumerate(self.blocks):
            hidden_states, history_k, history_v = blk(hidden_states,
                                                      attention_mask,
                                                      position_ids,
                                                      past_key,
                                                      past_value,
                                                      use_cache=use_cache,
                                                      )
            # history_ks.append(history_k)
            # history_vs.append(history_v)
        return hidden_states


class NormLmHead(nn.Module):
    def __init__(self, config, weights):
        self.config = config
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        update_tensor_data(self.norm.weights, weights.get(f'thinker.model.norm.weight'), f'normlm.norm.weight')

        self.lm_head = nn.Linear(hidden_size, config.vocab_size)
        key = 'thinker.lm_head.weight'
        if config.get('tie_word_embeddings', False):
            key = 'thinker.model.embed_tokens.weight'
        update_tensor_data(self.lm_head.weights, weights.get(key), f'normlm.lm.weight')

    @gen_subgraph(subgraph_name='lm')
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        h_shape = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [h_shape[0] * h_shape[1], h_shape[2]])
        lm_head = self.lm_head(hidden_states)
        out = ops.reshape(lm_head, [1, *list(lm_head.shape)])
        return out


@register_causallm(['qwen2.5thinker'])
class Qwen2_5OmniThinkerForConditionalGeneration(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        INFO(f"\t current num_hidden_layers = {config.num_hidden_layers}:")
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.model = Qwen2_5ThinkerModel(config, weights, prefix)
        self.norm_and_lm_head = NormLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")
        self.aipu_g = None
        self.prefix = prefix

    def forward(self, seq_len=None, use_cache=None, with_embed=False, past_len=None):  # prefix='thinker.model.' with_embed=True is prev graph
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        num_heads = self.config.num_attention_heads
        hidden_size = self.config.hidden_size
        # self.config.thinker_config['text_config']['head_dim']
        head_dim = hidden_size//num_heads

        with Graph('qwen') as self.aipu_g:

            hidden_states = ops.input([batch, seq_len, self.config.hidden_size],
                                      Dtype.FP32,
                                      name='inputs_embeds',
                                      exprs=['batch', 'seq_len', self.config.hidden_size])
            attention_mask = ops.input([batch, 1, seq_len, past_len + seq_len],
                                       Dtype.FP32,
                                       name='attention_mask',
                                       exprs=['batch', 1, 'seq_len', 'max_len'])
            position_ids = ops.input([3, batch, seq_len],
                                     Dtype.INT32,
                                     name='position_ids',
                                     exprs=[3, 'batch', 'seq_len'])

            scale = nn.Constant(
                np.array(1./np.sqrt(np.array(head_dim))).astype('float32').reshape([1]))()
            scale.op.constants['weights'].name = f"scale_for_normalization"
            kcache_module = nn.KVCacheModule(self.config, name='kcache')
            vcache_module = nn.KVCacheModule(self.config, name='vcache')
            hidden_states = self.model(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       past_key=kcache_module,
                                       past_value=vcache_module,
                                       use_cache=use_cache)
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
