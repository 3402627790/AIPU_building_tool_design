import numpy as np
import torch
import os
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


@register_causallm(['minicpm2_6_llm_embedding'])
class Minicpm2_6EmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get(f'llm.model.embed_tokens.weight')


class CPMllmRotaryEmbedding(nn.Module):
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
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=torch.int64).type_as(self.inv_freq)
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
        # sin_out_reshape = ops.reshape(
        #     gather_sin_out, [1, *gather_sin_out.shape])
        # cos_out_reshape = ops.reshape(
        #     gather_cos_out, [1, *gather_cos_out.shape])
        # sin_reshape_tile = ops.tile(sin_out_reshape, [1, self.num_heads, 1, 1])
        # cos_reshape_tile = ops.tile(cos_out_reshape, [1, self.num_heads, 1, 1])
        sin_new_shape = list(gather_sin_out.shape.symbol)[:1] + [1] + list(gather_sin_out.shape.symbol)[1:]
        cos_new_shape = list(gather_cos_out.shape.symbol)[:1] + [1] + list(gather_cos_out.shape.symbol)[1:]
        sin_out_reshape = gather_sin_out.reshape(sin_new_shape)
        cos_out_reshape = gather_cos_out.reshape(cos_new_shape)
        if self.name != '':
            sin_out.op.constants['weights'].name = f'{self.name}.rotary_emb.sin_weight'
            cos_out.op.constants['weights'].name = f'{self.name}.rotary_emb.cos_weight'
            sin_out.op.name = f"{self.name}.rotary_emb.sin_{sin_out.op.name}"
            cos_out.op.name = f"{self.name}.rotary_emb.cos_{cos_out.op.name}"
            gather_sin_out.name = f"{self.name}.{gather_sin_out.name}"
            gather_cos_out.name = f"{self.name}.{gather_cos_out.name}"

        return cos_out_reshape, sin_out_reshape


class CPMllmMLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        gate_proj_w = weights.get(
            f'llm.model.layers.{layer_id}.mlp.gate_proj.weight')
        up_proj_w = weights.get(
            f'llm.model.layers.{layer_id}.mlp.up_proj.weight')
        down_proj_w = weights.get(
            f'llm.model.layers.{layer_id}.mlp.down_proj.weight')
        update_tensor_data(self.gate_proj.weights, gate_proj_w,
                           name=f'{prefix}layers.{layer_id}.mlp.gate_proj.weight')
        update_tensor_data(self.up_proj.weights, up_proj_w,
                           name=f'{prefix}layers.{layer_id}.mlp.up_proj.weight')
        update_tensor_data(self.down_proj.weights, down_proj_w,
                           name=f'{prefix}layers.{layer_id}.mlp.down_proj.weight')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CPMllmAttention(nn.Module):

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

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size)

        q_proj_w = weights[f'llm.model.layers.{layer_id}.self_attn.q_proj.weight'].float()
        update_tensor_data(self.q_proj.weights, q_proj_w, f"{prefix}layers.{layer_id}.self_attn.q_proj.weight")
        q_proj_b = weights[f'llm.model.layers.{layer_id}.self_attn.q_proj.bias'].float()
        update_tensor_data(self.q_proj.biases, q_proj_b, f"{prefix}layers.{layer_id}.self_attn.q_proj.biases")

        k_proj_w = weights[f'llm.model.layers.{layer_id}.self_attn.k_proj.weight'].float()
        update_tensor_data(self.k_proj.weights, k_proj_w, f"{prefix}layers.{layer_id}.self_attn.k_proj.weight")
        k_proj_b = weights[f'llm.model.layers.{layer_id}.self_attn.k_proj.bias'].float()
        update_tensor_data(self.k_proj.biases, k_proj_b, f"{prefix}layers.{layer_id}.self_attn.k_proj.biases")

        v_proj_w = weights[f'llm.model.layers.{layer_id}.self_attn.v_proj.weight'].float()
        update_tensor_data(self.v_proj.weights, v_proj_w, f"{prefix}layers.{layer_id}.self_attn.v_proj.weight")
        v_proj_b = weights[f'llm.model.layers.{layer_id}.self_attn.v_proj.bias'].float()
        update_tensor_data(self.v_proj.biases, v_proj_b, f"{prefix}layers.{layer_id}.self_attn.v_proj.biases")

        o_proj_w = weights.get(f'llm.model.layers.{layer_id}.self_attn.o_proj.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.self_attn.o_proj.weight")
        o_proj_b = weights.get(f'llm.model.layers.{layer_id}.self_attn.o_proj.bias')
        update_tensor_data(self.o_proj.biases, o_proj_b, f"{prefix}layers.{layer_id}.self_attn.o_proj.biases")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        scales = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scales)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"
        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape.symbol
        bs_symbol, seq_len_symbol, _ = hidden_states.shape.symbol
        # bs, seq_len, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bs_symbol * seq_len_symbol, hidden_size])
        # crop0 = nn.Crop([[0, None], [0, None], [0, None], [self.head_dim//2, self.head_dim]])
        # crop1 = nn.Crop([[0, None], [0, None], [0, None], [0, self.head_dim//2]])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        self.rotary_emb = CPMllmRotaryEmbedding(self.num_heads,
                                                self.head_dim,
                                                max_position_embeddings=attention_mask.shape[-1],
                                                base=self.rope_theta,
                                                name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        self.q_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs_symbol, seq_len_symbol, self.num_key_value_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.q_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.kv_reshape(key_states))

        # query
        rtq_symbol0, rtq_symbol1, rtq_symbol2, _ = reshaped_trans_query.shape.symbol
        crop0 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [self.head_dim//2, self.head_dim]])
        crop1 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [0, self.head_dim//2]])
        q_crop_out0 = crop0(reshaped_trans_query)
        q_crop_out1 = crop1(reshaped_trans_query)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        # key
        rtq_symbol0, rtq_symbol1, rtq_symbol2, _ = reshaped_trans_key.shape.symbol
        kcrop0 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [self.head_dim//2, self.head_dim]])
        kcrop1 = nn.Crop([[0, rtq_symbol0], [0, rtq_symbol1], [0, rtq_symbol2], [0, self.head_dim//2]])
        k_crop_out0 = kcrop0(reshaped_trans_key)
        k_crop_out1 = kcrop1(reshaped_trans_key)
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
            k_values = ops.reshape(k_cache_out, [
                                   1, self.num_key_value_heads, 1, k_cache_out.shape[-2], k_cache_out.shape[-1]])
            k_values = ops.tile(k_values, [1, 1, self.num_key_value_groups, 1, 1])
            k_cache_out = ops.reshape(
                k_values, [1, self.num_heads, k_cache_out.shape[-2], k_cache_out.shape[-1]])
        kq_matmul = self.kq_matmul(q_add_out, k_cache_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"
        '''
        matmul has added params['fused_multiplier'] for scale,
        scale = 1.0 / np.sqrt(np.array(head_dim)).astype(np.float32) has merged to matmul
        '''
        softmax_masked = self.softmax(kq_matmul + attention_mask)

        reshaped_trans_value = self.qkv_transpose(
            self.kv_reshape(value_states))
        v_cache_out = v_cache(reshaped_trans_value, position_ids=position_ids,
                              block_idx=self.layer_id, name=f"past_key_value.{self.layer_id}.value")
        v_cache_out = v_cache_out[0]
        if self.num_key_value_groups > 1:
            history_v = v_cache_out
            v_values = ops.reshape(history_v, [
                                   1, self.num_key_value_heads, 1, history_v.shape[-2], history_v.shape[-1]])
            v_values = ops.tile(
                v_values, [1, 1, self.num_key_value_groups, 1, 1])
            v_cache_out = ops.reshape(
                v_values, [1, self.num_heads, history_v.shape[-2], history_v.shape[-1]])
        kqv_out = self.kqv_matmul(softmax_masked, v_cache_out)
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"
        # kqv_out.name = f"{self.matmul_name}_{kqv_out.name}"

        self.qkv_reshape = nn.Reshape(
            [bs_symbol * seq_len_symbol, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.o_proj(reshaped_trans)

        return o_t


class CPMllmDecoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'llm.model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = CPMllmAttention(config, weights, layer_id, prefix)

        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'llm.model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = CPMllmMLP(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            hidden_states = ops.reshape(hidden_states, hidden_states.shape)
            position_ids = ops.reshape(position_ids, position_ids.shape)
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
        hidden_states = hidden_states.reshape([bs_symbol * seq_len_symbol, hidden_states.shape[-1]])
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.reshape([bs_symbol, seq_len_symbol, hidden_states.shape[1]])
        hidden_states = residual + hidden_states
        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class CPMllmModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        # config.num_hidden_layers = 1
        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights[f'llm.model.embed_tokens.weight'].float()

        # self.decoders = [CPMllmDecoder(config, weights, 0)]
        self.decoders = [CPMllmDecoder(config, weights, idx, prefix)
                         for idx in gen_layer_ids(config.num_hidden_layers)]

        # self.norm.weights.name = f'{prefix}norm.weight'

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
                                       self.model_config.vocab_size, self.model_config.hidden_size])
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        else:
            input_embeds = input_ids
        if not self.model_config.only_embed:
            # attention_mask = ops.cast(attention_mask, Dtype.FP32, ignore_scale_zp=True)
            # attention_mask = ops.tile(
            #     attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

            hidden_states = input_embeds
            for i, decoder in enumerate(self.decoders):
                hidden_states, _, _ = decoder(hidden_states,
                                              attention_mask,
                                              position_ids,
                                              past_key,
                                              past_value,
                                              use_cache=use_cache,)

            return hidden_states
        else:

            return input_embeds


@register_causallm(['minicpm2_6_llm'])
class CPMllmCausalLM(BaseModel):
    def __init__(self, config, prefix='', only_embed=False):
        super().__init__(config)
        weights = self._load_state_dict()
        # config.num_hidden_layers = 32
        config.update({'weights': weights})
        config.update({'only_embed': only_embed})
        # config.update({'attn_layers': attn_layers})
        self.model = CPMllmModel(config, weights, prefix)
        if not config.only_embed:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            update_tensor_data(self.norm.weights, weights.get(
                f'llm.model.norm.weight'), f'normlm.norm.weight')
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
            self.lm_head.weights.betensor = weights[f'llm.lm_head.weight'].float(
            )
            self.lm_head.weights.sync_betensor()
            # self.lm_head.weights.name = f'lm_head.weight'
            self.lm_head.weights.name = f'normlm.lm.weight'
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.get('seq_len', self.config.get('max_position_embeddings', 5))
        past_len = self.config.past_len
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        self.num_hidden_layers = gen_layer_ids(self.config.get('num_hidden_layers'))
        model_name = 'cpmllm'

        if self.config.only_embed:
            model_name = 'embedding'
            with_embed = True
        with Graph(model_name) as self.aipu_g:
            if with_embed:
                hidden_states = ops.input([batch, seq_len], Dtype.FP32, name='input_ids', exprs=['batch', 'seq_len'])
            else:
                hidden_states = ops.input([batch, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded', exprs=[
                                          'batch', 'seq_len', self.config.hidden_size])

            if not self.config.only_embed:
                attention_mask = ops.input(
                    [batch, 1, seq_len, seq_len + past_len],
                    Dtype.FP32,
                    name='attention_mask',
                    exprs=['batch', 1, 'seq_len', 'seq_len'])

                position_ids = ops.input(
                    [batch, seq_len],
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

                # _ = self.norm_and_lm_head(hidden_states)

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
            else:
                hidden_states = self.model(hidden_states,
                                           use_cache=self.use_cache,
                                           with_embed=self.with_embed)
        return self.aipu_g
