import numpy as np
import torch
import os
from safetensors import safe_open
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder.logger import INFO

from AIPUBuilder.model import (QConf,
                               update_tensor_data,
                               register_causallm,
                               register_model_build,
                               register_model_quantize,
                               ModelBuild,
                               ModelQuantize,
                               BaseModel)


class Phi2RotaryEmbedding(nn.Module):
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


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Phi
class Phi2MLP(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        fc1_w = weights.get(f'model.layers.{layer_id}.mlp.fc1.weight')
        fc1_b = weights.get(f'model.layers.{layer_id}.mlp.fc1.bias')
        fc2_w = weights.get(f'model.layers.{layer_id}.mlp.fc2.weight')
        fc2_b = weights.get(f'model.layers.{layer_id}.mlp.fc2.bias')
        update_tensor_data(self.fc1.weights, fc1_w, name=f'{prefix}layers.{layer_id}.mlp.fc1.weight')
        update_tensor_data(self.fc1.biases, fc1_b, name=f'{prefix}layers.{layer_id}.mlp.fc1.bias')
        update_tensor_data(self.fc2.weights, fc2_w, name=f'{prefix}layers.{layer_id}.mlp.fc2.weight')
        update_tensor_data(self.fc2.biases, fc2_b, name=f'{prefix}layers.{layer_id}.mlp.fc2.bias')

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))


class Phi2Attention(nn.Module):

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
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        if (self.head_dim * self.num_heads) != self.hidden_size:
            ERROR(f"hidden_size must be divisible by num_heads (got `hidden_size`: "
                  f"{self.hidden_size} and `num_heads`: {self.num_heads}).")

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

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

        dense_w = weights.get(f'model.layers.{layer_id}.self_attn.dense.weight')
        dense_b = weights.get(f'model.layers.{layer_id}.self_attn.dense.bias')
        update_tensor_data(self.dense.weights, dense_w, f"{prefix}layers.{layer_id}.self_attn.dense.weight")
        update_tensor_data(self.dense.biases, dense_b, f"{prefix}layers.{layer_id}.self_attn.dense.bias")

        self.qkv_transpose = nn.Transpose([0, 2, 1, 3])

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        scales = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        if 1:
            self.scale = nn.Constant(np.array(1./np.sqrt(np.array(self.head_dim))).astype('float32').reshape([1]))
            self.scale.weights.name = f"scale_for_normalization"
        else:
            self.bn_q = nn.BatchNorm(self.head_dim)
            self.bn_q.weights.betensor = torch.full(list(self.bn_q.weights.shape), np.sqrt(
                np.array(scales))).to(self.bn_q.weights.betensor.device)
            self.bn_q.weights.sync_betensor()
            self.bn_q.weights.name = f"{prefix}layers.{layer_id}.self_attn.scaleq"

            self.bn_k = nn.BatchNorm(256, axis=2)
            self.bn_k.weights.betensor = torch.full(list(self.bn_k.weights.shape), np.sqrt(
                np.array(scales))).to(self.bn_k.weights.betensor.device)
            self.bn_k.weights.sync_betensor()
            self.bn_k.weights.name = f"{prefix}layers.{layer_id}.self_attn.scalek"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bs * seq_len, hidden_size])

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        self.rotary_emb = Phi2RotaryEmbedding(self.num_heads,
                                              int(self.head_dim *
                                                  self.partial_rotary_factor),
                                              max_position_embeddings=seq_len,
                                              base=self.rope_theta,
                                              name=f'{self.prefix}layers.{self.layer_id}.self_attn')
        cos_emb, sin_emb = self.rotary_emb(position_ids)

        self.q_reshape = nn.Reshape([bs, seq_len, self.num_heads, self.head_dim])
        self.kv_reshape = nn.Reshape([bs, seq_len, self.num_key_value_heads, self.head_dim])
        reshaped_trans_query = self.qkv_transpose(self.q_reshape(query_states))
        reshaped_trans_key = self.qkv_transpose(self.kv_reshape(key_states))

        query_rot, query_pass = ops.split(
            reshaped_trans_query, [self.rotary_emb.dim, self.head_dim - self.rotary_emb.dim], axis=-1)
        key_rot, key_pass = ops.split(
            reshaped_trans_key, [self.rotary_emb.dim, self.head_dim - self.rotary_emb.dim], axis=-1)

        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        # query
        crop0 = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [self.rotary_emb.dim//2, self.rotary_emb.dim]])
        crop1 = nn.Crop([[0, bs], [0, self.num_heads], [0, seq_len], [0, self.rotary_emb.dim//2]])
        q_crop_out0 = crop0(query_rot)
        q_crop_out1 = crop1(query_rot)
        neg_q_crop_out0 = ops.negative(q_crop_out0)
        rope_query = nn.Concat(-1)([neg_q_crop_out0, q_crop_out1])

        # key
        k_crop_out0 = crop0(key_rot)
        k_crop_out1 = crop1(key_rot)
        neg_k_crop_out0 = ops.negative(k_crop_out0)
        rope_key = nn.Concat(-1)([neg_k_crop_out0, k_crop_out1])

        neg_q_crop_out0.op.name = f"{self.matmul_name}.negative_0"
        neg_k_crop_out0.op.name = f"{self.matmul_name}.negative_1"

        q_cosine = query_rot * cos_emb
        k_cosine = key_rot * cos_emb
        q_r_sine = rope_query * sin_emb
        k_r_sine = rope_key * sin_emb

        q_add_out = q_cosine + q_r_sine
        k_add_out = k_cosine + k_r_sine

        q_add_out = nn.Concat(-1)([q_add_out, query_pass])
        k_add_out = nn.Concat(-1)([k_add_out, key_pass])

        history_k, history_v = None, None
        if use_cache:
            history_k = nn.Concat(2)([k_cache, k_add_out])
            k_add_out = history_k

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = list(k_add_out.shape)
            k_add_out = k_add_out.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            k_add_out = ops.tile(k_add_out, [1, 1, self.num_key_value_groups, slen, head_dim])
            k_add_out = k_add_out.reshape([batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        kq_matmul = self.kq_matmul(q_add_out, k_add_out)
        kq_matmul.op.name = f"{self.matmul_name}_matmul_kq"

        self.scale = nn.Constant(np.array(1./np.sqrt(np.array(self.head_dim))).astype('float32').reshape([1]))
        self.scale.weights.name = f"scale_for_normalization"
        scaled_kq = kq_matmul * self.scale()

        softmax_masked = self.softmax(scaled_kq + attention_mask)

        reshaped_trans_value = self.qkv_transpose(self.kv_reshape(value_states))
        if use_cache:
            history_v = nn.Concat(2)([v_cache, reshaped_trans_value])
            reshaped_trans_value = history_v

        if self.num_key_value_groups > 1:
            batch, num_key_value_heads, slen, head_dim = list(reshaped_trans_value.shape)
            reshaped_trans_value = reshaped_trans_value.reshape([batch, num_key_value_heads, 1, slen, head_dim])
            reshaped_trans_value = ops.tile(reshaped_trans_value, [1, 1, self.num_key_value_groups, slen, head_dim])
            reshaped_trans_value = reshaped_trans_value.reshape(
                [batch, num_key_value_heads * self.num_key_value_groups, slen, head_dim])

        kqv_out = self.kqv_matmul(softmax_masked, reshaped_trans_value)
        kqv_out.op.name = f"{self.matmul_name}_matmul_kqv"

        self.qkv_reshape = nn.Reshape([bs*seq_len, self.hidden_size])
        reshaped_trans = self.qkv_reshape(self.qkv_transpose(kqv_out))
        o_t = self.dense(reshaped_trans)

        return o_t, history_k, history_v


class Phi2Decoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")
        update_tensor_data(self.input_layernorm.biases,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.bias'),
                           f"{prefix}layers.{layer_id}.input_layernorm.bias")

        self.attention = Phi2Attention(config, weights, layer_id, prefix)

        self.mlp = Phi2MLP(config, weights, layer_id, prefix)

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):

        residual = hidden_states
        ln_out = self.input_layernorm(hidden_states)
        attn_outputs, k, v = self.attention(ln_out,
                                            attention_mask,
                                            position_ids,
                                            k_cache,
                                            v_cache,
                                            use_cache)
        mlp_out = self.mlp(ln_out.reshape([*list(hidden_states.shape)[1:]]))
        hidden_states = residual + mlp_out
        hidden_states = attn_outputs + hidden_states

        return hidden_states, k, v


class Phi2Model(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        # config.num_hidden_layers = 1
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights[f'model.embed_tokens.weight'].float()

        self.decoders = [Phi2Decoder(config, weights, idx, prefix) for idx in range(config.num_hidden_layers)]

        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        update_tensor_data(self.final_layernorm.weights, weights.get(
            f'model.final_layernorm.weight'), f'normlm.final_layernorm.weight')
        update_tensor_data(self.final_layernorm.biases, weights.get(
            f'model.final_layernorm.bias'), f'normlm.final_layernorm.bias')

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
            ERROR(f"when use_kv_cache, please provide the kv inputs")
        if use_cache:
            assert len(past_key) == self.model_config.num_hidden_layers
            assert len(past_value) == self.model_config.num_hidden_layers
        if not use_cache:
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

        attention_mask = ops.cast(attention_mask, Dtype.FP32, ignore_scale_zp=True)
        attention_mask = ops.tile(attention_mask, [1, self.model_config.num_key_value_heads, 1, 1])

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
            if use_cache:
                history_ks.append(history_k)
                history_vs.append(history_v)
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, history_ks, history_vs


@register_causallm(['phi2', 'phi'])
class Phi2CausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        # config.num_hidden_layers = 1
        config.update({'weights': weights})
        self.model = Phi2Model(config, weights, prefix)
        # self.decoder = Phi2Decoder(config, weights, 0)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        update_tensor_data(self.lm_head.weights, weights.get(f'lm_head.weight'), f'normlm.lm.weight')
        update_tensor_data(self.lm_head.biases, weights.get(f'lm_head.bias'), f'normlm.lm.bias')

        self.aipu_g = None
        self.prefix = prefix

    def forward(self, seq_len=None, use_cache=None, with_embed=False):  # prefix='model.' with_embed=True is prev graph
        seq_len = self.config.max_position_embeddings if seq_len is None else seq_len
        self.config.seq_len = seq_len
        self.with_embed = with_embed
        block_num = self.config.num_hidden_layers
        head_num = self.config.num_key_value_heads
        hidden_size = self.config.hidden_size
        self.use_cache = use_cache if use_cache is not None else self.config.use_cache
        with Graph('phi') as self.aipu_g:
            if self.with_embed:
                hidden_states = ops.input([1, seq_len], Dtype.FP32, name='input_ids')
            else:
                hidden_states = ops.input(
                    [1, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded')

            if self.use_cache:
                attention_mask = ops.input([1, 1, seq_len, seq_len * 2],
                                           Dtype.INT32, name='attention_mask')
            else:
                attention_mask = ops.input([1, seq_len], Dtype.INT32, name='attention_mask')
            # attention_mask = Tensor('attention_mask', [1, seq_len]), Dtype.INT32)
            position_ids = ops.input([1, seq_len], Dtype.INT32, name='position_ids')

            past_key, past_value = None, None
            if self.use_cache:
                past_key = [ops.input([1, head_num, seq_len, hidden_size//head_num],
                                      Dtype.FP32, name=f"past_key_value.{i}.key") for i in range(block_num)]
                past_value = [ops.input([1, head_num, seq_len, hidden_size//head_num],
                                        Dtype.FP32, name=f"past_key_value.{i}.value") for i in range(block_num)]

            hidden_states, history_k, history_v = self.model(hidden_states,
                                                             attention_mask,
                                                             position_ids,
                                                             past_key=past_key,
                                                             past_value=past_value,
                                                             use_cache=self.use_cache,
                                                             with_embed=self.with_embed)
            h_shape = hidden_states.shape
            hidden_states = ops.reshape(hidden_states, [h_shape[0]*h_shape[1], h_shape[2]])
            lm_head = self.lm_head(hidden_states)
            _ = ops.reshape(lm_head, [1, *list(lm_head.shape)])
            if self.use_cache:
                self.aipu_g.output_tensors = TensorList([t for t in self.aipu_g.output_tensors] + history_k + history_v)
                self.aipu_g.input_tensors = TensorList(
                    [t for t in self.aipu_g.input_tensors[:3]] + past_key + past_value)
        return self.aipu_g


@register_model_quantize(['phi_w4', 'phi2_w4', 'phi_w8'])
class Phi2W4ModelQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.bits = int(''.join(list(filter(str.isdigit, quantization_type))))
        if not self.bits in list(range(4, 17)):
            ERROR(f"quantization bits should be in {list(range(4, 17))}")
        self.hf_weights = hf_config.weights

    def update_qinfo(self):
        for n in self.aipu_g.nodes:

            if n.type in [OpType.Convolution, OpType.FullyConnected]:
                w = n.constants['weights']
                w.quantization.bits = self.bits
                w.quantization.scheme = Scheme.Asymmetric if self.bits < 8 else Scheme.Symmetric
                # w.quantization.scheme = Scheme.Symmetric
                w.quantization.granularity = Granularity.Per_Channel
                w.quantization.quantized_range = QuantizedRange.Full_Range
                w.quantization.aipu_dtype = Dtype.INT8 if self.bits < 9 else Dtype.INT16
                w.quantization.group = w.shape[0]
                trigger_mode = f"FLOAT16_ACT_INT{self.bits}_WHT"
                n.attrs['trigger_float_op'] = NodeParamValue(trigger_mode)

            else:
                n.attrs['trigger_float_op'] = NodeParamValue(str(Dtype.FP16))

            n.attrs['unquantifiable'] = True
            n.params['unquantifiable'] = True

            if n.type in [OpType.Softmax]:
                n.attrs['min_compatible_zhouyi_target'] = 'X3_1304'
                n.attrs['approx_params'] = NodeParamValue(int(1))
                n.attrs['lut_items_in_bits'] = 9
            if n.type in [OpType.Activation]:
                n.attrs['min_compatible_zhouyi_target'] = NodeParamValue('X3_1304')
                n.attrs['approx_params'] = NodeParamValue(0)
                n.attrs['lut_items_in_bits'] = 9
            if n.type == OpType.Input:
                if n.outputs[0].dtype.is_float:
                    n.outputs[0].dtype = Dtype.FP16

            for ot in n.outputs:
                if ot.io_tensor and ot.dtype.is_float:
                    ot.dtype = Dtype.FP16
