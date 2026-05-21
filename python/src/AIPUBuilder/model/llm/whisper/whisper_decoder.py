import numpy as np
import torch
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder._C._tongue import Symbol, Expr
from ...basemodel import BaseModel
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data
from ..llmutils import gen_layer_ids


def register_op_name(t, name):
    t.op.name = name


class MultiHeadAttention(nn.Module):
    def __init__(self, config, weights, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.cross_attn = False
        self.weights = weights
        self.config = config
        self.n_state = config.d_model
        self.n_ctx = config.max_source_positions
        self.n_head = config.decoder_attention_heads
        self.scale = (self.n_state // self.n_head) ** -0.25
        self.query = nn.Linear(self.n_state, self.n_state)
        self.key = nn.Linear(self.n_state, self.n_state)
        self.value = nn.Linear(self.n_state, self.n_state)
        self.out = nn.Linear(self.n_state, self.n_state)
        self.matmul = nn.MatMul()
        self.matmul_transb = nn.MatMul(trans_a=False, trans_b=True)
        self.softmax = nn.Softmax()
        self.update_fc_weights_bias()
        self.concat_d1 = nn.Concat(axis=1)
        self.concat_d0 = nn.Concat(axis=0)

    def update_fc_weights_bias(self):
        def get_weights_biases_name(self, prefix):
            if self.cross_attn:
                weights_name = f'model.decoder.layers.{self.layer_id}.encoder_attn.{prefix}_proj.weight'
                bias_name = f'model.decoder.layers.{self.layer_id}.encoder_attn.{prefix}_proj.bias'
            else:
                weights_name = f'model.decoder.layers.{self.layer_id}.self_attn.{prefix}_proj.weight'
                bias_name = f'model.decoder.layers.{self.layer_id}.self_attn.{prefix}_proj.bias'
            return weights_name, bias_name

        update_tensor_data(self.query.weights, self.weights.get(get_weights_biases_name(
            self, 'q')[0])['data'] * self.scale, get_weights_biases_name(self, 'q')[0])
        update_tensor_data(self.query.biases, self.weights.get(get_weights_biases_name(
            self, 'q')[1])['data'] * self.scale, get_weights_biases_name(self, 'q')[1])

        k_scale = 1 if self.cross_attn else self.scale
        update_tensor_data(self.key.weights, self.weights.get(get_weights_biases_name(
            self, 'k')[0])['data'] * k_scale, get_weights_biases_name(self, 'k')[0])

        update_tensor_data(self.value.weights, self.weights.get(
            get_weights_biases_name(self, 'v')[0])['data'], get_weights_biases_name(self, 'v')[0])
        update_tensor_data(self.value.biases, self.weights.get(
            get_weights_biases_name(self, 'v')[1])['data'], get_weights_biases_name(self, 'v')[1])

        update_tensor_data(self.out.weights, self.weights.get(
            get_weights_biases_name(self, 'out')[0])['data'], get_weights_biases_name(self, 'out')[0])
        update_tensor_data(self.out.biases, self.weights.get(
            get_weights_biases_name(self, 'out')[1])['data'], get_weights_biases_name(self, 'out')[1])

    def forward(self, x, k_cache, v_cache, position_ids=None):
        bs, seq_len, _ = x.shape
        bs_symbol, seq_len_symbol, _ = x.shape.symbol
        x = ops.reshape(x, [bs_symbol * seq_len_symbol, self.n_state])
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        register_op_name(q, f"/blocks.{self.layer_id}/attn/query/MatMul")
        register_op_name(k, f"/blocks.{self.layer_id}/attn/key/MatMul")
        register_op_name(v, f"/blocks.{self.layer_id}/attn/value/MatMul")

        if seq_len == 1:
            q = ops.reshape(q, [bs_symbol, self.n_head, seq_len_symbol, self.n_state // self.n_head])
            k = ops.reshape(k, [bs_symbol, self.n_head, seq_len_symbol, self.n_state//self.n_head])
        else:
            k = ops.reshape(k, [bs_symbol, seq_len_symbol, self.n_head, self.n_state // self.n_head])
            k = ops.transpose(k, [0, 2, 1, 3])
            q = ops.reshape(q, [bs_symbol, seq_len_symbol, self.n_head, self.n_state // self.n_head])
            q = ops.transpose(q, [0, 2, 1, 3])

        k_cache_out = k_cache(k, position_ids=position_ids, block_idx=self.layer_id)

        v = ops.reshape(v, [bs_symbol, self.n_head, seq_len_symbol, self.n_state // self.n_head])
        v_cache_out = v_cache(v, position_ids=position_ids, block_idx=self.layer_id)

        k = k_cache_out[0]
        if seq_len == 1:
            # [CP-19313]C=A*B=(B^T*A^T)^T, when B size >> A size, swap input can improve wtram reuse
            # qk = nn.Transpose([0, 1, 3, 2])(self.matmul_transb(k, q))
            qk = self.matmul_transb(k, q)
            _, _, kv_len, _ = qk.shape
            # for [CP-4660]
            # reshape is used instead of transpose because gsim will optimize softmax(axis=-1) to softmax(axis=-2), and currently, softmax only supports dynamic shape for axis=-1
            # Symbol(f"d5") represents max_seqlen. it is shown that setting ds_output_shape is to enable gb to derive the dynamic dimensions.
            # qk = ops.reshape(qk, [bs_symbol, self.n_head, seq_len_symbol, k_len_symbol])
            # When dynamic_input_dim is None, here does not require `ds_output_shape`; GB will automatically infer it. CP-29905
            qk = ops.reshape(qk, [bs, self.n_head, seq_len, kv_len])
            # qk.op.params['ds_output_shape'] = [[bs_symbol, self.n_head, seq_len_symbol, Expr(Symbol(f"max_len"))]]
        else:
            qk = self.matmul_transb(q, k)
        register_op_name(qk, f'/blocks.{self.layer_id}/attn/MatMul')
        w = self.softmax(qk)
        register_op_name(w, f'/blocks.{self.layer_id}/attn/Softmax')

        v = v_cache_out[0]
        qkv = self.matmul(w, v)
        register_op_name(qkv, f'/blocks.{self.layer_id}/attn/MatMul_1')

        qkv = ops.reshape(qkv, [bs_symbol * seq_len_symbol, self.n_state])
        out = self.out(qkv)
        register_op_name(out, f'/blocks.{self.layer_id}/attn/out/MatMul')
        out = ops.reshape(out, [bs_symbol, seq_len_symbol, self.n_state])

        return out


class CrossMultiHeadAttention(MultiHeadAttention):
    def __init__(self, config, weights, layer_id):
        super().__init__(config, weights, layer_id)
        self.config = config
        self.cross_attn = True
        self.update_fc_weights_bias()

    def forward(self, x, k_proc, v_proc):
        bs_symbol, seq_len_symbol, _ = x.shape.symbol
        x = ops.reshape(x, [bs_symbol * seq_len_symbol, self.n_state])
        q = self.query(x)
        register_op_name(q, f"/blocks.{self.layer_id}/cross_attn/query/MatMul")

        q = ops.reshape(q, [self.n_head, bs_symbol, self.n_state // self.n_head, seq_len_symbol])
        qk = self.matmul(k_proc, q)
        register_op_name(qk, f'/blocks.{self.layer_id}/cross_attn/MatMul')

        qk = ops.transpose(qk, [0, 1, 3, 2])
        w = self.softmax(qk)
        register_op_name(w, f'/blocks.{self.layer_id}/cross_attn/Softmax')

        qkv = self.matmul(w, v_proc)
        register_op_name(qkv, f'/blocks.{self.layer_id}/cross_attn/MatMul_1')

        qkv = ops.reshape(qkv, [bs_symbol * seq_len_symbol, self.n_state])
        out = self.out(qkv)
        register_op_name(out, f'/blocks.{self.layer_id}/cross_attn/out/MatMul')
        out = ops.reshape(out, [bs_symbol, seq_len_symbol, self.n_state])

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, config, weights, layer_id=0):
        self.layer_id = layer_id
        self.config = config
        self.n_state = config.d_model
        self.n_ctx = config.max_source_positions
        self.n_head = config.encoder_attention_heads
        self.attn = MultiHeadAttention(config, weights, layer_id)
        self.cross_attn = CrossMultiHeadAttention(config, weights, layer_id)

        self.attn_ln = nn.LayerNorm(self.n_state)
        update_tensor_data(self.attn_ln.weights, weights.get(
            f'model.decoder.layers.{layer_id}.self_attn_layer_norm.weight'), f'model.decoder.layers.{layer_id}.self_attn_layer_norm.weight')
        update_tensor_data(self.attn_ln.biases, weights.get(
            f'model.decoder.layers.{layer_id}.self_attn_layer_norm.bias'), f'model.decoder.layers.{layer_id}.self_attn_layer_norm.bias')

        self.cross_attn_ln = nn.LayerNorm(self.n_state)
        update_tensor_data(self.cross_attn_ln.weights, weights.get(
            f'model.decoder.layers.{layer_id}.encoder_attn_layer_norm.weight'), f'model.decoder.layers.{layer_id}.encoder_attn_layer_norm.weight')
        update_tensor_data(self.cross_attn_ln.biases, weights.get(
            f'model.decoder.layers.{layer_id}.encoder_attn_layer_norm.bias'), f'model.decoder.layers.{layer_id}.encoder_attn_layer_norm.bias')
        self.n_mlp = self.n_state * 4

        self.fc1 = nn.Linear(self.n_state, self.n_mlp)
        self.fc2 = nn.Linear(self.n_mlp, self.n_state)
        self.mlp_ln = nn.LayerNorm(self.n_state)
        update_tensor_data(self.mlp_ln.weights, weights.get(
            f'model.decoder.layers.{layer_id}.final_layer_norm.weight'), f'model.decoder.layers.{layer_id}.final_layer_norm.weight')
        update_tensor_data(self.mlp_ln.biases, weights.get(
            f'model.decoder.layers.{layer_id}.final_layer_norm.bias'), f'model.decoder.layers.{layer_id}.final_layer_norm.bias')

        update_tensor_data(self.fc1.weights, weights.get(
            f'model.decoder.layers.{layer_id}.fc1.weight'), f'model.decoder.layers.{layer_id}.fc1.weight')
        update_tensor_data(self.fc1.biases, weights.get(
            f'model.decoder.layers.{layer_id}.fc1.bias'), f'model.decoder.layers.{layer_id}.fc1.bias')

        update_tensor_data(self.fc2.weights, weights.get(
            f'model.decoder.layers.{layer_id}.fc2.weight'), f'model.decoder.layers.{layer_id}.fc2.weight')
        update_tensor_data(self.fc2.biases, weights.get(
            f'model.decoder.layers.{layer_id}.fc2.bias'), f'model.decoder.layers.{layer_id}.fc2.bias')

    def forward(self, x, k_cache, v_cache, k_proj, v_proj, position_ids):
        norm = self.attn_ln(x)
        register_op_name(norm, f"/blocks.{self.layer_id}/attn_ln/Add_1")

        attn_out = self.attn(norm, k_cache, v_cache, position_ids)
        x = nn.Add()(x, attn_out)
        register_op_name(x, f"/blocks.{self.layer_id}/Add")

        norm2 = self.cross_attn_ln(x)
        register_op_name(norm2, f"/blocks.{self.layer_id}/cross_attn_ln/Add_1")
        cross_attn_out = self.cross_attn(norm2, k_proj, v_proj)
        x = nn.Add()(x, cross_attn_out)
        register_op_name(x, f"/blocks.{self.layer_id}/Add_1")

        norm3 = self.mlp_ln(x)
        register_op_name(norm3, f"/blocks.{self.layer_id}/mlp_ln/Add_1")
        bs_symbol, seq_len_symbol, _ = norm3.shape.symbol
        norm3 = ops.reshape(norm3, [bs_symbol * seq_len_symbol, self.n_state])
        fc1 = self.fc1(norm3)
        register_op_name(fc1, f"/blocks.{self.layer_id}/mlp/mlp.0/MatMul")
        act = ops.gelu(fc1)
        register_op_name(act, f"/blocks.{self.layer_id}/mlp/mlp.1/Mul_1")
        fc2 = self.fc2(act)
        register_op_name(fc2, f"/blocks.{self.layer_id}/mlp/mlp.2/MatMul")
        fc2 = ops.reshape(fc2, [bs_symbol, seq_len_symbol, self.n_state])
        x = nn.Add()(x, fc2)
        register_op_name(x, f"/blocks.{self.layer_id}/Add_2")
        x = ops.reshape(x, [bs_symbol, seq_len_symbol, self.n_state])

        return x


class TextDecoder(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.config = config
        self.decoder_layers = config.decoder_layers
        self.n_state = config.d_model
        self.n_head = config.decoder_attention_heads
        self.vocab_size = config.vocab_size
        self.token_embedding = weights[f'model.decoder.embed_tokens.weight'].float().cpu().numpy()
        self.positional_embedding = weights[f'model.decoder.embed_positions.weight'].float().cpu().numpy()
        self.blocks = []
        for idx in range(self.decoder_layers):
            self.blocks.append(ResidualAttentionBlock(config, weights, idx))
        self.concat_d0 = nn.Concat(axis=0)

        self.ln = nn.LayerNorm(self.n_state)
        update_tensor_data(self.ln.weights, weights.get(
            f'model.decoder.layer_norm.weight'), f'model.decoder.layer_norm.weight')
        update_tensor_data(self.ln.biases, weights.get(
            f'model.decoder.layer_norm.bias'), f'model.decoder.layer_norm.bias')

        self.final_fc = nn.Linear(self.n_state, self.vocab_size)
        update_tensor_data(self.final_fc.weights, weights.get(
            f'model.decoder.embed_tokens.weight'), f'model.decoder.embed_tokens.weight.transpose')

    def forward(self, tokens, kcache_module, vcache_module, audio_features, position_ids):
        bs_symbol, seq_len_symbol, _ = tokens.shape.symbol
        for idx, block in enumerate(self.blocks):
            tokens = block(
                tokens, kcache_module, vcache_module, audio_features[2 * idx],
                audio_features[2 * idx + 1], position_ids)

        logits = self.ln(tokens)
        register_op_name(logits, f"/ln/Add_1")
        logits = ops.reshape(logits, [bs_symbol * seq_len_symbol, self.n_state])

        logits = self.final_fc(logits)
        register_op_name(logits, f"/MatMul")
        logits = ops.reshape(logits, [bs_symbol, seq_len_symbol, self.vocab_size])
        logits.name = 'logits'

        return logits


@register_update_config('whisper__decoder')
def whisper_decoder_update_config(config):
    if not hasattr(config, 'num_key_value_heads') and hasattr(config, 'decoder_attention_heads'):
        config.num_key_value_heads = config.decoder_attention_heads
    if not hasattr(config, 'hidden_size') and hasattr(config, 'd_model'):
        config.hidden_size = config.d_model
    if not hasattr(config, 'num_hidden_layers') and hasattr(config, 'decoder_layers'):
        config.num_hidden_layers = config.decoder_layers
    if not hasattr(config, 'num_attention_heads') and hasattr(config, 'decoder_attention_heads'):
        config.num_attention_heads = config.decoder_attention_heads


@register_causallm(['whisper__decoder'])
class WhisperDecoder(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.config = config
        self.n_state = config.d_model
        self.n_ctx = config.max_source_positions
        self.num_hidden_layers = config.decoder_layers
        self.n_head = config.decoder_attention_heads
        self.num_key_value_heads = config.decoder_attention_heads
        self.num_attention_heads = config.decoder_attention_heads
        self.model = TextDecoder(config, weights, prefix)
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        model_name = 'whisper__decoder'
        seq_len = self.config.get('seq_len', 1)
        past_len = self.config.get('past_len', 127)
        batch = self.config.get('batch', 1)
        input_tensor_list = []

        with Graph(model_name) as self.aipu_g:
            tokens = ops.input([batch, seq_len, self.n_state], Dtype.FP32,
                               name='/Add_1_output_0_0', exprs=['batch', 'seq_len', self.n_state])
            input_tensor_list.append(tokens)
            audio_features = []
            position_ids = None

            kcache_module = nn.KVCacheModule(self.config, name='kcache', cacheupdate_enable_ds=True)
            vcache_module = nn.KVCacheModule(self.config, name='vcache', cacheupdate_enable_ds=True)

            if self.config.kvcache_optimization:
                position_ids = ops.input([batch, seq_len], Dtype.INT32,
                                         name='position_ids', exprs=['batch', 'seq_len'])
                input_tensor_list.append(position_ids)

            for idx in gen_layer_ids(self.num_hidden_layers):
                input0 = ops.input([self.n_head, batch, self.n_ctx, self.n_state // self.n_head],
                                   Dtype.FP32, name=f'/blocks.{idx}/cross_attn/Transpose_1_output_0')
                input1 = ops.input([self.n_head, batch, self.n_ctx, self.n_state // self.n_head],
                                   Dtype.FP32, name=f'/blocks.{idx}/cross_attn/Transpose_2_output_0')
                register_op_name(input0, f'/blocks.{idx}/cross_attn/Transpose_1_output_0')
                register_op_name(input1, f'/blocks.{idx}/cross_attn/Transpose_2_output_0')
                input0.op.params['layout'] = 'NCHWC16'
                input1.op.params['layout'] = 'NCHWC16'
                audio_features.append(input0)
                audio_features.append(input1)
                input_tensor_list.append(input0)
                input_tensor_list.append(input1)

            input_tensor_list = input_tensor_list[:1] + kcache_module.cache_inputs + \
                vcache_module.cache_inputs + input_tensor_list[1:]
            logical = self.model(tokens, kcache_module, vcache_module, audio_features, position_ids)
            self.aipu_g.output_tensors = TensorList(
                kcache_module.cache_outputs + vcache_module.cache_outputs + [logical])
            self.aipu_g.input_tensors = TensorList(input_tensor_list)
        return self.aipu_g
