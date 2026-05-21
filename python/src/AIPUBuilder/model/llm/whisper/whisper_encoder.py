import numpy as np
import torch
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...register import register_causallm, gen_subgraph
from ...utils import update_tensor_data


def register_op_name(t, name):
    t.op.name = name


class ConvBlock(nn.Module):
    def __init__(self, config, weights):
        self.n_mels = config.num_mel_bins
        self.n_ctx = config.max_source_positions
        self.n_state = config.d_model
        self.n_head = config.encoder_attention_heads
        self.n_layer = config.encoder_layers
        self.conv1 = nn.Conv2d(self.n_mels, self.n_state, kernels=[1, 3], padding=[0, 0, 1, 1], name='/conv1/Conv')
        self.conv2 = nn.Conv2d(self.n_state, self.n_state, kernels=[1, 3], strides=[
                               2, 2], padding=[0, 0, 1, 1], name='/conv2/Conv')

        update_tensor_data(self.conv1.weights,
                           weights[f'model.encoder.conv1.weight'].unsqueeze_(1).permute([0, 1, 3, 2]),
                           f'model.encoder.conv1.weight')
        update_tensor_data(self.conv1.biases, weights.get(f'model.encoder.conv1.bias'), f'model.encoder.conv1.bias')
        update_tensor_data(self.conv2.weights, weights[f'model.encoder.conv2.weight'].unsqueeze_(
            1).permute([0, 1, 3, 2]), f'model.encoder.conv2.weight')
        update_tensor_data(self.conv2.biases, weights.get(f'model.encoder.conv2.bias'), f'model.encoder.conv2.bias')

        self.transpose = nn.Transpose([0, 1, 3, 2])

    def forward(self, hidden_states):
        hidden_states = self.transpose(hidden_states)
        hidden_states = self.conv1(hidden_states)
        register_op_name(hidden_states, '/conv1/Conv')
        conv1 = ops.gelu(hidden_states)
        register_op_name(conv1, '/Mul_1')
        conv2 = self.conv2(conv1)
        register_op_name(conv2, '/conv2/Conv')
        conv2 = ops.gelu(conv2)
        register_op_name(conv2, '/Mul_3')
        return conv2


class MultiHeadAttention(nn.Module):
    def __init__(self, config, weights, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.n_state = config.d_model
        self.n_ctx = config.max_source_positions
        self.n_head = config.encoder_attention_heads
        self.query = nn.Linear(self.n_state, self.n_state)
        self.key = nn.Linear(self.n_state, self.n_state)
        self.value = nn.Linear(self.n_state, self.n_state)
        self.out = nn.Linear(self.n_state, self.n_state)
        self.scale = (self.n_state // self.n_head) ** -0.25
        self.matmul = nn.MatMul()
        self.softmax = nn.Softmax()

        update_tensor_data(self.query.weights,
                           weights[f'model.encoder.layers.{layer_id}.self_attn.q_proj.weight'] * self.scale,
                           f'model.encoder.layers.{layer_id}.self_attn.q_proj.weight')
        update_tensor_data(self.query.biases,
                           weights[f'model.encoder.layers.{layer_id}.self_attn.q_proj.bias'] * self.scale,
                           f'model.encoder.layers.{layer_id}.self_attn.q_proj.bias')

        update_tensor_data(self.key.weights,
                           weights[f'model.encoder.layers.{layer_id}.self_attn.k_proj.weight'] * self.scale,
                           f'model.encoder.layers.{layer_id}.self_attn.k_proj.weight')

        update_tensor_data(self.value.weights, weights.get(
            f'model.encoder.layers.{layer_id}.self_attn.v_proj.weight'), f'model.encoder.layers.{layer_id}.self_attn.v_proj.weight')
        update_tensor_data(self.value.biases, weights.get(
            f'model.encoder.layers.{layer_id}.self_attn.v_proj.bias'), f'model.encoder.layers.{layer_id}.self_attn.v_proj.bias')

        update_tensor_data(self.out.weights, weights.get(
            f'model.encoder.layers.{layer_id}.self_attn.out_proj.weight'), f'model.encoder.layers.{layer_id}.self_attn.out_proj.weight')
        update_tensor_data(self.out.biases, weights.get(
            f'model.encoder.layers.{layer_id}.self_attn.out_proj.bias'), f'model.encoder.layers.{layer_id}.self_attn.out_proj.bias')

    def forward(self, x):
        x = ops.reshape(x, [self.n_ctx, self.n_state])
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        register_op_name(q, f"/blocks.{self.layer_id}/attn/query/MatMul")
        register_op_name(k, f"/blocks.{self.layer_id}/attn/key/MatMul")
        register_op_name(v, f"/blocks.{self.layer_id}/attn/value/MatMul")

        q = ops.reshape(q, [1, self.n_ctx, self.n_head, -1])
        q = ops.transpose(q, [2, 0, 1, 3])
        k = ops.reshape(k, [1, self.n_ctx, self.n_head, -1])
        k = ops.transpose(k, [2, 0, 3, 1])
        qk = self.matmul(q, k)
        register_op_name(qk, f'/blocks.{self.layer_id}/attn/MatMul')

        w = self.softmax(qk)
        register_op_name(w, f'/blocks.{self.layer_id}/attn/Softmax')

        v = ops.reshape(v, [1, self.n_ctx, self.n_head, -1])
        v = ops.transpose(v, [2, 0, 1, 3])
        qkv = self.matmul(w, v)
        register_op_name(qkv, f'/blocks.{self.layer_id}/attn/MatMul_1')

        qkv = ops.transpose(qkv, [1, 2, 0, 3])
        qkv = ops.reshape(qkv, [self.n_ctx, self.n_state])

        out = self.out(qkv)
        register_op_name(out, f'/blocks.{self.layer_id}/attn/out/MatMul')
        out = ops.reshape(out, [1, 1, self.n_ctx, self.n_state])
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(self, config, weights, layer_id=0):
        self.layer_id = layer_id
        self.config = config
        self.n_state = config.d_model
        self.n_ctx = config.max_source_positions
        self.n_head = config.encoder_attention_heads
        self.attn = MultiHeadAttention(config, weights, layer_id)
        self.attn_ln = nn.LayerNorm(self.n_state)
        update_tensor_data(self.attn_ln.weights, weights.get(
            f'model.encoder.layers.{layer_id}.self_attn_layer_norm.weight'), f'model.encoder.layers.{layer_id}.self_attn_layer_norm.weight')
        update_tensor_data(self.attn_ln.biases, weights.get(
            f'model.encoder.layers.{layer_id}.self_attn_layer_norm.bias'), f'model.encoder.layers.{layer_id}.self_attn_layer_norm.bias')
        self.n_mlp = self.n_state * 4

        self.fc1 = nn.Linear(self.n_state, self.n_mlp)
        self.fc2 = nn.Linear(self.n_mlp, self.n_state)
        self.mlp_ln = nn.LayerNorm(self.n_state)
        update_tensor_data(self.mlp_ln.weights, weights.get(
            f'model.encoder.layers.{layer_id}.final_layer_norm.weight'), f'model.encoder.layers.{layer_id}.final_layer_norm.weight')
        update_tensor_data(self.mlp_ln.biases, weights.get(
            f'model.encoder.layers.{layer_id}.final_layer_norm.bias'), f'model.encoder.layers.{layer_id}.final_layer_norm.bias')
        update_tensor_data(self.fc1.weights, weights.get(
            f'model.encoder.layers.{layer_id}.fc1.weight'), f'model.encoder.layers.{layer_id}.fc1.weight')
        update_tensor_data(self.fc1.biases, weights.get(
            f'model.encoder.layers.{layer_id}.fc1.bias'), f'model.encoder.layers.{layer_id}.fc1.bias')
        update_tensor_data(self.fc2.weights, weights.get(
            f'model.encoder.layers.{layer_id}.fc2.weight'), f'model.encoder.layers.{layer_id}.fc2.weight')
        update_tensor_data(self.fc2.biases, weights.get(
            f'model.encoder.layers.{layer_id}.fc2.bias'), f'model.encoder.layers.{layer_id}.fc2.bias')

    def forward(self, x):
        norm = self.attn_ln(x)
        register_op_name(norm, f"/blocks.{self.layer_id}/attn_ln/Add_1")
        attn_out = self.attn(norm)
        x = nn.Add()(x, attn_out)
        register_op_name(x, f"/blocks.{self.layer_id}/Add")
        norm2 = self.mlp_ln(x)
        register_op_name(norm2, f"/blocks.{self.layer_id}/mlp_ln/Add_1")
        norm2 = ops.reshape(norm2, [self.n_ctx, self.n_state])
        fc1 = self.fc1(norm2)
        register_op_name(fc1, f"/blocks.{self.layer_id}/mlp/mlp.0/MatMul")
        act = ops.gelu(fc1)
        register_op_name(act, f"/blocks.{self.layer_id}/mlp/mlp.1/Mul_1")
        fc2 = self.fc2(act)
        register_op_name(fc2, f"/blocks.{self.layer_id}/mlp/mlp.2/MatMul")
        fc2 = ops.reshape(fc2, [1, 1, self.n_ctx, self.n_state])
        x = nn.Add()(x, fc2)
        register_op_name(x, f"/blocks.{self.layer_id}/Add_1")
        return x


class DecoderFCBlock(nn.Module):
    def __init__(self, config, weights, layer_id=0):
        self.layer_id = layer_id
        self.config = config
        self.n_state = config.d_model
        self.n_ctx = config.max_source_positions
        self.n_head = config.encoder_attention_heads
        self.scale = (self.n_state // self.n_head) ** -0.25
        self.fc1 = nn.Linear(self.n_state, self.n_state)
        self.fc2 = nn.Linear(self.n_state, self.n_state)

        update_tensor_data(self.fc1.weights,
                           weights[f'model.decoder.layers.{layer_id}.encoder_attn.k_proj.weight'] * self.scale,
                           f'model.decoder.layers.{layer_id}.encoder_attn.k_proj.weight')
        update_tensor_data(self.fc2.weights, weights.get(
            f'model.decoder.layers.{layer_id}.encoder_attn.v_proj.weight'), f'model.decoder.layers.{layer_id}.encoder_attn.v_proj.weight')
        update_tensor_data(self.fc2.biases, weights.get(
            f'model.decoder.layers.{layer_id}.encoder_attn.v_proj.bias'), f'model.decoder.layers.{layer_id}.encoder_attn.v_proj.bias')

    def forward(self, x):
        out0 = self.fc1(x)
        out1 = self.fc2(x)
        register_op_name(out0, f"/blocks.{self.layer_id}/cross_attn/key/MatMul")
        register_op_name(out1, f"/blocks.{self.layer_id}/cross_attn/value/MatMul")

        out0 = ops.reshape(out0, [1, self.n_ctx, self.n_state])
        out1 = ops.reshape(out1, [1, self.n_ctx, self.n_state])

        return out0, out1


class AudioEncoder(nn.Module):
    def __init__(self, config, weights, prefix):
        super().__init__()
        self.config = config
        self.n_state = config.d_model
        self.n_ctx = config.max_source_positions
        self.n_head = config.encoder_attention_heads
        self.pre_conv = ConvBlock(config, weights)
        self.positional_embedding_weights = weights[f'model.encoder.embed_positions.weight'].float().cpu().numpy()
        self.blocks = []
        for idx in range(self.config.encoder_layers):
            self.blocks.append(ResidualAttentionBlock(config, weights, idx))
        self.ln_post = nn.LayerNorm(config.d_model)
        update_tensor_data(self.ln_post.weights, weights.get(
            f'model.encoder.layer_norm.weight'), f'model.encoder.layer_norm.weight')
        update_tensor_data(self.ln_post.biases, weights.get(
            f'model.encoder.layer_norm.bias'), f'model.encoder.layer_norm.bias')

        self.decoder_fc_blocks = []
        for idx in range(self.config.decoder_layers):
            self.decoder_fc_blocks.append(DecoderFCBlock(config, weights, idx))

    def forward(self, x: Tensor):
        x = self.pre_conv(x)
        positional_embedding = ops.constant(Tensor(self.positional_embedding_weights), [
                                            1, 1, self.config.max_source_positions, self.config.d_model])
        positional_embedding.op.constants['weights'].name = f"embed_tokens_weight"
        register_op_name(positional_embedding, 'onnx::Add_3089')
        x = nn.Add()(x, positional_embedding)
        register_op_name(x, '/Add_2')

        for block in self.blocks:
            x = block(x)

        ln_out = self.ln_post(x)
        register_op_name(ln_out, f"/ln_post/Add_1")
        ln_out.name = '/ln_post/Add_1_0'

        fc_blocks = [ln_out]
        ln_out = ops.reshape(ln_out, [self.n_ctx, self.n_state])
        for idx, block in enumerate(self.decoder_fc_blocks):
            fc_out0, fc_out1 = block(ln_out)
            fc_out0 = ops.reshape(fc_out0, [self.n_ctx, 1, self.n_head, self.n_state//self.n_head])
            fc_out0 = ops.transpose(fc_out0, [2, 1, 0, 3])
            fc_out1 = ops.reshape(fc_out1, [self.n_ctx, 1, self.n_head, self.n_state//self.n_head])
            fc_out1 = ops.transpose(fc_out1, [2, 1, 0, 3])
            fc_out0.name = f'/blocks.{idx}/cross_attn/key/MatMul_Transpose_output_0'
            fc_out1.name = f'/blocks.{idx}/cross_attn/value/MatMul_Transpose_output_1'
            fc_blocks.append(fc_out0)
            fc_blocks.append(fc_out1)

        return fc_blocks


@register_causallm(['whisper__encoder'])
class WhisperEncoder(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.model = AudioEncoder(config, weights, prefix)
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        model_name = 'whisper__encoder'
        with Graph(model_name) as self.aipu_g:
            hidden_states = ops.input(
                [1, 1, self.config.num_mel_bins, 3000], Dtype.FP32, name='mel')

            output_tensors = self.model(hidden_states)
            self.aipu_g.output_tensors = TensorList(output_tensors)
            self.aipu_g.input_tensors = TensorList([t for t in self.aipu_g.input_tensors[:1]])
        return self.aipu_g
