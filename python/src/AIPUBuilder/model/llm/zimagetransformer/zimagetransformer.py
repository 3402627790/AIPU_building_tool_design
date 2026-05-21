
import numpy as np
import os
import torch
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data, convert_expr
from ..llmutils import gen_layer_ids

ADALN_EMBED_DIM = 256


def apply_complex_rope(hidden_states, freqs_cis):
    # [bsz, seqlen, dim//2, 2], [1, seqlen, dim//2, 2]
    bsz, seqlen, nhead, hdim, _ = hidden_states.shape
    a = ops.crop(hidden_states, [[0, bsz], [0, seqlen], [0, nhead], [0, hdim], [0, 1]])
    b = ops.crop(hidden_states, [[0, bsz], [0, seqlen], [0, nhead], [0, hdim], [1, 2]])
    freqs_cis = ops.reshape(freqs_cis, [1, seqlen, 1, hdim, 2])
    c = ops.crop(freqs_cis, [[0, 1], [0, seqlen], [0, 1], [0, hdim], [0, 1]])
    d = ops.crop(freqs_cis, [[0, 1], [0, seqlen], [0, 1], [0, hdim], [1, 2]])
    res = ops.concat([a*c-b*d, a*d+b*c], axis=4)
    return ops.reshape(res, [bsz, seqlen, nhead, hdim * 2])


class ZImageBlock(nn.Module):

    def __init__(self, config, weights, layer_id, modulation: bool, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix
        self.modulation = modulation

        self.hidden_size = config.dim
        self.num_heads = config.n_heads
        config.rms_norm_eps = 1e-5
        self.head_dim = config.get('head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.n_kv_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.actfn = nn.ACT2FN["silu"]

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        self.q_norm = nn.RMSNorm(config.get("head_dim", self.head_dim), eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.get("head_dim", self.head_dim), eps=config.rms_norm_eps)

        q_proj_w = weights.get(f'{prefix}.{layer_id}.attention.to_q.weight')
        update_tensor_data(self.q_proj.weights, q_proj_w, f'{prefix}.{layer_id}.attention.to_q.weight')

        k_proj_w = weights.get(f'{prefix}.{layer_id}.attention.to_k.weight')
        update_tensor_data(self.k_proj.weights, k_proj_w, f'{prefix}.{layer_id}.attention.to_k.weight')

        v_proj_w = weights.get(f'{prefix}.{layer_id}.attention.to_v.weight')
        update_tensor_data(self.v_proj.weights, v_proj_w, f'{prefix}.{layer_id}.attention.to_v.weight')

        o_proj_w = weights.get(f'{prefix}.{layer_id}.attention.to_out.0.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f'{prefix}.{layer_id}.attention.to_out.0.weight')

        q_norm_w = weights.get(f'{prefix}.{layer_id}.attention.norm_q.weight')
        update_tensor_data(self.q_norm.weights, q_norm_w, f'{prefix}.{layer_id}.attention.norm_q.weight')

        k_norm_w = weights.get(f'{prefix}.{layer_id}.attention.norm_k.weight')
        update_tensor_data(self.k_norm.weights, k_norm_w, f'{prefix}.{layer_id}.attention.norm_k.weight')

        self.pre_attention_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.pre_attention_norm.weights,
                           weights.get(f'{prefix}.{layer_id}.attention_norm1.weight'),
                           f'{prefix}.{layer_id}.attention_norm1.weight')
        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'{prefix}.{layer_id}.attention_norm2.weight'),
                           f'{prefix}.{layer_id}.attention_norm2.weight')
        self.pre_ffn_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.pre_ffn_norm.weights,
                           weights.get(f'{prefix}.{layer_id}.ffn_norm1.weight'),
                           f'{prefix}.{layer_id}.ffn_norm1.weight')
        self.post_ffn_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_ffn_norm.weights,
                           weights.get(f'{prefix}.{layer_id}.ffn_norm2.weight'),
                           f'{prefix}.{layer_id}.ffn_norm2.weight')

        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size * 8 // 3)
        update_tensor_data(self.gate_proj.weights, weights.get(f'{prefix}.{layer_id}.feed_forward.w1.weight'),
                           f'{prefix}.{layer_id}.feed_forward.w1.weight')
        self.up_proj = nn.Linear(self.hidden_size, self.hidden_size * 8 // 3)
        update_tensor_data(self.up_proj.weights, weights.get(f'{prefix}.{layer_id}.feed_forward.w3.weight'),
                           f'{prefix}.{layer_id}.feed_forward.w3.weight')
        self.down_proj = nn.Linear(self.hidden_size * 8 // 3, self.hidden_size)
        update_tensor_data(self.down_proj.weights, weights.get(f'{prefix}.{layer_id}.feed_forward.w2.weight'),
                           f'{prefix}.{layer_id}.feed_forward.w2.weight')

        if self.modulation:
            self.adaln = nn.Linear(ADALN_EMBED_DIM, self.hidden_size * 4)
            update_tensor_data(self.adaln.weights, weights.get(f'{prefix}.{layer_id}.adaLN_modulation.0.weight'),
                               f'{prefix}.{layer_id}.adaLN_modulation.0.weight')
            update_tensor_data(self.adaln.biases, weights.get(f'{prefix}.{layer_id}.adaLN_modulation.0.bias'),
                               f'{prefix}.{layer_id}.adaLN_modulation.0.bias')

        scale = 1./np.sqrt(np.array(self.head_dim)).astype('float32')
        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=scale)
        self.kqv_matmul = nn.MatMul()
        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

        self.softmax = nn.Softmax(axis=3)

    def forward(self, hidden_states, attention_mask, freqs_cis, adaln_input):
        bsz, seqlen, _ = hidden_states.shape
        if self.modulation:
            ada_dim = self.hidden_size * 4
            adaln = self.adaln(adaln_input)
            adaln = ops.reshape(adaln, [bsz, 1, ada_dim])
            scale_msa = ops.crop(adaln, [[0, bsz], [0, 1], [0, self.hidden_size]])
            gate_msa = ops.crop(adaln, [[0, bsz], [0, 1], [self.hidden_size, self.hidden_size*2]])
            scale_mlp = ops.crop(adaln, [[0, bsz], [0, 1], [self.hidden_size*2, self.hidden_size*3]])
            gate_mlp = ops.crop(adaln, [[0, bsz], [0, 1], [self.hidden_size*3, self.hidden_size*4]])

            one = ops.constant(np.array([1.0]).astype(np.float32))
            gate_msa = ops.tanh(gate_msa)
            gate_mlp = ops.tanh(gate_mlp)
            scale_msa = scale_msa + one
            scale_mlp = scale_mlp + one

        attention_residual = hidden_states
        hidden_states = self.pre_attention_norm(hidden_states)
        if self.modulation:
            hidden_states = hidden_states * scale_msa
        hidden_states = ops.reshape(hidden_states, [bsz * seqlen, self.hidden_size])

        q = self.q_norm(ops.reshape(self.q_proj(hidden_states), [bsz, seqlen, self.num_heads, self.head_dim]))
        k = self.k_norm(ops.reshape(self.k_proj(hidden_states), [bsz, seqlen, self.num_heads, self.head_dim]))
        q = ops.reshape(q, [bsz, seqlen, self.num_heads, self.head_dim//2, 2])
        k = ops.reshape(k, [bsz, seqlen, self.num_heads, self.head_dim//2, 2])

        q = apply_complex_rope(q, freqs_cis)
        k = apply_complex_rope(k, freqs_cis)
        q = ops.transpose(q, [0, 2, 1, 3])
        k = ops.transpose(k, [0, 2, 1, 3])

        attn_weight = self.kq_matmul(q, k)
        attn_weight = attn_weight + attention_mask

        attn_score = self.softmax(attn_weight)

        v = self.v_proj(hidden_states)
        v = ops.reshape(v, [bsz, seqlen, self.num_heads, self.head_dim])
        v = ops.transpose(v, [0, 2, 1, 3])

        attn_output = self.kqv_matmul(attn_score, v)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [bsz * seqlen, self.hidden_size])
        attn_output = self.o_proj(attn_output)
        attn_output = self.post_attention_layernorm(attn_output)
        attn_output = ops.reshape(attn_output, [bsz, seqlen, self.hidden_size])
        if self.modulation:
            attn_output = attn_output * gate_msa

        hidden_states = attn_output + attention_residual
        ffn_residual = hidden_states

        hidden_states = self.pre_ffn_norm(hidden_states)
        if self.modulation:
            hidden_states = hidden_states * scale_mlp
        hidden_states = ops.reshape(hidden_states, [bsz * seqlen, self.hidden_size])
        gate = self.actfn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = self.down_proj(gate * up)
        hidden_states = self.post_ffn_norm(hidden_states)
        hidden_states = ops.reshape(hidden_states, [bsz, seqlen, self.hidden_size])
        if self.modulation:
            hidden_states = hidden_states * gate_mlp
        hidden_states = ffn_residual + hidden_states
        return hidden_states


class NoiseRefinerModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.hidden_size = config.dim
        self.in_channels = config.in_channels
        self.num_layers = config.n_refiner_layers

        self.x_embedder = nn.Linear(4 * self.in_channels, self.hidden_size)
        update_tensor_data(self.x_embedder.weights, weights.get('all_x_embedder.2-1.weight'),
                           'all_x_embedder.2-1.weight')
        update_tensor_data(self.x_embedder.biases, weights.get('all_x_embedder.2-1.bias'),
                           'all_x_embedder.2-1.bias')

        self.layers = [ZImageBlock(config, weights, i, True, prefix+"noise_refiner")
                       for i in gen_layer_ids(self.num_layers)]

    @gen_subgraph(subgraph_name="noise_refiner")
    def forward(self, hidden_states, attention_mask, freqs_cis, adaln_input):
        bsz, seqlen, dim = hidden_states.shape
        hidden_states = ops.reshape(hidden_states, [bsz * seqlen, dim])
        hidden_states = self.x_embedder(hidden_states)
        hidden_states = ops.reshape(hidden_states, [bsz, seqlen, self.hidden_size])
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, freqs_cis, adaln_input)
        return hidden_states


class ContextRefinerModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.hidden_size = config.dim
        self.in_channels = config.in_channels
        self.num_layers = config.n_refiner_layers
        self.cap_feat_dim = config.cap_feat_dim

        self.cap_embedder_norm = nn.RMSNorm(self.cap_feat_dim, eps=1e-5)
        update_tensor_data(self.cap_embedder_norm.weights, weights.get('cap_embedder.0.weight'),
                           'cap_embedder.0.weight')
        self.cap_embedder = nn.Linear(self.cap_feat_dim, self.hidden_size)
        update_tensor_data(self.cap_embedder.weights, weights.get('cap_embedder.1.weight'),
                           'cap_embedder.1.weight')
        update_tensor_data(self.cap_embedder.biases, weights.get('cap_embedder.1.bias'),
                           'cap_embedder.1.bias')

        self.layers = [ZImageBlock(config, weights, i, False, prefix+"context_refiner")
                       for i in gen_layer_ids(self.num_layers)]

    @gen_subgraph(subgraph_name="context_refiner")
    def forward(self, hidden_states, attention_mask, freqs_cis, adaln_input):
        bsz, seqlen, dim = hidden_states.shape
        hidden_states = self.cap_embedder_norm(hidden_states)
        hidden_states = ops.reshape(hidden_states, [bsz * seqlen, dim])
        hidden_states = self.cap_embedder(hidden_states)
        hidden_states = ops.reshape(hidden_states, [bsz, seqlen, self.hidden_size])
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, freqs_cis, adaln_input)
        return hidden_states


class UnifiedModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.config = config
        self.hidden_size = config.dim
        self.in_channels = config.in_channels
        self.num_layers = config.n_layers

        self.layers = [ZImageBlock(config, weights, i, True, prefix+"layers") for i in range(self.num_layers)]

    @gen_subgraph(subgraph_name="unified")
    def forward(self, hidden_states, attention_mask, freqs_cis, adaln_input):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, freqs_cis, adaln_input)
        return hidden_states


class FinalLayer(nn.Module):

    def __init__(self, weights, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.out_channels = out_channels
        update_tensor_data(self.linear.weights, weights.get('all_final_layer.2-1.linear.weight'),
                           'all_final_layer.2-1.linear.weight')
        update_tensor_data(self.linear.biases, weights.get('all_final_layer.2-1.linear.bias'),
                           'all_final_layer.2-1.linear.bias')

        self.adaLN_modulation_act = nn.ACT2FN["silu"]
        self.adaLN_modulation = nn.Linear(ADALN_EMBED_DIM, hidden_size)
        update_tensor_data(self.adaLN_modulation.weights, weights.get('all_final_layer.2-1.adaLN_modulation.1.weight'),
                           'all_final_layer.2-1.adaLN_modulation.1.weight')
        update_tensor_data(self.adaLN_modulation.biases, weights.get('all_final_layer.2-1.adaLN_modulation.1.bias'),
                           'all_final_layer.2-1.adaLN_modulation.1.bias')

    def forward(self, x, c):
        scale = ops.constant(np.array([1.0]).astype(np.float32)) + self.adaLN_modulation(self.adaLN_modulation_act(c))
        x = self.norm_final(x) * scale.unsqueeze(1)
        bsz, seqlen, dim = x.shape
        x = ops.reshape(x, [bsz * seqlen, dim])
        x = self.linear(x)
        x = ops.reshape(x, [bsz, seqlen, self.out_channels])
        return x


@register_causallm(['zimagetransformer'])
class ZImageTransformer(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        self.hidden_size = config.dim
        self.in_channels = config.in_channels
        self.cap_feat_dim = config.cap_feat_dim
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        weights = self._load_state_dict()
        self.noise_refiner = NoiseRefinerModel(config, weights, prefix)
        self.context_refiner = ContextRefinerModel(config, weights, prefix)
        self.unified_layers = UnifiedModel(config, weights, prefix)
        self.final_layer = FinalLayer(weights, self.hidden_size, 4 * self.in_channels)

    def forward(self):
        latent_size = self.config.get('latent_size', 64)
        cap_size = self.config.get('cap_size', 64)

        with Graph("zimagetransformer") as self.g:
            noise = ops.input([1, latent_size ** 2, self.in_channels * 4], dtype=Dtype.FP32, name='noise')
            noise_mask = ops.input([1, 1, latent_size ** 2, latent_size ** 2], dtype=Dtype.FP32, name='noise_mask')
            noise_rope = ops.input([1, latent_size ** 2, self.head_dim//2, 2], dtype=Dtype.FP32, name='noise_rope')
            adaln_input = ops.input([1, ADALN_EMBED_DIM], dtype=Dtype.FP32, name='adaln_input')
            cap_embeds = ops.input([1, cap_size, self.cap_feat_dim], dtype=Dtype.FP32, name='cap_embeds')
            cap_mask = ops.input([1, 1, cap_size, cap_size], dtype=Dtype.FP32, name='cap_mask')
            cap_rope = ops.input([1, cap_size, self.head_dim//2, 2], dtype=Dtype.FP32, name='cap_rope')
            unified_mask = ops.input([1, 1, latent_size ** 2 + cap_size, latent_size **
                                      2 + cap_size], dtype=Dtype.FP32, name='unified_mask')

            x = self.noise_refiner(noise, noise_mask, noise_rope, adaln_input)
            cap_feats = self.context_refiner(cap_embeds, cap_mask, cap_rope, None)

            unified_feats = ops.concat([x, cap_feats], axis=1)
            unified_rope = ops.concat([noise_rope, cap_rope], axis=1)
            unified_feats = self.unified_layers(unified_feats, unified_mask, unified_rope, adaln_input)

            unified = self.final_layer(unified_feats, adaln_input)
            self.g.output_tensors = TensorList([unified])
            self.g.input_tensors = TensorList([noise, noise_mask, noise_rope, cap_embeds,
                                               cap_mask, cap_rope, unified_mask, adaln_input])
        return self.g


@register_update_config(['zimagetransformer'])
def zimagetransformer_update_config(config):
    model_safetensors_index_json = 'diffusion_pytorch_model.safetensors.index.json'
    model_path = config.model_path
    index_json_path = os.path.join(model_path, model_safetensors_index_json)
    if not os.path.exists(index_json_path):
        WARN(
            f"zimage_transformer's {model_safetensors_index_json} does not exists, will use default filename = model.safetensors.index.json")
    else:
        config.update({'model_safetensors_index_json': model_safetensors_index_json})
