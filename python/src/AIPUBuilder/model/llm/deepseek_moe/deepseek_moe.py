import ast
import numpy as np
import functools
import torch
import math
from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO, DEBUG, WARN, ERROR
from AIPUBuilder import ops
from AIPUBuilder import nn
from AIPUBuilder._C._tongue import Symbol, Expr
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph
from ...utils import update_tensor_data, convert_expr
from ..llmutils import gen_layer_ids


def to_expr(eles):
    if isinstance(eles, (list, tuple)):
        return [to_expr(xx) for xx in eles]
    elif isinstance(eles, str):
        if '*' in eles:
            ele = eles.replace(' ', '').split('*')
            ele = [int(e) if not e.startswith('d') else e[1:] for e in ele]
            ele = to_expr(ele)
            return functools.reduce(lambda a, b: a*b, ele)
        else:
            return Expr(Symbol(eles))
    elif isinstance(eles, (int, float)):
        return Expr(eles)
    else:
        DEBUG(f"un-supported type: {type(eles)}, only support [list, tuple, str, int, float]")
        return eles


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

# Find dim range bounds based on rotations


def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


@register_causallm(['deepseek_moe_embedding'])
class DeepseekMoeEmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')


class DeepseekMoeRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device='cpu',
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = (emb.cos() * _mscale)
        self.sin_cached = (emb.sin() * _mscale)

    def forward(self, pos_ids):
        sin_out = ops.constant(Tensor(self.sin_cached))
        cos_out = ops.constant(Tensor(self.cos_cached))
        gather_sin_out = ops.gather(sin_out, pos_ids, axis=0)
        gather_cos_out = ops.gather(cos_out, pos_ids, axis=0)
        # sin_out_reshape = gather_sin_out.unsqueeze(dim=1)
        # cos_out_reshape = gather_cos_out.unsqueeze(dim=1)
        sin_out_reshape = ops.reshape(gather_sin_out, [1, 1, pos_ids.shape[1], self.sin_cached.shape[1]])
        cos_out_reshape = ops.reshape(gather_cos_out, [1, 1, pos_ids.shape[1], self.sin_cached.shape[1]])
        sin_reshape_tile = sin_out_reshape
        cos_reshape_tile = cos_out_reshape
        return cos_reshape_tile, sin_reshape_tile


class DeepseekMoeMLP(nn.Module):

    def __init__(self, config, weights, layer_id, prefix='', shared=False, expert_id=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if not shared:
            if expert_id is None:
                self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
                self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
                self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
            else:
                self.gate_proj = nn.Linear(self.hidden_size, config.moe_intermediate_size)
                self.up_proj = nn.Linear(self.hidden_size, config.moe_intermediate_size)
                self.down_proj = nn.Linear(config.moe_intermediate_size, self.hidden_size)
        else:
            inter_size = config.moe_intermediate_size * config.n_shared_experts
            self.gate_proj = nn.Linear(self.hidden_size, inter_size)
            self.up_proj = nn.Linear(self.hidden_size, inter_size)
            self.down_proj = nn.Linear(inter_size, self.hidden_size)
        self.act_fn = nn.ACT2FN[config.hidden_act]

        if not shared:
            if expert_id is None:
                gate_proj_w = weights.get(f'model.layers.{layer_id}.mlp.gate_proj.weight')
                up_proj_w = weights.get(f'model.layers.{layer_id}.mlp.up_proj.weight')
                down_proj_w = weights.get(f'model.layers.{layer_id}.mlp.down_proj.weight')
            else:
                gate_proj_w = weights.get(f'model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight')
                up_proj_w = weights.get(f'model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight')
                down_proj_w = weights.get(f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight')
        else:
            gate_proj_w = weights.get(f'model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight')
            up_proj_w = weights.get(f'model.layers.{layer_id}.mlp.shared_experts.up_proj.weight')
            down_proj_w = weights.get(f'model.layers.{layer_id}.mlp.shared_experts.down_proj.weight')
        update_tensor_data(self.gate_proj.weights, gate_proj_w, name=f'{prefix}layers.{layer_id}.mlp.gate_proj.weight')
        update_tensor_data(self.up_proj.weights, up_proj_w, name=f'{prefix}layers.{layer_id}.mlp.up_proj.weight')
        update_tensor_data(self.down_proj.weights, down_proj_w, name=f'{prefix}layers.{layer_id}.mlp.down_proj.weight')

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepseekMoeGate(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.topk = config.num_experts_per_tok

        if layer_id < config.first_k_dense_replace:
            self.mlp = DeepseekMoeMLP(config, weights, layer_id, prefix, False)
        else:
            self.shared_mlp = DeepseekMoeMLP(config, weights, layer_id, prefix, True)
            self.gate = nn.Linear(config.hidden_size, config.n_routed_experts)
            gate_w = weights.get(f"model.layers.{layer_id}.mlp.gate.weight")
            gate_w = gate_w['data'][:config.n_routed_experts]
            update_tensor_data(self.gate.weights, gate_w, name=f"{prefix}model.layers.{layer_id}.mlp.gate.weight")
            self.experts = [DeepseekMoeMLP(config, weights, layer_id, prefix, False, i)
                            for i in range(config.n_routed_experts)]

    def forward(self, hidden_states):
        seqlen, hidden_size = hidden_states.shape
        bsz = 1
        self.bsz = bsz
        self.seq_len = seqlen
        self.hidden_size = hidden_size
        if self.layer_id < self.config.first_k_dense_replace:
            return self.mlp(hidden_states)
        shared = self.shared_mlp(hidden_states)
        logits = self.gate(hidden_states)
        scores = ops.softmax(logits, 1)
        topk_weight, topk_idx = ops.topk(scores, self.topk, 1, sorted=False, select_index='last')
        identity = hidden_states

        hidden_states = ops.reshape(hidden_states, [bsz*seqlen, hidden_size])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        y = y + shared
        return y

    def moe_infer(self, x, topk_idx, topk_weight):
        cnts = ops.constant(np.zeros([self.bsz * self.seq_len, len(self.experts)]).astype('float32'))
        src = ops.constant(np.ones([self.bsz * self.seq_len, len(self.experts)]).astype('float32'))
        cnts = ops.scatter_elements(cnts, topk_idx, src, axis=1)
        tokens_per_expert = ops.reduce_sum(cnts, [0], keep_dim=False)
        tokens_per_expert_idx, iter_num = ops.nonzero(tokens_per_expert)
        tokens_per_expert_idx = ops.reshape(tokens_per_expert_idx, [len(self.experts)])
        tokens_per_expert = ops.gather(tokens_per_expert, tokens_per_expert_idx, 0, 0)
        tokens_per_expert_idx = ops.cast(tokens_per_expert_idx, Dtype.INT32, ignore_scale_zp=True)
        idxs = ops.reshape(topk_idx, [self.bsz * self.seq_len * self.config.num_experts_per_tok])
        topk_val = ops.shape(idxs)
        idxs = ops.topk(idxs, topk_val, 0, largest=False, sorted=True, select_index='last')[1]
        idxs_dtype = idxs.dtype
        idxs_dtype_bits = idxs_dtype.bits
        idxs_max = 2 ** idxs_dtype_bits - 1 if not idxs_dtype.is_signed else 2 ** (idxs_dtype_bits - 1) - 1
        const_dtype = idxs_dtype.np
        if idxs_max < self.config.num_experts_per_tok:
            const_bits = math.ceil(math.log2(self.config.num_experts_per_tok) + 0.5)
            dtype_bits = math.ceil(const_bits / 8.0) * 8
            const_dtype = Dtype(f'int{dtype_bits}' if dtype_bits <= 2 else f'int32')
            idxs = ops.cast(idxs, const_dtype, True, "TRUNCATION")
        selected = ops.divmod(idxs, nn.Constant(
            np.array([self.config.num_experts_per_tok]).astype(const_dtype))(), 'TRUNC')[0]
        sorted_tokens = ops.gather(x, selected, 0, 0)
        cond_in = nn.Constant(np.array([1]).astype(np.uint8))()
        start_idx = nn.Constant(np.array([0]).astype(np.int32))()
        self.loop_subgraph = self.create_loop_subgraph(self.config, self.layer_id)
        outs = ops.loop([iter_num, cond_in, start_idx,
                         sorted_tokens, tokens_per_expert, tokens_per_expert_idx], self.loop_subgraph,
                        1, 1, True, len(self.experts))[1]
        outs = ops.reshape(outs, [len(self.experts) * self.bsz * self.seq_len *
                           self.config.num_experts_per_tok, self.hidden_size])
        outs = ops.crop(outs, [(0, self.bsz * self.seq_len * self.config.num_experts_per_tok), (0, self.hidden_size)])
        outs.op.params['ds_crops'] = to_expr(
            [(0, f"{self.bsz} * dseq_len * {self.config.num_experts_per_tok}"), (0, self.hidden_size)])

        outs = ops.reshape(outs, [self.bsz * self.seq_len * self.config.num_experts_per_tok, self.hidden_size])
        idxs = ops.reshape(idxs, [self.bsz * self.seq_len * self.config.num_experts_per_tok, 1])
        outs = ops.scatter_nd(outs, idxs, outs, 'None')
        outs = ops.reshape(outs, [self.bsz * self.seq_len, self.config.num_experts_per_tok, self.hidden_size])
        outs.op.params['ds_output_shape'] = to_expr(
            [[f'{self.bsz}*dseq_len', self.config.num_experts_per_tok, self.hidden_size]])
        residual = ops.reshape(topk_weight, [self.bsz * self.seq_len, self.config.num_experts_per_tok, 1])
        residual.op.params['ds_output_shape'] = to_expr(
            [[f'{self.bsz}*dseq_len', self.config.num_experts_per_tok, 1]])
        outs = outs * residual
        outs = ops.reduce_sum(outs, [1], keep_dim=False)
        return outs

    def create_loop_subgraph(self, config, layer_id):
        def construct_if_subgraph(layer_id, expert_id):
            with Graph(f'layer_{layer_id}_mlp_expert_{expert_id}_valid_subgraph') as valid_g:
                tokens_for_this_expert = ops.input([self.bsz * self.seq_len * self.config.num_experts_per_tok, self.hidden_size],
                                                   Dtype.FP32, name=f'token_in_layer_{layer_id}_mlp_expert_{expert_id}_then')
                out = self.experts[expert_id](tokens_for_this_expert)
            valid_g.input_tensors = TensorList([tokens_for_this_expert])
            valid_g.output_tensors = TensorList([out])
            with Graph(f'layer_{layer_id}_mlp_expert_{expert_id}_invalid_subgraph') as invalid_g:
                tokens_for_this_expert = ops.empty([self.bsz * self.seq_len * self.config.num_experts_per_tok, self.hidden_size],
                                                   Dtype.FP32, exprs=[0, self.hidden_size])
            invalid_g.input_tensors = TensorList([])
            invalid_g.output_tensors = TensorList([tokens_for_this_expert])
            return valid_g, invalid_g

        with Graph(f'layer_{layer_id}_mlp_loop_subgraph') as subgraph:
            iter_num = ops.input([1], Dtype.INT32, name=f'iter_num_graph_{layer_id}')
            cond_in = ops.input([1], Dtype.UINT8, name=f'cond_in_graph_{layer_id}')
            start_idx = ops.input([1], Dtype.INT32, name=f'start_idx_graph_{layer_id}')
            sorted_tokens = ops.input([self.bsz * self.seq_len * self.config.num_experts_per_tok,
                                      self.hidden_size], Dtype.FP32, name=f'sorted_tokens_graph_{layer_id}')
            tokens_per_expert = ops.input([len(self.experts)], Dtype.FP32, name=f'tokens_per_exp_graph_{layer_id}')
            tokens_per_expert_idx = ops.input([len(self.experts)], Dtype.INT32,
                                              name=f'tokens_per_exp_graph_idx_{layer_id}')
            GROUP_NUMBER = self.config.get('loop_subgraph_split_num', 2)
            num_tokens = ops.gather(tokens_per_expert, iter_num, 0, 0)
            expert_idx = ops.gather(tokens_per_expert_idx, iter_num, 0, 0)
            end_idx = start_idx + ops.cast(num_tokens, Dtype.INT32, ignore_scale_zp=True)
            tokens_for_this_expert_slice = ops.dynamic_slice([sorted_tokens, start_idx, end_idx, nn.Constant(
                np.zeros([1]).astype(np.int32))(), nn.Constant(np.ones([1]).astype(np.int32))()])[0]
            result = []
            scan_out = []
            for group in range(GROUP_NUMBER):
                group_size = len(self.experts) // GROUP_NUMBER
                for i in range(len(self.experts) // GROUP_NUMBER):
                    valid = ops.greater(num_tokens, nn.Constant(np.array([0]).astype(np.float32))())
                    match = ops.equal(expert_idx, nn.Constant(np.array([group_size*group+i]).astype(np.float32))())
                    if_cond = valid * match
                    if_cond = ops.cast(if_cond, Dtype.UINT8, True, "TRUNCATION")
                    valid_g, invalid_g = construct_if_subgraph(layer_id, group_size*group+i)
                    y = ops.If([if_cond, tokens_for_this_expert_slice],
                               valid_g, invalid_g)[0]
                    final_out = ops.dummy([y])[0]
                    result.append(final_out)
                result = ops.switch(result)
                scan_out.append(result)
                result = []
            scan_out = ops.switch(scan_out)
            start_idx = end_idx
            cond_out = ops.less(iter_num, nn.Constant(np.array([len(self.experts) - 1]).astype(np.int32))())
            cond_out = ops.cast(cond_out, Dtype.UINT8)
            cond_in_out = ops.dummy([cond_in, cond_out])
        subgraph.input_tensors = TensorList(
            [iter_num, cond_in, start_idx, sorted_tokens, tokens_per_expert, tokens_per_expert_idx])
        subgraph.output_tensors = TensorList([cond_out, start_idx, scan_out])
        # subgraph.serialize("loop_graph.def", "loop_graph.bin")
        return subgraph


class DeepseekMoeAttention(nn.Module):

    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.prefix = prefix

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.rope_scaling = config.rope_scaling
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim,
                                name=f"model.layers.{layer_id}.self_attn.q_proj")
        q_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.q_proj.weight')
        q_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.q_proj.bias')
        update_tensor_data(self.q_proj.weights, q_proj_w, f"{prefix}layers.{layer_id}.self_attn.q_proj.weight")
        update_tensor_data(self.q_proj.biases, q_proj_b, f"{prefix}layers.{layer_id}.self_attn.q_proj.bias")

        self.kv_a_proj = nn.Linear(self.hidden_size, self.qk_rope_head_dim + self.kv_lora_rank,
                                   name=f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa")
        kv_a_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight')
        kv_a_proj_b = weights.get(f'model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.bias')
        update_tensor_data(self.kv_a_proj.weights, kv_a_proj_w,
                           f"{prefix}layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight")
        update_tensor_data(self.kv_a_proj.biases, kv_a_proj_b,
                           f"{prefix}layers.{layer_id}.self_attn.kv_a_proj_with_mqa.bias")

        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=1e-5)
        update_tensor_data(self.kv_a_layernorm.weights, weights.get(
            f'model.layers.{layer_id}.self_attn.kv_a_layernorm.weight'), f"{prefix}layers.{layer_id}.self_attn.kv_a_layernorm.weight")

        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_heads * (self.q_head_dim -
                                   self.qk_rope_head_dim + self.v_head_dim), name=f"model.layers.{layer_id}.self_attn.kv_b_proj")
        kv_b_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.kv_b_proj.weight')
        update_tensor_data(self.kv_b_proj.weights, kv_b_proj_w, f"{prefix}layers.{layer_id}.self_attn.kv_b_proj.weight")

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size,
                                name=f"model.layers.{layer_id}.self_attn.o_proj")
        o_proj_w = weights.get(f'model.layers.{layer_id}.self_attn.o_proj.weight')
        update_tensor_data(self.o_proj.weights, o_proj_w, f"{prefix}layers.{layer_id}.self_attn.o_proj.weight")

        self.rotary_emb = DeepseekMoeRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=self.config.rope_scaling["factor"],
            base=config.rope_theta,
            beta_fast=config.rope_scaling['beta_fast'],
            beta_slow=config.rope_scaling['beta_slow'],
            mscale=config.rope_scaling['mscale'],
            mscale_all_dim=config.rope_scaling['mscale_all_dim']
        )

        self.softmax_scale = self.q_head_dim ** -0.5
        if self.rope_scaling is not None:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.kq_matmul = nn.MatMul(trans_a=False, trans_b=True, fused_multiplier=self.softmax_scale)
        self.kqv_matmul = nn.MatMul()

        self.matmul_name = f"{prefix}layers.{layer_id}.self_attn"

    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        bs, seq_len, hidden_size = hidden_states.shape.symbol
        bs_symbol, seq_len_symbol, _ = hidden_states.shape.symbol
        hidden_states = ops.reshape(hidden_states, [bs_symbol * seq_len_symbol, hidden_size])

        cos, sin = self.rotary_emb(position_ids)

        q_proj = self.q_proj(hidden_states)
        comp_kv = self.kv_a_proj(hidden_states)

        q_proj = ops.reshape(q_proj, [bs, seq_len, self.num_heads, self.q_head_dim])
        q_proj_nope = ops.crop(q_proj, [[0, bs], [0, seq_len], [0, self.num_heads], [0, self.qk_nope_head_dim]])
        q_proj_nope.op.params['ds_crops'] = to_expr(
            [[0, bs], [0, 'seq_len'], [0, self.num_heads], [0, self.qk_nope_head_dim]])

        q_proj_nope = ops.transpose(q_proj_nope, [0, 2, 1, 3])

        q_proj_rope = ops.crop(q_proj, [[0, bs], [0, seq_len], [0, self.num_heads],
                               [self.qk_nope_head_dim, self.q_head_dim]])
        q_proj_rope.op.params['ds_crops'] = to_expr([[0, bs], [0, 'seq_len'], [0, self.num_heads], [
                                                    self.qk_nope_head_dim, self.q_head_dim]])

        q_proj_rope = ops.transpose(q_proj_rope, [0, 2, 1, 3])
        q_proj_rope = ops.reshape(q_proj_rope, [bs, self.num_heads, seq_len, self.qk_rope_head_dim//2, 2])
        q_proj_rope = ops.transpose(q_proj_rope, [0, 1, 2, 4, 3])
        q_proj_rope = ops.reshape(q_proj_rope, [bs, self.num_heads, seq_len, self.qk_rope_head_dim])

        q_proj_rope_origin = ops.crop(q_proj_rope, [[0, bs], [0, self.num_heads], [
                                      0, seq_len], [0, self.qk_rope_head_dim // 2]])
        q_proj_rope_origin.op.params['ds_crops'] = to_expr(
            [[0, bs], [0, self.num_heads], [0, "seq_len"], [0, self.qk_rope_head_dim // 2]])

        q_proj_rope_neg = ops.crop(q_proj_rope, [[0, bs], [0, self.num_heads], [0, seq_len], [
                                   self.qk_rope_head_dim // 2, self.qk_rope_head_dim]])
        q_proj_rope_neg.op.params['ds_crops'] = to_expr([[0, bs], [0, self.num_heads], [0, "seq_len"], [
                                                        self.qk_rope_head_dim // 2, self.qk_rope_head_dim]])

        q_proj_rope_neg = ops.negative(q_proj_rope_neg)
        q_proj_rope_rotate = ops.concat([q_proj_rope_neg, q_proj_rope_origin], 3)

        q_embed = (q_proj_rope * cos) + (q_proj_rope_rotate * sin)

        query_states = ops.concat([q_proj_nope, q_embed], 3)

        comp_kv = ops.reshape(comp_kv, [bs, seq_len, (self.qk_rope_head_dim + self.kv_lora_rank)])
        comp_kv_res = ops.crop(comp_kv, [[0, bs], [0, seq_len], [0, self.kv_lora_rank]])
        comp_kv_res.op.params['ds_crops'] = to_expr([[0, bs], [0, "seq_len"], [0, self.kv_lora_rank]])

        k_pe = ops.crop(comp_kv, [[0, bs], [0, seq_len], [self.kv_lora_rank,
                        self.qk_rope_head_dim + self.kv_lora_rank]])
        k_pe.op.params['ds_crops'] = to_expr(
            [[0, bs], [0, "seq_len"], [self.kv_lora_rank, self.qk_rope_head_dim + self.kv_lora_rank]])

        kv = self.kv_a_layernorm(comp_kv_res)
        kv = ops.reshape(kv, [bs*seq_len, self.kv_lora_rank])
        kv = self.kv_b_proj(kv)
        kv = ops.reshape(kv, [bs, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim])
        kv = ops.transpose(kv, [0, 2, 1, 3])
        k_nope = ops.crop(kv, [[0, bs], [0, self.num_heads], [0, seq_len], [0, self.qk_nope_head_dim]])
        k_nope.op.params['ds_crops'] = to_expr(
            [[0, bs], [0, self.num_heads], [0, "seq_len"], [0, self.qk_nope_head_dim]])

        value_states = ops.crop(kv, [[0, bs], [0, self.num_heads], [0, seq_len], [
                                self.qk_nope_head_dim, self.qk_nope_head_dim + self.v_head_dim]])
        value_states.op.params['ds_crops'] = to_expr([[0, bs], [0, self.num_heads], [0, "seq_len"], [
                                                     self.qk_nope_head_dim, self.qk_nope_head_dim + self.v_head_dim]])

        k_pe = ops.reshape(k_pe, [1, 1, seq_len, self.qk_rope_head_dim//2, 2])
        k_pe = ops.transpose(k_pe, [0, 1, 2, 4, 3])
        k_pe = ops.reshape(k_pe, [1, 1, seq_len, self.qk_rope_head_dim])

        k_pe_origin = ops.crop(k_pe, [[0, bs], [0, 1], [0, seq_len], [0, self.qk_rope_head_dim // 2]])
        k_pe_origin.op.params['ds_crops'] = to_expr([[0, bs], [0, 1], [0, "seq_len"], [0, self.qk_rope_head_dim // 2]])

        k_pe_neg = ops.crop(k_pe, [[0, bs], [0, 1], [0, seq_len], [self.qk_rope_head_dim // 2, self.qk_rope_head_dim]])
        k_pe_neg.op.params['ds_crops'] = to_expr(
            [[0, bs], [0, 1], [0, "seq_len"], [self.qk_rope_head_dim // 2, self.qk_rope_head_dim]])

        k_pe_neg = ops.negative(k_pe_neg)
        k_pe_rotate = ops.concat([k_pe_neg, k_pe_origin], 3)

        k_embed = (k_pe * cos) + (k_pe_rotate * sin)
        k_embed = ops.tile(k_embed, [1, self.num_heads, 1, 1])

        key_states = ops.concat([k_nope, k_embed], 3)

        key_states = k_cache(key_states, position_ids=position_ids, block_idx=self.layer_id,
                             name=f'past_key_value.{self.layer_id}.key')[0]
        value_states = v_cache(value_states, position_ids=position_ids, block_idx=self.layer_id,
                               name=f'past_key_value.{self.layer_id}.key')[0]

        attn_weights = self.kq_matmul(query_states, key_states)

        attn_weights += attention_mask
        attn_weights = ops.softmax(attn_weights, 3)

        attn_output = self.kqv_matmul(attn_weights, value_states)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(attn_output, [bs*seq_len, self.num_heads * self.v_head_dim])
        attn_output = self.o_proj(attn_output)

        return attn_output


class DeepseekMoeDecoder(nn.Module):
    def __init__(self, config, weights, layer_id, prefix=''):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.input_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.input_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.input_layernorm.weight")

        self.attention = DeepseekMoeAttention(config, weights, layer_id, prefix)

        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.post_attention_layernorm.weights,
                           weights.get(f'model.layers.{layer_id}.post_attention_layernorm.weight'),
                           f"{prefix}layers.{layer_id}.post_attention_layernorm.weight")

        self.mlp = DeepseekMoeGate(config, weights, layer_id, prefix)

    @gen_subgraph(subgraph_name='attnblk')
    def forward(self, hidden_states, attention_mask, position_ids, k_cache=None, v_cache=None, use_cache=False):
        is_gen_subgraph = True and hasattr(self.config, 'enable_subgraph') and self.config.enable_subgraph
        if is_gen_subgraph:
            """
            when enable_subgraph=true, will do some special operations:
            1. avoiding the constant output tensor is subgraph's input tensor, will re-generate one constant in subgraph.
            2. if one tensor is multi ops input, the subgraph will insert the duplicate input tensor into subgraph input tensors,
            which will cause big change for llm runtime. in order to avoid this graph, will insert reshape op, its output tensors will
            be as multi ops input, and reshape is including in this subgraph.
            """
            hidden_states = ops.reshape(hidden_states, hidden_states.shape)
            attention_mask = ops.tile(attention_mask, [1, self.config.num_attention_heads, 1, 1])
            position_ids = ops.reshape(position_ids, position_ids.shape)

        INFO(f"\t build the {self.layer_id} Decoder block.")
        residual = hidden_states
        bs, seq_len, hidden_size = hidden_states.shape.symbol
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states,
                                       attention_mask,
                                       position_ids,
                                       k_cache,
                                       v_cache,
                                       use_cache)
        hidden_states = hidden_states.reshape([bs, seq_len, hidden_size])
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.reshape([bs * seq_len, hidden_size])
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states.reshape([bs, seq_len, hidden_size])
        hidden_states = residual + hidden_states

        kcache_outs = [k_cache.cache_outputs[-1]] if len(k_cache.cache_outputs) else [None]
        vcache_outs = [v_cache.cache_outputs[-1]] if len(v_cache.cache_outputs) else [None]
        return [hidden_states] + kcache_outs + vcache_outs


class DeepseekMoeModel(nn.Module):

    def __init__(self, config, weights, prefix):
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights.get('model.embed_tokens.weight')['data'].float()

        # self.decoders = [DeepseekMoeDecoder(config, weights, 0)]
        self.decoders = [DeepseekMoeDecoder(config, weights, idx, prefix)
                         for idx in gen_layer_ids(config.num_hidden_layers)]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key=[],
                past_value=[],
                use_cache=None,
                with_embed=True,
                with_preprocess_mask=False,
                ):

        # input_embeds = self.embed_tokens(input_ids)
        if with_embed:
            embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
                                       self.model_config.vocab_size, self.model_config.hidden_size])
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        else:
            input_embeds = input_ids

        # is_gen_subgraph = hasattr(self.model_config, 'enable_subgraph') and self.model_config.enable_subgraph
        # if not is_gen_subgraph:
        #     attention_mask = ops.tile(attention_mask, [1, self.model_config.num_attention_heads, 1, 1])

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
        if config.get('tie_word_embeddings', False):
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


@register_causallm(['deepseek_moe'])
class DeepseekMoeCausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        INFO(f"\t current num_hidden_layers = {config.num_hidden_layers}:")
        weights = self.config.get('weights', self._load_state_dict())
        self.model = DeepseekMoeModel(config, weights, prefix)
        self.norm_and_lm_head = NormLmHead(self.config, weights) if self.config.get(
            'with_lm', True) else lambda i: INFO(f"\t[LLM] Not including lm_head part.")
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.seq_len
        past_len = self.config.past_len
        with_embed = self.config.get('with_embed', False)
        use_cache = self.config.get('use_cache', True) and (past_len > 0)
        batch = self.config.get('batch', 1)

        num_heads = self.config.num_attention_heads
        hidden_size = self.config.hidden_size
        qk_nope_head_dim = self.config.get("qk_nope_head_dim", hidden_size//num_heads)
        qk_rope_head_dim = self.config.get("qk_rope_head_dim", hidden_size//num_heads)
        v_head_dim = self.config.get("v_head_dim", hidden_size//num_heads)

        with Graph('deepseek_moe') as self.aipu_g:
            if with_embed:
                hidden_states = ops.input([batch, seq_len], Dtype.FP32, name='input_ids', exprs=['batch', 'seq_len'])
            else:
                hidden_states = ops.input([batch, seq_len, self.config.hidden_size], Dtype.FP32, name='embeded',
                                          exprs=['batch', 'seq_len', self.config.hidden_size])
            if use_cache:
                attention_mask = ops.input([batch, 1, seq_len, seq_len + past_len],
                                           Dtype.FP32, name='attention_mask',
                                           exprs=['batch', 1, 'seq_len', 'max_len'])
            else:
                attention_mask = ops.input([batch, 1, seq_len, seq_len], Dtype.FP32, name='attention_mask',
                                           exprs=['batch', 1, 'seq_len', 'seq_len'])
                # attention_mask = ops.input([1, 1, 1, seq_len], Dtype.FP32, name='attention_mask')
            # attention_mask = Tensor('attention_mask', TensorShape([1, seq_len]), Dtype.INT32)
            position_ids = ops.input([batch, seq_len], Dtype.INT32, name='position_ids',  exprs=['batch', 'seq_len'])

            kcache_module = nn.KVCacheModule(self.config, name='kcache', head_dim=qk_rope_head_dim + qk_nope_head_dim)
            vcache_module = nn.KVCacheModule(self.config, name='vcache', head_dim=v_head_dim)

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
