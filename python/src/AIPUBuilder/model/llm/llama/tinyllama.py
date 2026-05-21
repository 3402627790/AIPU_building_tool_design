import numpy as np
import torch
from AIPUBuilder.core import *  # noqa
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel, BaseEmbedding
from ...register import register_causallm, gen_subgraph, register_update_config
from ...utils import update_tensor_data
from .llama2 import Llama2MLP, Llama2Attention, Llama2Decoder, NormLmHead, Llama2Model


@register_causallm(['llama2_tiny_embedding'])
class LLama2TinyEmbCausalLM(BaseEmbedding):
    def __init__(self, config, prefix=''):
        super().__init__(config, prefix)
        self.embed_weights = self.config.weights.get('model.embed_tokens.weight')


class TinyLlama2RotaryEmbedding(nn.Module):
    def __init__(self, num_heads, dim, max_position_embeddings=2048, base=10000., rope_scaling=None, device=None, name=''):
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
        sin_out_reshape = gather_sin_out.unsqueeze(dim=1)
        cos_out_reshape = gather_cos_out.unsqueeze(dim=1)
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


def _set_default_values(config):
    _defv = {
        'norm_epsilon': config.get('rms_norm_eps', 0.00001),
        'num_key_value_heads': 'num_attention_heads',
        'rope_theta': config.get('rope_theta', 10000.0),
        'rope_scaling': None,
    }

    for k, v in _defv.items():
        if not hasattr(config, k):
            if not isinstance(v, str):
                config.update({k: v})
            else:
                config.update({k: config.get(v)})


@register_causallm(['llama2_tiny'])
class TinyLlama2CausalLM(BaseModel):
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

        hidden_size = self.config.hidden_size
        num_attention_heads = self.config.num_attention_heads
        self.rotary_emb = TinyLlama2RotaryEmbedding(num_attention_heads,
                                                    self.config.get('head_dim', hidden_size//num_attention_heads),
                                                    max_position_embeddings=self.config.seq_len+self.config.past_len,
                                                    base=self.config.get('rope_theta', 10000.0),
                                                    rope_scaling=self.config.get('rope_scaling', None),
                                                    name=f'rotary')

        with Graph('llama2_tiny') as self.aipu_g:
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
