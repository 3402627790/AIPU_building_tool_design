
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import *
from ...register import register_causallm
from ...utils import update_tensor_data


@register_causallm(['ds671b_lm'])
class DeepseekR1Head(BaseModel):
    def __init__(self, config, prefix='', only_embed=False, attn_layers=True):
        super().__init__(config)
        search_safe_tensors_key = 'layers.'+str(60)+'.'
        weights = self._load_state_dict(search_safe_tensors_key)
        self.config.update({'ir_prefix': ''})
        config.update({'weights': weights})
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.norm.weights,
                           weights.get(f'model.norm.weight'), f"{prefix}model.norm.weight")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        update_tensor_data(self.lm_head.weights,
                           weights.get(f'lm_head.weight'), f"{prefix}lm_head.weight")
        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        seq_len = self.config.seq_len
        model_name = 'deepseek_lm'
        with Graph(model_name+'_head') as self.aipu_g:
            hidden_states = ops.input(
                [1, seq_len, self.config.hidden_size], Dtype.FP32, name='last_hidden_state')
            # aipu_g.input_tensors = TensorList(hidden_states)
            hidden_states = self.norm(hidden_states)
            h_shape = hidden_states.shape
            hidden_state = ops.reshape(
                hidden_states, [h_shape[0]*h_shape[1], h_shape[2]])
            lm_head = self.lm_head(hidden_state)
            _ = ops.reshape(lm_head, [1, *list(lm_head.shape)])

        return self.aipu_g
