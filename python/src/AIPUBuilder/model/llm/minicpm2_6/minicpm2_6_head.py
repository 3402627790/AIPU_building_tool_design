from AIPUBuilder.core import *
from AIPUBuilder.logger import INFO
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import BaseModel
from ...register import register_causallm
from ...utils import update_tensor_data


@register_causallm(['minicpm2_6_head'])
class Minicpm2_6HeadCausalLM(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.aipu_g = None
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        update_tensor_data(self.norm.weights, weights.get(
            f'llm.model.norm.weight'), f'normlm.norm.weight')
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_head.weights.betensor = weights[f'llm.lm_head.weight'].float()
        self.lm_head.weights.sync_betensor()
        self.lm_head.weights.name = f'normlm.lm.weight'

    def forward(self):
        seq_len = self.config.seq_len
        model_name = 'head'
        with Graph(model_name+'_head') as self.aipu_g:
            hidden_states = ops.input(
                [1, seq_len, self.config.hidden_size], Dtype.FP32, name='last_hidden_state')
            hidden_states = self.norm(hidden_states)
            h_shape = hidden_states.shape
            hidden_state = ops.reshape(
                hidden_states, [h_shape[0]*h_shape[1], h_shape[2]])
            lm_head = self.lm_head(hidden_state)
            _ = ops.reshape(lm_head, [1, *list(lm_head.shape)])
        return self.aipu_g
