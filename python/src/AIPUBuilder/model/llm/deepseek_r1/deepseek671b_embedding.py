
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from AIPUBuilder.core import *
from AIPUBuilder import ops
from AIPUBuilder import nn
from ...basemodel import *
from ...register import register_causallm


@register_causallm(['ds671b_embedding'])
class DeepseekR1Embedding(BaseModel):
    def __init__(self, config, prefix='', only_embed=False, attn_layers=True):
        super().__init__(config)
        search_safe_tensors_key = 'layers.'+str(0)+'.'
        weights = self._load_state_dict(search_safe_tensors_key)
        config.update({'ir_prefix': ''})
        config.update({'weights': weights})
        self.model_config = config
        self.padding_idx = config.pad_token_id if hasattr(
            config, 'pad_token_id') else None
        self.vocab_size = config.vocab_size
        self.prefix = prefix
        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_weights = weights[f'model.embed_tokens.weight'].float()

        self.aipu_g = None
        self.prefix = prefix

    def forward(self):
        model_name = 'deepseek'
        with Graph(model_name+'_embedding') as self.aipu_g:
            input_ids = ops.input(
                [1, self.config.seq_len], Dtype.INT32, name='input_ids')
            embed_const = ops.constant(Tensor(self.embed_weights.cpu().numpy()), [
                self.model_config.vocab_size, self.model_config.hidden_size])
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            input_embeds = ops.gather(embed_const, input_ids, 0, 0)
        return self.aipu_g
