from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import os
import numpy as np
import torch
import inspect
from AIPUBuilder.core import Graph, Tensor, Dtype, TensorShape
from AIPUBuilder import ops
from AIPUBuilder.logger import ERROR, INFO, WARN
from AIPUBuilder.model.config import *
from AIPUBuilder.model.loadweight import LoadWeight
from AIPUBuilder.model.utils import update_tensor_data
from AIPUBuilder.nn import Module
from safetensors import safe_open


class BaseModel(Module):
    llm_path = ''

    def __init__(self, config=None):
        self.config = config
        if self.config.get("num_hidden_layers", False):
            from AIPUBuilder.model.llm.llmutils import gen_layer_ids
            msg = f"\t current num_hidden_layers = {config.num_hidden_layers}"
            nhl = self.config.get('num_hidden_layers')
            if isinstance(nhl, list):
                msg += f", its layer block index = {gen_layer_ids(nhl)}"
            INFO(msg)

    def _load_state_dict(self, path=''):
        weights = None
        if self.config.get('weights', None) is not None and isinstance(self.config.weights, LoadWeight):
            weights = self.config.weights
        else:
            weights = LoadWeight(self.config)
            llminfo = weights.llminfo()
            self.config.update({'weights': weights})
            self.config.update(llminfo)
        return weights

        # state_dict = {}
        # if path == '' and self.llm_path != '':
        #     path = self.llm_path
        # if not os.path.exists(path):
        #     raise Exception(f"{path} is not exists when loading stat dict")
        # if os.path.isdir(path):
        #     INFO(f"\t loading huggingface weight.")
        #     safetensor_files = []
        #     torchbin_files = []
        #     for i in os.listdir(path):
        #         if i.endswith(".safetensors"):
        #             safetensor_files.append(i)
        #         if i.endswith('.bin'):
        #             torchbin_files.append(i)

        #     if len(safetensor_files):
        #         for safetensor in safetensor_files:
        #             try:
        #                 state_dict.update(self._load_state_dict(os.path.join(path, safetensor)))
        #             except Exception as e:
        #                 ERROR(f"parse the {safetensor} failed in load state dict, {e}")
        #     elif len(torchbin_files):
        #         for torchbin in torchbin_files:
        #             try:
        #                 w = torch.load(os.path.join(path, torchbin))
        #                 state_dict.update(w)
        #             except Exception as e:
        #                 ERROR(f"parse the {torchbin} failed in load state dict, {e}")
        #     else:
        #         ERROR(f"safetensor or torch bin both donot founded in {path}, please check it.")
        #     return state_dict
        # with safe_open(path, framework="pt", device="cpu") as f:
        #     for key in f.keys():
        #         state_dict[key] = f.get_tensor(key)
        # return state_dict

    @classmethod
    def load_config(cls, pretrained_config_path):
        cls.llm_path = pretrained_config_path
        config_path = os.path.join(pretrained_config_path, 'config.json')
        pretrained_config = PretrainedConfig.from_json_file(config_path)
        return pretrained_config

    @classmethod
    def from_pretrained(cls, pretrained_path_or_config, kwargs=None):
        pretrained_config = None
        if isinstance(pretrained_path_or_config, str):
            cls.llm_path = pretrained_path_or_config
            config_path = os.path.join(
                pretrained_path_or_config, 'config.json')
            pretrained_config = PretrainedConfig.from_json_file(config_path)

        if isinstance(pretrained_path_or_config, PretrainedConfig):
            cls.llm_path = pretrained_path_or_config.get('model_path', '')
            pretrained_config = pretrained_path_or_config

        if not isinstance(pretrained_config, PretrainedConfig):
            raise Exception(
                f'wrong pretrained_config type, now is {type(pretrained_config)}, which should be PretrainedConfig')
        if kwargs is not None:
            for k, v in kwargs.items():
                pretrained_config.__setattr__(k, v)
        model = cls(pretrained_config)
        return model, pretrained_config


class BaseEmbedding(BaseModel):
    def __init__(self, config, prefix=''):
        super().__init__(config)
        weights = self._load_state_dict()
        config.update({'weights': weights})
        self.aipu_g = None
        self.prefix = prefix
        self.embed_weights = None

    def embedding_data(self, force_float=True):
        dd = self.embed_weights
        if self.embed_weights is not None and isinstance(self.embed_weights, dict):
            if len(self.embed_weights) > 1 and (force_float or self.config.get('force_float_ir', False)):
                if 'data' in self.embed_weights:
                    if 'scale' in self.embed_weights and 'zerop' in self.embed_weights:
                        d = self.embed_weights['data']
                        s = self.embed_weights['scale']
                        z = self.embed_weights['zerop']
                        scales = s.reshape(d.shape[0], -1)
                        zeros = z.reshape(d.shape[0], -1)
                        if self.config.hf_quantization_config.get('group_size', None) is not None:
                            group_size = self.config.hf_quantization_config.get('group_size')
                        else:
                            group_size = d.shape[-1] // scales.shape[-1]

                        scales = scales.repeat_interleave(group_size, dim=-1)
                        zeros = zeros.repeat_interleave(group_size, dim=-1)
                        dd = (d.float() + zeros.float()) / scales.float()
                    else:
                        dd = self.embed_weights['data']
            else:
                dd = self.embed_weights['data']
        return dd

    def forward(self):
        seq_len = self.config.seq_len
        with Graph('embedding') as self.aipu_g:
            input_ids = ops.input([1, seq_len], Dtype.INT32, name='input_ids')
            t = Tensor('embedding_data', TensorShape(list(self.embed_weights['data'].shape)), Dtype.FP32)
            update_tensor_data(t, self.embed_weights)
            embed_const = ops.constant(t)
            embed_const.op.constants['weights'].name = f"{self.prefix}embed_tokens.weight"
            _ = ops.gather(embed_const, input_ids, 0, 0)
        return self.aipu_g


class TCVisionHub(object):
    """torch model automatic download"""

    def __init__(self):
        self.model = None

    def __call__(self):
        return self.model
