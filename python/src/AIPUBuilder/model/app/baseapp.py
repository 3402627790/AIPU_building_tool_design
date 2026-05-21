from abc import ABC, abstractmethod
import torch
from AIPUBuilder.logger import ERROR


class BaseApp(ABC):
    def __init__(self):
        self.aipu_bins = None
        self.configs = None
        self.config_basic_infos = ''

    def set_configs(self, configs):
        self.configs = configs
        self.aipu_bins = [config.get('aipu_binary_path', '') for config in self.configs]
        self.config_basic_infos = ''.join([c.get_basic_info_from_config() for c in self.configs])

    @abstractmethod
    def forward(self, inputs=None, *args, **kwargs):
        pass

    def _remove_betensor(self, model):
        from AIPUBuilder.core import use_cuda
        for n in model.nodes:
            for ot in n.outputs:
                ot.remove_betensor()
            for ck, ct in n.constants.items():
                ct.remove_betensor()

        model = None
        if use_cuda():
            torch.cuda.empty_cache()

    def __repr__(self):
        s = f'{self.__class__.__name__}:\n'
        for k, v in self.__dict__.items():
            s += f'{k}: {v}\n'
        return s
