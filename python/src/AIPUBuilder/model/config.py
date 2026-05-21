import copy
import json
import os


class PretrainedConfig(object):
    def __init__(self, **kwargs):
        self.config_name = 'unknow'
        self.update(kwargs)
        self.hf_quantization_config = kwargs.get('quantization_config', None)

    def set_if_not_exist(self, key, value):
        if not hasattr(self, key):
            setattr(self, key, value)

    def update(self, kv):
        for k, v in kv.items():
            self.__setattr__(k, v)

    def get(self, k, default=None):
        if hasattr(self, k):
            return self.__getattribute__(k)
        else:
            return default

    @classmethod
    def from_dict(cls, config):
        config = copy.deepcopy(config)
        this = cls(**config)
        return this

    @classmethod
    def from_json_file(cls, config_file: str):
        if config_file.endswith('.json') and os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
        else:
            config = {}
        pcfg = PretrainedConfig.from_dict(config)
        return pcfg

    def __contains__(self, key):
        return key in self.__dict__

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        for k, v in self.__dict__.items():
            yield k, v

    def pop(self, key):
        if key in self:
            return self.__dict__.pop(key)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def get_basic_info_from_config(self):
        info_str = "config info:{"
        info_str += f"model_path: {self.get('model_path', '')}, "
        info_str += f"model_type: {self.get('model_type', '')}, "
        info_str += f"target: {self.get('target', '')}, "
        if self.get('app', '') != '':
            info_str += f"app: {self.get('app', '')}, "
            info_str += f"engine: {self.get('engine', '')},"
            if self.get('engine', '') == 'sim':
                info_str += f"simulator: {self.get('simulator')},"
        info_str += '}\n'
        return info_str

    def __repr__(self):
        multi_str = ""
        for k, v in self.__dict__.items():
            if "weights" in k:
                continue
            multi_str += f"{k}:{v} "
        return multi_str

    @property
    def has_aipu_binary(self):
        ret = False
        if 'aipu_binary_path' in self:
            aipu_bin = self.get('aipu_binary_path')
            ret = isinstance(aipu_bin, str) and (aipu_bin.endswith('.bin') or aipu_bin.endswith('.zip')) and os.path.exists(aipu_bin)
        return ret

    @property
    def has_quantized_ir(self):
        ret = False
        ir_path = self.get('quantized_ir_path', None)
        aipu_binary_path = self.get('aipu_binary_path', None)
        if ir_path is not None:
            ret = isinstance(ir_path, str) and ir_path.endswith('.txt') and os.path.exists(ir_path)
        elif aipu_binary_path is not None:
            ret = isinstance(aipu_binary_path, str) and aipu_binary_path.endswith(
                '.txt') and os.path.exists(aipu_binary_path)
        return ret

    @property
    def has_float_ir(self):
        ret = False
        ir_path = self.get('float_ir_path', None)
        if ir_path is not None:
            ret = isinstance(ir_path, str) and ir_path.endswith('.txt') and os.path.exists(ir_path)
        return ret

    @property
    def has_ir(self):
        return self.has_float_ir or self.has_quantized_ir
