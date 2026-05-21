import copy
import numpy as np
import torch
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.core import Dtype
from AIPUBuilder.model.app.baseapp import BaseApp
from AIPUBuilder.model.utils import parse_string_to_pairs
from AIPUBuilder.model.register import register_app, ENGINES


class Analysis(object):
    @staticmethod
    def cosine_distance(a, b):
        x = torch.tensor(a, dtype=torch.float64) if not isinstance(a, torch.Tensor) else a.double()
        y = torch.tensor(b, dtype=torch.float64) if not isinstance(b, torch.Tensor) else b.double()

        t1 = x.flatten()
        t2 = y.flatten()
        t1_m = torch.norm(t1, p=2)
        t2_m = torch.norm(t2, p=2)
        t1t2 = torch.dot(t1, t2)
        t1t2_m = t1_m * t2_m
        if t1t2_m.item() == 0.0:
            if t1_m == t2_m:
                return 1.0
            else:
                return 0.0
        else:
            if t1t2 == t1t2_m:
                return 1.0
            else:
                return (t1t2 / t1t2_m).item()

    @staticmethod
    def l1_metric(data1, data2):
        data1 = np.array(data1).flatten()
        data2 = np.array(data2).flatten()
        data_1 = data1.astype(np.float32)
        data_2 = data2.astype(np.float32)
        number = 0
        error = 0
        max_diff = -512
        number = data_2[data_2 != 0].size
        # for data in data_2:
        #     if data != 0:
        #         number += 1

        if len(data_1) != len(data_2):
            length = min(len(data_1), len(data_2))
            op = np.sum(np.abs(data_1[:length] - data_2[:length]))
            if number == 0:
                print('classify number zero error!')
            else:
                error = op / number
            print('Length Gap Exists')
        else:
            op = np.sum(np.abs(data_1 - data_2))
            max_diff = np.max(np.abs(data_1 - data_2))
            if number == 0:
                print('classify number zero error!')
            else:
                error = op / number
                # print(f"error={error}, op={op}, number={number}")
        return op, error, number, max_diff


@register_app()
class Similarity(BaseApp):

    def __init__(self):
        """Similarity Compare constructor
        the similarity compare mode:
            egine="exe,opt", will compare the quantized IR between engine=exe and engine=opt.
            egine="exe,sim", will compare aipu binary between engine=exe and engine=sim.
            egine=opt, will compare between the float IR and quantized IR with engine=opt.

        Parameters
        ---------
        mode: str
        inputs: str
            the input data path, a numpy or input binary splitted by ','.
        """
        super().__init__()

        self.input_datas = []
        self.engines = []

        # floatIR and quantIR similarity in same engine, like engine=opt
        # sameIR in differenct engines, like quantIR in opt, exe and sim

    def _get_input_datas(self):

        for config_idx, config in enumerate(self.configs):
            input_paths = config.get('input', None)
            input_shapes = config.get('input_shape', None)
            input_dtypes = self.engines[config_idx].input_dtypes
            if input_paths is None:
                ERROR(f"please set input data paths of {config.config_name}")
            if input_shapes is not None:
                input_shapes = parse_string_to_pairs(f"input_shapes={input_shapes}")["input_shapes"]

            input_paths = input_paths.replace(" ", '')
            if ',' in input_paths:
                input_paths = input_paths.split(',')
            else:
                input_paths = [input_paths]

            input_datas = []
            for input_idx, ipath in enumerate(input_paths):
                if ipath.endswith(".npy"):
                    input_datas.append(np.load(ipath))
                elif ipath.endswith(".bin") and len(input_shapes):
                    in_shape = input_shapes[input_idx]
                    in_dtype = Dtype(input_dtypes[input_idx].lower()).np
                    input_datas.append(np.fromfile(
                        ipath, dtype=in_dtype).reshape(in_shape))
                else:
                    ERROR(
                        f"please check input file type={ipath}, now only support file name endswith .npy or .bin, and if use .bin, please set the --input_shape")
            self.input_datas.append(input_datas)
            INFO(f"{config.config_name} has len(input_datas) = {len(input_datas)}.")
            config.update({'input_datas': input_datas})

    def _gen_engines(self):
        if len(self.configs) > 1:
            '''
            # multi-configs will use its config to compare
            case1:
                [Common]
                ...
                engine=opt
                input=xxx.npy

                [fir]
                model_name=xx

                [qir]
                model_name=xx
                quant_type=w4

            case2:
                [Common]
                ...
                model_name=xx
                quant_type=w4
                input=xxx.npy

                [qir0]
                engine=opt

                [qir1]
                engine=exe
            '''
            for config in self.configs:
                engine_key = config.get("engine")
                if isinstance(engine_key, (tuple, list)):
                    engine_key = engine_key[0]
                self.engines.append(ENGINES[engine_key](config))
        elif len(self.configs) == 1:
            '''
                now only supports one_config with one engine = opt to compare similarity between floatIR and quantized IR.
            '''
            config = self.configs[0]
            engine_key = config.get("engine")
            if config.has_float_ir and config.has_quantized_ir:
                self.engines.append(ENGINES[engine_key](config))
                config.config_name = 'quantized_ir'

                config_cloned = copy.deepcopy(config)
                config_cloned.config_name = 'float_ir'
                self.configs.append(config_cloned)
                config_cloned.pop('quantized_ir_path')
                if config.has_quantized_ir:
                    config_cloned.pop('aipu_binary_path')
                self.engines.append(ENGINES[engine_key](config_cloned))
        else:
            WARN(f"when using{self.__class__.__name__} should be configurated config section.")

        if len(self.engines) == 0:
            ERROR(f"when run similarity app, getting len(engines) == 0.")

    def _engine_forward(self):
        for engine_idx, engine in enumerate(self.engines):
            out = engine.forward(self.input_datas[engine_idx])
            self.configs[engine_idx].update({'output_datas': out})

    def _compare(self):
        out0 = self.configs[0].output_datas
        if not all([len(out0) == len(config.output_datas) for config in self.configs]):
            ERROR(
                f"all configs should have the len(output_datas), but now is {[len(config.output_datas) for config in self.configs]}")

        for config in self.configs[1:]:
            outx = config.output_datas
            for o0, ox in zip(out0, outx):
                l1_error = Analysis.l1_metric(o0, ox)
                cos = Analysis.cosine_distance(o0, ox)
                INFO(
                    f"compare between output0(from config_name={self.configs[0].config_name}, and engine={self.configs[0].engine}) and output_x(from config_name={config.config_name}, and engine=config.engine), the l1_error = {l1_error}, cos = {cos}.")

    def forward(self, inputs=None):
        """
        step1: gen_engines: multi-configs, one-config
        step2: get_input_datas
        step3: engine.forward
        step4: compare the result
        """
        self._gen_engines()
        self._get_input_datas()
        self._engine_forward()
        self._compare()
