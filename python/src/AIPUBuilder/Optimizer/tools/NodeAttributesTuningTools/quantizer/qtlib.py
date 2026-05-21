from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.config.parser import CfgParser
from AIPUBuilder.Optimizer.optmaster import *
from AIPUBuilder.Optimizer.config.cfg_fields import ALL_FIELDS, DEFAULT_FIELDS
import types
import re
import os


def parser_override(self, databus, attrs):
    for k, v in ALL_FIELDS.items():
        v = v.default()
        if v.upper() == "FALSE":
            v = False
        elif v.upper() == "TRUE":
            v = True
        elif re.findall('^[-+]?\d+$', v):
            v = int(v)
        elif re.findall('^[-+]?[0-9]+\.?[0-9]*$', v):
            v = float(v)
        self.argv[k] = v

    self.argv["graph"] = databus["graph"]
    self.argv["bin"] = databus["bin"]
    self.argv['model_name'] = "aipuattr_model"
    self.argv['output_dir'] = databus['dump_dir']
    self.argv['out_ir_name'] = "aipuattr_quant"

    for k, v in databus["quantizer_hparams"].items():
        self.argv[k] = v

    if "attr_entry" in databus and databus["attr_entry"] == "hparam":
        for k, v in attrs:
            self.argv[k] = v
        self.argv["out_ir_name"] = "aipuattr_quant_" + str(hash(str(attrs)))
    return self.argv


class CfgArg(dict):
    opt_config = ''

    def __init__(self, d):
        super().__init__(d)

    def __getattr__(self, k):
        return self.__getitem__(k)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)


class QtlibQuantizer:

    def __init__(self, databus, attrs: [(str, object)]):
        self.databus = databus
        self.parser = CfgParser("", {}, {})
        self.attrs = attrs
        self.parser.parser_override = parser_override
        self.args = parser_override(self.parser, self.databus, self.attrs)
        self.graph = QuantizeGraph.parse(self.databus.graph, self.databus.bin)

    def quantize(self):
        cfgarg = CfgArg(self.args)
        for k, v in self.args.items():
            self.databus[k] = v
        optimizer = OptMaster(self.graph, cfgarg)
        optimizer()
        self.databus["quantized_IR"]["qtlib"].append({
            "attr": self.attrs,
            "graph": os.path.join(self.databus["output_dir"], f"{self.args['out_ir_name']}.txt"),
            "bin": os.path.join(self.databus["output_dir"], f"{self.args['out_ir_name']}.bin")
        })
        dtype = self.graph.output_tensors[0].dtype
        self.databus.simulator_output_type = dtype2nptype(dtype)
        self.databus.gt_output_type = dtype2nptype(dtype)
