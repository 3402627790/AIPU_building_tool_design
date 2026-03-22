import sys
# from AIPUBuilder.core._builder import build_graph, Target
from AIPUBuilder.core import *
from AIPUBuilder.plugin_loader import *
from AIPUBuilder.logger import *

import numpy as np


OpType_Transpose = register_optype("Transpose")
@register_plugin(PluginType.Builder, 0)
class TransposePlugin(BuilderOpPlugin):
    def get_graph_pattern(self):
        # print("checking and using params for dummy Transpose")
        return ([("0", OpType.Transpose)], [])

    def get_score(self):
        return 32767

    def set_target(self, target: str):
        if target not in ['X2_1204']:
            return False
        self.target = target.lower()
        return True

    def check_params(self, nodes):
        return True

    def setup(self, sgnode, nodes):
        sgnode.attrs["keeping_layout"] = False
        valid_in = [[DataLayout.NHWC], [DataLayout.NCHWC32]]
        sgnode.attrs["valid_input_layouts"] = valid_in
        sgnode.attrs["valid_output_layouts"] = valid_in
        return True

    def generate_params(self, sgnode, nodes):
        INFO("Dummy Transpose plugin ro generation...")
        inp_1 = sgnode.inputs[0]
        inp_2 = 0x5a5a
        out_1 = sgnode.outputs[0]
        print(nodes[0].inputs[0].shape.symbol)
        ih = nodes[0].inputs[0].shape.symbol.h
        iw = nodes[0].inputs[0].shape.symbol.w
        print(type(ih))
        dsize = nodes[0].inputs[0].shape.symbol.size()
        ro = BuilderParams()
        ro.append(0x4)
        ro.append(ih)
        ro.append(0x5a5a4)
        ro.append(iw)
        ro.append(dsize)
        ro.append(inp_1)
        ro.append(inp_2)
        ro.append(out_1)
        return ro

    def generate_code_file(self, sgnode, nodes):
        name = f"test_ds_kernel.cl"
        return name

    def generate_descriptor(self, sgnode, nodes):
        return BuilderParams()
