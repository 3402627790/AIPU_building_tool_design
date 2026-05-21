import os
import json
import sys
sys.path.insert(0, "/project/ai/scratch01/tomgao_01/work/new_space/AIPU_building_tool_design_cpp/python/src")
from AIPUBuilder.plugin_loader import *
from AIPUBuilder.core import *
# from AIPUBuilder.core._builder import build_graph, Target
from AIPUBuilder.core._checker import *
NewOp = register_optype("AddOne")

@register_plugin(PluginType.Checker, 0)
class AddoneCheckerPlugin(CheckerOpPlugin):
    CFLAG = ["-g", "-O0"]

    def get_op_type(self):
        return OpType.AddOne


    def check(self, sgnode):
        # print("check_IR for NewOP=============================================================")
        errors = []
        errors.append(IRError(IRErrorType.InvalidParam, "stride_x", "invalid params"))
        return errors
    def infer_shape(self, sgnode):
        inp = sgnode.inputs[0]
        # out = sgnode.outputs[0]
        print(inp)
        shape = inp.shape

        return [shape]

def main():
    # err = IRError(IRErrorType.InvalidParam, "kernel_x", "invalid params")
    # print(err.type(), err.key(), err.msg())
    import numpy as np
    g=Graph()
    print(os.getcwd())
    file_name = "/project/ai/scratch01/tomgao_01/work/new_space/AIPU_building_tool_design_cpp/tutorial/checker/graph_addone.def"
    check_ir(file_name)

if __name__ == "__main__":
    main()