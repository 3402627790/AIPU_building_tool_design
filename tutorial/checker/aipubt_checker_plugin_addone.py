import sys
import os
CUR_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT_DIR=os.path.join(CUR_DIR,"../../")
sys.path.insert(0,os.path.join(ROOT_DIR,"python/src"))
from AIPUBuilder.plugin_loader import *
from AIPUBuilder.core import *
NewOp = register_optype("AddOne")

@register_plugin(PluginType.Checker, 0)
class AddoneCheckerPlugin(CheckerOpPlugin):
    CFLAG = ["-g", "-O0"]

    def get_op_type(self):
        return OpType.AddOne

    # return error list
    def check(self, node):
        errors = []
        if not hasattr(node, "inputs"):
            errors.append(IRError(IRErrorType.MissingParam,
                          "layer_bottom",
                          "Inputs required"))
        elif len(node.inputs) != 1:
            errors.append(IRError(IRErrorType.InvalidParam,
                          "layer_bottom",
                          "Invalid inputs num " +
                           str(len(node.inputs)) +
                           ", expected 1"))
        return errors

    # return shape list
    def infer_shape(self, node):
        inp = node.inputs[0]
        # out = node.outputs[0]
        shape = TensorShape(inp.shape)
        shape.layout = DataLayout.NHWC

        return [shape]

def main():
    print("Tutorial for IRChecker Python Plugin API")
    file_name = os.path.join(CUR_DIR,"graph_addone.def")
    from AIPUBuilder._C._checker import check_ir
    check_ir(file_name)

if __name__ == "__main__":
    main()