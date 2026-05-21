from AIPUBuilder.CGBuilder._core import *
from AIPUBuilder.CGBuilder._builder import *
from AIPUBuilder.CGBuilder._plugin_loader import register_plugin,register_optype

NewOp=register_optype("NewOp")
@register_plugin(0)
class MyPyLUTPlugin(BuilderOpPlugin):
    def get_graph_pattern(self):
        return ([("0", NewOp)], [])

    def get_score(self):
        return 100

    def set_target(self, target: str):
        print(target)
        return True

    def check_params(self, nodes):
        print("checking params")
        method=str(nodes[0].params["method"])
        if method=="MAX":
            return True
        print("failed~~~~")
        return False
    
    def setup(self, sgnode, nodes):
        print("Setting up")
        return True

    def generate_params(self, sgnode, nodes):
        print("generating ro")
        inp=sgnode.inputs[0]
        out=sgnode.outputs[0]
        params=nodes[0].params
        ro=BuilderParams()
        ro.append(inp.shape.w)
        ro.append(inp.shape.h)
        ro.append(inp.shape.c)
        ro.append(params["kernel_x"])
        ro.append(params["kernel_y"])
        ro.append(params["stride_x"])
        ro.append(params["stride_y"])
        ro.append(params["pad_left"])
        ro.append(params["pad_right"])
        ro.append(params["pad_top"])
        ro.append(params["pad_bottom"])
        ro.append(inp)
        ro.append(out)
        print(ro[-1])
        print(ro)
        return ro

    def generate_code_name(self, sgnode, nodes):
        print("generating code")
        inp=sgnode.inputs[0]
        if inp.type == Dtype.UINT8:
            return   "maxpool_u8.bin"
        elif inp.type == Dtype.INT8:
            return "maxpool_s8.bin";
        return ""
    def generate_descriptor(self,sgnode,nodes):
        print("generating desc")
        return BuilderParams()

