from AIPUBuilder.core import *
from AIPUBuilder.plugin_loader import *

NewOp=register_optype("NewOp")
@register_plugin(PluginType.Builder,0)
class MyPyLUTPlugin(BuilderOpPlugin):
    def get_graph_pattern(self):
        return ([("0", NewOp)], [])

    def get_score(self):
        return 100

    def set_target(self, target: str):
        print(target)
        return True

    def check_params(self, nodes):
        print("checking and using params for NewOP")
        return True
 
    
    def setup(self, sgnode, nodes):
        print("Setting up")
        sgnode.attrs["keeping_layout"]=True
        valid_in=[
            [DataLayout.NCHWC32],
            [DataLayout.NCHWC16],
            [DataLayout.NCHW],
            [DataLayout.NHWC],
        ]
        sgnode.attrs["valid_input_layouts"]=valid_in
        sgnode.attrs["valid_output_layouts"]=valid_in
        return True

    def generate_params(self, sgnode, nodes):
        print("generating ro")
        inp=sgnode.inputs[0]
        out=sgnode.outputs[0]
        start_idx= 0 if inp.shape.dim()<4 else 1
        ro=BuilderParams()
        for i in range(3):
            if i<inp.shape.dim():
                ro.append(inp.shape[start_idx+i])
            else:
                ro.append(1)
        ro.append(sgnode.constants[nodes[0].name+"/lut"])
        ro.append(inp)
        ro.append(out)
        return ro
    
    def genreate_code_file(self,sgnode,nodes):
        return "sigmoid.c"
   