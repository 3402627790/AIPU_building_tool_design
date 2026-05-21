from collections import OrderedDict
import numpy as np
from AIPUBuilder.Parser.plugin_op import ParserOp, ParserSubgraph
from AIPUBuilder.Parser.node import TensorShape
from AIPUBuilder.plugin_loader import register_plugin, PluginType



@register_plugin(PluginType.Parser, '0.1')
class DprnnSubgraph(ParserSubgraph):

    def __init__(self, framework, params):
        super(DprnnSubgraph, self).__init__(framework, params)

    def replacing_nodes(self):
        ret = {}
        nodes = self.params['nodes']
        edges = self.params['edges']
        input_names = self.params['input_nodes']
        output_names = self.params['output_nodes']

        if input_names \
                and output_names \
                and all([in_name in nodes for in_name in input_names]) \
                and all([out_name in nodes for out_name in output_names]):
            input_concat = input_names[0]
            output_reshape = output_names[0]
            input_concat_node = nodes[input_concat]
            output_reshape_node = nodes[output_reshape]
            if input_concat_node.get('type', '') == 'ConcatV2' and output_reshape_node.get('type', '') == 'Reshape':
                outside_in_names, outside_out_names = [], []
                input_shapes, output_shapes = [], []
                for e, e_attr in edges.items():
                    if e[1] == input_concat:
                        input_shapes.append(e_attr.get('shape', []))
                        outside_in_names.append(e[0])
                    if e[0] == output_reshape:
                        output_shapes.append(e_attr.get('shape', []))
                        outside_out_names.append(e[1])

                if len(input_shapes) == 2 \
                        and len(output_shapes) >= 1 \
                        and input_shapes[0] == [1, 250, 256, 64] \
                        and input_shapes[1] == [1, 249, 256, 64] \
                        and output_shapes[0] == [499, 256, 64]:
                    replacing_nodes, replacing_edges = OrderedDict(), OrderedDict()
                    replacing_names = ['replacing_' + name for name in ['concat', 'reshape0', 'pad', 'reshape1', 'transpose', 'reshape2', 'crop', 'reshape3']]
                    in_names_map = {input_concat: replacing_names[0]}
                    out_names_map = {output_reshape: replacing_names[-1]}

                    for e, e_attr in edges.items():
                        if e[0] in outside_in_names:
                            replacing_edges.update({(e[0], in_names_map[e[1]], e[2]): e_attr})
                        if e[1] in outside_out_names:
                            replacing_edges.update({(out_names_map[e[0]], e[1], e[2]): e_attr})

                    for i in range(len(replacing_names) - 1):
                        shape = []
                        if i == 0:      # replacing_concat
                            shape = [1, 499, 256, 64]
                        elif i == 1:    # replacing_reshape0
                            shape = [1, 998, 128, 64]
                        elif i == 2:    # replacing_pad
                            shape = [1, 1000, 128, 64]
                        elif i == 3:    # replacing_reshape1
                            shape = [2, 250, 256, 64]
                        elif i == 4:    # replacing_transpose
                            shape = [250, 2, 256, 64]
                        elif i == 5:    # replacing_reshape2
                            shape = [1, 500, 256, 64]
                        elif i == 6:    # replacing_crop
                            shape = [1, 499, 256, 64]
                        elif i == 7:    # replacing_reshape3
                            shape = [499, 256, 64]
                        attr = {'src_out_port': 0, 'dst_in_port': 0, 'shape': shape}
                        replacing_edges.update({(replacing_names[i], replacing_names[i+1], 0): attr})

                    replacing_nodes.update({'replacing_concat': {'type': 'ConcatV2', 'axis': 1, 'N': 2}})
                    replacing_nodes.update({'replacing_reshape0': {'type': 'Reshape', 'dim': [1, 998, 128, 64]}})
                    replacing_nodes.update({'replacing_pad': {'type': 'Pad', 'padding': np.array([[0, 0], [0, 2], [0, 0], [0, 0]], np.int32)}})
                    replacing_nodes.update({'replacing_reshape1': {'type': 'Reshape', 'dim': [2, 250, 256, 64]}})
                    replacing_nodes.update({'replacing_transpose': {'type': 'Transpose', 'perm': [1, 0, 2, 3]}})
                    replacing_nodes.update({'replacing_reshape2': {'type': 'Reshape', 'dim': [1, 500, 256, 64]}})
                    replacing_nodes.update({'replacing_crop': {'type': 'StridedSlice',  'begin': np.array([0, 0, 0, 0], np.int32), 'end': np.array([1, 499, 256, 64], np.int32)}})
                    replacing_nodes.update({'replacing_reshape3': {'type': 'Reshape', 'dim': [499, 256, 64]}})
                    ret = {'nodes': replacing_nodes,
                           'edges': replacing_edges,
                           'input_nodes': ['replacing_concat'],
                           'output_nodes': ['replacing_reshape3']
                           }
                else:
                    print('Warning: Parser subgraph plugin meets invalid input/output shapes!')
            else:
                print('Warning: Parser subgraph plugin meets invalid input/output types!')
        else:
            print('Warning: Parser subgraph plugin meets invalid input_names or output_names!')
        return ret
