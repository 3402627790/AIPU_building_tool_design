# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import torch
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import *
from AIPUBuilder.Optimizer.utils import *
# layer_name=RefineDetDetion0
# layer_type=RefineDetDetection
# layer_bottom=[arm_conf_data,conf_data,arm_loc_data,loc_data,anchor]
# layer_bottom_shape=[[1,16320,2],[1,16320,60],[1,16320,4],[1,16320,4],[1,16320,4]]
# layer_bottom_type=[uint8,uint8,int8,int8,int16]
# layer_top=[output]
# layer_top_shape=[[1,200,4]]
# layer_top_datalayout=[NHWC]
# layer_top_type=[int16]
# proposal_lut_type=int32
# proposal_lut_offset=0
# proposal_lut_size=1024
# proposal_lut_shape=[256]
# proposal_scale_type=uint16,uint16,uint16
# proposal_scale_value=[16579,21059,21152]
# proposal_shift_type=int8,int8,int8
# proposal_shift_value=[24,30,14]
# decodebox_lut_type=int32
# decodebox_lut_offset=1024
# decodebox_lut_size=1024
# decodebox_lut_shape=[256]
# decodebox_scale_type=uint16,uint16,uint16
# decodebox_scale_value=[19775,19030,16384]
# decodebox_shift_type=int8,int8,int8
# decodebox_shift_value=[25,30,14]
# obj_thresh = 255
# obj_scale_type=[uint16,uint16]
# obj_scale_value= [25599,256]
# conf_thresh= 255
# conf_scale_type=[uint16,uint16]
# conf_scale_value= [25600,256]
# pre_nms_topk=1000
# post_nms_topk=200
# nms_scale_value=16384
# nms_scale_type=int16
# nms_shift_type=int8
# nms_shift_value=14
# nms_iou_thresh_shift=8
# nms_iou_thresh=115


register_optype("RefineDetDetection")


class OptSubgraph(object):

    def __init__(self, subgraph_name='opt_subgraph'):
        self.subgraph_name = subgraph_name
        self.subgraph = QuantizeGraph(subgraph_name)
        self.quant_subgraph = None

    @staticmethod
    def get_node_param(graph, node_name, param_name):
        ret = None
        for sg_node in graph.nodes:
            if node_name in sg_node.name or sg_node.name in node_name:
                ret = sg_node.get_param(param_name)
        if ret is None:
            OPT_ERROR(f"the node_name={node_name} is not in subgraph nodes, please recheck the node name")
        return ret

    @staticmethod
    def get_node_constant(graph, node_name, constant_name):
        ret = None
        for sg_node in graph.nodes:
            if node_name in sg_node.name or sg_node.name in node_name:
                ret = sg_node.get_constant(constant_name)
        if ret is None:
            OPT_ERROR(f"the node_name={node_name} is not in subgraph nodes, please recheck the node name")
        return ret

    @staticmethod
    def set_node_param(node, params):
        # float IR params for float;
        for key, p in params.items():
            node.params[key] = p

    @staticmethod
    def set_node_attr(src_node, dst_node, quant_attr_cfgs=None):
        # init attrs
        def init_attrs(key, default_value=None):
            if src_node is None:
                dst_node.attrs[key] = default_value
            else:
                dst_node.attrs[key] = src_node.attrs[key] if key in src_node.attrs else default_value

        init_attrs('layer_id')
        dst_node.attrs['layer_top_type_original'] = [dtype2str(dst_node.outputs[0].dtype)]
        init_attrs('q_mode_activation', "per_tensor_symmetric_full_range")
        init_attrs('q_mode_weight',     "per_tensor_symmetric_restricted_range")
        # bias and weight better to have the same quantize_method
        init_attrs('q_mode_bias', dst_node.attrs['q_mode_weight'])
        init_attrs('q_bits_activation', 8)
        init_attrs('q_bits_weight', 8)
        if dst_node.type == OpType.BatchNorm:
            dst_node.attrs['q_bits_weight'] = 16
        init_attrs('q_bits_bias', 32)
        init_attrs('q_strategy_activation', "mean")
        init_attrs('q_strategy_weight', "extrema")
        init_attrs('q_strategy_bias', "extrema")
        init_attrs('running_statistic_momentum', 0.9)
        init_attrs('histc_bins', 2048)
        init_attrs('debug_fake_quantize', False)
        init_attrs('resize_degrade_to_nearest', False)
        init_attrs('with_winograd', False),
        init_attrs('lut_items_in_bits', 8)
        dst_node.attrs['optimization_info'] = {}
        dst_node.attrs['batch_size_in_IR'] = 1
        dst_node.attrs['calculate_running_time'] = False
        dst_node.attrs['multiplier_bits'] = 8

        # update attrs by config dict: QuantAttrsCfg
        #    1. cfg items error.
        if quant_attr_cfgs is not None:
            cfgs_dict = quant_attr_cfgs.get(dst_node.name, None)
            #    2. not empty dict.
            if cfgs_dict:
                for k, value in cfgs_dict.items():
                    if k not in dst_node.attrs:
                        OPT_WARN(f"dst_node={dst_node.name} does not have key={k} in node.attrs.")
                    dst_node.attrs[k] = value
            # else:
            #     OPT_DEBUG(f"set attrs error in node: {dst_node.name}")

    @staticmethod
    def create_node(node_name, op_type, inputs: list, outputs: list, params: dict):
        # step1. new node
        node = PyNode(node_name, op_type)  # -> str, OpType

        # step2. add input
        node.additional = True
        for i in inputs:
            if op_type in [OpType.Constant, ]:
                if len(inputs) != 1:
                    OPT_ERROR(f"weights tensor number must be one, please check the Constant op inputs.")
                node.constants["weights"] = i
                break
            else:
                node.add_input(i)

        # step3. add output
        for o in outputs:
            output_dtype = Dtype.FP32
            if len(o) == 3:  # output_name, output_shape, output_dtype:default=Dtype.FP32
                output_dtype = o[2]
            out_tensor = PyTensor(o[0], TensorShape(o[1]), output_dtype)
            out_tensor.ir_shape = TensorShape(o[1])
            node.add_output(out_tensor)

        # step4. set params(IR params)
        OptSubgraph.set_node_param(node, params)
        return node

    def add_node_to_subgraph(self, node):
        if self.subgraph is None:
            self.subgraph = QuantizeGraph(self.subgraph_name)
        self.subgraph.add_node(node)


class RefineDetSubgraph(OptSubgraph):

    def __init__(self, subgraph_name='refinedet_subgraph'):
        super().__init__(subgraph_name)
        self.quant_attr_cfgs = {
            "SNRD_input_arm_conf": {"q_bits_activation": 8},
            "SNRD_input_odm_conf": {"q_bits_activation": 8},
            "SNRD_input_arm_loc": {"q_bits_activation": 8},
            "SNRD_input_odm_loc": {"q_bits_activation": 8},
            "SNRD_input_anchors": {"q_bits_activation": 16},
            "SNRD_boundingbox_arm": {"q_bits_activation": 8},
            "SNRD_boundingbox_odm": {"q_bits_activation": 8},
            "SNRD_slice_l7":        {},
            "SNRD_slice_l8":        {},
            "SNRD_constant_l9":     {},
            "SNRD_logical_l10":     {},
            "SNRD_reshape_l11":     {},
            "SNRD_topk_l12":        {},
            "SNRD_gantherl13":      {},
            "SNRD_ganther_l14":     {},
            "SNRD_reshape_l14_1":   {},
            "SNRD_constant_l15":    {},
            "SNRD_transpose_l16":   {},
            "SNRD_reshape_l17":     {},
            "SNRD_logical_l18":     {},
            "SNRD_topk_l19":        {},
            "SNRD_gather_l20":      {},
            "SNRD_reshape_l21":     {},
            "SNRD_histogram_l22":   {},
            "SNRD_constant_l23":    {"q_bits_activation": 16},
            "SNRD_NMS_l24":         {},
            "SNRD_reshape_l25":     {},
            "SNRD_reshape_l26":     {},
            "SNRD_topk_l27":        {},
            "SNRD_reshape_l28":     {},
            "SNRD_gather_l29":      {},
            "SNRD_constant_l30":    {},
            "SNRD_gather_l31":      {},
            "SNRD_concat_l32":      {"q_bits_activation": 16},
        }

        self.attrs_update_done = False
        # OPT_INFO(f"init refinedet subgraph OK.")

    def add_to_subgraph(self, node_name, op_type, inputs: list, outputs: list, params: dict):
        new_node = self.create_node(node_name, op_type, inputs, outputs, params)
        self.add_node_to_subgraph(new_node)
        return new_node

    def build_subgraph(self, obj_thresh=0.01, conf_thresh=0.01):
        # 1. build constant
        # constant tensor
        constant_tensor_l9 = PyTensor("SNRD_obj_thresh_const", torch.tensor(obj_thresh).reshape([1, 1, 1]).float())
        constant_tensor_l15 = PyTensor("SNRD_conf_thresh_const", torch.tensor(conf_thresh).reshape([1, 1]).float())
        constant_tensor_l23 = PyTensor("SNRD_class_num_const", torch.arange(1, 60).reshape([59, 1]).int())
        constant_tensor_l30 = PyTensor("SNRD_id_repeat_const", torch.arange(
            1, 60).reshape([1, 59, 1]).repeat(1, 1000, 1).int())

        # 2. build the sub_graph(add new attrs)
        # Params:
        #       node_name, optype, inputs:[input0, input1,...], outputs:[[output_name, output_shape], .....], params:{}
        # SNRD: SubNodeRefineDectect
        arm_conf_node = self.add_to_subgraph("SNRD_input_arm_conf", OpType.Input,
                                             [],
                                             [['arm_conf_score_post_reshape', [1, 16320, 2]], ], {})
        odm_conf_node = self.add_to_subgraph("SNRD_input_odm_conf", OpType.Input,
                                             [],
                                             [['odm_conf_score_post_reshape', [1, 16320, 60]], ], {})
        arm_loc_node = self.add_to_subgraph("SNRD_input_arm_loc", OpType.Input,
                                            [],
                                            [['arm_loc_post_reshape', [1, 16320, 4]], ], {})
        odm_loc_node = self.add_to_subgraph("SNRD_input_odm_loc", OpType.Input,
                                            [],
                                            [['odm_loc_post_reshape', [1, 16320, 4]], ], {})
        anchors_node = self.add_to_subgraph("SNRD_input_anchors", OpType.Input,
                                            [],
                                            [['anchors', [1, 16320, 4]], ], {})
        boundingbox0 = self.add_to_subgraph("SNRD_boundingbox_arm", OpType.BoundingBox,
                                            [anchors_node.outputs[0], arm_loc_node.outputs[0]],
                                            [["SNRD_boundbox0_out", [1, 16320, 4]], ['arm_no_use0', [1, 1]], ['arm_no_use1', [1, 1]]], {})
        boundingbox1 = self.add_to_subgraph("SNRD_boundingbox_odm", OpType.BoundingBox,
                                            [boundingbox0.outputs[0], odm_loc_node.outputs[0]],
                                            [["SNRD_boundbox1_out", [1, 16320, 4]], ['odm_no_use0', [1, 1]], ['odm_no_use1', [1, 1]]], {})
        slice_l7 = self.add_to_subgraph("SNRD_slice_l7", OpType.Slice,
                                        [arm_conf_node.outputs[0]],
                                        [["SNRD_slice_l7_out", [1, 16320, 1]]],
                                        {"strides": [1, 1, 1], "begin": [0, 0, 1], "end": [1, 16320, 2]})
        slice_l8 = self.add_to_subgraph("SNRD_slice_l8", OpType.Slice,
                                        [odm_conf_node.outputs[0]],
                                        [["SNRD_slice_l8_out", [1, 16320, 59]], ],
                                        {"strides": [1, 1, 1], "begin": [0, 0, 1], "end": [1, 16320, 60]})
        constant_l9 = self.add_to_subgraph("SNRD_constant_l9", OpType.Constant,
                                           [constant_tensor_l9],
                                           [["constant_l9_out", [1, 1, 1]]], {})
        logical_l10 = self.add_to_subgraph("SNRD_logical_l10", OpType.Logical,
                                           [slice_l7.outputs[0], constant_l9.outputs[0]],
                                           [["SNRD_object_index", [1, 16320, 1]], ],
                                           {"method": "GREATER_EQUAL"})
        reshape_l11 = self.add_to_subgraph("SNRD_reshape_l11", OpType.Reshape,
                                           [logical_l10.outputs[0]],
                                           [["SNRD_object_index_reshape", [1, 16320]]], {})
        topk_l12 = self.add_to_subgraph("SNRD_topk_l12", OpType.TopK,
                                        [reshape_l11.outputs[0]],
                                        [["SNRD_topk_score", [1, 1000]], ["SNRD_topk_index", [1, 1000]]],
                                        {"k": 1000, "axis": 1, "largest": True, "sorted": True, "select_index": "random"})
        gather_l13 = self.add_to_subgraph("SNRD_gantherl13", OpType.Gather,
                                          [slice_l8.outputs[0], topk_l12.outputs[1]],
                                          [["SNRD_odm_sorted_score", [1, 1000, 59]]],
                                          {"axis": 1, "batch_dims": 0})  # sort odm score
        reshape_l13_1 = self.add_to_subgraph("SNRD_reshape_l13_1", OpType.Reshape,
                                             [gather_l13.outputs[0]],
                                             [["SNRD_odm_sorted_after_topk_reshape", [1, 1, 1000, 59]]], {})
        gather_l14 = self.add_to_subgraph("SNRD_ganther_l14", OpType.Gather,
                                          [boundingbox1.outputs[0], topk_l12.outputs[1]],
                                          [["SNRD_decode_box_after_topk", [1, 1000, 4]]],
                                          {"axis": 1, "batch_dims": 0})
        # reshape_l14_1 = self.add_to_subgraph("SNRD_reshape_l14_1", OpType.Reshape,
        #                                      [gather_l14.outputs[0]],
        #                                      [["SNRD_decode_box_after_topk", [1, 1000, 4]]], {})
        constant_l15 = self.add_to_subgraph("SNRD_constant_l15", OpType.Constant,
                                            [constant_tensor_l15],
                                            [["SNRD_constant_l15_out", [1, 1]]], {})
        transpose_l16 = self.add_to_subgraph("SNRD_transpose_l16", OpType.Transpose,
                                             [reshape_l13_1.outputs[0]],
                                             [["SNRD_odm_conf_transpose", [1, 1, 59, 1000]]],
                                             {"perm": [0, 1, 3, 2]})
        reshape_l17 = self.add_to_subgraph("SNRD_reshape_l17", OpType.Reshape,
                                           [transpose_l16.outputs[0]],
                                           [["SNRD_odm_conf_reshape", [59, 1000]], ], {})
        logical_l18 = self.add_to_subgraph("SNRD_logical_l18", OpType.Logical,
                                           [reshape_l17.outputs[0], constant_l15.outputs[0]],
                                           [["SNRD_score_index_stage2", [59, 1000]]],
                                           {"method": "GREATER_EQUAL"})
        topk_l19 = self.add_to_subgraph("SNRD_topk_l19", OpType.TopK,
                                        [reshape_l17.outputs[0]],
                                        [["SNRD_topk_score2", [59, 1000]], ["SNRD_topk_index2", [59, 1000]]],
                                        {"k": 1000, "axis": 1, "largest": True, "sorted": True, "select_index": "random"})
        gather_l20 = self.add_to_subgraph("SNRD_gather_l20", OpType.Gather,
                                          # [reshape_l14_1.outputs[0], topk_l19.outputs[1]],
                                          [gather_l14.outputs[0], topk_l19.outputs[1]],
                                          [["SNRD_decodebox_after_gather_1", [1, 59, 1000, 4]]],
                                          {"axis": 1, "batch_dims": 0})
        reshape_l21 = self.add_to_subgraph("SNRD_reshape_l21", OpType.Reshape,
                                           [gather_l20.outputs[0]],
                                           [["SNRD_decode_box_after_gather", [59, 1000, 4]]], {})
        histogram_l22 = self.add_to_subgraph("SNRD_histogram_l22", OpType.Count,
                                             [logical_l18.outputs[0]],
                                             [["SNRD_histogram_out", [59, 59]]],
                                             {"nbins": 59, "min": 1, "max": 59, "discrete": True})
        constant_l23 = self.add_to_subgraph("SNRD_constant_l23", OpType.Constant,
                                            [constant_tensor_l23],
                                            [["SNRD_constant_l23_out", [59, 1], Dtype.UINT16]], {})
        NMS_l24 = self.add_to_subgraph("SNRD_NMS_l24", OpType.NMS,
                                       [reshape_l21.outputs[0], histogram_l22.outputs[0],
                                           constant_l23.outputs[0], topk_l19.outputs[0]],
                                       [["SNRD_nms_out_box", [59, 1000, 4]], ["SNRD_nms_out_perbox", [59, 59]], [
                                           "SNRD_nms_out_score", [59, 1000]], ["SNRD_nms_out_box_index", [59, 1000]]],
                                       {"iou_threshold": 0.45})
        reshape_l25 = self.add_to_subgraph("SNRD_reshape_l25", OpType.Reshape,
                                           [NMS_l24.outputs[2]],
                                           [["SNRD_nms_out_score_reshape", [1, 59000]]], {})
        reshape_l26 = self.add_to_subgraph("SNRD_reshape_l26", OpType.Reshape,
                                           [NMS_l24.outputs[0]],
                                           [["SNRD_nms_out_box_reshape", [1, 59000, 4]]], {})
        topk_l27 = self.add_to_subgraph("SNRD_topk_l27", OpType.TopK,
                                        [reshape_l25.outputs[0]],
                                        [["SNRD_final_sorted_out_score", [1, 200]], [
                                            "SNRD_final_sorted_out_index", [1, 200]]],
                                        {"k": 200, "axis": 1, "largest": True, "sorted": True, "select_index": "random"})
        reshape_l28 = self.add_to_subgraph("SNRD_reshape_l28", OpType.Reshape,
                                           [topk_l27.outputs[0]],
                                           [["SNRD_final_score_after_reshape", [1, 200, 1]]], {})
        gather_l29 = self.add_to_subgraph("SNRD_gather_l29", OpType.Gather,
                                          [reshape_l26.outputs[0], topk_l27.outputs[1]],
                                          [["SNRD_gather_box_loc_output", [1, 200, 4]]],
                                          {"axis": 1, "batch_dims": 0})
        constant_l30 = self.add_to_subgraph("SNRD_constant_l30", OpType.Constant,
                                            [constant_tensor_l30],
                                            [["SNRD_constant_l30_out", [1, 59000, 1], Dtype.UINT16]], {})
        gather_l31 = self.add_to_subgraph("SNRD_gather_l31", OpType.Gather,
                                          [constant_l30.outputs[0], topk_l27.outputs[1]],
                                          [["SNRD_gather_label_output", [1, 200, 1]], ],
                                          {"axis": 1, "batch_dims": 0})
        concat_l32 = self.add_to_subgraph("SNRD_concat_l32", OpType.Concat,
                                          [gather_l29.outputs[0], gather_l31.outputs[0], reshape_l28.outputs[0]],
                                          [["SNRD_output", [1, 200, 6]]],
                                          {"axis": 2})

        self.subgraph.input_tensors = (arm_conf_node.outputs[0], odm_conf_node.outputs[0], arm_loc_node.outputs[0],
                                       odm_loc_node.outputs[0], anchors_node.outputs[0])
        self.subgraph.output_tensors = (concat_l32.outputs[0], )
        return self.subgraph


'''
Instance a global RefineDetSubgraph() when importing this file, and we only have one instance. When forwarding, the input tensors of
each RefineDetDetection op will update to the inputs of refinedet_subgraph, and the outputs of nodes
in subgraph will save to each RefineDetDetection op's placeholders.

#TODO: if more than one RefineDetDetection op in CompassIR, the placeholders mode will casue issue.
'''
refinedet_subgraph = RefineDetSubgraph()
graph = refinedet_subgraph.build_subgraph()


@op_register(OpType.RefineDetDetection)
def refinedet_detection(self, *args):
    refinedet_subgraph.subgraph.reset_edge_tensors_ref_count()
    if self.quantized:
        refinedet_subgraph.quant_subgraph.reset_edge_tensors_ref_count()
    arm_conf = self.inputs[0].betensor
    odm_conf = self.inputs[1].betensor
    # caffe ssd type [x,y,w,h] to tensorflow fasterrcnn type [y,x,h,w]
    arm_loc = torch.stack((self.inputs[2].betensor[:, :, 1], self.inputs[2].betensor[:, :, 0],
                           self.inputs[2].betensor[:, :, 3], self.inputs[2].betensor[:, :, 2]), 2)
    odm_loc = torch.stack((self.inputs[3].betensor[:, :, 1], self.inputs[3].betensor[:, :, 0],
                           self.inputs[3].betensor[:, :, 3], self.inputs[3].betensor[:, :, 2]), 2)
    anchors = torch.stack((self.inputs[4].betensor[:, :, 1], self.inputs[4].betensor[:, :, 0],
                           self.inputs[4].betensor[:, :, 3], self.inputs[4].betensor[:, :, 2]), 2)
    rd_input_tensors = [arm_conf, odm_conf, arm_loc, odm_loc, anchors]

    # step1. update attrs to subgraph node
    if not refinedet_subgraph.attrs_update_done:
        for sg_node in refinedet_subgraph.subgraph.nodes:
            refinedet_subgraph.set_node_attr(self, sg_node, refinedet_subgraph.quant_attr_cfgs)
        refinedet_subgraph.attrs_update_done = True

    # step2. update inputs to subgraph.input_tensors and forward
    if self.quantized:
        for input_t, sg_input_t in zip(rd_input_tensors, refinedet_subgraph.quant_subgraph.input_tensors):
            sg_input_t.betensor = input_t
        for sg_q_node in refinedet_subgraph.quant_subgraph.nodes:
            if sg_q_node.type in [OpType.Input]:
                sg_q_node.quantized = False
            else:
                sg_q_node.quantized = True
            sg_q_node.forward()
        ret_betensor = refinedet_subgraph.quant_subgraph.output_tensors[-1].betensor
    else:
        ret_betensor = refinedet_subgraph.subgraph.forward(rd_input_tensors)[-1].betensor

    # step3. add statistic tensor
    if len(self.placeholders) == 0 and not self.quantized:
        for n in refinedet_subgraph.subgraph.nodes:
            for ot in n.outputs:
                self.placeholders.append(ot)
    elif self.quantized:
        self.placeholders.clear()
        for qn in refinedet_subgraph.quant_subgraph.nodes:
            for ot in qn.outputs:
                self.placeholders.append(ot)
    else:
        OPT_DEBUG(f"add statistic tensor")

    # tensorflow fasterrcnn type [y,x,h,w] change back to caffe ssd type [x,y,w,h]
    ret_betensor = torch.stack((ret_betensor[:, :, 1],
                                ret_betensor[:, :, 0],
                                ret_betensor[:, :, 3],
                                ret_betensor[:, :, 2],
                                ret_betensor[:, :, 4],
                                ret_betensor[:, :, 5],
                                ), 2)
    self.outputs[0].betensor = ret_betensor[:, :, :4]
    refinedet_subgraph.subgraph.reset_edge_tensors_ref_count()
    if self.quantized:
        refinedet_subgraph.quant_subgraph.reset_edge_tensors_ref_count()
    return self.outputs[0].betensor


@quant_register(OpType.RefineDetDetection)
def refinedet_detection_quantize(self, *args):

    if refinedet_subgraph.quant_subgraph is None:
        refinedet_subgraph.quant_subgraph = refinedet_subgraph.subgraph.clone()
    else:
        tmp_dict = {}
        for qn in refinedet_subgraph.quant_subgraph.nodes:
            tmp_dict.update({qn.name: qn})

        for n in refinedet_subgraph.subgraph.nodes:
            qn = tmp_dict[n.name]
            for k, v in n.params.items():
                qn.params[k] = v

    quant_subgraph = refinedet_subgraph.quant_subgraph
    for i_t, sg_i_t in zip(self.inputs, quant_subgraph.input_tensors):
        default_tensor_property = get_tensor_default_property()
        for k in default_tensor_property+['dtype']:
            sg_i_t.__setattr__(k, i_t.__getattribute__(k))

    if not refinedet_subgraph.attrs_update_done:
        for sg_node in refinedet_subgraph.quant_subgraph.nodes:
            refinedet_subgraph.set_node_attr(None, sg_node, refinedet_subgraph.quant_attr_cfgs)
        refinedet_subgraph.attrs_update_done = True

    for sg_node in quant_subgraph.nodes:
        sg_node.quantized = False
        sg_node.quantize()

    default_tensor_property = get_tensor_default_property()
    sg_last_output = quant_subgraph.tensors('SNRD_output')
    for k in default_tensor_property+['dtype']:
        self.outputs[0].__setattr__(k, sg_last_output.__getattribute__(k))

    self.params['proposal_scale_value'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_arm", "box_scale_value")
    self.params['proposal_scale_type'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_arm", "box_scale_type")
    self.params['proposal_shift_value'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_arm", "box_shift_value")
    self.params['proposal_shift_type'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_arm", "box_shift_type")
    self.constants['proposal_th_lut'] = RefineDetSubgraph.get_node_constant(
        quant_subgraph, "SNRD_boundingbox_arm", "th_lut")
    self.constants['proposal_tw_lut'] = RefineDetSubgraph.get_node_constant(
        quant_subgraph, "SNRD_boundingbox_arm", "tw_lut")

    self.params['decodebox_scale_value'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_odm", "box_scale_value")
    self.params['decodebox_scale_type'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_odm", "box_scale_type")
    self.params['decodebox_shift_value'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_odm", "box_shift_value")
    self.params['decodebox_shift_type'] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_boundingbox_odm", "box_shift_type")
    self.constants['decodebox_th_lut'] = RefineDetSubgraph.get_node_constant(
        quant_subgraph, "SNRD_boundingbox_odm", "th_lut")
    self.constants['decodebox_tw_lut'] = RefineDetSubgraph.get_node_constant(
        quant_subgraph, "SNRD_boundingbox_odm", "tw_lut")

    self.params["obj_thresh"] = RefineDetSubgraph.get_node_constant(
        quant_subgraph, "SNRD_constant_l9", "weights").betensor.reshape(1).int().item()
    self.params["obj_scale_type"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_logical_l10", "scale_type")
    self.params["obj_scale_value"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_logical_l10", "scale_value")

    self.params["conf_thresh"] = RefineDetSubgraph.get_node_constant(
        quant_subgraph, "SNRD_constant_l15", "weights").betensor.reshape(1).int().item()
    self.params["conf_scale_type"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_logical_l18", "scale_type")
    self.params["conf_scale_value"] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_logical_l18", "scale_value")

    self.params["pre_nms_topk"] = 1000
    self.params["post_nms_topk"] = 200
    self.params["nms_scale_value"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_NMS_l24", "scale_value")
    self.params["nms_scale_type"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_NMS_l24", "scale_type")
    self.params["nms_shift_type"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_NMS_l24", "shift_type")
    self.params["nms_shift_value"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_NMS_l24", "shift_value")
    self.params["nms_iou_tresh"] = RefineDetSubgraph.get_node_param(quant_subgraph, "SNRD_NMS_l24", "iou_threshold")
    self.params["nms_iou_thresh_shift"] = RefineDetSubgraph.get_node_param(
        quant_subgraph, "SNRD_NMS_l24", "iou_thresh_shift")


def refinedet_combine(arm_loc, arm_conf, prior_box, odm_loc, odm_conf):

    def apply_box_deltas(boxes, deltas, variances):
        ############################
        # caffe ssd/refinedet model: [x,y,w,h]
        # tensorflow fasterrcnn/maskrcnn model: [y,x,h,w]
        xcenter_a = (boxes[:, 0] + boxes[:, 2]) / 2
        ycenter_a = (boxes[:, 1] + boxes[:, 3]) / 2
        wa = boxes[:, 2] - boxes[:, 0]
        ha = boxes[:, 3] - boxes[:, 1]

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        dx /= 10
        dy /= 10
        dh /= 5
        dw /= 5

        xcenter = dx * wa + xcenter_a
        ycenter = dy * ha + ycenter_a
        w = np.exp(dw) * wa
        h = np.exp(dh) * ha
        xmin = xcenter - w / 2.
        ymin = ycenter - h / 2.
        xmax = xcenter + w / 2.
        ymax = ycenter + h / 2.

        box = np.stack([xmin, ymin, xmax, ymax], axis=0).transpose((1, 0))
        return box

    def NMS_F(boxes, scores, nms_thresh, top_k):

        order = np.argsort(scores, axis=0)[::-1]  # get descending index

        if len(order) > top_k and top_k > -1:
            order = order[:top_k]

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2-x1) * (y2-y1)
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(np.zeros_like(xx1), xx2 - xx1)
            h = np.maximum(np.zeros_like(xx1), yy2 - yy1)

            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]

        return np.stack(keep)

    num_classes = 60
    background_label = 0
    top_k = 1000
    keep_top_k = 500
    nms_threshold = 0.45
    conf_thresh = 0.01
    objectness_thre = 0.01

    batch = odm_loc.shape[0]
    output = []
    for i in range(batch):
        default = apply_box_deltas(prior_box[0], arm_loc[i], [10, 5])
        decoded_boxes = apply_box_deltas(default, odm_loc[i], [10, 5])

        arm_object_conf = arm_conf[i, :, 1:].squeeze()  # exclude background
        object_index = np.where(arm_object_conf > objectness_thre)  # 1D index 16320 --> 14440;
        # conf_data[no_object_index.expand_as(conf_data)] = 0

        # for each class, perform nms
        conf_preds = odm_conf[i, :, :][object_index]  # [14440, 21]
        conf_scores = conf_preds.transpose((1, 0))   # [num_class, num_prior]=[21, 14440]
        decoded_boxes = decoded_boxes[object_index]

        for cl in range(1, num_classes):
            score_index = np.where(conf_scores[cl] > conf_thresh)
            scores = conf_scores[cl][score_index]

            # c_mask = conf_scores[cl].gt(self.conf_thresh)
            # scores = conf_scores[cl][c_mask]
            # print(scores.dim())
            if scores.size == 0:
                continue

            boxes = decoded_boxes[score_index]
            # idx of highest scoring and non-overlapping boxes per class
            #print(boxes, scores)
            ids = NMS_F(boxes, scores, nms_threshold, top_k)
            count = len(ids)

            labels = np.repeat(cl, count).reshape((-1, 1))
            box_label = np.concatenate((boxes[ids], labels), axis=1)
            box_label_score = np.concatenate((box_label, np.expand_dims(scores[ids], 1)), axis=1)
            # label_score = np.concatenate((box_label, np.expand_dims(scores[ids], 1)), axis=1)
            # label_score_box = np.concatenate((label_score, boxes[ids]), axis=1)
            out = np.expand_dims(box_label_score, 0)

            if len(output) == 0:
                output = out
            else:
                output = np.concatenate((output, out), axis=1)
        if output.shape[1] > 100:
            ids = np.argsort(output[:, :, 1], axis=1)[::-1]
            keep_out = output[:, ids[0, :100], :]  # todo
        else:
            keep_out = output

        return keep_out

# def show_box(GT_bbox, out_bbox, topN=5, title='float'):
#     # Function(): final box show;
#     # cat_dict['bbox'][i] = (x1,y1,x2,y2), date type: 0.0~1.0
#     cat_dict = {"GT_bbox": [
#         [0.1, 0.1, 0.3, 0.3],
#     ],
#         "out_bbox": [
#         [0.2, 0.2, 0.4, 0.4],
#     ],
#     }
#     if 1:
#         cat_dict["GT_bbox"] = GT_bbox[:topN, :4]
#         cat_dict["out_bbox"] = out_bbox[:topN, :4] if title == 'float' else out_bbox[:topN, :4] / 32767.

#     # my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']  # red, green, blue, black, etc.
#     color = 'r' if title == 'float' else 'b'

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     plt.title(title)
#     for i in range(topN):
#         plt.gca().add_patch(plt.Rectangle(xy=(cat_dict['out_bbox'][i][1], cat_dict['out_bbox'][i][0]),
#                                           width=cat_dict['out_bbox'][i][3] - cat_dict['out_bbox'][i][1],
#                                           height=cat_dict['out_bbox'][i][2] - cat_dict['out_bbox'][i][0],
#                                           edgecolor=color, alpha=0.3,
#                                           fill=False, linewidth=2))
#     plt.show()

# if __name__ == "__main__":
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # register_optype("RefineDetDetection")


#     arm_loc = np.loadtxt(os.path.join(bp, "arm_loc")).reshape([1, 16320, 4])
#     arm_conf = np.loadtxt(os.path.join(bp, "arm_conf_flatten")).reshape([1, 16320, 2])
#     prior_box = np.loadtxt(os.path.join(bp, "arm_priorbox"))[:16320*4].reshape([1, 16320, 4])
#     odm_loc = np.loadtxt(os.path.join(bp, "odm_loc")).reshape([1, 16320, 4])
#     odm_conf = np.loadtxt(os.path.join(bp, "odm_conf_flatten")).reshape([1, 16320, 60])
#     GT = np.loadtxt(os.path.join(bp, "detection_out"))  # [500, 6]

#     # 1. build float graph

#     rfd_graph = QuantizeGraph('refinedet_graph')
#     arm_conf_node = OptSubgraph.create_node("SNRD_input_arm_conf", OpType.Input,
#                                             [],
#                                             [['arm_conf_score_post_reshape', [1, 16320, 2]], ], {})
#     odm_conf_node = OptSubgraph.create_node("SNRD_input_odm_conf", OpType.Input,
#                                             [],
#                                             [['odm_conf_score_post_reshape', [1, 16320, 60]], ], {})
#     arm_loc_node = OptSubgraph.create_node("SNRD_input_arm_loc", OpType.Input,
#                                            [],
#                                            [['arm_loc_post_reshape', [1, 16320, 4]], ], {})
#     odm_loc_node = OptSubgraph.create_node("SNRD_input_odm_loc", OpType.Input,
#                                            [],
#                                            [['odm_loc_post_reshape', [1, 16320, 4]], ], {})
#     anchors_node = OptSubgraph.create_node("SNRD_input_anchors", OpType.Input,
#                                            [],
#                                            [['anchors', [1, 16320, 4]], ], {})
#     refinedet_detection_node = OptSubgraph.create_node("refinedet_detection", OpType.RefineDetDetection,
#                                                        [arm_conf_node.outputs[0], odm_conf_node.outputs[0],
#                                                            arm_loc_node.outputs[0], odm_loc_node.outputs[0], anchors_node.outputs[0]],
#                                                        [["refinedet_detection_output", [1, 200, 6]]],
#                                                        {"obj_thresh": 0.01, "conf_thresh": 0.01, "pre_nms_topk": 1000, "post_nms_topk": 200})

#     rfd_graph.add_node(arm_conf_node)
#     rfd_graph.add_node(odm_conf_node)
#     rfd_graph.add_node(arm_loc_node)
#     rfd_graph.add_node(odm_loc_node)
#     rfd_graph.add_node(anchors_node)
#     rfd_graph.add_node(refinedet_detection_node)

#     for i, n in enumerate(rfd_graph.nodes):
#         n.attrs['layer_id'] = str(i)
#     ########################################################################
#     #
#     # unit test: float flow.
#     #
#     ########################################################################

#     UNIT_FLOAT = 1
#     if UNIT_FLOAT:
#         input_datas = [arm_conf, odm_conf, arm_loc, odm_loc, prior_box]
#         for d, i_t in zip(input_datas, rfd_graph.input_tensors):
#             i_t.betensor = d
#         for n in rfd_graph.nodes:
#             n.forward()
#         out = rfd_graph.output_tensors[0].betensor
#         show_box(GT, out.cpu().numpy().reshape(200, 6), topN=6)

#     ############################################################
#     #
#     #
#     #  unit test: quantization
#     #
#     ############################################################
#     UNIT_QUANT = 1
#     if UNIT_QUANT:
#         # update attr
#         for n in rfd_graph.nodes:
#             OptSubgraph.set_node_attr(None, n, refinedet_subgraph.quant_attr_cfgs)

#         # calibrate
#         for n in rfd_graph.nodes:
#             for t in n.inputs + n.outputs:
#                 t.min = t.betensor.min()
#                 t.max = t.betensor.max()
#                 t.extrema_min = t.min
#                 t.extrema_max = t.max
#             for t in n.placeholders:
#                 t.min = t.betensor.min()
#                 t.max = t.betensor.max()
#                 t.extrema_min = t.min
#                 t.extrema_max = t.max

#         # clone graph
#         rfd_quant_graph = rfd_graph.clone()
#         rfd_quant_graph.topological_sort()

#         # quantize
#         for n in rfd_quant_graph.nodes:
#             n.quantize()
#             n.quantized = True

#         # quantize forward
#         input_datas = [arm_conf, odm_conf, arm_loc, odm_loc, prior_box]
#         for d, i_t in zip(input_datas, rfd_quant_graph.input_tensors):
#             i_t.betensor = PyTensor('tmp', d).betensor
#         for qn in rfd_quant_graph.nodes:
#             qn.forward()
#         quant_out = rfd_quant_graph.output_tensors[0].betensor
#         show_box(GT, quant_out.cpu().numpy().reshape(200, 6), topN=7, title='quant')

#     quant_name_node = {}
#     for q_n in rfd_quant_graph.nodes:
#         quant_name_node.update({q_n.name: q_n})
#     for f_n in rfd_graph.nodes:
#         q_n = quant_name_node[f_n.name]
#         OPT_INFO(f"\nop_type={f_n.type}, and f_op_name={f_n.name}, q_op_name={q_n.name}")
#         for f_o, q_o in zip(f_n.outputs, q_n.outputs):
#             cos = cosine_distance(f_o.betensor, q_o.betensor)
#             OPT_INFO(f"f_o.name={f_o.name}, cosdis={cos}")

#     OPT_INFO(f"begin to show nodes'cosine distance of subgraph.")
#     quant_subgraph_name_node = {}
#     for q_n in refinedet_subgraph.quant_subgraph.nodes:
#         quant_subgraph_name_node.update({q_n.name: q_n})
#     for f_n in refinedet_subgraph.subgraph.nodes:
#         q_n = quant_subgraph_name_node[f_n.name]
#         for f_o, q_o in zip(f_n.outputs, q_n.outputs):
#             cos = cosine_distance(f_o.betensor, q_o.betensor)
#             OPT_INFO(f"subgraph: layer_type={f_n.type}, output={f_o.name}, cos={cos}")

#     f_index = out[:, :, 4]
#     q_index = quant_out[:, :, 4]
#     f_score = out[:, :, 5]
#     q_score = quant_out[:, :, 5]
#     f_coord = out[:, :, :4]
#     q_coord = quant_out[:, :, :4]
#     index_cos = cosine_distance(f_index, q_index)
#     score_cos = cosine_distance(f_score, q_score)
#     coord_cos = cosine_distance(f_coord, q_coord)
#     OPT_INFO(f"index_cos={index_cos}, score_cos={score_cos}, coord_cos={coord_cos}")
