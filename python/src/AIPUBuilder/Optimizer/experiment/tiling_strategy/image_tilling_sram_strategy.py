# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from pickle import FALSE
import sys
import os
from AIPUBuilder.Optimizer.framework import *
from AIPUBuilder.Optimizer.logger import tqdm
from AIPUBuilder.Optimizer.utils.quant_tool_utils import cosine_distance
from AIPUBuilder.Optimizer.optmaster import *
from AIPUBuilder.Optimizer.utils import *
from AIPUBuilder import ops
from AIPUBuilder.Optimizer.logger import OPT_WARN, OPT_ERROR, OPT_FATAL, OPT_DEBUG

from re import A, split
import sys
import os
import torch
import numpy as np
import copy
from abc import ABCMeta, abstractmethod

################################# shape infer Manager(Rigister + Factory) #########################


class ShapeInferManager(dict):
    """
    Shape Infer Register.
    """

    def __init__(self, name, *args, **kwargs):
        super(ShapeInferManager, self).__init__(*args, **kwargs)
        self._name = name
        self._factory = {}
        self._split_strategy = None
        self._split_params = None
        self._all_blocks_size = None
        self._sram_restriction = None
        self._adj_matrix = None
        self._device = torch.device("cpu")
    ######################## shape infer attrs ################################

    @property
    def adj_matrix(self):
        if self._adj_matrix:
            return self._adj_matrix
        else:
            OPT_ERROR(self._name + " adj_matirx is not seted, pls call shape_infer.set_adj_matrix(graph)")

    @adj_matrix.setter
    def adj_matrix(self, Adj):
        if Adj != self._adj_matrix and len(Adj) == len(self._adj_matrix):
            self._adj_matrix = copy.deepcopy(Adj)
        else:
            OPT_ERROR("Adj-matrix shape is not the same.")

    @property
    def tranparent_op(self):
        return [OpType.Reshape,
                OpType.Cast,
                OpType.Activation,
                OpType.Transpose]

    @property
    def sram_restriction(self):
        return self._sram_restriction if self._sram_restriction and self._split_strategy == "SRAM" else None

    @sram_restriction.setter
    def sram_restriction(self, KB):
        self._sram_restriction = KB

    @property
    def split_h(self):
        return self._split_params[0]

    @property
    def split_w(self):
        return self._split_params[1]

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def split_params(self):
        """
        input:
            pairs: torch.tensor([3, 4]).
        """
        return self._split_params

    @split_params.setter
    def split_params(self, params):
        """
        input:
            params: torch.tensor([split_h, split_w]).
        """
        if isinstance(params, type(torch.tensor)):
            self._split_params = params.to(self._device)
        else:
            self._split_params = torch.tensor(params, device=self._device)

    @property
    def split_strategy(self):
        return self._split_strategy

    @split_strategy.setter
    def split_strategy(self, strategy_str):
        supported_strategy = ["SRAM", "TILL"]
        if strategy_str in supported_strategy:
            self._split_strategy = strategy_str
        else:
            OPT_ERROR(
                f"{strategy_str} not in supported tilling strategy: [{supported_strategy}].")

    @property
    def all_blocks_size(self):
        if isinstance(self._all_blocks_size, torch.Tensor):
            return self._all_blocks_size
        else:
            OPT_ERROR("shape_infer block size should be initied.")

    @all_blocks_size.setter
    def all_blocks_size(self, input_shape):
        """
                    ______________
         basic_block|    |    |  |
                    |____|____|__|
                    |    |    |  |
                    |____|____|__|
                    |____|____|__|corner_block

        input_shape: [H,W] dims.
        """
        # if self._all_blocks_size:
        #     return self._all_blocks_size

        split_h, split_w = self._split_params
        all_block_size = torch.ones((split_h, split_w, 2), device=self._device)

        normal_block_size = input_shape // self._split_params
        remain = input_shape - normal_block_size * self._split_params
        corner_block_size = normal_block_size + remain

        all_block_size = all_block_size * normal_block_size

        # replace corner block's shape.
        all_block_size[:, split_w-1, 1] = corner_block_size[1]
        all_block_size[split_h-1, :, 0] = corner_block_size[0]

        # record
        if not self._all_blocks_size or self._all_blocks_size != all_block_size:
            self._all_blocks_size = all_block_size

        return all_block_size

    def __getitem__(self, op_type):
        return self._factory.get(op_type)  # not found return None.

    def __setitem__(self, op_type, shape_class):
        self._factory.update({op_type: shape_class})

    def __str__(self):
        return str(self._factory)

    def __contains__(self, key):
        return key in self._factory

    def __repr__(self):
        return "shape infer name: {}, split_strategy: {}, split_params: {}, ".format(
            self._name, self._split_strategy, self._split_params)

    ######################### shape infer base function ########################
    def register(self, op_type):  # decorator params
        def add_register_item(shape_class):
            if op_type not in self._factory:
                self.__setitem__(op_type, shape_class)

        return add_register_item

    def update_adj_matrix(self, graph):
        """
        un-Direced Acyclic Graph
        """
        node_num = len(graph.nodes)
        Adj = np.zeros((node_num, node_num), dtype=np.int8)
        for i in range(node_num):
            for j in range(node_num):
                for child in graph.nodes[i].children:
                    if child == graph.nodes[j]:
                        Adj[i][j] = 1
                for parent in graph.nodes[i].parents:
                    if parent == graph.nodes[j]:
                        Adj[i][j] = 1

        # init Adjacent matirx.
        self._adj_matrix = Adj

    def get_children_ids(self, layer_id):
        all_connected_node = list(np.where(self._adj_matrix[layer_id, :] == 1)[0])
        children = []
        for idx in all_connected_node:
            if idx > layer_id:
                children.append(idx)
        return children

    def get_parents_ids(self, layer_id):
        all_connected_node = list(np.where(self._adj_matrix[layer_id, :] == 1)[0])
        parents = []
        for idx in all_connected_node:
            if idx < layer_id:
                parents.append(idx)
        return parents

    def get_multi_input_4_dead_node(self, layer_id):
        if not len(self._adj_matrix):
            OPT_ERROR("pls init first: shape_infer.update_adj_matrix(graph)")
            return

        class splited_input(object):
            def __init__(self, start_id, end_id) -> None:
                self.start_id = start_id
                self.end_id = end_id

        multi_input = []
        for i in range(layer_id):
            children = self.get_children_ids(i)
            for c in children:
                if c > layer_id:
                    multi_input.append(splited_input(i, c))

    def is_dead_node(self, node):

        if node.type == OpType.Pooling:
            _, H, W, _ = list(node.inputs[0].shape[0])
            kernel_x = int(node.get_param("kernel_x"))
            kernel_y = int(node.get_param("kernel_y"))
            if H == kernel_y and W == kernel_x:
                return True

        return False


# global Variance.
shape_infer = ShapeInferManager(name="image_tilling")

################################## shape infer Ops ################################################


class ShapeOpBase(metaclass=ABCMeta):
    def __init__(self, *args):
        super(ShapeOpBase, self).__init__(*args)

    @abstractmethod
    def forward_shape(self, node):
        """
        split:
        ================runtime start===============================

                           input
                     / ......|...... \
                    |                 |
                  crop_1   ......   crop_n
                    |                 |
                intput_1   ......   input_n

        =================AIPU start==============================
                _________        _________
                |         |      |         |
                | Graph_1 |      | Graph_n |
                |_________|      |_________|
                intput_1 ....... input_n
                    |                |
                crop_1 ......... crop_n
                    |                |
                conv_1 ......... conv_n
                    |                |
                pooling_1 ....... pooling_n
                    .      .......   .
                    .      .......   .
                    .      .......   .
                  fc_1 .......... fc_n
                    |    .........   |
                  out_1 ......... out_n

        ================runtime start=============================

                    out_1 ......... out_n
                      \    .........  /
                       \             /
                           concat
                             |
                            out

        output: multi-core graph; multi-graphs list;
        """
        OPT_ERROR("this is the parent method. should not come here.")

    @abstractmethod
    def backward_shape(self, node):
        """
        merge:
        ==========AIPU start===================================
                    intput
                 /    |    \
                /     |     \
            crop_1..crop..crop_n
              /       |       \
             /        |        \
          conv_1....conv.....conv_n
             |        |         |
             |        |         |
        pooling_1...pooling...pooling_n
             . ...... . ....... .
             . ...... . ....... .
             . ...... . ....... .
             . ...... . ....... .
           fc_1.......fc........fc_n
             \ .......|........./
              \.......|......../
                   concat
                      |
                     out
        """
        OPT_ERROR("this is the parent method. should not come here.")

    def is_splitable(self, node):
        return False
        # OPT_ERROR("this is a abstract method.")

    def footprint(self, node):
        # unit: 1KB = 1024B
        output_shape = list(node.outputs[0].shape)
        featuremap_size = output_shape[1]*output_shape[2]*output_shape[3] // 1024  # H*W*Cout

        constant_size = 0
        if len(node.constants) > 0:
            count = 1
            for k, v in node.constants.items():
                shape = list(v.shape)
                for i in range(1, len(shape)):
                    count *= shape[i]

            constant_size = count

        return featuremap_size, constant_size

    def get_padding_params(self, node):
        pass

##### for every single op#################


@shape_infer.register(OpType.Convolution)
class Conv2d(ShapeOpBase):
    """
    typical HW split;

    get _lens, _ins, _outs:
    |<------------len----------|--->|
    |<----------in-------------|
    |<----------out------------|
    """
    # padding layout format[top,left,bottom,right]

    def fill_padding_param_for_all_blocks(self, node):

        padding_left = node.params['pad_left']
        padding_right = node.params['pad_right']
        padding_top = node.params['pad_top']
        padding_bottom = node.params['pad_bottom']
        stride_x = node.params['stride_x']
        kernel_x = node.params['kernel_x']
        stride_y = node.params['stride_y']
        kernel_y = node.params['kernel_y']

        if node.type == OpType.ConvTranspose:
            padding_left = kernel_x - padding_left - 1
            padding_right = kernel_x - padding_right - 1
            padding_top = kernel_y - padding_top - 1
            padding_bottom = kernel_y - padding_bottom - 1
            stride_x = 1
            stride_y = 1

        padding = torch.zeros((self.block_num_h, self.block_num_h, 4),
                              dtype=torch.int16, device=self.device_type)
        w = node.inputs[0].ir_shape[2]
        # some conv padding is useless, such as input w=10, k=3,s=2,padding left=1, padding right=1
        # o = (10-3+2)/2 + 1 = 5, to get this 5 point, k window slide to indice 7,8,9, right padding is useless
        padding_right = padding_right - \
            (w - kernel_x + padding_left + padding_right) % stride_x
        padding_right = padding_right if padding_right > 0 else 0

        h = node.inputs[0].ir_shape[1]
        padding_bottom = padding_bottom - \
            (h - kernel_y + padding_top + padding_bottom) % stride_y
        padding_bottom = padding_bottom if padding_bottom > 0 else 0

        # padding = torch.zeros((item_num[0],item_num[1],4),dtype=torch.int16,device=device_type)
        # left top corner
        padding[0, 0, 0:2] = torch.tensor([padding_top, padding_left])
        # right bottom corner
        padding[self.block_num_h - 1, self.block_num_w - 1,
                2:4] = torch.tensor([padding_bottom, padding_right])
        # right top corner
        padding[0, self.block_num_w - 1, 3] = torch.tensor([padding_right])
        padding[0, self.block_num_w - 1, 0] = torch.tensor([padding_top])
        # left bottom corner
        padding[self.block_num_w - 1, 0, 2] = torch.tensor([padding_bottom])
        padding[self.block_num_w - 1, 0, 1] = torch.tensor([padding_left])
        # top row
        padding[0, 1:self.block_num_w - 1, 0] = torch.tensor([padding_top])
        # bottom row
        padding[self.block_num_w - 1, 1:self.block_num_w -
                1, 2] = torch.tensor([padding_bottom])

        # left col
        padding[1:self.block_num_w - 1, 0, 1] = torch.tensor([padding_left])
        # right col
        padding[1:self.block_num_w - 1, self.block_num_w -
                1, 3] = torch.tensor([padding_right])
        stride = torch.tensor([stride_y, stride_x], device=self.device_type)
        return padding, stride

    def get_blocks_len_by_output(self, out_size, k_size, stride, padding):

        o = ((out_size - 1) * stride + k_size -
             padding[..., 0:2] - padding[..., 2:4])

        return o

    def get_kernel_size(self, node):
        dl_x = node.get_param('dilation_x', optional=True, default_value=1)
        dl_y = node.get_param('dilation_y', optional=True, default_value=1)
        k_size = torch.tensor([(node.params['kernel_y']-1) * dl_y + 1,
                               (node.params['kernel_x']-1) * dl_x + 1], device=self.device_type)
        return k_size

    def get_blocks_overlap(self, w_len, next_layer_w_in, stride, padding):
        """
        overlap = w_len - (next_layer_w_in * stride - padding)
        """
        return (w_len - (next_layer_w_in * stride - padding[..., 0:2] - padding[..., 2:4]))

    def deal_pooling_op(self, node, blocks_len):
        # pooling ceil mode, if according to formula:
        #            expect_out = (h - k + p) // s + 1< IR's out shape
        # pooling op, out = (hw - k + padding) / stride + 1
        # if (hw - k + padding) % stride != 0, ceil_mode=true,need adjust.

        def pooling_ceil_mode_padding(node):
            left = node.params['pad_left']
            right = node.params['pad_right']
            top = node.params['pad_top']
            bottom = node.params['pad_bottom']
            stride_x = node.params['stride_x']
            kernel_x = node.params['kernel_x']
            stride_y = node.params['stride_y']
            kernel_y = node.params['kernel_y']

            # [top,left,bottom,right]
            padding = torch.tensor(
                [top, left, bottom, right], device=self.device_type)
            kernel_size = torch.tensor(
                [kernel_y, kernel_x], device=self.device_type)
            stride = torch.tensor([stride_y, stride_x],
                                  device=self.device_type)
            hw_in = torch.tensor(
                [node.inputs[0].ir_shape[1], node.inputs[0].ir_shape[2]], device=device_type)
            hw_out = torch.tensor(
                [node.outputs[0].ir_shape[1], node.outputs[0].ir_shape[2]], device=device_type)
            expect_out = self.get_blocks_len_by_output(
                hw_in, kernel_size, stride, padding)
            return hw_out - expect_out
            ###############################################################################################
            # calculate partition Ops input shape and output shape, neightbor part slice offset(work_items_in)
            # bottom shape work_items_len
            # top shape work_items_out
            #############################################################################################

        error = pooling_ceil_mode_padding(node)

        blocks_len[:, self.block_num_w - 1, 1] = blocks_len[:,
                                                            self.block_num_w - 1, 1] - error[1]
        blocks_len[self.block_num_h - 1, :,
                   0] = blocks_len[self.block_num_h - 1, :, 0] - error[0]

    def deal_conv_transpose(self, node, blocks_len, remain, kernel_size, overlap):
        stride = torch.tensor(
            [node.params['stride_y'], node.params['stride_x']], device=self.device_type)
        temp_len = torch.zeros_like(blocks_len)
        left_top = torch.tensor([0, 0], device=self.device_type)
        for item_y in range(self.block_num_h):
            left_top[1] = 0
            for item_x in range(self.block_num_w):
                hw = (blocks_len[item_y, item_x, :] - left_top) // stride
                compensate_point = (
                    blocks_len[item_y, item_x, :]-left_top) % stride != 0
                hw = hw + compensate_point
                remain[:] = (kernel_size[:] - (blocks_len[item_y, item_x, :] - left_top[:]) %
                             stride[:]) % stride[:]
                left_top[1] = remain[1] - 1 + stride[1]*(remain[1] == 0)
                overlap[item_y, item_x, 1] = overlap[item_y, item_x, 1]//stride[1] + \
                    ((overlap[item_y, item_x, 1]-left_top[1]) % stride[1] != 0)
                temp_len[item_y, item_x, :] = hw
            remain[0] = (kernel_size[0]-(blocks_len[item_y, 0, 0]-left_top[0]) %
                         stride[0]) % stride[0]
            left_top[0] = remain[0]-1 + stride[0]*(remain[0] == 0)
            overlap[item_y, :, 0] = overlap[item_y, :, 0]//stride[0] + \
                ((overlap[item_y, :, 0]-left_top[0]) % stride[0] != 0)
        blocks_len[:] = temp_len[:]

    ##############################################################################################
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        self.device_type = node.outputs[0].betensor.device
        out_shape = torch.tensor(
            [node.outputs[0].shape[1], node.outputs[0].shape[2]], device=device_type)

        block_num = shape_infer._split_params  # [split_h, split_w]
        self.block_num_h,  self.block_num_w = block_num
        # split_h, split_w,  (h,w)
        blocks_len = torch.zeros(
            (self.block_num_h, self.block_num_w, 2), device=self.device_type)
        blocks_out = torch.ones(
            (self.block_num_h,  self.block_num_w, 2), device=self.device_type)

        basic_block_shape = out_shape // block_num
        remain = out_shape - basic_block_shape * block_num
        corner_block_shape = basic_block_shape + remain

        blocks_out = blocks_out * basic_block_shape
        blocks_out[:, self.block_num_w, 1] = corner_block_shape[1]
        blocks_out[self.block_num_h, :, 0] = corner_block_shape[0]

        basic_block_in = blocks_out[0, 0, :]
        overlap = torch.zeros((self.block_num_h, self.block_num_w, 2),
                              dtype=torch.int16, device=self.device_type)

        kernel_size = self.get_kernel_size(node)
        padding, stride = self.fill_padding_param_for_all_blocks(node)
        blocks_len = self.get_blocks_len_by_output(
            node, kernel_size, stride, padding)

        if 'ceil_mode' in node.params and node.params['ceil_mode']:
            self.deal_pooling_op(node, blocks_len)

        overlap = self.get_blocks_overlap(
            blocks_len, basic_block_in, stride, padding)

        if node.type == OpType.ConvTranspose:
            self.deal_conv_transpose(
                node, blocks_len, remain, kernel_size, overlap)

    def is_splitable(self, node):
        """
        ir_shape: NHWC
        """

        H = node.inputs[0].ir_shape[1]
        W = node.inputs[0].ir_shape[2]
        C = node.inputs[0].ir_shape[3]
        split_h = int(shape_infer.split_h)
        split_w = int(shape_infer.split_w)
        sram_size = shape_infer.sram_restriction

        if shape_infer.split_strategy == "SRAM":
            splitable = H // split_h > node.params['kernel_y']           \
                and (H // split_h > 0)                           \
                and (W // split_w > node.params['kernel_x'])      \
                and ((H*W*C >= sram_size*512) or (H*W*C >= sram_size*512))
        else:
            splitable = 'kernel_x' in node.params                  \
                        and H // split_h > node.params['kernel_y'] \
                        and (W // split_w > 0)                     \
                        and W // split_w > node.params['kernel_x']

        return splitable


@shape_infer.register(OpType.FullyConnected)
class Fc(ShapeOpBase):
    def __init__(self, *args):
        super(Pooling, self).__init__(*args)

    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)

    def is_splitable(self, node):
        """
        FC only support h dim split.
        """

        not_slidewin_op_split = (node.inputs[0].ir_shape[-2] > shape_infer.split_h
                                 ) and (node.outputs[0].ir_shape[-1] > shape_infer.split_w)
        return not_slidewin_op_split


@shape_infer.register(OpType.Pooling)
class Pooling(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        node.params["kernel_x"] = min(
            node.inputs[0].ir_shape[2], node.params["kernel_x"])
        node.params["kernel_y"] = min(
            node.inputs[0].ir_shape[1], node.params["kernel_y"])

        ret = OP_DICT[node.type](node)

        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)

    def is_splitable(self, node):
        return Conv2d().is_splitable(node)


@shape_infer.register(OpType.Input)
class Input(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)


@shape_infer.register(OpType.Tile)
class Tile(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        reps = self.get_params('repeats')
        h, w = shape_infer.get
        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)


@shape_infer.register(OpType.Eltwise)
class Eltwise(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)

    def is_splitable(self, node):
        return False


@shape_infer.register(OpType.DepthwiseConv)
class DepthwiseConv(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return Conv2d().backward_shape(node)

    def is_splitable(self, node):
        return Conv2d().is_splitable(node)


@shape_infer.register(OpType.ConvTranspose)
class ConvTranspose(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return Conv2d().backward_shape(node)

    def is_splitable(self, node):
        H = node.inputs[0].ir_shape[1]

        splitable = H > node.params['kernel_y']

        return splitable


@shape_infer.register(OpType.DepthToSpace)
class DepthToSpace(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return True

    def is_splitable(self, node):
        not_slidewin_op_split = True
        return not_slidewin_op_split

########### transparent ops #############


@shape_infer.register(OpType.Transpose)
class Transpose(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)

    def is_splitable(self, node):
        # flag =  'kernel_x' in  node.parents[0].params or \
        #          node.inputs[0].shape == node.outputs[0].shape
        if node.type in shape_infer.tranparent_op and 'kernel_x' in node.parents[0].params:
            if node.inputs[0].shape == node.outputs[0].shape or node.type == OpType.Transpose:
                return True
        return False


@shape_infer.register(OpType.Activation)
class Activation(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)

    def is_splitable(self, node):
        return Transpose().is_splitable(node)


@shape_infer.register(OpType.Reshape)
class Reshape(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)

    def is_splitable(self, node):
        return Transpose().is_splitable(node)


@shape_infer.register(OpType.Cast)
class Cast(ShapeOpBase):
    def forward_shape(self, node):
        if node.type not in OP_DICT:
            OPT_FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        ret = OP_DICT[node.type](node)

        # update ir_shape by tensor shape.
        node.outputs[0].ir_shape = TensorShape(list(ret.shape))

        return True

    def backward_shape(self, node):
        return super().backward_shape(node)

    def is_splitable(self, node):
        return Transpose().is_splitable(node)

############################### mem analysis(DAG Color algorithm)##################################


class BlcokedParams(object):
    def __init__(self):
        self.start_ids = []
        self.colors = []
        self.end_ids = []

    def record(self, color, id0, id1):
        assert len(self.colors) == len(set(self.colors)), "error in colotblocked function."

        if color in self.colors:
            return

        self.colors.append(color)
        self.start_ids.append(id0)
        self.end_ids.append(id1)

    def IsDone(self, Adj, target_node_id, cur_node_id):
        assert target_node_id != cur_node_id, "check point 1."
        max_child_id = max(list(np.where(Adj[target_node_id, :] == 1)[0]))
        return cur_node_id >= max_child_id

    def release(self, cur_id, Adj):
        """
                        node0
                        /       \
                    /         |
                    node1        |
                    |  \       |
                    .   \      |
                    .    \     |
                    node90  |    |
                    / \    |   /
                    /   \   |  /
                /     node100
                |        |
        self.start_ids:
            _____________________________________________________
            |___|___| 0 |___|____| 1 |____|____|____|____|____|90 |
        self.end_ids:
            _____________________________________________________
            |___|___|100|___|____|100|____|____|____|____|____|100|

        """
        assert len(self.colors) == len(set(self.colors)), "error in color blocked function."

        if cur_id not in self.end_ids and len(self.colors) > 0:
            return

        released_ids = []
        matched_ids = list(np.where(np.array(self.end_ids) == cur_id)[0])
        for idx in matched_ids:
            parent_id = self.start_ids[idx]
            if self.IsDone(Adj, parent_id, cur_id):
                released_ids.append(idx)

        if len(released_ids) > 0 or len(ColorGraph.GetParentsIds(Adj, cur_id)) == 0:
            # update
            updated_colors = []
            updated_start_ids = []
            updated_end_ids = []
            for i in range(len(self.colors)):
                if i not in released_ids:
                    updated_colors.append(self.colors[i])
                    updated_start_ids.append(self.start_ids[i])
                    updated_end_ids.append(self.end_ids[i])

            self.colors = updated_colors
            self.start_ids = updated_start_ids
            self.end_ids = updated_end_ids


class ColorGraph(object):
    def __init__(self, node_num, color_num, Adj, mode="strict") -> None:
        self._node_num = node_num
        self._color_in_nodes = node_num * [-1]  # color index for each node;
        self._color_num = color_num
        self._color_schemes = []  # total color schemes: [_color_in_nodes0[:],_color_in_nodes1[:] .....]
        self._color_match_num = 0  # len(_color_schemes)
        self._Adj = Adj
        self._blocked_params = BlcokedParams()
        self._mode = mode

    @classmethod
    def GetChildrenIds(cls, Adj, cur_id):
        all_connected_node = list(np.where(Adj[cur_id, :] == 1)[0])
        children = []
        for idx in all_connected_node:
            if idx > cur_id:
                children.append(idx)
        return children

    @classmethod
    def GetParentsIds(cls, Adj, cur_id):
        all_connected_node = list(np.where(Adj[cur_id, :] == 1)[0])
        parents = []
        for idx in all_connected_node:
            if idx < cur_id:
                parents.append(idx)
        return parents

    def to_png(self, graph=None, Adj=None):
        colors_idx = self._color_schemes[0]
        from graphviz import Digraph
        dot = Digraph(comment='The color graph')
        colors = ["palegreen", "red",      "blue",          "violet",     "orange",
                  "wheat4",   "lightpink", "yellow",        "aquamarine", "brown",
                  "teal",     "olive",    "palevioletred4", "indigo",     "fuchsia",
                  "gray"]

        def add_node(dot, layer_id, label, color_idx):
            # dot.node(name=layer_id, label= label, shape="box", style="filled, rounded", color=colors[color_idx])
            dot.node(name=layer_id, label=label, shape="record", style="filled, rounded", fillcolor=colors[color_idx])

        if graph:
            node_num = len(graph.nodes)

            for i in range(node_num):
                label = "{layer_" + str(i) + " | " + \
                    str(graph.nodes[i].type) + " | " + str(graph.nodes[i].outputs[0].shape) + " }"
                # label = f"layer_{str(i)} \n {str(graph.nodes[i].type)} \n {str(graph.nodes[i].outputs[0].shape)}"
                add_node(dot, str(i), label, colors_idx[i])

            for i in range(node_num):
                for j in range(node_num):
                    for child in graph.nodes[i].children:
                        if child == graph.nodes[j]:
                            dot.edge(str(i), str(j))
        elif len(Adj):
            node_num = len(Adj)
            for i in range(node_num):
                label = "{layer_" + str(i) + " | " + "Attrs:" + " }"
                add_node(dot, str(i), label, colors_idx[i])

            for i in range(node_num):
                children_ids = ColorGraph.GetChildrenIds(Adj, i)
                for c_id in children_ids:
                    dot.edge(str(i), str(c_id))
        else:
            assert False, "color graph to_png fail."
        # dot.edges(['12', '13', '23'])
        # dot.edge('2', '3', 'test')

        # print(dot.source)
        dot.view()
        # dot.render('./test-table.gv', view=True)

    def IsNodeBlocked(self, cur_node_idx):
        """
        current node with self._color_in_nodes[cur_node_idx] color is blocked ?
        """

        # step 1:
        cur_color = self._color_in_nodes[cur_node_idx]
        if cur_color in self._blocked_params.colors:
            return True

        # is blocked node ?
        '''
              node_i
             /     \
            /       \
        node_(i+1) node_(i+100)
            |
            |
        node_(i+2)
        '''
        children_ids = ColorGraph.GetChildrenIds(self._Adj, cur_node_idx)
        max_node_id = max(children_ids) if children_ids else cur_node_idx

        # step 2: clean blocked colors when node comsumed.
        #         current color is availabe(after IsSamecolor() function.).
        self._blocked_params.release(cur_node_idx, self._Adj)

        # step 3: record
        if len(children_ids) > 1 and max_node_id - cur_node_idx > 1:
            self._blocked_params.record(cur_color, cur_node_idx, max_node_id)

        return False

    def IsSameColor(self, layer_i):
        neighbours_id = list(np.where(self._Adj[layer_i, :] == 1)[0])
        for n_id in neighbours_id:
            if self._color_in_nodes[layer_i] == self._color_in_nodes[n_id]:
                return True
        return False

    def IsOk(self, node_i):
        # just need 1 color scheme.
        one_candidate = (self._color_match_num == 0)
        if not one_candidate:
            return False

        # is same color with neighbours ?
        if self.IsSameColor(node_i):
            return False

        # is node blocked ?
        if self.IsNodeBlocked(node_i):
            return False

        return True

    def BackTrace(self, i):
        if i >= self._node_num:
            self._color_match_num += 1
            # print("color_num: %d, color scheme matched: %d"%(self._color_num, self._color_match_num))
            self._color_schemes.append(copy.deepcopy(self._color_in_nodes))
            return

        else:
            for j in range(self._color_num):

                self._color_in_nodes[i] = j

                if self.IsOk(i):
                    self.BackTrace(i + 1)

                self._color_in_nodes[i] = -1  # NULL color reset.

##################################tilling base functions############################################


class TillingBase(object):
    # def __init__(self, name):
    #     self._name = name
    ########### mem analysis(DAG Color algorithm)################
    def DAG_color_algorithm(graph):
        """mColoring optimization problem.
        input:
            graph    : origin graph;
            sram_size: sram size restriction to get mini ddr footprint.

        output:
            min color_nums;
            max featuremap size.
        """
        # make Un-Directed Graph Adj matrix.
        node_num = len(graph.nodes)
        Adj = np.zeros((node_num, node_num), dtype=np.int8)
        for i in range(node_num):
            for j in range(node_num):
                for child in graph.nodes[i].children:
                    if child == graph.nodes[j]:
                        Adj[i][j] = 1
                for parent in graph.nodes[i].parents:
                    if parent == graph.nodes[j]:
                        Adj[i][j] = 1
        # if os.path.exists("./adj.npy"):
        MAX_COLOR_NUM = 100
        DFS = ColorGraph(node_num, MAX_COLOR_NUM, Adj)
        DFS.BackTrace(0)
        if (DFS._color_match_num > 0):
            need_color_num = len(set(DFS._color_schemes[0]))
            OPT_INFO(f"need {need_color_num} colors to allocate mem.")
            DFS.to_png(graph=graph)
        else:
            assert False, "can not allocate colors."

        # OPT_INFO("DAG color algorithm done.")

        block_size = shape_infer.sram_restriction / need_color_num

        featuremap_size = []
        constant_size = []
        for n in graph.nodes:
            Fsize, Csize = shape_infer[n.type]().footprint(n)  # unit: KB == 1024B
            featuremap_size.append(Fsize)
            constant_size.append(Csize)

        import math
        max_size = max(featuremap_size)
        OPT_INFO(f"max featuremap size is: {max_size} KB")

        split_len = max_size // block_size
        split_hw = [math.ceil(split_len / 2), math.floor(split_len / 2)]
        OPT_INFO(f"need to split_HW: {split_hw}")

        return split_hw

    def DAG_algorithm_unittest():
        Adj = np.load("./adj.npy")
        DFS = ColorGraph(135, 30, Adj)
        DFS.BackTrace(0)
        if (DFS._color_match_num > 0):
            need_color_num = len(set(DFS._color_schemes[0]))
            print("need %d colors to allocate mem." % need_color_num)
            DFS.to_png(DFS._color_schemes[0], Adj=Adj)
        else:
            print("can not allocate colors.")
        print("1")

    @classmethod
    def _deal_split_params(cls, graph, split_params):
        """
        split_params:
            int: sram size.
            list[int, int]: blocks_num.

        return:
            torch.tensor([int, int]): [split_h, split_w]
        """
        if isinstance(split_params, int):
            shape_infer.split_strategy = "SRAM"
            shape_infer.sram_restriction = split_params
            shape_infer.split_params = super().DAG_color_algorithm(graph)
            return shape_infer.split_params
        elif isinstance(split_params, list) and len(split_params) == 2:
            shape_infer.split_strategy = "TILL"
            shape_infer.split_params = split_params
            return split_params
        else:
            OPT_ERROR(
                "split params type not supported. should: int or list[int, int].")

    ########### get splited shape ################################
    # shape backward infer
    def calc_work_item_size_by_output(out_size, k_size, stride, padding):
        return ((out_size - 1) * stride + k_size - padding[..., 0:2] - padding[..., 2:4])
    # shape forward infer

    def calc_work_item_size_output(in_size, k_size, stride, padding):
        return ((in_size - k_size + padding[..., 0:2] + padding[..., 2:4]) // stride + 1)
    # overlap

    def calc_work_item_overlap(w_len, next_layer_w_in, stride, padding):
        # overlap = w_len - (next_layer_w_in * stride - padding)
        return (w_len - (next_layer_w_in * stride - padding[..., 0:2] - padding[..., 2:4]))

    ########################### utils ############################
    # for slide window op: conv/deconv
    def copy_quant_paramters(new_outp_tensor, replaced_node):
        new_outp_tensor.qmin = replaced_node.outputs[0].qmin
        new_outp_tensor.qmax = replaced_node.outputs[0].qmax
        new_outp_tensor.similarity = replaced_node.outputs[0].similarity
        new_outp_tensor.scale = replaced_node.outputs[0].scale
        new_outp_tensor.zerop = replaced_node.outputs[0].zerop
        new_outp_tensor.qbits = replaced_node.outputs[0].qbits
        new_outp_tensor.ir_dtype = replaced_node.outputs[0].ir_dtype
        new_outp_tensor.dtype = replaced_node.outputs[0].dtype
        new_outp_tensor.qinvariant = replaced_node.outputs[0].qinvariant
        new_outp_tensor.key_axis = replaced_node.outputs[0].key_axis
    # get eltwise input

    def get_eltwise_input0(cur_node_idx, num_x, num_y, node_num, eltwise_inputs):
        eltwise_inputs.clear()
        tot_slice = num_x*num_y
        for i in range(tot_slice):
            eltwise_inputs.append(cur_node_idx+tot_slice+(node_num-1)+i*node_num)
    # deconv padding

    def fill_deconv_padding_param_for_all_items(node, layerid, all_work_items_len, each_node_work_items_out):
        item_num = shape_infer.split_params
        device_type = shape_infer.device

        padding_left = node.params['pad_left']
        padding_right = node.params['pad_right']
        padding_top = node.params['pad_top']
        padding_bottom = node.params['pad_bottom']
        stride_x = node.params['stride_x']
        kernel_x = node.params['kernel_x']
        stride_y = node.params['stride_y']
        kernel_y = node.params['kernel_y']

        padding = torch.zeros((item_num[0], item_num[1], 4), dtype=torch.int16, device=device_type)
        for item_y in range(item_num[0]):
            h = each_node_work_items_out[layerid, item_y, 0, 0].int().item()
            h_in = all_work_items_len[layerid, item_y, 0, 0].int().item()
            padding_bottom = kernel_y-(h+padding_top-(h_in-1)*stride_y)
            padding_left = node.params['pad_left']
            for item_x in range(item_num[1]):

                w = each_node_work_items_out[layerid, item_y, item_x, 1].int().item()
                w_in = all_work_items_len[layerid, item_y, item_x, 1].int().item()
                # modify center's part's padding
                padding_right = kernel_x-(w+padding_left-(w_in-1)*stride_x)
                # [top,left,bottom,right]
                padding[item_y, item_x, :] = torch.tensor(
                    [padding_top, padding_left, padding_bottom, padding_right], device=device_type)
                padding_left = kernel_x-((stride_x-(kernel_x-padding_right)-1)+stride_x *
                                         ((stride_x-(kernel_x-padding_right)-1) < 0))-1

            padding_top = kernel_y-((stride_y-(kernel_y-padding_bottom)-1)+stride_y *
                                    ((stride_y-(kernel_y-padding_bottom)-1) < 0))-1
        return padding
    # normal op padding.

    def fill_padding_param_for_all_items(node):
        # padding layout format[top,left,bottom,right]
        device_type = shape_infer.device
        padding_left = node.params['pad_left']
        padding_right = node.params['pad_right']
        padding_top = node.params['pad_top']
        padding_bottom = node.params['pad_bottom']
        stride_x = node.params['stride_x']
        kernel_x = node.params['kernel_x']
        stride_y = node.params['stride_y']
        kernel_y = node.params['kernel_y']

        if node.type == OpType.ConvTranspose:
            padding_left = kernel_x - padding_left - 1
            padding_right = kernel_x - padding_right - 1
            padding_top = kernel_y - padding_top - 1
            padding_bottom = kernel_y - padding_bottom - 1
            stride_x = 1  # ?
            stride_y = 1  # ?

        """
        (split_h, split_w, 4):
            4: (padding_top, padding_left, padding_bottom, padding_right)
        """
        padding = torch.zeros((shape_infer.split_h, shape_infer.split_w, 4), dtype=torch.int16, device=device_type)
        w = node.inputs[0].ir_shape[2]  # input_w

        # update padding_bottom, padding_right:
        #   some conv padding is useless, such as input w=10, k=3,s=2,padding left=1, padding right=1
        #   o = (10-3+2)/2 + 1 = 5, to get this 5 point, k window slide to indice 8,9, right padding is useless
        padding_right = padding_right - (w - kernel_x + padding_left + padding_right) % stride_x
        padding_right = padding_right if padding_right > 0 else 0

        h = node.inputs[0].ir_shape[1]
        padding_bottom = padding_bottom - (h - kernel_y + padding_top + padding_bottom) % stride_y
        padding_bottom = padding_bottom if padding_bottom > 0 else 0

        # left top corner
        padding[0, 0, 0:2] = torch.tensor([padding_top, padding_left])
        # right bottom corner
        padding[shape_infer.split_h - 1, shape_infer.split_w - 1, 2:4] = torch.tensor([padding_bottom, padding_right])
        # right top corner
        padding[0, shape_infer.split_w - 1, 3] = torch.tensor([padding_right])
        padding[0, shape_infer.split_w - 1, 0] = torch.tensor([padding_top])
        # left bottom corner
        padding[shape_infer.split_h - 1, 0, 2] = torch.tensor([padding_bottom])
        padding[shape_infer.split_h - 1, 0, 1] = torch.tensor([padding_left])
        # top row
        padding[0, 1:shape_infer.split_w - 1, 0] = torch.tensor([padding_top])
        # bottom row
        padding[shape_infer.split_h - 1, 1:shape_infer.split_w - 1, 2] = torch.tensor([padding_bottom])

        # left col
        padding[1:shape_infer.split_h - 1, 0, 1] = torch.tensor([padding_left])
        # right col
        padding[1:shape_infer.split_h - 1, shape_infer.split_h - 1, 3] = torch.tensor([padding_right])
        stride = torch.tensor([stride_y, stride_x], device=device_type)

        return padding, stride
    # pooling ceil.

    def pooling_ceil_mode_padding(node):
        device_type = shape_infer.device

        left = node.params['pad_left']
        right = node.params['pad_right']
        top = node.params['pad_top']
        bottom = node.params['pad_bottom']
        stride_x = node.params['stride_x']
        kernel_x = node.params['kernel_x']
        stride_y = node.params['stride_y']
        kernel_y = node.params['kernel_y']

        # [top,left,bottom,right]
        padding = torch.tensor([top, left, bottom, right], device=device_type)
        kernel_size = torch.tensor([kernel_y, kernel_x],       device=device_type)
        stride = torch.tensor([stride_y, stride_x],       device=device_type)
        hw_in = torch.tensor([node.inputs[0].ir_shape[1],  node.inputs[0].ir_shape[2]],  device=device_type)
        hw_out = torch.tensor([node.outputs[0].ir_shape[1], node.outputs[0].ir_shape[2]], device=device_type)

        expect_out = calc_work_item_size_output(hw_in, kernel_size, stride, padding)

        return hw_out - expect_out
    # show partition detail.

    def disp_be_able_partion_node(graph, end_split_idx_list, start_split_idx_list):
        for i, idx in enumerate(start_split_idx_list):
            OPT_INFO('layer from ' + str(idx) + ' to ' + str(end_split_idx_list[i]) + ' are spliting')
            for m in range(end_split_idx_list[i]-idx+1):
                node = graph.nodes[idx + m]
                OPT_INFO('layer id ' + str(idx + m) + ' ' + node.name + ' can be split')

    ########### insert partial ops #################################
    # insert slice op, fc case,x direction shape is at constants(weights,so slice item_x is 1)
    def insert_crop_op(graph, all_work_items_len, all_work_items_in, insert_node_pos, inserted_op_list, parents_id=0):
        parts_num_y = shape_infer.split_h
        parts_num_x = shape_infer.split_w
        if graph.nodes[insert_node_pos].type == OpType.FullyConnected:
            parts_num_x = 1

        # inserted_op_list = []
        crop_begin_pos = [0, 0]
        crop_end_pos = [0, 0]
        for item_y in range(parts_num_y):
            crop_begin_pos[1] = 0
            for item_x in range(parts_num_x):
                current_node = graph.nodes[insert_node_pos]
                parent = current_node.parents[parents_id]  # self.nodes[insert_node_pos-1]

                crop_name = current_node.name + '_crop_item_y'+str(item_y)+'_item_x'+str(item_x)
                crop = PyNode(graph.get_valid_node_name(crop_name), OpType.Crop)

                shape_dim = len(current_node.inputs[0].shape)
                # fill input and outputs tensor
                # add inputs from curren inserted node
                # slice.add_input(current_node.inputs[0])
                crop.add_input(parent.outputs[0])
                # if len(slice.parents) == 0:
                #     slice.add_parent(parent)
                # new slice output
                crop_outp_tensor_name = graph.get_valid_tensor_name(
                    current_node.inputs[0].name+'_crop'+'_item_y'+str(item_y)+'_item_x'+str(item_x))
                # h,w size
                h = all_work_items_len[0, item_y, item_x, 0].int().item()
                w = all_work_items_len[0, item_y, item_x, 1].int().item()
                if shape_dim == 4:
                    outshape = TensorShape([current_node.inputs[0].shape[0], h, w, current_node.inputs[0].shape[3]])
                else:
                    outshape = TensorShape([h, w])
                out_tensor = PyTensor(crop_outp_tensor_name, outshape, current_node.inputs[0].dtype)
                out_tensor.ir_shape = outshape
                out_tensor.qmin = current_node.inputs[0].qmin
                out_tensor.qmax = current_node.inputs[0].qmax
                crop.add_output(out_tensor)
                # begin=[0,0,0,0]
                # end=[-1,165,272,64]
                crops = [[0, 0], [0, 0], [0, 0], [0, 0]]

                # need setting crops paramsters:[('begin'),('end')]
                if shape_dim == 4:
                    crop_end_pos[0] = crop_begin_pos[0]+all_work_items_len[0, item_y, item_x, 0].int().item()
                    crop_end_pos[1] = crop_begin_pos[1]+all_work_items_len[0, item_y, item_x, 1].int().item()

                    crops[0][1] = out_tensor.ir_shape[0]

                    crops[1][0] = crop_begin_pos[0]
                    crops[1][1] = crop_end_pos[0]
                    crops[2][0] = crop_begin_pos[1]
                    crops[2][1] = crop_end_pos[1]
                    crops[3][1] = out_tensor.ir_shape[3]
                    crop.params['crops'] = crops
                else:
                    crops = [[0, 0], [0, 0]]
                    crop_end_pos[0] = crop_begin_pos[0]+all_work_items_len[0, item_y, item_x, 0].int().item()
                    crop_end_pos[1] = crop_begin_pos[1]+all_work_items_len[0, item_y, item_x, 1].int().item()
                    crops[0][0] = crop_begin_pos[0]
                    crops[0][1] = crop_end_pos[0]
                    crops[1][0] = crop_begin_pos[1]
                    crops[1][1] = crop_end_pos[1]
                    crop.params['crops'] = crops

                crop.attrs.update(parent.attrs.clone())
                # x direction offset update
                crop_begin_pos[1] = crop_begin_pos[1] + all_work_items_in[item_y, item_x, 1].int().item()
                inserted_op_list.append(crop)
            # y direction offset update
            crop_begin_pos[0] = crop_begin_pos[0]+all_work_items_in[item_y, 0, 0].int().item()
    # insert matmul two inputs slice, matmul partition number = item_num[0]+item_num[1]

    def insert_matmul_slice_op(graph, all_work_items_len, all_work_items_in, insert_node_pos, inserted_op_list):
        # inserted_op_list = []
        crop_begin_pos = [0, 0]
        crop_end_pos = [0, 0]
        crop_begin_pos[1] = 0
        current_node = graph.nodes[insert_node_pos]
        item_num = shape_infer.split_params

        for i in range(1, -1, -1):
            parent = current_node.parents[i]
            if i == 0:  # h dir
                parts_num_y = item_num[i]
                parts_num_x = 1
                crop_end_pos[1] = current_node.inputs[0].shape[-1]
                crop_begin_pos[0] = 0
            else:  # w dir
                parts_num_x = item_num[i]
                parts_num_y = 1
                crop_end_pos[0] = current_node.inputs[1].shape[-2]
                crop_begin_pos[1] = 0
            for item_y in range(parts_num_y):
                for item_x in range(parts_num_x):
                    slice_name = current_node.name + '_slice_item_y'+str(item_y+parts_num_x)+'_item_x'+str(item_x)
                    slice = PyNode(graph.get_valid_node_name(slice_name), OpType.Slice)
                    shape_dim = len(current_node.inputs[i].shape)
                    # fill input and outputs tensor
                    # new slice output
                    slice_outp_tensor_name = graph.get_valid_tensor_name(
                        current_node.inputs[i].name+'_slice'+'_item_y'+str(item_y)+'_item_x'+str(item_x))
                    # h,w size
                    h_w = all_work_items_len[0, item_y, item_x, i].int().item()
                    outshape = TensorShape(current_node.inputs[i].shape)
                    outshape[-2+i] = h_w
                    out_tensor = PyTensor(slice_outp_tensor_name, outshape, current_node.inputs[i].dtype)
                    out_tensor.ir_shape = outshape
                    out_tensor.qmin = current_node.inputs[i].qmin
                    out_tensor.qmax = current_node.inputs[i].qmax
                    slice.add_output(out_tensor)

                    # add inputs from curren inserted node
                    slice.add_input(parent.outputs[0])
                    # if len(slice.parents) == 0:
                    #     slice.add_parent(parent)
                    # slice.add_input(current_node.inputs[i])
                    # need setting slice paramsters:('strides'),('begin'),('end')
                    slice.params['strides'] = [1]*shape_dim
                    begin = [0, 0, 0, 0]
                    begin[-2+i] = crop_begin_pos[i]
                    slice.params['begin'] = begin

                    crop_end_pos[i] = crop_begin_pos[i]+all_work_items_len[0, item_y, item_x, i].int().item()
                    slice.params['end'] = [-1, -1, crop_end_pos[0], crop_end_pos[1]]

                    slice.params['upper_bound'] = True
                    slice.attrs.update(parent.attrs.clone())
                    # x direction offset update
                    crop_begin_pos[i] = crop_begin_pos[i]+all_work_items_in[item_y, item_x, i].int().item()
                    inserted_op_list.append(slice)
            # parent.remove_child(parent.children[0])
    # insert slide window op : conv, pooling...

    @classmethod
    def insert_slide_window_op(cls, graph, inputs, each_node_work_items_out, all_work_items_len, first_node_idx, node, inserted_op_list, item_x, item_y):
        optype = graph.nodes[node].type
        replaced_node = graph.nodes[node]
        new_layer_name = replaced_node.name
        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y' +
                                                str(item_y)+'_item_x'+str(item_x)+'_node'+str(node)), optype)
        # add input tensor
        part.add_input(inputs)
        # add output tensor
        h = each_node_work_items_out[node-first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[node-first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y'+str(item_y)+'_item_x'+str(item_x)+'_node'+str(node))
        newshape = TensorShape([replaced_node.outputs[0].shape[0], h, w, replaced_node.outputs[0].shape[3]])
        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape

        cls.copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        inputs = part.outputs[0]

        # add conv paramers
        for k, v in replaced_node.params.items():
            part.params[k] = copy.deepcopy(v)
        part.attrs.update(replaced_node.attrs.clone())
        for k, v in replaced_node.constants.items():
            part.constants[k] = replaced_node.constants[k]
        # padding need update
        if replaced_node.type == OpType.ConvTranspose:
            padding = cls.fill_deconv_padding_param_for_all_items(
                replaced_node, node-first_node_idx, all_work_items_len, each_node_work_items_out)
        else:
            padding, _ = cls.fill_padding_param_for_all_items(replaced_node)
        part.params['pad_top'] = padding[item_y, item_x, 0].item()
        part.params['pad_left'] = padding[item_y, item_x, 1].item()
        part.params['pad_bottom'] = padding[item_y, item_x, 2].item()
        part.params['pad_right'] = padding[item_y, item_x, 3].item()

        inserted_op_list.append(part)
        return inputs
    # insert non-slide window op :

    @classmethod
    def insert_non_slide_window_op(cls, graph, inputs, each_node_work_items_out, first_node_idx, node, inserted_op_list, item_x, item_y):

        optype = graph.nodes[node].type
        replaced_node = graph.nodes[node]
        new_layer_name = replaced_node.name
        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y' +
                                                str(item_y)+'_item_x'+str(item_x)+'_node'+str(node)), optype)
        # add input tensor
        part.add_input(inputs)
        # add output tensor
        shape_dim = len(replaced_node.outputs[0].shape)
        h = each_node_work_items_out[node-first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[node-first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y'+str(item_y)+'_item_x'+str(item_x)+'_node'+str(node))
        if replaced_node.type == OpType.Transpose:
            newshape = TensorShape([replaced_node.outputs[0].shape[0], replaced_node.outputs[0].shape[1], h, w])
        elif shape_dim == 4:
            newshape = TensorShape([replaced_node.outputs[0].shape[0], h, w, replaced_node.outputs[0].shape[3]])
        else:
            newshape = TensorShape([h, w])
        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape
        cls.copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        inputs = part.outputs[0]

        # add not conv type paramers
        if replaced_node.type == OpType.Reshape:
            part.params['shape'] = newshape.clone()
        else:
            for k, v in replaced_node.params.items():
                if type(v) != TensorShape:
                    part.params[k] = copy.deepcopy(v)
                else:
                    part.params[k] = v.clone()
        part.attrs.update(replaced_node.attrs.clone())
        for k, v in replaced_node.constants.items():
            part.constants[k] = replaced_node.constants[k]

        inserted_op_list.append(part)
        return inputs
    # insert fc op split_x = 1

    @classmethod
    def insert_fc_op(cls, graph, inputs, each_node_work_items_out, first_node_idx, cur_node_idx, inserted_op_list, item_y, parts_num_x):

        optype = graph.nodes[cur_node_idx].type
        replaced_node = graph.nodes[cur_node_idx]
        new_layer_name = replaced_node.name
        last_w = 0
        for item_x in range(parts_num_x):
            part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y'+str(item_y) +
                                                    '_item_x'+str(item_x)+'_node'+str(cur_node_idx)), optype)
            # add input tensor
            part.add_input(inputs)
            # add output tensor
            h = each_node_work_items_out[cur_node_idx-first_node_idx, item_y, item_x, 0].int().item()
            w = each_node_work_items_out[cur_node_idx-first_node_idx, item_y, item_x, 1].int().item()
            new_outp_tensor_name = graph.get_valid_tensor_name(
                replaced_node.outputs[0].name + '_item_y'+str(item_y)+'_item_x'+str(item_x)+'_node'+str(cur_node_idx))

            newshape = TensorShape([h, w])
            new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
            new_outp_tensor.ir_shape = newshape
            cls.copy_quant_paramters(new_outp_tensor, replaced_node)
            part.add_output(new_outp_tensor)
            # inputs = part.outputs[0]
            # add conv paramers
            for k, v in replaced_node.params.items():
                part.params[k] = copy.deepcopy(v)
            part.attrs.update(replaced_node.attrs.clone())
            # partition constants weight
            for k, v in replaced_node.constants.items():
                newshape = (replaced_node.constants[k].ir_shape)
                newshape[0] = w
                tmp = PyTensor(replaced_node.constants[k].name, newshape, replaced_node.constants[k].dtype)
                tmp = replaced_node.constants[k]  # tmp copy from replaced_node.constants[k],return old tmp
                part.constants[k] = tmp
                part.constants[k].betensor = copy.deepcopy(
                    replaced_node.constants[k].betensor[last_w:last_w+w, ...])
                part.constants[k].ir_shape[0] = w
            last_w = last_w + w
            inserted_op_list.append(part)
        return inputs
    # inser matmul op, split_x = 1

    @classmethod
    def insert_matmul_op(cls, graph, inputs, each_node_work_items_out, first_node_idx, cur_node_idx, inserted_op_list, item_y, item_x):

        optype = graph.nodes[cur_node_idx].type
        replaced_node = graph.nodes[cur_node_idx]
        new_layer_name = replaced_node.name

        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y'+str(item_y) +
                                                '_item_x'+str(item_x)+'_node'+str(cur_node_idx)), optype)
        # add input tensor
        part.add_input(inputs[0])
        part.add_input(inputs[1])
        # add output tensor

        h = each_node_work_items_out[cur_node_idx-first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[cur_node_idx-first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y'+str(item_y)+'_item_x'+str(item_x)+'_node'+str(cur_node_idx))

        newshape = TensorShape(replaced_node.outputs[0].shape)
        newshape[-2] = h
        newshape[-1] = w
        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape
        cls.copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        # add  paramers
        for k, v in replaced_node.params.items():
            part.params[k] = copy.deepcopy(v)
        part.attrs.update(replaced_node.attrs.clone())
        for k, v in replaced_node.constants.items():
            part.constants[k] = replaced_node.constants[k]
        inserted_op_list.append(part)
        return inputs
    # for multi-input and input size different.

    @classmethod
    def insert_eltwise_op(cls, graph, inputs, each_node_work_items_out, first_node_idx, cur_node_idx, inserted_op_list, item_y, item_x):

        optype = graph.nodes[cur_node_idx].type
        replaced_node = graph.nodes[cur_node_idx]
        new_layer_name = replaced_node.name
        shape_dim = len(replaced_node.outputs[0].shape)
        part = PyNode(graph.get_valid_node_name(new_layer_name + '_item_y'+str(item_y) +
                                                '_item_x'+str(item_x)+'_node'+str(cur_node_idx)), optype)
        # add input tensor
        part.add_input(inputs[0])
        part.add_input(inputs[1])
        # add output tensor

        h = each_node_work_items_out[cur_node_idx-first_node_idx, item_y, item_x, 0].int().item()
        w = each_node_work_items_out[cur_node_idx-first_node_idx, item_y, item_x, 1].int().item()
        new_outp_tensor_name = graph.get_valid_tensor_name(
            replaced_node.outputs[0].name + '_item_y'+str(item_y)+'_item_x'+str(item_x)+'_node'+str(cur_node_idx))

        if shape_dim == 4:
            newshape = TensorShape([replaced_node.outputs[0].shape[0], h, w, replaced_node.outputs[0].shape[3]])
        else:
            newshape = TensorShape([h, w])

        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, replaced_node.outputs[0].dtype)
        new_outp_tensor.ir_shape = newshape
        cls.copy_quant_paramters(new_outp_tensor, replaced_node)
        part.add_output(new_outp_tensor)
        # add not conv type paramers

        for k, v in replaced_node.params.items():
            if type(v) != TensorShape:
                part.params[k] = copy.deepcopy(v)
            else:
                part.params[k] = v.clone()
        part.attrs.update(replaced_node.attrs.clone())

        inserted_op_list.append(part)
        return inputs


class TillingBackward(TillingBase):
    # high level logic code.
    @classmethod
    def _get_partition_params_by_back_propagate(cls, graph, end_split_node_idx, start_split_node_idx):
        # each_node_work_items_len, all_work_items_in, each_node_work_items_out
        # calculate conv partition params by back propagate
        device_type = shape_infer.device
        v_item_num = shape_infer.split_params
        nodes_num = end_split_node_idx-start_split_node_idx+1

        last_node = graph.nodes[end_split_node_idx]

        # get fused out shape.
        if 'kernel_x' in last_node.params:
            fused_out = torch.tensor([last_node.outputs[0].shape[1],
                                      last_node.outputs[0].shape[2]], device=device_type)
        elif last_node.type == OpType.DepthToSpace or last_node.type in shape_infer.tranparent_op or last_node.type == OpType.Eltwise:
            fused_out = torch.tensor(
                [last_node.inputs[0].shape[1], last_node.inputs[0].shape[2]], device=device_type)
        # fc has one input, A[H,W]*WEIGHT[W,I]=C[H,I],W can't be splited
        elif last_node.type == OpType.FullyConnected:
            fused_out = torch.tensor([last_node.inputs[0].shape[-2],
                                      last_node.outputs[0].shape[-1]], device=device_type)
        # matmul has 2 inputs matrix A[...,H,W]*B[...,W,I]=C[...,H,I]
        elif last_node.type == OpType.MatMul:
            fused_out = torch.tensor([last_node.inputs[0].shape[-2],
                                      last_node.inputs[1].shape[-1]], device=device_type)

        work_items_len = torch.zeros((shape_infer.split_h, shape_infer.split_w, 2), device=device_type)
        work_items_out = torch.ones((shape_infer.split_h, shape_infer.split_w, 2), device=device_type)
        each_node_work_items_out = torch.ones(
            (nodes_num, shape_infer.split_h, shape_infer.split_w, 2), device=device_type)
        each_node_work_items_len = torch.ones(
            (nodes_num, shape_infer.split_h, shape_infer.split_w, 2), device=device_type)

        # last node output.no overlap
        each_item_out_size = torch.div(fused_out, v_item_num, rounding_mode='floor')
        remain = fused_out - each_item_out_size * v_item_num
        last_item_out_len = each_item_out_size + remain
        work_items_out = work_items_out*each_item_out_size

        work_items_out[:, shape_infer.split_h - 1, 1] = last_item_out_len[1]
        work_items_out[shape_infer.split_w - 1, :, 0] = last_item_out_len[0]
        all_work_items_in = work_items_out[0, 0, :]
        overlap = torch.zeros((shape_infer.split_h, shape_infer.split_w, 2), dtype=torch.int16, device=device_type)
        for idx in range(end_split_node_idx, start_split_node_idx-1, -1):
            node = graph.nodes[idx]
            if 'kernel_x' in node.params:
                dl_x = node.get_param('dilation_x', optional=True, default_value=1)
                dl_y = node.get_param('dilation_y', optional=True, default_value=1)
                kernel_size = torch.tensor([(node.params['kernel_y'] - 1) * dl_y + 1,
                                            (node.params['kernel_x'] - 1) * dl_x + 1], device=device_type)
                padding, stride = super().fill_padding_param_for_all_items(node)
                each_node_work_items_out[idx-start_split_node_idx] = work_items_out

                work_items_len = super().calc_work_item_size_by_output(work_items_out, kernel_size, stride, padding)
                # pooling op:
                #       out = (hw - k + padding) / stride + 1
                #
                # if (hw - k + padding) % stride !=0 and ceil_mode=true --> need adjust
                if 'ceil_mode' in node.params and node.params['ceil_mode']:
                    error = super().pooling_ceil_mode_padding(node)
                    # right coloum.
                    work_items_len[:, shape_infer.split_w - 1,
                                   1] = work_items_len[:, shape_infer.split_w - 1, 1] - error[1]
                    # bottom row.
                    work_items_len[shape_infer.split_h - 1, :,
                                   0] = work_items_len[shape_infer.split_h - 1, :, 0] - error[0]

                overlap = super().calc_work_item_overlap(work_items_len, all_work_items_in, stride, padding)

                ######deconv case####################################
                # 1. transfer deconv to conv:
                #       w      = (w - 1) * stride + 1
                #       newpad = k-padding-1
                #       stride = 1
                # 2. calc work_items_len and overlap with conv layer's method
                # 3. update conv paramter to deconv param
                if node.type == OpType.ConvTranspose:
                    stride = torch.tensor([node.params['stride_y'], node.params['stride_x']], device=device_type)
                    temp_len = torch.zeros_like(work_items_len)
                    left_top = torch.tensor([0, 0], device=device_type)
                    for item_y in range(shape_infer.split_h):
                        left_top[1] = 0
                        for item_x in range(shape_infer.split_w):
                            hw = (work_items_len[item_y, item_x, :] - left_top) // stride
                            compensate_point = (work_items_len[item_y, item_x, :]-left_top) % stride != 0
                            hw = hw + compensate_point
                            remain[:] = (kernel_size[:]-(work_items_len[item_y, item_x, :]-left_top[:]) %
                                         stride[:]) % stride[:]
                            left_top[1] = remain[1]-1 + stride[1]*(remain[1] == 0)
                            overlap[item_y, item_x, 1] = overlap[item_y, item_x, 1]//stride[1] + \
                                ((overlap[item_y, item_x, 1]-left_top[1]) % stride[1] != 0)
                            temp_len[item_y, item_x, :] = hw
                        remain[0] = (kernel_size[0]-(work_items_len[item_y, 0, 0]-left_top[0]) %
                                     stride[0]) % stride[0]
                        left_top[0] = remain[0]-1 + stride[0]*(remain[0] == 0)
                        overlap[item_y, :, 0] = overlap[item_y, :, 0]//stride[0] + \
                            ((overlap[item_y, :, 0]-left_top[0]) % stride[0] != 0)
                    work_items_len[:] = temp_len[:]

            elif node.type == OpType.DepthToSpace:
                work_items_len[:] = work_items_out[:]
                block_size_x = node.params['block_size_x']
                # block_size_y = self.get_param('block_size_y')
                each_node_work_items_out[idx-start_split_node_idx, :] = work_items_len*block_size_x
            elif node.type in shape_infer.ranparent_op or node.type == OpType.Eltwise:
                work_items_len[:] = work_items_out[:]
                each_node_work_items_out[idx-start_split_node_idx, :] = work_items_len
            elif node.type == OpType.FullyConnected:
                work_items_len[:] = work_items_out[:]
                inw = node.inputs[0].shape[-1]
                work_items_len[..., -1] = inw
                each_node_work_items_out[idx-start_split_node_idx, :] = work_items_out
            elif node.type == OpType.MatMul:
                work_items_len[:] = work_items_out[:]
                each_node_work_items_out[idx-start_split_node_idx, :] = work_items_out

            all_work_items_in = (work_items_len - overlap)
            # upodate for next shape backward.
            work_items_out[:] = work_items_len[:]
            # backup calculated params.
            each_node_work_items_len[idx-start_split_node_idx, :] = work_items_len

        return each_node_work_items_len, all_work_items_in, each_node_work_items_out

    @classmethod
    def _get_partition_segments(cls, graph, item_num, ratio):
        def node_child_is_eltwise(n):
            if len(graph.nodes[n].children) > 0 and graph.nodes[n].children[0].type == OpType.Eltwise:
                return int(graph.nodes[n].children[0].attrs['layer_id'])
            else:
                return 0

        def check_valid_split_size(n):
            return ('kernel_x' in n.params and n.inputs[0].ir_shape[1]//item_num[0] > n.params['kernel_y']
                    and (n.outputs[0].ir_shape[1] // item_num[0] > 0)
                    and n.inputs[0].ir_shape[2] // item_num[1] > n.params['kernel_x'])

        def save_split_end_idx(node_idx):
            end_split_idx_list.append(node_idx)
            end_eltwise_idx_list.append(node_child_is_eltwise(node_idx))

        def save_split_start_idx(node_idx):
            start_split_idx_list.append(node_idx)

        node_idx = 0
        end_idx = len(graph.nodes)
        start_split_idx_list = []
        end_split_idx_list = []
        end_eltwise_idx_list = []
        search_start_idx_flag = True

        if shape_infer.split_strategy == "SRAM":
            item_num = [2, 2]

        while node_idx < end_idx:
            n = graph.nodes[node_idx]
            k_in_range_of_feature = False
            not_slidewin_op_split = False

            is_splitable = shape_infer[n.type]().is_splitable(n)

            if n.type in [OpType.DepthToSpace, OpType.FullyConnected, OpType.MatMul]:
                not_slidewin_op_split = is_splitable
            else:
                k_in_range_of_feature = is_splitable

            if k_in_range_of_feature:
                # node_idx = int(n.attrs["layer_id"])
                if search_start_idx_flag:
                    save_split_start_idx(node_idx)
                    search_start_idx_flag = False
                    continue

                # check overlap, if overlap is too lage, stop here.
                w_len, w_in,  _ = cls._get_partition_params_by_back_propagate(graph, node_idx, start_split_idx_list[-1])
                overlap_rate = 1 - w_in[0, 0, :] / w_len[0, 0, 0, :]

                # overlap overflow.
                if overlap_rate[0].item() > ratio or overlap_rate[1].item() > ratio:
                    if (len(end_split_idx_list) == 0 or end_split_idx_list[-1] != node_idx-1) and search_start_idx_flag:
                        node_idx = node_idx - 1

                    search_start_idx_flag = True
                    # save_split_end_idx(node_idx)
                    save_split_end_idx(node_idx)

                    # if overlap is out_of_range and only one op, it indicates this op can't be split
                    if start_split_idx_list[-1] == node_idx:
                        del start_split_idx_list[-1]
                        del end_split_idx_list[-1]

                elif (node_idx + 1) != end_idx:  # not the last one.
                    # check conv's output name, it maybe next several layers' input
                    # if it is several-conv's layer input, seperated operation should stop here
                    if len(n.children) == 0 or (n.type == OpType.Eltwise and check_valid_split_size(n.parents[0])) or \
                            (n.children[0].type not in shape_infer.tranparent_op and 'kernel_x' not in n.children[0].params):
                        search_start_idx_flag = True
                        # end_split_idx_list.append(node_idx)
                        save_split_end_idx(node_idx)
                    elif n.type == OpType.Eltwise and 'kernel_x' in n.parents[1].params:
                        search_start_idx_flag = True
                        # end_split_idx_list.append(node_idx-1)
                        save_split_end_idx(node_idx-1)
                    elif len(n.children) > 1:
                        search_start_idx_flag = True
                        # end_split_idx_list.append(node_idx)
                        save_split_end_idx(node_idx)
                    elif len(n.children[0].inputs) > 1:
                        # check child node if eltwise is or not
                        if n.children[0].parents[1].name != n.name:
                            search_start_idx_flag = True
                            # end_split_idx_list.append(node_idx)
                            save_split_end_idx(node_idx)
            else:
                if (not search_start_idx_flag):
                    search_start_idx_flag = True
                    # end_split_idx_list.append(node_idx-1)
                    save_split_end_idx(node_idx - 1)

            # split op alone
            if not_slidewin_op_split:
                save_split_start_idx(node_idx)
                save_split_end_idx(node_idx)
                search_start_idx_flag = True

            node_idx += 1

        if not search_start_idx_flag:
            save_split_end_idx(node_idx - 1)

        # re-check if eltwise op can be merged
        for i in range(len(end_eltwise_idx_list)):
            if end_eltwise_idx_list[i] > 0:
                idx = np.where(end_eltwise_idx_list[i] == np.array(end_eltwise_idx_list))
                # the other input is not conv, so ignore
                if idx[0].size == 1:
                    if graph.nodes[end_eltwise_idx_list[i]].parents[0].type != OpType.Transpose \
                            and graph.nodes[end_eltwise_idx_list[i]].parents[1].type != OpType.Transpose:
                        end_split_idx_list[idx[0][0]] = end_eltwise_idx_list[i]
                        end_eltwise_idx_list[i] = end_eltwise_idx_list[i] + len(graph.nodes)
                    else:
                        end_eltwise_idx_list[i] = 0
                else:
                    end_split_idx_list[idx[0][1]] = end_eltwise_idx_list[i]

        super().disp_be_able_partion_node(graph, end_split_idx_list, start_split_idx_list)
        return end_split_idx_list, start_split_idx_list, end_eltwise_idx_list

    @classmethod
    def run(cls, base_graph, split_prams, ratio, tilling_list):
        device_type = base_graph.nodes[0].outputs[0].betensor.device
        shape_infer.device = device_type
        tot_nodes_num = len(base_graph.nodes)
        blocks_num = super()._deal_split_params(base_graph, split_prams)

        # get each seperatable segment
        if len(tilling_list) == 0:
            end_split_idx_list, start_split_idx_list, end_eltwise_idx_list = cls._get_partition_segments(
                base_graph, blocks_num, ratio)

        if len(tilling_list) == 0 and len(end_split_idx_list) == 0:
            return

        node_offset = 0
        #  to generate sigle core partition to debug memory footprint
        if len(tilling_list) > 0:
            start_split_idx_list = [int(tilling_list[segment*4])for segment in range(len(tilling_list)//4)]
            end_split_idx_list = [int(tilling_list[segment*4+1])for segment in range(len(tilling_list)//4)]
        segment_items_size = torch.ones((len(end_split_idx_list), 2), device=device_type).int()

        if len(tilling_list) > 0:
            for segment in range(len(tilling_list) // 4):
                segment_items_size[segment, 0] = int(tilling_list[segment * 4 + 2])
                segment_items_size[segment, 1] = int(tilling_list[segment * 4 + 3])
        elif shape_infer.split_strategy == "SRAM":
            sram_size = shape_infer.sram_restriction
            for segment in range(len(end_split_idx_list)):
                n = base_graph.nodes[end_split_idx_list[segment]]
                if 'kernel_x' in n.params or end_eltwise_idx_list[segment] > 0:
                    if end_split_idx_list[segment] > start_split_idx_list[segment]:
                        if end_eltwise_idx_list[segment] > 0:
                            weight_size = sram_size * 512
                        else:
                            weight_size = torch.prod(torch.tensor(n.constants['weights'].shape)).item()
                        itemy = max((n.inputs[0].ir_shape[1] * n.inputs[0].ir_shape[2] * n.inputs[0].ir_shape[3] / (sram_size * 1024 - weight_size)),
                                    (n.outputs[0].ir_shape[1] * n.outputs[0].ir_shape[2] * n.outputs[0].ir_shape[3] / (sram_size * 1024 - weight_size)))
                        itemy = torch.ceil(torch.tensor(itemy, device=device_type)).int()
                        if (n.outputs[0].ir_shape[1] - itemy) // itemy > 1 or \
                                (n.inputs[0].ir_shape[1] - itemy) // itemy > 1:
                            itemy = itemy + 2
                        segment_items_size[segment, 0] = itemy
                elif n.type == OpType.Transpose:
                    itemy = max((n.inputs[0].ir_shape[1] * n.inputs[0].ir_shape[2] * n.inputs[0].ir_shape[3] / (sram_size * 1024)),
                                (n.outputs[0].ir_shape[1] * n.outputs[0].ir_shape[2] * n.outputs[0].ir_shape[3] / (sram_size * 1024)))
                    itemy = torch.ceil(torch.tensor(itemy, device=device_type)).int()
                    segment_items_size[segment, 0] = itemy
            # re-calculate items
            for segment in range(len(end_eltwise_idx_list)):
                if end_eltwise_idx_list[segment] > 0:
                    idx = np.where(end_eltwise_idx_list[segment] == np.array(end_eltwise_idx_list))
                    if idx[0].size == 2:
                        segment_items_size[segment, 0] = max(
                            [segment_items_size[idx[0][0], 0], segment_items_size[idx[0][1], 0]])
        else:
            segment_items_size[:, ] = torch.tensor(shape_infer.split_params.clone().detach(), device=device_type)

        # insert partitiated op each segment
        eltwise_inputs = []
        all_nodes_eltwise = []
        for segment in range(len(end_split_idx_list)):
            item_num = segment_items_size[segment]
            parts_num_y = item_num[0].item()
            parts_num_x = item_num[1].item()
            if parts_num_y * parts_num_x == 1:
                continue
            last_node_idx = end_split_idx_list[segment] + node_offset
            first_node_idx = start_split_idx_list[segment] + node_offset
            all_work_items_len, all_work_items_in, each_node_work_items_out = \
                cls._get_partition_params_by_back_propagate(base_graph, last_node_idx, first_node_idx)
            #######################################################################################
            # insert slice op, get each block size
            #######################################################################################
            inserted_op_list = []
            eltwise_insert_pos_idx = 0
            if base_graph.nodes[last_node_idx].type != OpType.MatMul:
                super().insert_crop_op(base_graph, all_work_items_len, all_work_items_in, first_node_idx, inserted_op_list)
            else:
                super().insert_matmul_slice_op(base_graph, all_work_items_len, all_work_items_in, first_node_idx, inserted_op_list)
            ##########################################################################################
            # start add seperated conv node op, parts_num_y,parts_num_x conv op
            ##########################################################################################
            if base_graph.nodes[first_node_idx].type == OpType.FullyConnected:
                for item_y in range(parts_num_y):
                    inputs = inserted_op_list[item_y].outputs[0]
                    for node in range(first_node_idx, last_node_idx+1, 1):
                        inputs = super().insert_fc_op(base_graph, inputs, each_node_work_items_out, first_node_idx,
                                                      node, inserted_op_list, item_y, parts_num_x)
            elif base_graph.nodes[first_node_idx].type == OpType.MatMul:
                for item_y in range(parts_num_y):
                    for item_x in range(parts_num_x):
                        inputs = [inserted_op_list[item_y+parts_num_x].outputs[0], inserted_op_list[item_x].outputs[0]]
                        for node in range(first_node_idx, last_node_idx+1, 1):
                            inputs = super().insert_matmul_op(base_graph, inputs, each_node_work_items_out, first_node_idx,
                                                              node, inserted_op_list, item_y, item_x)
            else:
                if base_graph.nodes[last_node_idx].type == OpType.Eltwise:
                    if end_eltwise_idx_list[segment] < tot_nodes_num:
                        eltwise_input = all_nodes_eltwise.pop()
                        end_eltwise_idx_list[segment] = 0
                    else:  # the other side input is from crop
                        eltwise_work_items_len, eltwise_work_items_in, eltwise_node_work_items_out = super(
                        ).get_partition_params_by_back_propagate(last_node_idx, last_node_idx)
                        parents_id = base_graph.nodes[last_node_idx].parents[0].type == OpType.Convolution or base_graph.nodes[
                            last_node_idx].parents[0].type in shape_infer.tranparent_op
                        super().insert_crop_op(base_graph, eltwise_work_items_len, eltwise_work_items_in, last_node_idx, inserted_op_list, parents_id)
                        eltwise_insert_pos_idx = len(inserted_op_list)-parts_num_x*parts_num_y

                for item_y in range(parts_num_y):
                    for item_x in range(parts_num_x):
                        inputs = inserted_op_list[item_y*parts_num_x+item_x].outputs[0]
                        for node in range(first_node_idx, last_node_idx+1, 1):
                            if 'kernel_x' in base_graph.nodes[node].params:
                                inputs = super().insert_slide_window_op(
                                    base_graph, inputs, each_node_work_items_out, all_work_items_len, first_node_idx, node, inserted_op_list, item_x, item_y)
                            elif base_graph.nodes[node].type == OpType.Eltwise:
                                # assure eltwise is last node
                                if end_eltwise_idx_list[segment] > tot_nodes_num:
                                    input1 = inserted_op_list[item_y*parts_num_x +
                                                              item_x+eltwise_insert_pos_idx].outputs[0]
                                else:
                                    input1 = base_graph.nodes[eltwise_input[item_y*parts_num_x+item_x]].outputs[0]
                                inputs = [inputs, input1]
                                inputs = super().insert_eltwise_op(base_graph, inputs, each_node_work_items_out,
                                                                   first_node_idx, node, inserted_op_list, item_y, item_x)
                            else:
                                inputs = super().insert_non_slide_window_op(
                                    base_graph, inputs, each_node_work_items_out, first_node_idx, node, inserted_op_list, item_x, item_y)

            # ###################################################################################
            # last node output.no overlap
            # current node need change name, input and output shape
            ####################################################################################

            current_node = base_graph.nodes[last_node_idx]

            if current_node.type == OpType.FullyConnected:
                slice_num = parts_num_y
            elif current_node.type == OpType.MatMul:
                slice_num = parts_num_y+parts_num_x
            else:
                slice_num = parts_num_y*parts_num_x
            slice_num = slice_num+eltwise_insert_pos_idx
            node_num = last_node_idx-first_node_idx+1
            shape_dim = len(current_node.inputs[0].shape)
            axis_shift = 0
            if shape_dim == 2:
                axis_shift = 1
            if current_node.type == OpType.MatMul:
                axis_shift = 3
            elif current_node.type == OpType.Transpose:
                axis_shift = -1
            if end_eltwise_idx_list[segment] > 0 and end_eltwise_idx_list[segment] < tot_nodes_num:

                end_eltwise_idx_list[segment] = 0
                super().get_eltwise_input0(first_node_idx, parts_num_x, parts_num_y, node_num, eltwise_inputs)
                all_nodes_eltwise.append(np.array(eltwise_inputs))
            else:
                #########################################################################
                # there are items in only x or y direction
                if parts_num_x == 1 or parts_num_y == 1:
                    concat_name = current_node.name + '_item0'+'to'+str(parts_num_y*parts_num_x) + "_merge"
                    # new concat op
                    concat = PyNode(base_graph.get_valid_node_name(concat_name), OpType.Concat)
                    # concat's inputs number is parts_num_y*parts_num_x
                    for item_y in range(parts_num_y):
                        for item_x in range(parts_num_x):
                            inputs = inserted_op_list[slice_num+(item_y*parts_num_x+item_x+1)*node_num-1].outputs[0]
                            concat.add_input(inputs)

                    # new concat output
                    # current node's output are as concat node's output
                    concat.add_output(current_node.outputs[0])
                    concat.attrs.update(current_node.attrs.clone())
                    # params need is concat related
                    if parts_num_x == 1:
                        concat.params['axis'] = 1-axis_shift
                    else:
                        concat.params['axis'] = 2-axis_shift
                    # quantized IR, need scale and shift
                    inserted_op_list.append(concat)
                else:
                    #########################################################################
                    # there are items in x and y direction
                    concat_name = current_node.name + '_axis_y' + "_concat"
                    concat_h = PyNode(base_graph.get_valid_node_name(concat_name), OpType.Concat)
                    for item_y in range(parts_num_y):
                        concat_name = current_node.name + '_item_x'+str(item_y) + "_concat"
                        # new concat op
                        concat = PyNode(base_graph.get_valid_node_name(concat_name), OpType.Concat)
                        for item_x in range(parts_num_x):
                            inputs = inserted_op_list[slice_num+(item_y*parts_num_x+item_x+1)*node_num-1].outputs[0]
                            concat.add_input(inputs)

                        # get w direction concat result, then take it as h direction concat input
                        if axis_shift == 1:
                            h = inputs.ir_shape[0]
                            w = current_node.outputs[0].ir_shape[1]
                            new_outp_tensor_name = base_graph.get_valid_tensor_name(
                                current_node.outputs[0].name + '_item_y'+str(item_y))
                            newshape = TensorShape([h, w])
                        elif axis_shift == 3:
                            h = inputs.ir_shape[-2]
                            w = current_node.outputs[0].ir_shape[-1]
                            new_outp_tensor_name = base_graph.get_valid_tensor_name(
                                current_node.outputs[0].name + '_item_y'+str(item_y))
                            newshape = TensorShape([current_node.outputs[0].shape[0],
                                                    current_node.outputs[0].shape[1], h, w])
                        # transpose case
                        elif axis_shift == -1:
                            h = inputs.ir_shape[2]
                            w = current_node.outputs[0].ir_shape[3]
                            new_outp_tensor_name = base_graph.get_valid_tensor_name(
                                current_node.outputs[0].name + '_item_y'+str(item_y))
                            newshape = TensorShape([current_node.outputs[0].shape[0],
                                                    current_node.outputs[0].shape[1], h, w])
                        else:
                            h = inputs.ir_shape[1]
                            w = current_node.outputs[0].ir_shape[2]
                            new_outp_tensor_name = base_graph.get_valid_tensor_name(
                                current_node.outputs[0].name + '_item_y'+str(item_y))
                            newshape = TensorShape([current_node.outputs[0].shape[0], h,
                                                    w, current_node.outputs[0].shape[3]])
                        new_outp_tensor = PyTensor(new_outp_tensor_name, newshape, current_node.outputs[0].dtype)
                        new_outp_tensor.ir_shape = newshape
                        new_outp_tensor.qmin = current_node.outputs[0].qmin
                        new_outp_tensor.qmax = current_node.outputs[0].qmax
                        new_outp_tensor.similarity = current_node.outputs[0].similarity
                        concat.add_output(new_outp_tensor)
                        concat.params['axis'] = 2-axis_shift
                        inserted_op_list.append(concat)
                        # set  default value
                        concat.attrs.update(current_node.attrs.clone())
                        # add h dir concat input
                        concat_h.add_input(new_outp_tensor)

                    #################################################################
                    # set  default value
                    concat_h.attrs.update(current_node.attrs.clone())
                    # params need is concat related
                    concat_h.params['axis'] = 1 - axis_shift

                    concat_h.add_output(current_node.outputs[0])

                    # quantized IR, need scale and shift
                    inserted_op_list.append(concat_h)
            # self.remove_node(self.nodes[last_node_idx])
            tot_node = len(inserted_op_list)
            # for i in range(len(self.nodes[last_node_idx].children)):
            #     if self.nodes[last_node_idx].children[i].type == OpType.Eltwise:
            #         get_eltwise_input0(first_node_idx, parts_num_x, parts_num_y, node_num, eltwise_inputs)
            #         break

            # insert new op to nodes
            for node in range(tot_node-1, -1, -1):
                base_graph.nodes.insert(last_node_idx+1, inserted_op_list[node])
                inserted_op_list[node].quantized = True
            # remove original nodes
            # to remove this layer parent's child, child is replaced by silce

            base_graph.nodes[first_node_idx].remove_input(base_graph.nodes[first_node_idx].inputs[0])
            base_graph.nodes[last_node_idx].remove_output(base_graph.nodes[last_node_idx].outputs[0])
            for node in range(first_node_idx, last_node_idx+1, 1):
                base_graph.remove_node(base_graph.nodes[first_node_idx])

            node_offset = node_offset + tot_node - node_num

        # update each layer id
        for node in range(len(base_graph.nodes)):
            base_graph.nodes[node].attrs['layer_id'] = str(int(node))


class TillingForward(TillingBase):
    """
    support:
        1. mainly support multi-graphs(multi-core) support.
        2. split Vertical/horizontal/block support.
    """

    @classmethod
    def run(cls, base_graph, split_prams, ratio, tilling_list, split_mode="E2E", dump_path=None):
        def trans_blocks_pos_2_offset(hi, wj, blocks_size):
            """
            translate block position (hi, wj) to blocks mem offset.
            """
            x0 = 0
            x1 = 0
            y0 = 0
            y1 = 0
            split_h, split_w = self._split_params
            for h in range(split_h):
                if h < hi:
                    y0 += int(blocks_size[h, 0, 0])

            for w in range(split_w):
                if w < wj:
                    x0 += int(blocks_size[0, w, 1])

            x1 = x0 + int(blocks_size[0, wj, 1])
            y1 = y0 + int(blocks_size[hi, 0, 0])

            return x0, x1, y0, y1

        def get_split_tensor(t, blocks_size):
            """
            block_size: [split_h, split_w, 2], 2 means len([h_size, w_size]).
            t: the tensor to be split.

            t.layout must be: NHWC
            """

            assert len(t.stride()) == 4, "only support 4 dims for now."

            split_h, split_w = shape_infer.split_params

            all_split_tensors = []
            for h in range(split_h):
                row_tensors = []
                for w in range(split_w):
                    x0, x1, y0, y1 = trans_blocks_pos_2_offset(
                        h, w, blocks_size)
                    row_tensors.append(t[:, y0:y1, x0:x1, :].detach().clone())
                all_split_tensors.append(row_tensors)

            return all_split_tensors

        def generate_split_input(ir_shape):
            pass

        def fake_runtime_check(base_graph, split_graphs):
            # GT part.
            GT_inp = []
            for inp in base_graph.input_tensors:
                inp_random = torch.reshape(torch.rand(
                    inp.ir_shape.size())*127, list(inp.ir_shape))
                # replace betensor.
                inp.betensor = inp_random
                GT_inp.append(inp_random)

            GT_out = base_graph.forward(GT_inp)

            # step2: split part.
            split_h, split_w = shape_infer.get_split_params()
            blocks_size = shape_infer.all_blocks_size
            split_inputs = get_split_tensor(
                GT_inp[0], blocks_size)  # return a 2d list

            split_out_all = []
            for i in range(split_h):
                split_out_w = []
                for j in range(split_w):
                    graph = split_graphs[i][j]
                    inp_tensor = split_inputs[i][j]
                    graph.forward(inp_tensor)
                    split_out_w.append(graph.output_tensors[0])
                split_out_all.append(split_out_w)

            # start combine(concate) output:
            combine_output = torch.zeros_like(GT_out)
            for i in range(split_h):
                for j in range(split_w):
                    t = split_out_all[i, j]
                    x0, x1, y0, y1 = trans_blocks_pos_2_offset(
                        i, j, blocks_size)
                    if (y1-y0) == t.shape[1] and (x1-x0) == t.shape[2]:
                        combine_output[:, y0:y1, x0:x1, :] = t
                    else:
                        OPT_ERROR("need aligned.")

            print(f"max(GT_out - combine_out) = {torch.max(GT_out - combine_output)}")

        def dump_graph(splited_graph, dump_path):
            splited_graph.serialize_scale_zp = True
            splited_graph.serialize(
                dump_path + splited_graph.name+".txt", dump_path + splited_graph.name+".bin")
            OPT_INFO(f'IR {splited_graph.name} has been saved into {dump_path}')

        def _get_split_graph_parts(graph, item_num, ratio):
            shape_infer.update_adj_matrix(graph)

        def _generate_split_graph(base_graph, split_graph_name, split_inp_size):
            cloned_graph = base_graph.clone(split_graph_name)

            for inp in cloned_graph.input_tensors:
                # updata ir_shape
                inp.ir_shape[1] = split_inp_size[0]
                inp.ir_shape[2] = split_inp_size[1]
                inp_random = torch.reshape(torch.rand(
                    inp.ir_shape.size())*127, list(inp.ir_shape))
                # replace betensor.
                inp.betensor = inp_random

            # forward shape infer
            with tqdm(total=len(cloned_graph.nodes), desc='forward shape infer', file=sys.stdout, leave=True, disable=False) as pbar:
                for n in cloned_graph.nodes:
                    if not shape_infer[n.type]().forward_shape(n):
                        OPT_INFO(n.name + " shape forward infer failed.")
                    pbar.update(1)
                pbar.refresh()

            return cloned_graph

        device_type = base_graph.nodes[0].outputs[0].betensor.device
        shape_infer.device = device_type
        tot_nodes_num = len(base_graph.nodes)
        blocks_num = super()._deal_split_params(base_graph, split_prams)
        input_hw_size = torch.tensor([base_graph.input_tensors[0].ir_shape[1],
                                      base_graph.input_tensors[0].ir_shape[2]], device=device_type)
        all_block_size = shape_infer.all_blocks_size(input_hw_size)

        split_graphs = []
        for h in range(blocks_num[0]):
            split_graphs_w = []
            for w in range(blocks_num[1]):
                split_inp_size = all_block_size[h, w, :]
                split_graph_name = f"split_graph_H{h}_W{w}"
                splited_graph = _generate_split_graph(base_graph, split_graph_name, split_inp_size)
                splited_graph.name = split_graph_name
                split_graphs_w.append(splited_graph)
                dump_graph(splited_graph, dump_path)
            split_graphs.append(split_graphs_w)

        # # unit-test.
        # if not fake_runtime_check(base_graph, split_graphs):
        #     assert False, "graph_split_failed."

####################################### API #######################################################


def generate_split_graph_api(base_graph, split_prams, ratio, tilling_list: list, split_mode="backward"):
    """
    Params:
        ratio:
            1. restrict overlap params.

        split_prams:
            1. type(int), means SRAM/DDR footprint size restriction, unit = KB.
            2. type(list), type(torch.tensor([...]))

        split_mode:
            1. split_ (end-to-end split)
                    a. forward : force end-to-end forward split.
                    b. backward: force end-to-end backward split.
            2. autotune_split_ (block size split)
                    a. forward : based on net-structure to auto block split.
                    b. backward: based on net-structure to auto block split.
    Work Flow:
        base on split_prams generate blocks_num, then base on split_mode call logical specifical functions.
    """
    dump_path = "./"
    if split_mode == "forward":
        TillingForward.run(base_graph, split_prams, ratio, tilling_list, split_mode, dump_path)
    elif split_mode == "backward":
        TillingBackward.run(base_graph, split_prams, ratio, tilling_list)
    else:
        OPT_ERROR(f"split mode: {split_mode} not supported(should be forward/backward).")
