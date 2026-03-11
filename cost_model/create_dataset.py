import argparse
import os
import numpy as np
import tempfile
import configparser
import json
import AIPUBuilder.core as core
import AIPUBuilder.ops as ops
from AIPUSimProfiler.profile.perf_generator import get_graph_json, get_config, GeneratePerf


def get_op_graph(op_name, **kargs):
    input_np = []
    if op_name in ['sigmoid', 'leakyrelu', 'prelu', 'tanh', 'silu', 'hardswish']:
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        if op_name in ['leakyrelu', 'silu', 'hardswish']:
            in0 = ops.input(in0_shape, core.Dtype.FP32)
        else:
            in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        if op_name == 'sigmoid':
            out0 = ops.sigmoid(in0)
        elif op_name == 'leaky_relu':
            out0 = ops.leaky_relu(in0, negative_slope=0.01)
        elif op_name == 'prelu':
            alpha = core.Tensor(np.random.rand(in0_shape[-1]))
            out0 = ops.prelu(in0, alpha)
        elif op_name == 'tanh':
            out0 = ops.tanh(in0)
        elif op_name == 'silu':
            out0 = ops.silu(in0)
        else:
            out0 = ops.hard_swish(in0)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name in ['conv2d', 'dwconv2d']:
        output_c = kargs['output_c']
        group = kargs['group']
        kernel_y = kargs['kernel_y']
        kernel_x = kargs['kernel_x']
        stride_y = kargs['stride_y']
        stride_x = kargs['stride_x']
        dilation_y = kargs['dilation_y']
        dilation_x = kargs['dilation_x']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        if group == in0_shape[3]:
            weight = core.Tensor(np.random.rand(output_c, kernel_y, kernel_x, 1))
            conv2d = ops.depthwise_conv2d
        else:
            weight = core.Tensor(np.random.rand(output_c, kernel_y, kernel_x, in0_shape[3]))
            conv2d = ops.conv2d
        out0 = conv2d(in0,
                      weight,
                      strides=[stride_y, stride_x],
                      padding=[0, 0, 0, 0],
                      dilations=[dilation_y, dilation_x])
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name in ['avgpool2d', 'maxpool2d']:
        kernel_y = kargs['kernel_y']
        kernel_x = kargs['kernel_x']
        stride_y = kargs['stride_y']
        stride_x = kargs['stride_x']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        if op_name == 'avgpool2d':
            pool2d = ops.avg_pool
        else:
            pool2d = ops.max_pool
        out0 = pool2d(in0,
                      kernel=[kernel_y, kernel_x],
                      strides=[stride_y, stride_x],
                      dilations=[1, 1],
                      padding=[0, 0, 0, 0])
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'softmax':
        axis = kargs['axis']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.FP32)
        in0.name = 'input0'
        out0 = ops.softmax(in0, axis=axis)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'fc':
        output_c = kargs['output_c']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        weight = core.Tensor(np.random.rand(output_c, in0_shape[1]))
        out0 = ops.fully_connected(in0, weight)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name in ['ewadd', 'ewmul', 'ewsub', 'ewmin']:
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in1_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        input_np.append(in1_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        in1 = ops.input(in0_shape, core.Dtype.INT8)
        in1.name = 'input1'
        if op_name == 'ewadd':
            out0 = ops.elementwise_add(in0, in1)
        elif op_name == 'ewmul':
            out0 = ops.elementwise_mul(in0, in1)
        elif op_name == 'ewsub':
            out0 = ops.elementwise_sub(in0, in1)
        else:
            out0 = ops.elementwise_min(in0, in1)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'matmul':
        in0_shape = kargs['in_shapes'][0]
        in1_shape = [in0_shape[0], in0_shape[1], in0_shape[3], in0_shape[2]]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in1_np = np.random.randint(low=0, high=128, size=in1_shape)
        input_np.append(in0_np)
        input_np.append(in1_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        in1 = ops.input(in1_shape, core.Dtype.INT8)
        in1.name = 'input1'
        out0 = ops.matmul(in0, in1)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name in ['add', 'sub', 'mul']:
        in0_shape = kargs['in_shapes'][0]
        in1_shape = kargs['in_shapes'][1]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in1_np = np.random.randint(low=0, high=128, size=in1_shape)
        input_np.append(in0_np)
        input_np.append(in1_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        in1 = ops.input(in1_shape, core.Dtype.INT8)
        in1.name = 'input1'
        if op_name == 'add':
            out0 = ops.add(in0, in1)
        elif op_name == 'sub':
            out0 = ops.sub(in0, in1)
        else:
            out0 = ops.mul(in0, in1)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'sd':
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in1_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        input_np.append(in1_np)
        in0 = ops.input(in0_shape, core.Dtype.FP32)
        in0.name = 'input0'
        in1 = ops.input(in0_shape, core.Dtype.FP32)
        in1.name = 'input1'
        out0 = ops.squared_difference(in0, in1)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'b2s':
        block_shape = kargs['block_shape']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.batch_to_space(in0, block_shape=block_shape)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'cast':
        dtype_map = {
            'int8' : core.Dtype.INT8,
            'int16' : core.Dtype.INT16,
            'int32' : core.Dtype.INT32,
        }
        dtype = kargs['dtype']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in0 = ops.input(in0_shape, dtype_map[dtype[0]])
        in0.name = 'input0'
        input_np.append(in0_np.astype(in0.dtype.np))
        out0 = ops.cast(in0, dtype_map[dtype[1]])
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'rsqrt':
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.rsqrt(in0)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'rs':
        in0_shape = kargs['in_shapes'][0]
        in1_shape = [in0_shape[1]]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in1_np = np.random.randint(low=0, high=1, size=in1_shape)
        input_np.append(in0_np)
        input_np.append(in1_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        in1 = ops.input(in1_shape, core.Dtype.INT32)
        in1.name = 'input1'
        out0 = ops.reverse_sequence(in0, in1, seq_axis=0, batch_axis=1)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'crop':
        crops = kargs['crops']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.crop(in0, crops)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'pad':
        pads = kargs['pads']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.pad(in0, pads, mode="Constant", constant_value=0.0)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'd2s':
        block_size = kargs['block_size'][0]
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.depth_to_space(in0, block_size=block_size)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'tile':
        repeats = kargs['repeats']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.tile(in0, repeats)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'resize':
        scale = kargs['scale']
        method = kargs['method']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.resize(in0, scale=scale, method=method)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'concat':
        axis = kargs['axis']
        in_shapes = kargs['in_shapes']
        inps = []
        for k, shape in enumerate(in_shapes):
            input_np.append(np.random.randint(low=0, high=128, size=shape))
            inps.append(ops.input(shape, core.Dtype.INT8))
            inps[k].name = f'input{k}'
        out0 = ops.concat(inps, axis=axis)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'split':
        axis = kargs['axis']
        splits = kargs['splits']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        outs = ops.split(in0, splits, axis)
        for k, out in enumerate(outs):
            out.name = f'output{k}'
        outs[0].op.name = op_name
    elif op_name == 's2d':
        in0_shape = kargs['in_shapes'][0]
        block_size = kargs['block_size'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.space_to_depth(in0, block_size=block_size)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'gather':
        axis = kargs['axis']
        in1_np = kargs['indexs']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        input_np.append(in1_np)
        in0 = ops.input(in0_shape, core.Dtype.FP32)
        in0.name = 'input0'
        in1 = ops.input(in1_np.shape, core.Dtype.FP32)
        in1.name = 'input1'
        out0 = ops.gather(in0, in1, axis=axis, batch_dims=0)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'transpose':
        perm = kargs['perm']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.transpose(in0, perm)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'layoutcvt':
        from_layout, to_layout = kargs['layouts']
        in0_shape = kargs['in_shapes'][0]
        out0_shape = kargs['out_shapes'][0]
        graph = f"""
model_name=layoutcvt
layer_number=2
input_tensors=[input0]
output_tensors=[output0]

layer_id=0
layer_name=input
layer_type=Input
layer_bottom=[]
layer_bottom_shape=[]
layer_bottom_type=[]
layer_top=[input0]
layer_top_shape=[[{",".join([str(x) for x in in0_shape])}]]
layer_top_datalayout=[{from_layout}]
layer_top_type=[int8]
layer_top_scale=[1.0]
layer_top_zp=[0]

layer_id=1
layer_name=layoutcvt
layer_type=LayoutConvertor
layer_bottom=[input0]
layer_bottom_shape=[[{",".join([str(x) for x in in0_shape])}]]
layer_bottom_type=[int8]
layer_top=[output0]
layer_top_shape=[[{",".join([str(x) for x in out0_shape])}]]
layer_top_type=[int8]
layer_top_datalayout=[{to_layout}]
layer_top_scale=[1.0]
layer_top_zp=[0]
from={from_layout}
to={to_layout}
"""
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        return graph, input_np
    elif op_name in ['l1norm', 'l2norm']:
        axis = kargs['axis']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        if op_name == 'l1norm':
            out0 = ops.l1_norm(in0, axis=axis, epsilon=1e-5)
        else:
            out0 = ops.l2_norm(in0, axis=axis, epsilon=1e-5)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'bn':
        axis = kargs['axis']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        gamma = core.Tensor(np.random.rand(in0_shape[axis]))
        beta = core.Tensor(np.random.rand(in0_shape[axis]))
        mean = core.Tensor(np.random.rand(in0_shape[axis]))
        variance = core.Tensor(np.random.rand(in0_shape[axis]))
        out0 = ops.batch_norm(in0, gamma, beta, mean, variance, axis=axis, epsilon=1e-5)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'gn':
        axis = kargs['axis']
        group = kargs['group']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        gamma = core.Tensor(np.random.rand(in0_shape[axis]))
        beta = core.Tensor(np.random.rand(in0_shape[axis]))
        out0 = ops.group_norm(in0, gamma, beta, axis=axis, group=group, epsilon=1e-5)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'ln':
        axis = kargs['axis']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.FP32)
        in0.name = 'input0'
        gamma = core.Tensor(np.random.rand(*[in0_shape[x] for x in axis]).astype(np.float32))
        beta = core.Tensor(np.random.rand(*[in0_shape[x] for x in axis]).astype(np.float32))
        out0 = ops.layer_norm(in0, gamma, beta, axis=axis, epsilon=1e-5, quantization=(1.0, 0.0))
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'in':
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.FP32)
        in0.name = 'input0'
        gamma = core.Tensor(np.ones(in0_shape[-1]).astype(np.float32))
        beta = core.Tensor(np.zeros(in0_shape[-1]).astype(np.float32))
        out0 = ops.instance_norm(in0, gamma, beta, epsilon=1e-5, quantization=(1.0, 0.0))
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'gs':
        method = kargs['method']
        in0_shape = kargs['in_shapes'][0]
        in1_shape = kargs['in_shapes'][1]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in1_np = np.random.randint(low=0, high=128, size=in1_shape)
        input_np.append(in0_np)
        input_np.append(in1_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in1 = ops.input(in1_shape, core.Dtype.INT16)
        in0.name = 'input0'
        in1.name = 'input1'
        out0 = ops.grid_sample(in0,
                               in1,
                               align_corners=True,
                               method=method,
                               padding_mode="border")
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'slice':
        begin = kargs['begin']
        end = kargs['end']
        strides = kargs['strides']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.slice(in0, begin, end, strides)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'reduce':
        axis = kargs['axis']
        method = kargs['method']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.reduce(in0, method=method, axis=axis, keep_dim=True)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'deconv2d':
        output_c = kargs['output_c']
        group = kargs['group']
        kernel_y = kargs['kernel_y']
        kernel_x = kargs['kernel_x']
        stride_y = kargs['stride_y']
        stride_x = kargs['stride_x']
        dilation_y = kargs['dilation_y']
        dilation_x = kargs['dilation_x']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        weight = core.Tensor(np.random.rand(output_c, kernel_y, kernel_x, in0_shape[3]))
        out0 = ops.conv2d_transpose(in0,
                                    weight,
                                    strides=[stride_y, stride_x],
                                    padding=[0, 0, 0, 0],
                                    group=group,
                                    dilations=[dilation_y, dilation_x])
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name == 'lstm':
        in_shapes = kargs['in_shapes']
        in_size = in_shapes[0][-1]
        cell_size = in_shapes[1][-1]
        inps = []
        for k, shape in enumerate(in_shapes):
            input_np.append(np.random.randint(low=0, high=128, size=shape))
            inps.append(ops.input(shape, core.Dtype.INT8))
            inps[k].name = f'input{k}'
        weight = core.Tensor(np.random.rand(4 * cell_size, in_size + cell_size))
        bias = core.Tensor(np.random.rand(4 * cell_size))
        outs = ops.basic_lstm(inps[0],
                              inps[1],
                              inps[2],
                              weight,
                              bias,
                              activations=["Sigmoid", "Tanh", "Tanh"],
                              out_sequence="H_C")
        for k, out in enumerate(outs):
            out.name = f'output{k}'
        outs[0].op.name = op_name
    elif op_name == 'cs':
        group = kargs['group']
        axis = kargs['axis']
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        out0 = ops.channel_shuffle(in0, group, axis=axis, split=1)[0]
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name in ['sine', 'cosine']:
        in0_shape = kargs['in_shapes'][0]
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        input_np.append(in0_np)
        in0 = ops.input(in0_shape, core.Dtype.INT8)
        in0.name = 'input0'
        if op_name == 'sine':
            out0 = ops.sine(in0)
        else:
            out0 = ops.cosine(in0)
        out0.op.name = op_name
        out0.name = 'output0'
    elif op_name in ['quantize', 'dequantize']:
        dtype_map = {
            'float16' : core.Dtype.FP16,
            'int8' : core.Dtype.INT8,
            'int16' : core.Dtype.INT16,
        }
        dtype = kargs['dtype']
        in0_shape = kargs['in_shapes'][0]
        if op_name == 'quantize':
            in_dtype = 'float16'
            out_dtype = dtype
        else:
            in_dtype = dtype
            out_dtype = 'float16'
        in0_np = np.random.randint(low=0, high=128, size=in0_shape)
        in0 = ops.input(in0_shape, dtype_map[in_dtype])
        in0.name = 'input0'
        input_np.append(in0_np.astype(in0.dtype.np))
        if op_name == 'quantize':
            out0 = ops.quantize(in0, dtype_map[out_dtype])
        else:
            out0 = ops.dequantize(in0, dtype_map[out_dtype])
        out0.op.name = op_name
        out0.name = 'output0'
    else:
        raise ValueError(f'Unknown op_name: {op_name}')
    graph = ops.get_default_graph()
    ops.new_default_graph()
    graph.name = op_name
    return graph, input_np


def gen_opt_config(op_name, inputs):
    config = configparser.ConfigParser()
    config.add_section('Common')
    common = config['Common']
    common['model_name'] = op_name
    common['graph'] = f'{op_name}.txt'
    common['bin'] = f'{op_name}.bin'
    batch_size = inputs[0].shape[0]
    common['calibration_batch_size'] = str(batch_size)
    common['metric_batch_size'] = '1'
    common['weight_bits'] = '8'
    common['bias_bits'] = '32'
    common['activation_bits'] = '8'
    common['dump'] = 'False'
    common['output_dir'] = './'
    common['quant_ir_name'] = f'{op_name}_quant'
    common['cast_dtype_for_lib'] = 'True'
    with open('opt.cfg', 'w') as fd:
        config.write(fd)


def gen_input_bin(op_name, inputs):
    bin_files = []
    for k, inp in enumerate(inputs):
        bin_file = f'input{k}.bin'
        if op_name in ['cast', 'quantize', 'dequantize']:
            inp.tofile(bin_file)
        else:
            inp.astype(np.int8).tofile(bin_file)
        bin_files.append(bin_file)
    return bin_files


def get_perf_node_info(perf_info):
    perf_node_info = dict()
    if 'nodes' in perf_info:
        for node_info in perf_info['nodes']:
            for name, info in get_perf_node_info(node_info).items():
                perf_node_info[name] = info
    else:
        perf_node_info[perf_info['name']] = perf_info
    return perf_node_info


def parse_profile_data(case_path, frequency=1000, ddr_rd_delay=0, ddr_wr_delay=0):
    graph_json = get_graph_json(case_path, 'graph.json')
    config = get_config(graph_json)
    perf = GeneratePerf(config, graph_json)
    perf.set_frequency(frequency)
    perf.update_params(DDR_RD_DELAY=ddr_rd_delay, DDR_WR_DELAY=ddr_wr_delay)
    perf_data = perf.get_perf_data(case_path)

    graph_perf_data = dict()
    for perf_info in perf_data['profile_data']:
        for node_name, node_info in get_perf_node_info(perf_info).items():
            runtime = node_info['runtime_information']
            # utilization = busy_cycle / count_cycle * 100
            count_cycle = str(runtime['count_cycle'])
            read_bytes = str(runtime['AIPU_read_bytes'])
            write_bytes = str(runtime['AIPU_write_bytes'])
            aiff_cycle = str(runtime['AIFF_busy_cycle'])
            dma_cycle = str(runtime['DMA_busy_cycle'])
            graph_perf_data[node_name] = [count_cycle, read_bytes, write_bytes, aiff_cycle, dma_cycle]
    return graph_perf_data


def get_plugin_node_info(node_info):
    plugin_node_info = dict()
    if node_info['op_type'] == 'SubGraph':
        if node_info['plugin']:
            plugin_node_info[node_info['layer node name']] = node_info
        else:
            for node in node_info['internal_nodes']:
                for name, info in get_plugin_node_info(node).items():
                    plugin_node_info[name] = info
    else:
        plugin_node_info[node_info['layer node name']] = node_info
    return plugin_node_info


def get_graph_node_info(graph_json):
    with open(graph_json, 'r') as fd:
        graph_info = json.load(fd)

    tensor_info = graph_info['tensors']
    graph_node_info = dict()
    for node_info in graph_info['nodes']:
        for name, info in get_plugin_node_info(node_info).items():
            input_info = []
            if info['inputs'] is not None:
                for inp in info['inputs']:
                    input_info.append(tensor_info[inp])
            info['inputs'] = input_info

            output_info = []
            for outp in info['outputs']:
                output_info.append(tensor_info[outp])
            info['outputs'] = output_info
            graph_node_info[name] = info
    return graph_node_info


def run_simulator(op_name, target, **kargs):
    temp_dir = tempfile.TemporaryDirectory()
    cur_path = os.getcwd()
    output_dir = os.path.abspath(temp_dir.name)
    os.chdir(output_dir)

    graph, inputs = get_op_graph(op_name, **kargs)
    if op_name == 'layoutcvt':
        with open(f'{op_name}_quant.txt', 'w') as fd:
            fd.write(graph)
        os.system(f'touch {op_name}_quant.bin')
    else:
        graph_def, graph_bin = graph.serialize()
        with open(f'{op_name}.txt', 'w') as fd:
            fd.write(graph_def)
        with open(f'{op_name}.bin', 'wb') as fd:
            fd.write(graph_bin)

        gen_opt_config(op_name, inputs)
        aipuopt_cmd = 'aipuopt -c opt.cfg'
        os.system(aipuopt_cmd)

    if op_name in ['avgpool2d', 'maxpool2d']:
        disable_pass = ['split_avg_pool_kernel', 'split_maxpool_kernel']
    elif op_name in ['conv2d', 'dwconv2d']:
        disable_pass = ['split_conv_pad', 'split_conv_input_c', 
                        'split_conv_output_c', 'split_conv_kernel', 'conv_full_tiling',
                        'stride_kernel_size_adaptor', 'atrous_conv_to_conv']
    elif op_name == 'fc':
        disable_pass = ['batch_FC_to_conv']
    elif op_name == 'cast':
        disable_pass = ['cast_to_leakyrelu']
    elif op_name == 'crop':
        disable_pass = ['remove_crop', 'replace_crop_to_conv']
    elif op_name == 'resize':
        disable_pass = ['resize_to_dwconv']
    elif op_name == 'concat':
        disable_pass = ['remove_concat']
    elif op_name == 'split':
        disable_pass = ['remove_split']
    elif op_name == 'slice':
        disable_pass = ['slice_to_crop', 'remove_dummy_slice']
    elif op_name == 'gn':
        disable_pass = ['decompose_groupnorm']
    elif op_name == 'reduce':
        disable_pass = ['reduce_to_reshape', 'reduce_to_pool']
    else:
        disable_pass = []

    if op_name in ['cast', 'quantize', 'dequantize']:
        graph_txt = f'{op_name}.txt'
        graph_bin = f'{op_name}.bin'
    else:
        graph_txt = f'{op_name}_quant.txt'
        graph_bin = f'{op_name}_quant.bin'

    in_files = gen_input_bin(op_name, inputs)
    aipurun_cmd = f'aipurun {graph_txt} -w {graph_bin} -i {",".join(in_files)}' + \
                  f' -o output.bin --target {target} --dumpIR dumpir --profile'
    if len(disable_pass) > 0:
        aipurun_cmd += f' --disable-pass {",".join(disable_pass)}'
    os.system(aipurun_cmd)

    graph_perf_data = parse_profile_data(output_dir)
    graph_node_info = get_graph_node_info('graph.json')
    matched_node_info = dict()
    for node_name in graph_node_info:
        if node_name in graph_perf_data:
            node_info = graph_node_info[node_name]
            node_info['perf_data'] = graph_perf_data[node_name]
            matched_node_info[node_name] = node_info

    os.chdir(cur_path)
    temp_dir.cleanup()
    return matched_node_info


def update_op_dataset(op_name, target, case_info):
    file_root = os.path.dirname(__file__)
    if target == 'X1_1204':
        data_file = f'{op_name}_x1.json'
    else:
        data_file = f'{op_name}_x2.json'
    file_path = os.path.join(file_root, 'ops_dataset', data_file)

    if os.path.exists(file_path):
        with open(file_path, 'r') as fd:
            op_case_info = json.load(fd)
        if case_info:
            tmp_path = file_path.replace('.json', '.tmp')
            os.rename(file_path, tmp_path)
            for case_id, case_data in case_info.items():
                op_case_info[case_id] = case_data
            with open(file_path, 'w') as fd:
                json.dump(op_case_info, fd, indent=2)
            os.remove(tmp_path)
    else:
        with open(file_path, 'w') as fd:
            json.dump(case_info, fd, indent=2)
        op_case_info = case_info
    return set([x for x in op_case_info])


def get_case_data(node_info, sg_name=None, skip_cvt=True):
    if sg_name is not None or skip_cvt:
        filtered_info = dict()
        for name, info in node_info.items():
            if sg_name is not None and name != sg_name:
                continue
            if skip_cvt and info['op_type'] == 'LayoutConvertor':
                continue
            filtered_info[name] = info
        node_info = filtered_info

    if len(node_info) != 1:
        raise ValueError(f'Unsupported case: {node_info}')
    case_data = dict()
    for name, info in node_info.items():
        case_data['perf_data'] = info['perf_data']
        if info['op_type'] == 'SubGraph':
            if len(info['internal_nodes']) != 1:
                raise ValueError(f'Multiple nodes in case: {node_info}')
            node = info['internal_nodes'][0]
            case_data['op_type'] = node['op_type']
            case_data['params'] = node['params']
        else:
            case_data['op_type'] = info['op_type']
            case_data['params'] = info['params']
        case_data['inputs'] = info['inputs']
        case_data['outputs'] = info['outputs']
    return case_data


def gen_op_dataset(op_name, target):
    case_info = dict()
    op_case_id = update_op_dataset(op_name, target, case_info)
    if op_name in ['sigmoid', 'leakyrelu', 'prelu', 'tanh', 'silu', 'hardswish']:
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name, target, in_shapes=[[bz, fh, fw, ic]])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name == 'conv2d':
        for bz in [1]:
            for ky, kx in [[1, 1], [3, 3], [5, 5] ,[7, 7]]:
                for sy, sx in [[1, 1], [2, 2]]:
                    for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                        for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                            for ic in [32, 64, 96, 128, 160, 192, 224, 256]:
                                for oc in [32, 64, 96, 128, 160, 192, 224, 256]:
                                    case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_oc_{}_ky_{}_kx_{}_sy_{}_sx_{}'.format(
                                        op_name, bz, fh, fw, ic, oc, ky, kx, sy, sx)
                                    if case_id in op_case_id:
                                        continue
                                    node_info = run_simulator(op_name,
                                                              target,
                                                              in_shapes=[[bz, fh, fw, ic]],
                                                              output_c=oc,
                                                              group=1,
                                                              kernel_y=ky,
                                                              kernel_x=kx,
                                                              stride_y=sy,
                                                              stride_x=sx,
                                                              dilation_y=1,
                                                              dilation_x=1)
                                    case_info[case_id] = get_case_data(node_info)
                                    if len(case_info) % 100 == 0:
                                        op_case_id = update_op_dataset(op_name, target, case_info)
                                        case_info = dict()
    elif op_name == 'dwconv2d':
        for bz in [1]:
            for ky, kx in [[3, 3], [5, 5] ,[7, 7]]:
                for sy, sx in [[1, 1], [2, 2]]:
                    for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                        for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                            for ic in [32, 64, 96, 128, 160, 192, 224, 256]:
                                case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_g_{}_ky_{}_kx_{}_sy_{}_sx_{}'.format(
                                    op_name, bz, fh, fw, ic, ic, ky, kx, sy, sx)
                                if case_id in op_case_id:
                                    continue
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=[[bz, fh, fw, ic]],
                                                          output_c=ic,
                                                          group=ic,
                                                          kernel_y=ky,
                                                          kernel_x=kx,
                                                          stride_y=sy,
                                                          stride_x=sx,
                                                          dilation_y=1,
                                                          dilation_x=1)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 'avgpool2d':
        for bz in [1]:
            for ky in [3, 5, 7, 9, 11, 13]:
                for kx in [3, 5, 7, 9, 11, 13]:
                    for sy, sx in [[1, 1], [2, 2]]:
                        for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                            for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                                for ic in [32, 64, 96, 128, 160, 192, 224, 256]:
                                    case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_ky_{}_kx_{}_sy_{}_sx_{}'.format(
                                        op_name, bz, fh, fw, ic, ky, kx, sy, sx)
                                    if case_id in op_case_id:
                                        continue
                                    node_info = run_simulator(op_name,
                                                              target,
                                                              in_shapes=[[bz, fh, fw, ic]],
                                                              kernel_y=ky,
                                                              kernel_x=kx,
                                                              stride_y=sy,
                                                              stride_x=sx)
                                    case_info[case_id] = get_case_data(node_info)
                                    if len(case_info) % 100 == 0:
                                        op_case_id = update_op_dataset(op_name, target, case_info)
                                        case_info = dict()
    elif op_name == 'maxpool2d':
        for bz in [1]:
            for ky, kx in [[3, 3], [5, 5], [7, 7], [9, 9]]:
                for sy, sx in [[1, 1], [2, 2]]:
                    for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                        for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                            for ic in [32, 64, 96, 128, 160, 192, 224, 256]:
                                case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_ky_{}_kx_{}_sy_{}_sx_{}'.format(
                                    op_name, bz, fh, fw, ic, ky, kx, sy, sx)
                                if case_id in op_case_id:
                                    continue
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=[[bz, fh, fw, ic]],
                                                          kernel_y=ky,
                                                          kernel_x=kx,
                                                          stride_y=sy,
                                                          stride_x=sx)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 'softmax':
        for bz in [1, 2, 3, 4, 5, 6, 7, 8]:
            for ic in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                       544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                case_id = '{}_bz_{}_ic_{}'.format(op_name, bz, ic)
                if case_id in op_case_id:
                    continue
                node_info = run_simulator(op_name,
                                          target,
                                          in_shapes=[[bz, ic]],
                                          axis=1)
                case_info[case_id] = get_case_data(node_info)
                if len(case_info) % 100 == 0:
                    op_case_id = update_op_dataset(op_name, target, case_info)
                    case_info = dict()
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fh, fw, ic]],
                                                  axis=3)
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name == 'fc':
        for bz in [1, 2, 3, 4, 5, 6, 7, 8]:
            for ic in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                       544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                for oc in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                           544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                    case_id = '{}_bz_{}_ic_{}_oc_{}'.format(op_name, bz, ic, oc)
                    if case_id in op_case_id:
                        continue
                    node_info = run_simulator(op_name,
                                              target,
                                              in_shapes=[[bz, ic]],
                                              output_c=oc)
                    case_info[case_id] = get_case_data(node_info)
                    if len(case_info) % 100 == 0:
                        op_case_id = update_op_dataset(op_name, target, case_info)
                        case_info = dict()
    elif op_name in ['ewadd', 'ewmul', 'ewsub', 'ewmin']:
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fh, fw, ic]])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name == 'fc':
        for bz in [1]:
            for fh in [1, 3, 5, 7, 9, 11, 13, 15]:
                for fw in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                           544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                    for ic in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                               544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fh, fw, ic]])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name in ['add', 'sub', 'mul']:
        for bz in [1]:
            for fz in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                       544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                for ic in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                           544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                    for bc in [1, ic]:
                        for swap in [False, True]:
                            case_id = '{}_bz_{}_fz_{}_ic_{}_bc_{}_swap_{}'.format(
                                op_name, bz, fz, ic, bc, swap)
                            if case_id in op_case_id:
                                continue
                            if swap:
                                in_shapes = [[bz, fz, ic], [bc]]
                            else:
                                in_shapes = [[bc], [bz, fz, ic]]
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=in_shapes)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for bc in [1, ic]:
                            for swap in [False, True]:
                                case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_bc_{}_swap_{}'.format(
                                    op_name, bz, fh, fw, ic, bc, swap)
                                if case_id in op_case_id:
                                    continue
                                if swap:
                                    in_shapes = [[bz, fh, fw, ic], [bc]]
                                else:
                                    in_shapes = [[bc], [bz, fh, fw, ic]]
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=in_shapes)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 'sd':
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fh, fw, ic]])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name == 'b2s':
        for bz, bs_y, bs_x in [[4, 2, 2], [16, 4, 4]]:
            for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        case_id = '{}_bz_{}_bs_y_{}_bs_x_{}_fh_{}_fw_{}_ic_{}'.format(
                            op_name, bz, bs_y, bs_x, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fh, fw, ic]],
                                                  block_shape=[bs_y, bs_x])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name == 'cast':
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for id, od in [['int32', 'int8'], ['int16', 'int8']]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_id_{}_od_{}'.format(
                                op_name, bz, fh, fw, ic, id, od)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      dtype=[id, od])
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'rsqrt':
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name, target, in_shapes=[[bz, fh, fw, ic]])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name == 'rs':
        for ts in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]:
            for bz in [1, 2, 3, 4, 5, 6, 7, 8]:
                for ic in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                           544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                    case_id = '{}_ts_{}_bz_{}_ic_{}'.format(op_name, ts, bz, ic)
                    if case_id in op_case_id:
                        continue
                    node_info = run_simulator(op_name, target, in_shapes=[[ts, bz, ic]])
                    case_info[case_id] = get_case_data(node_info)
                    if len(case_info) % 100 == 0:
                        op_case_id = update_op_dataset(op_name, target, case_info)
                        case_info = dict()
    elif op_name == 'crop':
        def gen_crop(max_value, crop_ratio=0.25):
            max_crop = int(max_value * crop_ratio)
            low = np.random.randint(0, high=max_crop + 1, size=1)[0]
            min_high = max(low + 1, max_value - max_crop)
            high = np.random.randint(min_high, high=max_value + 1, size=1)[0]
            return [int(low), int(high)]
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for crops in [[[0, bz], gen_crop(fh), [0, fw], [0, ic]],
                                      [[0, bz], [0, fh], gen_crop(fw), [0, ic]],
                                      [[0, bz], [0, fh], [0, fw], gen_crop(ic)],
                                      [[0, bz], gen_crop(fh), gen_crop(fw), [0, ic]]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_crops_{}'.format(
                                op_name, bz, fh, fw, ic, crops)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name, 
                                                      target, 
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      crops=crops)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'pad':
        def gen_pad(max_value, pad_ratio=0.25):
            max_pad = int(max_value * pad_ratio)
            pad_left = np.random.randint(0, high=max_pad + 1, size=1)[0]
            pad_right = np.random.randint(0, high=max_pad + 1, size=1)[0]
            return [int(pad_left), int(pad_right)]
        for bz in [1]:
            for fz in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                       544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                for ic in [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 
                           544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]:
                    for pads in [[[0, 0], gen_pad(fz), [0, 0]],
                                 [[0, 0], [0, 0], gen_pad(ic)]]:
                        case_id = '{}_bz_{}_fz_{}_ic_{}_pads_{}'.format(op_name, bz, fz, ic, pads)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name, 
                                                  target, 
                                                  in_shapes=[[bz, fz, ic]],
                                                  pads=pads)
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for pads in [[[0, 0], gen_pad(fh), [0, 0], [0, 0]],
                                     [[0, 0], [0, 0], gen_pad(fw), [0, 0]],
                                     [[0, 0], [0, 0], [0, 0], gen_pad(ic)],
                                     [[0, 0], gen_pad(fh), gen_pad(fw), [0, 0]]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_pads_{}'.format(
                                op_name, bz, fh, fw, ic, pads)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name, 
                                                      target, 
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      pads=pads)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'd2s':
        for bz in [1]:
            for bs_y, bs_x in [[2, 2], [3, 3], [4, 4], [8, 8]]:
                for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                    for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                        for ic in [64, 128, 192, 256, 320, 384, 448, 512, 
                                   576, 640, 704, 768, 832, 896, 960, 1024]:
                            multiple = bs_y * bs_x
                            if ic % multiple != 0:
                                ic = (ic // multiple + 1) * multiple
                            case_id = '{}_bz_{}_bs_y_{}_bs_x_{}_fh_{}_fw_{}_ic_{}'.format(
                                op_name, bz, bs_y, bs_x, fh, fw, ic)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name, 
                                                      target, 
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      block_size=[bs_y, bs_x])
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'tile':
        def gen_repeat(max_value):
            value = np.random.randint(1, high=max_value + 1, size=1)[0]
            return int(value)
        for bz in [1]:
            for fh in [1, 3, 5, 7, 9, 11, 13, 15]:
                for fw in [1, 3, 5, 7, 9, 11, 13, 15]:
                    for ic in [16, 32, 48, 64, 80, 96, 112, 128]:
                        for repeats in [[gen_repeat(8), 1, 1, 1],
                                        [1, gen_repeat(64), 1, 1],
                                        [1, 1, gen_repeat(64), 1],
                                        [1, gen_repeat(64), gen_repeat(64), 1],
                                        [1, 1, 1, gen_repeat(8)]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_repeats_{}'.format(
                                op_name, bz, fh, fw, ic, repeats)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name, 
                                                      target, 
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      repeats=repeats)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'resize':
        for bz in [1]:
            for fh in [16, 32, 48, 64, 80, 96, 112, 128]:
                for fw in [16, 32, 48, 64, 80, 96, 112, 128]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for scale in [2, 3, 4, 5]:
                            for method in ['BILINEAR', 'NEAREST']:
                                case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_scale_{}_method_{}'.format(
                                    op_name, bz, fh, fw, ic, scale, method)
                                if case_id in op_case_id:
                                    continue
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=[[bz, fh, fw, ic]],
                                                          scale=[scale, scale],
                                                          method=method)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 'concat':
        def gen_shape(shape, axis):
            low = max(1, shape[axis] // 2)
            high = shape[axis] + 1
            shape[axis] = int(np.random.randint(low, high=high, size=1)[0])
            return shape
        for bz in [1]:
            for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                    for ic in [32, 64, 96, 128, 160, 192, 224, 256]:
                        for axis in [0, 1, 2, 3]:
                            for num in [2, 3, 4, 5, 7, 8, 9, 10]:
                                case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}_num_{}'.format(
                                    op_name, bz, fh, fw, ic, axis, num)
                                if case_id in op_case_id:
                                    continue
                                in_shapes = []
                                for _ in range(num):
                                    in_shapes.append(gen_shape([bz, fh, fw, ic], axis))
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=in_shapes,
                                                          axis=axis)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 'split':
        def gen_split(max_value):
            value = np.random.randint(max(1, max_value // 2), high=max_value + 1, size=1)[0]
            return int(value)
        for bz in [1]:
            for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                    for ic in [32, 64, 96, 128, 160, 192, 224, 256]:
                        for axis in [0, 1, 2, 3]:
                            for num in [2, 3, 4, 5, 7, 8, 9, 10]:
                                case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}_num_{}'.format(
                                    op_name, bz, fh, fw, ic, axis, num)
                                if case_id in op_case_id:
                                    continue
                                in_shape = [bz, fh, fw, ic]
                                splits = []
                                for _ in range(num):
                                    splits.append(gen_split(in_shape[axis]))
                                in_shape[axis] = sum(splits)
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=[in_shape],
                                                          axis=axis,
                                                          splits=splits)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 's2d':
        for bz in [1]:
            for bs_y, bs_x in [[2, 2], [3, 3], [4, 4]]:
                for fh in [16, 32, 48, 64, 80, 96, 112, 128]:
                    for fw in [16, 32, 48, 64, 80, 96, 112, 128]:
                        for ic in [4, 8, 12, 16, 20, 24, 28, 32]:
                            in_fh = fh * bs_y
                            in_fw = fw * bs_x
                            case_id = '{}_bz_{}_bs_y_{}_bs_x_{}_fh_{}_fw_{}_ic_{}'.format(
                                op_name, bz, bs_y, bs_x, in_fh, in_fw, ic)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name, 
                                                      target, 
                                                      in_shapes=[[bz, in_fh, in_fw, ic]],
                                                      block_size=[bs_y, bs_x])
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'gather':
        def get_indexs(shape, axis, dim):
            max_value = max(1, shape[axis] // 2)
            size = []
            for _ in range(dim):
                size.append(int(np.random.randint(1, high=max_value + 1, size=1)[0]))
            return np.random.randint(0, high=shape[axis], size=size)
        for bz in [1, 2, 4, 8]:
            for ic in [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 
                       272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512]:
                for axis in [0, 1]:
                    for dim in [1, 2]:
                        case_id = '{}_bz_{}_ic_{}_axis_{}_dim_{}'.format(
                            op_name, bz, ic, axis, dim)
                        if case_id in op_case_id:
                            continue
                        in_shape = [bz, ic]
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[in_shape],
                                                  indexs=get_indexs(in_shape, axis, dim),
                                                  axis=axis)
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
        for bz in [1]:
            for fz in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for axis in [0, 1, 2]:
                        for dim in [1, 2]:
                            case_id = '{}_bz_{}_fz_{}_ic_{}_axis_{}_dim_{}'.format(
                                op_name, bz, fz, ic, axis, dim)
                            if case_id in op_case_id:
                                continue
                            in_shape = [bz, fz, ic]
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[in_shape],
                                                      indexs=get_indexs(in_shape, axis, dim),
                                                      axis=axis)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'transpose':
        for bz in [1, 2, 4, 8]:
            for fz in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for perm in [[2, 0, 1]]:
                        case_id = '{}_bz_{}_fz_{}_ic_{}_perm_{}'.format(
                            op_name, bz, fz, ic, perm)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fz, ic]],
                                                  perm=perm)
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
        for bz in [1, 2, 4, 8]:
            for fh in [16, 32, 48, 64, 80, 96, 112, 128]:
                for fw in [16, 32, 48, 64, 80, 96, 112, 128]:
                    for ic in [16, 32, 48, 64, 80, 96, 112, 128]:
                        for perm in [[3, 1, 2, 0], [0, 2, 1, 3], [0, 2, 3, 1]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_perm_{}'.format(
                                op_name, bz, fh, fw, ic, perm)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      perm=perm)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
        for bz in [1, 2, 4, 8]:
            for ic in [4, 8, 12, 16, 20, 24, 28, 32]:
                for fh in [8, 16, 24, 32, 40, 48, 56, 64]:
                    for fw in [8, 16, 24, 32, 40, 48, 56, 64]:
                        for cx in [8, 16, 32]:
                            for perm in [[0, 1, 3, 2, 4], [0, 3, 1, 2, 4]]:
                                case_id = '{}_bz_{}_ic_{}_fh_{}_fw_{}_cx_{}_perm_{}'.format(
                                    op_name, bz, ic, fh, fw, cx, perm)
                                if case_id in op_case_id:
                                    continue
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=[[bz, ic, fh, fw, cx]],
                                                          perm=perm)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 'layoutcvt':
        def convert_shape(shape, layout):
            if layout == 'NCHW':
                shape = [shape[0], shape[3], shape[1], shape[2]]
            elif layout == 'NCHWC16':
                shape = [shape[0], shape[3] // 16, shape[1], shape[2], 16]
            elif layout == 'NCHWC32':
                shape = [shape[0], shape[3] // 32, shape[1], shape[2], 32]
            return shape
        for bz in [1, 2, 4, 8]:
            for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                    for ic in [64, 128, 192, 256, 320, 384, 448, 512]:
                        for from_layout, to_layout in [['NHWC', 'NCHW'],
                                                       ['NHWC', 'NCHWC16'],
                                                       ['NHWC', 'NCHWC32'],
                                                       ['NCHW', 'NHWC'],
                                                       ['NCHW', 'NCHWC16'],
                                                       ['NCHW', 'NCHWC32'],
                                                       ['NCHWC16', 'NHWC'],
                                                       ['NCHWC16', 'NCHW'],
                                                       ['NCHWC32', 'NHWC'],
                                                       ['NCHWC32', 'NCHW']]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_from_{}_to_{}'.format(
                                op_name, bz, fh, fw, ic, from_layout, to_layout)
                            if case_id in op_case_id:
                                continue
                            in_shape = convert_shape([bz, fh, fw, ic], from_layout)
                            out_shape = convert_shape([bz, fh, fw, ic], to_layout)
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[in_shape],
                                                      out_shapes=[out_shape],
                                                      layouts=[from_layout, to_layout])
                            case_info[case_id] = get_case_data(node_info, skip_cvt=False)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'l1norm':
        for bz in [1]:
            for fh in [16, 32, 48, 64, 80, 96, 112, 128]:
                for fw in [16, 32, 48, 64, 80, 96, 112, 128]:
                    for ic in [16, 32, 48, 64, 80, 96, 112, 128]:
                        for axis in [[0]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}'.format(
                                op_name, bz, fh, fw, ic, axis)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      axis=axis)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'l2norm':
        for bz in [1]:
            for fh in [8, 16, 24, 32, 40, 48, 56, 64]:
                for fw in [8, 16, 24, 32, 40, 48, 56, 64]:
                    for ic in [8, 16, 24, 32, 40, 48, 56, 64]:
                        for axis in [[0]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}'.format(
                                op_name, bz, fh, fw, ic, axis)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      axis=axis)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'bn':
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for axis in [3]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}'.format(
                                op_name, bz, fh, fw, ic, axis)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      axis=axis)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'gn':
        for bz in [1]:
            for fh in [64, 128, 192, 256, 320, 384, 448, 512]:
                for fw in [64, 128, 192, 256, 320, 384, 448, 512]:
                    for ic in [64, 128, 192, 256, 320, 384, 448, 512]:
                        for axis in [3]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}'.format(
                                op_name, bz, fh, fw, ic, axis)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      axis=axis,
                                                      group=32)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'ln':
        for bz in [1]:
            for fh in [16, 32, 48, 64, 80, 96, 112, 128]:
                for fw in [16, 32, 48, 64, 80, 96, 112, 128]:
                    for ic in [16, 32, 48, 64, 80, 96, 112, 128]:
                        for axis in [[1, 2, 3]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}'.format(
                                op_name, bz, fh, fw, ic, axis)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      axis=axis)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'in':
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(
                            op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fh, fw, ic]])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name == 'gs':
        for bz in [1]:
            for fh in [16, 32, 48, 64, 80, 96, 112, 128]:
                for fw in [16, 32, 48, 64, 80, 96, 112, 128]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for method in ['bilinear']:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_method_{}'.format(
                                op_name, bz, fh, fw, ic, method)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic], [bz, fh, fw, 2]],
                                                      method=method)
                            case_info[case_id] = get_case_data(node_info, sg_name='subgraph_gs')
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'slice':
        def gen_slice(max_value, slice_ratio=0.25):
            max_slice = int(max_value * slice_ratio)
            low = np.random.randint(0, high=max_slice + 1, size=1)[0]
            min_high = max(low + 1, max_value - max_slice)
            high = np.random.randint(min_high, high=max_value + 1, size=1)[0]
            return [int(low), int(high)]
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for slices in [[[0, bz], gen_slice(fh), [0, fw], [0, ic]],
                                       [[0, bz], [0, fh], gen_slice(fw), [0, ic]],
                                       [[0, bz], [0, fh], [0, fw], gen_slice(ic)]]:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_slices_{}'.format(
                                op_name, bz, fh, fw, ic, slices)
                            if case_id in op_case_id:
                                continue
                            begin = [s[0] for s in slices]
                            end = [s[1] for s in slices]
                            strides = [1] * len(slices)
                            node_info = run_simulator(op_name,
                                                      target,
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      begin=begin,
                                                      end=end,
                                                      strides=strides)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    elif op_name == 'reduce':
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for axis in [[1], [2], [3]]:
                            for method in ['MEAN']:
                                case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_axis_{}_method_{}'.format(
                                    op_name, bz, fh, fw, ic, axis, method)
                                if case_id in op_case_id:
                                    continue
                                node_info = run_simulator(op_name,
                                                          target,
                                                          in_shapes=[[bz, fh, fw, ic]],
                                                          axis=axis,
                                                          method=method)
                                case_info[case_id] = get_case_data(node_info)
                                if len(case_info) % 100 == 0:
                                    op_case_id = update_op_dataset(op_name, target, case_info)
                                    case_info = dict()
    elif op_name == 'deconv2d':
        for bz in [1]:
            for ky, kx in [[2, 2], [3, 3]]:
                for sy, sx in [[2, 2]]:
                    for fh in [32, 64, 96, 128, 160, 192, 224, 256]:
                        for fw in [32, 64, 96, 128, 160, 192, 224, 256]:
                            for ic in [32, 64, 96, 128, 160, 192, 224, 256]:
                                for oc in [32, 64, 96, 128, 160, 192, 224, 256]:
                                    case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_oc_{}_ky_{}_kx_{}_sy_{}_sx_{}'.format(
                                        op_name, bz, fh, fw, ic, oc, ky, kx, sy, sx)
                                    if case_id in op_case_id:
                                        continue
                                    node_info = run_simulator(op_name,
                                                              target,
                                                              in_shapes=[[bz, fh, fw, ic]],
                                                              output_c=oc,
                                                              group=1,
                                                              kernel_y=ky,
                                                              kernel_x=kx,
                                                              stride_y=sy,
                                                              stride_x=sx,
                                                              dilation_y=1,
                                                              dilation_x=1)
                                    case_info[case_id] = get_case_data(node_info)
                                    if len(case_info) % 100 == 0:
                                        op_case_id = update_op_dataset(op_name, target, case_info)
                                        case_info = dict()
    elif op_name == 'lstm':
        for bz in [1, 2, 3, 4, 5, 6, 7, 8]:
            for ic in [64, 128, 192, 256, 320, 384, 448, 512]:
                for oc in [64, 128, 192, 256, 320, 384, 448, 512]:
                    case_id = '{}_bz_{}_ic_{}_oc_{}'.format(op_name, bz, ic, oc)
                    if case_id in op_case_id:
                        continue
                    node_info = run_simulator(op_name,
                                              target,
                                              in_shapes=[[bz, 1, ic], [bz, oc], [bz, oc]])
                    case_info[case_id] = get_case_data(node_info)
                    if len(case_info) % 100 == 0:
                        op_case_id = update_op_dataset(op_name, target, case_info)
                        case_info = dict()
    elif op_name == 'cs':
        for bz in [1]:
            for fz in [64, 128, 192, 256, 320, 384, 448, 512]:
                for ic in [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 
                           1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 
                           3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968, 4096]:
                    for group in [ic]:
                        case_id = '{}_bz_{}_fz_{}_ic_{}_group_{}'.format(op_name, bz, fz, ic, group)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name,
                                                  target,
                                                  in_shapes=[[bz, fz, ic]],
                                                  group=group,
                                                  axis=2)
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name in ['sine', 'cosine']:
        for bz in [1]:
            for fh in [16, 32, 48, 64, 80, 96, 112, 128]:
                for fw in [16, 32, 48, 64, 80, 96, 112, 128]:
                    for ic in [16, 32, 48, 64, 80, 96, 112, 128]:
                        case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}'.format(op_name, bz, fh, fw, ic)
                        if case_id in op_case_id:
                            continue
                        node_info = run_simulator(op_name, target, in_shapes=[[bz, fh, fw, ic]])
                        case_info[case_id] = get_case_data(node_info)
                        if len(case_info) % 100 == 0:
                            op_case_id = update_op_dataset(op_name, target, case_info)
                            case_info = dict()
    elif op_name in ['quantize', 'dequantize']:
        for bz in [1]:
            for fh in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                for fw in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                    for ic in [16, 32, 64, 96, 128, 192, 256, 320, 384, 448]:
                        for dtype in ['int8', 'int16']:
                            case_id = '{}_bz_{}_fh_{}_fw_{}_ic_{}_dtype_{}'.format(
                                op_name, bz, fh, fw, ic, dtype)
                            if case_id in op_case_id:
                                continue
                            node_info = run_simulator(op_name, 
                                                      target, 
                                                      in_shapes=[[bz, fh, fw, ic]],
                                                      dtype=dtype)
                            case_info[case_id] = get_case_data(node_info)
                            if len(case_info) % 100 == 0:
                                op_case_id = update_op_dataset(op_name, target, case_info)
                                case_info = dict()
    else:
        raise ValueError(f'Unknown op_name: {op_name}')
    update_op_dataset(op_name, target, case_info)


def main():
    args = argparse.ArgumentParser('Dataset')
    args.add_argument('--op_name',
                      type=str,
                      default='',
                      required=True,
                      help='Operator name to collect data')
    args.add_argument('--target',
                      type=str,
                      default='X2_1204',
                      required=True,
                      help='AIPU target of operator')
    opt = args.parse_args()
    assert opt.target in ['X1_1204', 'X2_1204']
    gen_op_dataset(opt.op_name, opt.target)


if __name__ == '__main__':
    main()
