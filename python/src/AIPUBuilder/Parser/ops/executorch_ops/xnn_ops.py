# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.
import numpy as np
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class XNNAbsOp(SameShapeOp, LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNAbsOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNAbsOp, attr_dict)
        assert self.check_required(), 'XNNAbsOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNAbsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.abs(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Abs', 'version': 6}


class XNNAddOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNAddOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNAddOp, attr_dict)
        assert self.check_required(), 'XNNAddOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNAddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 2, 'The input length is invalid in XNNAddOp infer shape.'
        out_tensor = np.add(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 7}


class XNNAvgPooling2dOp(OpHasPaddingStrides, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'kernel_shape': {'type': AttrType.INTS, 'required': True},
                'dilations': {'type': AttrType.INTS, 'required': True},
                'strides': {'type': AttrType.INTS, 'required': True},
                'padding': {'type': AttrType.INTS, 'required': True},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNAvgPooling2dOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNAvgPooling2dOp, attr_dict)
        assert self.check_required(), 'XNNAvgPooling2dOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNAvgPooling2dOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.is_all_inputs_const():
            padding = np.reshape(np.array(self.padding), (4, 2)).tolist()
            out_tensor = tf.nn.avg_pool(inputs[0],
                                        ksize=self.kernel_shape,
                                        strides=[1] + self.strides + [1],
                                        padding=padding,
                                        data_format='NHWC').numpy()
        else:
            input_symbol = self.get_input_symbols(local=False if self.ds_mode else True)[0]
            symbol_map = dict(zip(input_symbol, list(inputs[0].shape)))
            in_symbol = input_symbol[1:-1]
            spatial_padding = np.transpose(np.array(self.padding)).flatten().tolist()
            out_symbol = BaseOnnxPoolOp.cal_out_shape(
                in_symbol, spatial_padding, self.strides, self.kernel_shape, 'NOTSET', dilations=self.dilations,
                ceil_mode=False)
            output_symbol = input_symbol[:1] + out_symbol + input_symbol[-1:]
            out_tensor_shape = [s.subs(symbol_map) if not isinstance(s, int) else s for s in output_symbol]
            out_tensor = np.random.ranf(out_tensor_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'AveragePool', 'version': 11}


class XNNBatchMatrixMultiplyOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {
                    'flag': {'type': AttrType.INT, 'default': 0},
                }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNBatchMatrixMultiplyOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNBatchMatrixMultiplyOp, attr_dict)
        assert self.check_required(), 'XNNBatchMatrixMultiplyOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNBatchMatrixMultiplyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'The input length is invalid in XNNBatchMatrixMultiplyOp infer shape.'

        input0, input1 = inputs
        if (self.flag & 0x2) != 0:
            input0 = np.swapaxes(input0, -1, -2)
        if (self.flag & 0x1) != 0:
            input1 = np.swapaxes(input1, -1, -2)
        out_tensor = np.matmul(input0, input1)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MatMul', 'version': 9}


class XNNCeilingOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNCeilingOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNCeilingOp, attr_dict)
        assert self.check_required(), 'XNNCeilingOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNCeilingOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.ceil(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Ceil', 'version': 13}


class XNNClampOp(SameShapeOp, LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'OutputMin': {'type': AttrType.FLOAT, 'default': -3.402823e+38},
                'OutputMax': {'type': AttrType.FLOAT, 'default': 3.402823e+38},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNClampOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNClampOp, attr_dict)
        assert self.check_required(), 'XNNClampOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNClampOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.clip(inputs[0], self.OutputMin, self.OutputMax)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Clip', 'version': 6}


class XNNConcatOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {
                    'axis': {'type': AttrType.INT, 'required': True},
                }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNConcatOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNConcatOp, attr_dict)
        assert self.check_required(), 'XNNConcatOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNConcatOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.concatenate(inputs, axis=self.axis)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Concat', 'version': 11}


class XNNConcatenate2Op(XNNConcatOp):
    pass


class XNNConcatenate3Op(XNNConcatOp):
    pass


class XNNConcatenate4Op(XNNConcatOp):
    pass


class XNNConcatenate5Op(XNNConcatOp):
    pass


class XNNConv2dOp(BaseConvOp, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {
                    'kernel_shape': {'type': AttrType.INTS, 'required': True},
                    'dilations': {'type': AttrType.INTS, 'required': True},
                    'strides': {'type': AttrType.INTS, 'required': True},
                    'padding': {'type': AttrType.INTS, 'required': True},
                    'adj': {'type': AttrType.INTS, 'required': True},
                    'group': {'type': AttrType.INT, 'required': True},
                    'group_ic': {'type': AttrType.INT, 'required': True},
                    'group_oc': {'type': AttrType.INT, 'required': True},
                }
        }

    @classmethod
    def perm_xnn_to_onnx(cls):
        return [0, 3, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(XNNConv2dOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNConv2dOp, attr_dict)
        assert self.check_required(), 'XNNConv2dOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNConv2dOp, self).infer_shape()
        inputs = self.get_input_tensors()

        self.auto_pad = 'NOTSET'

        spatial_padding = np.transpose(np.array(self.padding)).flatten().tolist()
        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[1:-1],
                                             spatial_padding,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NHWC')
        full_out_shape = [inputs[0].shape[0]] + out_shape + [self.weights.shape[0]]
        out_tensor = np.random.ranf(size=full_out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}


class XNNCopyOp(SameShapeOp, LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNCopyOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNCopyOp, attr_dict)
        assert self.check_required(), 'XNNCopyOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNCopyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0], copy=True)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Identity', 'version': 1}


class XNNCosOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNCosOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNCosOp, attr_dict)
        assert self.check_required(), 'XNNCosOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNCosOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.cos(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Cos', 'version': 7}


class XNNDepthwiseConv2dOp(BaseConvOp, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {
                    'kernel_shape': {'type': AttrType.INTS, 'required': True},
                    'dilations': {'type': AttrType.INTS, 'required': True},
                    'strides': {'type': AttrType.INTS, 'required': True},
                    'padding': {'type': AttrType.INTS, 'required': True},
                    'adj': {'type': AttrType.INTS, 'required': True},
                    'group': {'type': AttrType.INT, 'required': True},
                    'group_ic': {'type': AttrType.INT, 'required': True},
                    'group_oc': {'type': AttrType.INT, 'required': True},
                }
        }

    @classmethod
    def perm_xnn_to_onnx(cls):
        return [3, 0, 1, 2]

    def __init__(self, graph, attr_dict=None):
        super(XNNDepthwiseConv2dOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNDepthwiseConv2dOp, attr_dict)
        assert self.check_required(), 'XNNDepthwiseConv2dOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNDepthwiseConv2dOp, self).infer_shape()
        inputs = self.get_input_tensors()
        '''
        # out_tensor = tf.nn.depthwise_conv2d(inputs[0],
        #                                     np.transpose(self.weights, axes=type(self).perm_lite_to_tf(self.multiplier)),
        #                                     strides=[1] + self.strides + [1],
        #                                     padding='VALID' if self.auto_pad in ('VALID', 'NOTSET') else 'SAME',
        #                                     data_format='NHWC',
        #                                     rate=self.dilations)
        # out_tensor = tf.nn.bias_add(out_tensor, self.biases, data_format='NHWC').numpy()
        # out_tensor = self.cal_activation(out_tensor)
        '''
        self.auto_pad = 'NOTSET'
        spatial_padding = np.transpose(np.array(self.padding)).flatten().tolist()
        out_shape = BaseConvOp.cal_out_shape(inputs[0].shape[1:-1],
                                             spatial_padding,
                                             self.strides,
                                             self.kernel_shape,
                                             self.auto_pad,
                                             dilations=self.dilations,
                                             data_format='NHWC')
        out_shape = [inputs[0].shape[0]] + out_shape + [self.num_output]
        out_tensor = np.random.ranf(size=out_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Conv', 'version': 1}

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'num_output':
                if self.__dict__['_attr'][item].value is not None:
                    ret = self.__dict__['_attr'][item].value
                elif self.weights is not None:
                    ret = self.weights.shape[-1]
                    self.__dict__['_attr'][item].value = ret
                elif self.biases is not None:
                    ret = self.biases.shape[0]
                    self.__dict__['_attr'][item].value = ret
                else:
                    ret = None
        except:
            ret = None
        if ret is None:
            ret = super(XNNDepthwiseConv2dOp, self).__getattr__(item)
        return ret


class XNNELUOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'Alpha': {'type': AttrType.FLOAT, 'default': 1.0},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNELUOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNELUOp, attr_dict)
        assert self.check_required(), 'XNNELUOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNELUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.array(inputs[0])
        mask = out_tensor <= 0
        out_tensor[mask] = self.Alpha * (np.exp(out_tensor[mask]) - 1)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Elu', 'version': 6}


class XNNExpOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNExpOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNExpOp, attr_dict)
        assert self.check_required(), 'XNNExpOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNExpOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.exp(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Exp', 'version': 13}


class XNNFloorOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNFloorOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNFloorOp, attr_dict)
        assert self.check_required(), 'XNNFloorOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNFloorOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.floor(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Floor', 'version': 13}


class XNNFullyConnectedOp(BaseLinearOp, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {'flag': {'type': AttrType.INT, 'required': True}},
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNFullyConnectedOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNFullyConnectedOp, attr_dict)
        assert self.check_required(), 'XNNFullyConnectedOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNFullyConnectedOp, self).infer_shape()
        inputs = self.get_input_tensors()
        filter_shape = self.weights.shape if self.weights is not None else inputs[1].shape
        if self.is_all_inputs_const():
            if self.weights is not None:
                inp = np.reshape(inputs[0], (-1, self.weights.shape[-1]))
                out_tensor = np.matmul(inp, np.transpose(
                    self.weights, axes=type(self).perm_lite_to_tf()) if self.flag == 0 else self.weights)
            else:
                inp = np.reshape(inputs[0], (-1, inputs[1].shape[-1]))
                out_tensor = np.matmul(inp, np.transpose(inputs[1]) if self.flag == 0 else inputs[1])
            if self.biases is not None:
                out_tensor = out_tensor + self.biases
        else:
            assert inputs[0].shape[-1] == filter_shape[-1] if self.flag == 0 else filter_shape[0], 'input0 shape[-1] MUST be equal to filter shape[-1].'
            out_shape = list(inputs[0].shape[:-1]) + [filter_shape[0] if self.flag == 0 else filter_shape[-1]]
            out_tensor = np.random.ranf(tuple(out_shape)).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


class XNNGeluOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNGeluOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNGeluOp, attr_dict)
        assert self.check_required(), 'XNNGeluOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNGeluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.nn.gelu(inputs[0].astype(np.float32), approximate=False).numpy()
        self.set_out_tensor(out_tensor.astype(inputs[0].dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'Gelu', 'version': 20}


class XNNGlobalAvgPooling2dOp(LayoutConcernedOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNGlobalAvgPooling2dOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNGlobalAvgPooling2dOp, attr_dict)
        assert self.check_required(), 'XNNGlobalAvgPooling2dOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNGlobalAvgPooling2dOp, self).infer_shape()
        inputs = self.get_input_tensors()
        axes = [1, 2]
        out_tensor = np.mean(inputs[0], axis=tuple(axes), keepdims=True)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'GlobalAveragePool', 'version': 1}


class XNNHardswishOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNHardswishOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNHardswishOp, attr_dict)
        assert self.check_required(), 'XNNHardswishOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNHardswishOp, self).infer_shape()
        inputs = self.get_input_tensors()
        x = inputs[0]
        relu6 = np.minimum(np.maximum(x + 3, 0), 6)
        out_tensor = x * relu6 / 6
        self.set_out_tensor(out_tensor.astype(x.dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'HardSwish', 'version': 1}


class XNNLeakyReLUOp(BaseReluOp, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'NegativeSlope': {'type': AttrType.FLOAT, 'default': 0.0},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNLeakyReLUOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNLeakyReLUOp, attr_dict)
        assert self.check_required(), 'XNNLeakyReLUOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNLeakyReLUOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = inputs[0]
        out_tensor = np.where(inp > 0, inp, inp * self.NegativeSlope)
        self.set_out_tensor(out_tensor.astype(inp.dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'LeakyRelu', 'version': 6}


class XNNLogOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNLogOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNLogOp, attr_dict)
        assert self.check_required(), 'XNNLogOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNLogOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.log(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Log', 'version': 13}


class XNNMaximumOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNMaximumOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNMaximumOp, attr_dict)
        assert self.check_required(), 'XNNMaximumOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNMaximumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'The input length is invalid in XNNMaximumOp infer shape.'
        out_tensor = np.maximum(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Max', 'version': 6}


class XNNMaxPooling2dOp(OpHasPaddingStrides, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {
                    'kernel_shape': {'type': AttrType.INTS, 'required': True},
                    'dilations': {'type': AttrType.INTS, 'required': True},
                    'strides': {'type': AttrType.INTS, 'required': True},
                    'padding': {'type': AttrType.INTS, 'required': True},
                }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNMaxPooling2dOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNMaxPooling2dOp, attr_dict)
        assert self.check_required(), 'XNNMaxPooling2dOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNMaxPooling2dOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if self.is_all_inputs_const():
            padding = np.reshape(np.array(self.padding), (4, 2)).tolist()
            out_tensor = tf.nn.max_pool(inputs[0],
                                        ksize=self.kernel_shape,
                                        strides=[1] + self.strides + [1],
                                        padding=padding,
                                        data_format='NHWC').numpy()
        else:
            input_symbol = self.get_input_symbols(local=False if self.ds_mode else True)[0]
            symbol_map = dict(zip(input_symbol, list(inputs[0].shape)))
            in_symbol = input_symbol[1:-1]
            spatial_padding = np.transpose(np.array(self.padding)).flatten().tolist()
            out_symbol = BaseOnnxPoolOp.cal_out_shape(
                in_symbol, spatial_padding, self.strides, self.kernel_shape, 'NOTSET', dilations=self.dilations,
                ceil_mode=False)
            output_symbol = input_symbol[:1] + out_symbol + input_symbol[-1:]
            out_tensor_shape = [s.subs(symbol_map) if not isinstance(s, int) else s for s in output_symbol]
            out_tensor = np.random.ranf(out_tensor_shape).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'MaxPool', 'version': 11}


class XNNMinimumOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNMinimumOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNMinimumOp, attr_dict)
        assert self.check_required(), 'XNNMinimumOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNMinimumOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'The input length is invalid in XNNMinimumOp infer shape.'
        out_tensor = np.minimum(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Min', 'version': 6}


class XNNMultiplyOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNMultiplyOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNMultiplyOp, attr_dict)
        assert self.check_required(), 'XNNMultiplyOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNMultiplyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(
            inputs) == 2, 'The input length is invalid in XNNMultiplyOp infer shape.'
        out_tensor = np.multiply(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Mul', 'version': 7}


class XNNNegateOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNNegateOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNNegateOp, attr_dict)
        assert self.check_required(), 'XNNNegateOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNNegateOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.negative(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Neg', 'version': 6}


class XNNPowOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNPowOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNPowOp, attr_dict)
        assert self.check_required(), 'XNNPowOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNPowOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) in (1, 2), 'The input length is invalid in XNNPowOp infer shape.'
        if len(inputs) == 1:
            out_tensor = np.power(inputs[0], 2)
        else:
            out_tensor = np.power(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pow', 'version': 7}


class XNNReluOp(BaseReluOp, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNReluOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNReluOp, attr_dict)
        assert self.check_required(), 'XNNReluOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNReluOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.maximum(inputs[0], 0)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Relu', 'version': 6}


class XNNReciprocalSquareRootOp(SameShapeOp, LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNReciprocalSquareRootOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNReciprocalSquareRootOp, attr_dict)
        assert self.check_required(), 'XNNReciprocalSquareRootOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNReciprocalSquareRootOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.reciprocal(np.sqrt(inputs[0]))
        self.set_out_tensor(out_tensor)


class XNNSigmoidOp(SameShapeOp, LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNSigmoidOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNSigmoidOp, attr_dict)
        assert self.check_required(), 'XNNSigmoidOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNSigmoidOp, self).infer_shape()
        inputs = self.get_input_tensors()
        input_dtype = str(inputs[0].dtype)
        inp = inputs[0].astype(np.float32) if input_dtype != 'float32' else inputs[0]
        out_tensor = tf.sigmoid(inp).numpy()
        if input_dtype != 'float32':
            out_tensor = out_tensor.astype(input_dtype)
        out_symbol = self.cal_output_symbol()
        self.set_out_tensor(out_tensor, shape_symbol=out_symbol)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sigmoid', 'version': 6}


class XNNSinOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNSinOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNSinOp, attr_dict)
        assert self.check_required(), 'XNNSinOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNSinOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sin(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sin', 'version': 7}


class XNNSoftmaxOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNSoftmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNSoftmaxOp, attr_dict)
        assert self.check_required(), 'XNNSoftmaxOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNSoftmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        x = inputs[0]
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_val)
        out_tensor = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.set_out_tensor(out_tensor.astype(x.dtype))

    @property
    def correspond_onnx_op(self):
        return {'type': 'Softmax', 'version': 13}


class XNNSquareOp(SameShapeOp, LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNSquareOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNSquareOp, attr_dict)
        assert self.check_required(), 'XNNSquareOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNSquareOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.square(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Mul', 'version': 7}


class XNNSquareRootOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNSquareRootOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNSquareRootOp, attr_dict)
        assert self.check_required(), 'XNNSquareRootOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNSquareRootOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.sqrt(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sqrt', 'version': 6}


class XNNSqueezeOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNSqueezeOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNSqueezeOp, attr_dict)
        assert self.check_required(), 'XNNSqueezeOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNSqueezeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.squeeze(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Squeeze', 'version': 13}


class XNNStaticSliceOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'offset': {'type': AttrType.INTS, 'required': True},
                'size': {'type': AttrType.INTS, 'required': True},
                'flag': {'type': AttrType.INT, 'default': 0},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNStaticSliceOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNStaticSliceOp, attr_dict)
        assert self.check_required(), 'XNNStaticSliceOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNStaticSliceOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = inputs[0]
        rank = len(inp.shape)

        assert len(self.offset) == len(self.size) and len(self.offset) > 0, \
            'The offsets/sizes are invalid in XNNStaticSliceOp infer shape.'
        assert len(self.offset) <= rank, 'The slice rank is invalid in XNNStaticSliceOp infer shape.'

        slices = [slice(None)] * rank
        for i, (start, size) in enumerate(zip(self.offset, self.size)):
            start = int(start)
            size = int(size)
            end = inp.shape[i] if size == 0 else start + size
            slices[i] = slice(start, end)

        out_tensor = inp[tuple(slices)]
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 13}


class XNNStaticReshapeOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'NewShapeAsNumpy': {'type': AttrType.INTS, 'default': []},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNStaticReshapeOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNStaticReshapeOp, attr_dict)
        assert self.check_required(), 'XNNStaticReshapeOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNStaticReshapeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(self.NewShapeAsNumpy) > 0, 'The new shape is invalid in XNNStaticReshapeOp infer shape.'
        out_tensor = np.reshape(inputs[0], self.NewShapeAsNumpy)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 13}


class XNNStaticConstantPadOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {
                'pads': {'type': AttrType.INTS, 'required': True},
                'pad_value': {'type': AttrType.FLOAT, 'default': 0.0},
                'flag': {'type': AttrType.INT, 'default': 0},
            }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNStaticConstantPadOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNStaticConstantPadOp, attr_dict)
        assert self.check_required(), 'XNNStaticConstantPadOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNStaticConstantPadOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp = inputs[0]
        rank = len(inp.shape)

        if rank * 2 != len(self.pads):
            raise RuntimeError(
                "The number of elements in pads should be 2 times the number of rank"
            )

        pad_width = [(0, 0)] * rank
        for i, axis in enumerate(list(range(rank))):
            pad_begin = self.pads[i]
            pad_end = self.pads[rank + i]
            pad_width[axis] = (pad_begin, pad_end)

        out_tensor = np.pad(
            inp, pad_width=pad_width, mode='constant', constant_values=float(self.pad_value)
        ).astype(inp.dtype)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Pad', 'version': 11}


class XNNStaticResizeBilinear2DOp(LayoutConcernedOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {
                    'new_height': {'type': AttrType.INT, 'required': True},
                    'new_width': {'type': AttrType.INT, 'required': True},
                }
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNStaticResizeBilinear2DOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNStaticResizeBilinear2DOp, attr_dict)
        assert self.check_required(), 'XNNStaticResizeBilinear2DOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNStaticResizeBilinear2DOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = tf.compat.v1.image.resize_bilinear(inputs[0], [self.new_height, self.new_width]).numpy()
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Resize', 'version': 11}


class XNNStaticTransposeOp(OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {'perm': {'type': AttrType.INTS, 'default': [], 'required': True}}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNStaticTransposeOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNStaticTransposeOp, attr_dict)
        assert self.check_required(), 'XNNStaticTransposeOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNStaticTransposeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.transpose(inputs[0], self.perm)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Transpose', 'version': 1}


class XNNSubtractOp(LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNSubtractOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNSubtractOp, attr_dict)
        assert self.check_required(), 'XNNSubtractOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNSubtractOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) == 2, 'The input length is invalid in XNNSubtractOp infer shape.'
        out_tensor = np.subtract(*inputs)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Sub', 'version': 7}


class XNNTanhOp(SameShapeOp, LayoutUnawareOp, OpHasOneOutPort, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {}
        }

    def __init__(self, graph, attr_dict=None):
        super(XNNTanhOp, self).__init__(graph, attr_dict)
        self.update_attributes(XNNTanhOp, attr_dict)
        assert self.check_required(), 'XNNTanhOp is missing a required parameter.'

    def infer_shape(self):
        super(XNNTanhOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_tensor = np.tanh(inputs[0])
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Tanh', 'version': 6}
