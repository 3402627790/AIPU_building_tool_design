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


class XNNFullyConnectedOp(BaseLinearOp, XnnOp):
    @classmethod
    def attributes(cls):
        return {
            1: {},
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
                    self.weights, axes=type(self).perm_lite_to_tf()))
            else:
                inp = np.reshape(inputs[0], (-1, inputs[1].shape[-1]))
                out_tensor = np.matmul(inp, np.transpose(inputs[1]))
            if self.biases is not None:
                out_tensor = out_tensor + self.biases
        else:
            assert inputs[0].shape[-1] == filter_shape[-1], 'input0 shape[-1] MUST be equal to filter shape[-1].'
            out_shape = list(inputs[0].shape[:-1]) + [filter_shape[-2]]
            out_tensor = np.random.ranf(tuple(out_shape)).astype(inputs[0].dtype)
        self.set_out_tensor(out_tensor)


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
