# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.
import numpy as np

from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Aten_to_copyOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Aten_to_copyOp, self).__init__(graph, attr_dict)
        self.update_attributes(Aten_to_copyOp, attr_dict)
        assert self.check_required(), 'Aten_to_copyOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'non_blocking':
                inputs = self.get_input_tensors()
                ret = bool(inputs[1])
        except:
            ret = None
        if ret is None:
            ret = super(Aten_to_copyOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Aten_to_copyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = inputs[0]
        self.set_out_tensor(output_tensor)


class Aten_local_scalar_denseOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Aten_local_scalar_denseOp, self).__init__(graph, attr_dict)
        self.update_attributes(Aten_local_scalar_denseOp, attr_dict)
        assert self.check_required(), 'Aten_local_scalar_denseOp is missing a required parameter.'

    def infer_shape(self):
        super(Aten_local_scalar_denseOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.array(inputs[0].item())
        self.set_out_tensor(output_tensor)


class AtenabsOp(OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(AtenabsOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtenabsOp, attr_dict)
        assert self.check_required(), 'AtenabsOp is missing a required parameter.'

    def infer_shape(self):
        super(AtenabsOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.abs(inputs[0])
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Abs', 'version': 6}


class AtenaddOp(OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'alpha':
                inputs = self.get_input_tensors()
                ret = inputs[2].item() if len(inputs) > 2 else 1.0
        except:
            ret = None
        if ret is None:
            ret = super(AtenaddOp, self).__getattr__(item)
        return ret

    def __init__(self, graph, attr_dict=None):
        super(AtenaddOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtenaddOp, attr_dict)
        assert self.check_required(), 'AtenaddOp is missing a required parameter.'

    def infer_shape(self):
        super(AtenaddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.add(inputs[0], inputs[1] * self.alpha)
        self.set_out_tensor(output_tensor)


class Atenalias_copyOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Atenalias_copyOp, self).__init__(graph, attr_dict)
        self.update_attributes(Atenalias_copyOp, attr_dict)
        assert self.check_required(), 'Atenalias_copyOp is missing a required parameter.'

    def infer_shape(self):
        super(Atenalias_copyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = inputs[0]
        self.set_out_tensor(output_tensor)


class AtenarangeOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    def __init__(self, graph, attr_dict=None):
        super(AtenarangeOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtenarangeOp, attr_dict)
        assert self.check_required(), 'AtenarangeOp is missing a required parameter.'

    def infer_shape(self):
        super(AtenarangeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        if len(inputs) == 1:
            start = 0
            end = inputs[0].item()
            step = 1
        else:
            start = inputs[0].item()
            end = inputs[1].item()
            step = inputs[2].item() if len(inputs) > 2 else 1
        output_tensor = np.arange(start, end, step)
        self.set_out_tensor(output_tensor)

    def correspond_onnx_op(self):
        return {'type': 'Range', 'version': 11}


class AtencatOp(OpHasAxis, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(AtencatOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtencatOp, attr_dict)
        assert self.check_required(), 'AtencatOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axis':
                inputs = self.get_input_tensors()
                ret = int(inputs[1])
        except:
            ret = None
        if ret is None:
            ret = super(AtencatOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(AtencatOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.concatenate(inputs[:-1], axis=self.axis)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Concat', 'version': 11}


class AtenembeddingOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(AtenembeddingOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtenembeddingOp, attr_dict)
        assert self.check_required(), 'AtenembeddingOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'dim':
                inputs = self.get_input_tensors()
                ret = int(inputs[1])
        except:
            ret = None
        if ret is None:
            ret = super(AtenembeddingOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(AtenembeddingOp, self).infer_shape()
        inputs = self.get_input_tensors()
        weights, indices = torch.from_numpy(inputs[0]), torch.from_numpy(inputs[1])
        assert len(weights.shape) == 2, 'weights should be a 2D tensor.'
        output_shape = list(indices.shape) + [weights.shape[-1]]
        output_tensor = torch.index_select(weights, 0, indices.reshape(-1)).view(output_shape).numpy()
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Gather', 'version': 11}


class AtenindexOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    def __init__(self, graph, attr_dict=None):
        super(AtenindexOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtenindexOp, attr_dict)
        assert self.check_required(), 'AtenindexOp is missing a required parameter.'

    def infer_shape(self):
        super(AtenindexOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp, indices = inputs[0], inputs[1]
        output_tensor = inp[indices]
        self.set_out_tensor(output_tensor)


class AtenmaxOp(LayoutUnawareOp, OpHasVariableOutPorts, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(AtenmaxOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtenmaxOp, attr_dict)
        assert self.check_required(), 'AtenmaxOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'dim':
                inputs = self.get_input_tensors()
                ret = int(inputs[1].item())
            elif item == 'keepdim':
                inputs = self.get_input_tensors()
                ret = bool(inputs[2].item())
        except:
            ret = None
        if ret is None:
            ret = super(AtenmaxOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(AtenmaxOp, self).infer_shape()
        inputs = self.get_input_tensors()
        out_ports = self.get_out_ports()
        out_tensors = []
        if 0 in out_ports:
            output_tensor = np.max(inputs[0], axis=self.dim, keepdims=self.keepdim)
            out_tensors.append(output_tensor)
        if 1 in out_ports:
            output_tensor = np.argmax(inputs[0], axis=self.dim, keepdims=self.keepdim)
            out_tensors.append(output_tensor)
        self.set_out_tensor(out_tensors)


class Aten_native_batch_norm_legit_no_trainingOp(LayoutUnawareOp, OpHasVariableOutPorts, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Aten_native_batch_norm_legit_no_trainingOp, self).__init__(graph, attr_dict)
        self.update_attributes(Aten_native_batch_norm_legit_no_trainingOp, attr_dict)
        assert self.check_required(), 'Aten_native_batch_norm_legit_no_trainingOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            inputs = self.get_input_tensors()
            if item == 'momentum':
                ret = float(inputs[5].item()) if len(inputs) > 5 and inputs[5] is not None else 0.1
            elif item == 'epsilon':
                ret = float(inputs[6].item()) if len(inputs) > 6 and inputs[6] is not None else 1e-5
        except:
            ret = None
        if ret is None:
            ret = super(Aten_native_batch_norm_legit_no_trainingOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Aten_native_batch_norm_legit_no_trainingOp, self).infer_shape()
        inputs = self.get_input_tensors()
        assert len(inputs) >= 5, 'Aten_native_batch_norm_legit_no_trainingOp expects at least 5 inputs.'

        x = inputs[0]
        scale = inputs[1]
        bias = inputs[2]
        mean = inputs[3]
        var = inputs[4]

        calc_dtype = np.float32
        x_f = x.astype(calc_dtype)
        scale_f = scale.astype(calc_dtype)
        bias_f = bias.astype(calc_dtype)
        mean_f = mean.astype(calc_dtype)
        var_f = var.astype(calc_dtype)

        # aten batch_norm uses channel on axis=1 for rank>=2. Broadcast affine and stats to input rank.
        reshape_dims = [1] * x_f.ndim
        if x_f.ndim >= 2:
            reshape_dims[1] = scale_f.shape[0]
        else:
            reshape_dims[0] = scale_f.shape[0]

        scale_f = np.reshape(scale_f, reshape_dims)
        bias_f = np.reshape(bias_f, reshape_dims)
        mean_f = np.reshape(mean_f, reshape_dims)
        var_f = np.reshape(var_f, reshape_dims)

        y = (x_f - mean_f) / np.sqrt(var_f + self.epsilon)
        y = y * scale_f + bias_f
        y = y.astype(x.dtype, copy=False)
        self.set_out_tensor([y])

    @property
    def correspond_onnx_op(self):
        return {'type': 'BatchNormalization', 'version': 14}


class AtenmeanOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(AtenmeanOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtenmeanOp, attr_dict)
        assert self.check_required(), 'AtenmeanOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'dim':
                inputs = self.get_input_tensors()
                ret = tuple(inputs[1].tolist())
            elif item == 'keepdim':
                inputs = self.get_input_tensors()
                ret = bool(inputs[2])
        except:
            ret = None
        if ret is None:
            ret = super(AtenmeanOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(AtenmeanOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.mean(inputs[0], axis=self.dim, keepdims=self.keepdim)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'ReduceMean', 'version': 11}


class Atenselect_copyOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Atenselect_copyOp, self).__init__(graph, attr_dict)
        self.update_attributes(Atenselect_copyOp, attr_dict)
        assert self.check_required(), 'Atenselect_copyOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'dim':
                inputs = self.get_input_tensors()
                ret = int(inputs[1])
        except:
            ret = None
        if ret is None:
            ret = super(Atenselect_copyOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Atenselect_copyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.take(inputs[0], inputs[2], self.dim)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Gather', 'version': 11}


class Atenslice_copyOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Atenslice_copyOp, self).__init__(graph, attr_dict)
        self.update_attributes(Atenslice_copyOp, attr_dict)
        assert self.check_required(), 'Atenslice_copyOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'dim':
                inputs = self.get_input_tensors()
                ret = int(inputs[1])
            elif item == 'start':
                inputs = self.get_input_tensors()
                ret = int(inputs[2])
            elif item == 'end':
                inputs = self.get_input_tensors()
                ret = int(inputs[3])
            elif item == 'step':
                inputs = self.get_input_tensors()
                ret = int(inputs[4])
        except:
            ret = None
        if ret is None:
            ret = super(Atenslice_copyOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Atenslice_copyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        slices = [slice(0, a) for a in inputs[0].shape]
        slices[self.dim] = slice(self.start, self.end, self.step)
        output_tensor = inputs[0][tuple(slices)]
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Slice', 'version': 11}


class AtensubOp(OpHasOneOutPort, AtenOp):
    def __getattr__(self, item):
        ret = None
        try:
            if item == 'alpha':
                inputs = self.get_input_tensors()
                ret = inputs[2].item() if len(inputs) > 2 else 1.0
        except:
            ret = None
        if ret is None:
            ret = super(AtensubOp, self).__getattr__(item)
        return ret

    def __init__(self, graph, attr_dict=None):
        super(AtensubOp, self).__init__(graph, attr_dict)
        self.update_attributes(AtensubOp, attr_dict)
        assert self.check_required(), 'AtensubOp is missing a required parameter.'

    def infer_shape(self):
        super(AtensubOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.subtract(inputs[0], inputs[1] * self.alpha)
        self.set_out_tensor(output_tensor)


class Atensym_sizeOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Atensym_sizeOp, self).__init__(graph, attr_dict)
        self.update_attributes(Atensym_sizeOp, attr_dict)
        assert self.check_required(), 'Atensym_sizeOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'dim':
                inputs = self.get_input_tensors()
                ret = int(inputs[1])
        except:
            ret = None
        if ret is None:
            ret = super(Atensym_sizeOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Atensym_sizeOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp_shape = inputs[0].shape
        out_tensor = np.array(inp_shape[self.dim], dtype=np.int64)
        self.set_out_tensor(out_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Shape', 'version': 15}


class Atenunsqueeze_copyOp(OpHasAxis, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Atenunsqueeze_copyOp, self).__init__(graph, attr_dict)
        self.update_attributes(Atenunsqueeze_copyOp, attr_dict)
        assert self.check_required(), 'Atenunsqueeze_copyOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'axes':
                inputs = self.get_input_tensors()
                ret = [int(inputs[1])]
        except:
            ret = None
        if ret is None:
            ret = super(Atenunsqueeze_copyOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Atenunsqueeze_copyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.expand_dims(inputs[0], self.axes)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Unsqueeze', 'version': 11}


class Dim_order_ops_to_dim_order_copyOp(OpHasOneOutPort, AtenOp):

    def __init__(self, graph, attr_dict=None):
        super(Dim_order_ops_to_dim_order_copyOp, self).__init__(graph, attr_dict)
        self.update_attributes(Dim_order_ops_to_dim_order_copyOp, attr_dict)
        assert self.check_required(), 'Dim_order_ops_to_dim_order_copyOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'perm':
                inputs = self.get_input_tensors()
                if len(inputs) > 2:
                    ret = inputs[2].tolist()
                else:
                    inp_rank = len(inputs[0].shape)
                    ret = list(range(inp_rank))
        except:
            ret = None
        if ret is None:
            ret = super(Dim_order_ops_to_dim_order_copyOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Dim_order_ops_to_dim_order_copyOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.transpose(inputs[0], self.perm)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Transpose', 'version': 13}
