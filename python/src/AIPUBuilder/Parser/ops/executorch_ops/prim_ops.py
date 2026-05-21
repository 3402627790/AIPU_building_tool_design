# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.
import numpy as np
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Executorch_primaddOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Executorch_primaddOp, self).__init__(graph, attr_dict)
        self.update_attributes(Executorch_primaddOp, attr_dict)
        assert self.check_required(), 'Executorch_primaddOp is missing a required parameter.'

    def infer_shape(self):
        super(Executorch_primaddOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.add(*inputs)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 11}


class Executorch_primet_viewOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Executorch_primet_viewOp, self).__init__(graph, attr_dict)
        self.update_attributes(Executorch_primet_viewOp, attr_dict)
        assert self.check_required(), 'Executorch_primet_viewOp is missing a required parameter.'

    def __getattr__(self, item):
        ret = None
        try:
            if item == 'size':
                inputs = self.get_input_tensors()
                ret = inputs[1].tolist()
        except:
            ret = None
        if ret is None:
            ret = super(Executorch_primet_viewOp, self).__getattr__(item)
        return ret

    def infer_shape(self):
        super(Executorch_primet_viewOp, self).infer_shape()
        inputs = self.get_input_tensors()
        inp_shape = inputs[0].shape
        new_shape = self.size.copy()
        if 0 in new_shape:
            idx = new_shape.index(0)
            new_shape[idx] = inp_shape[idx]
        output_tensor = inputs[0].reshape(new_shape)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Reshape', 'version': 5}
