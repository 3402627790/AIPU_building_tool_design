# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025-2026 Arm Technology (China) Co. Ltd.
import numpy as np
from ..op import *
from ...logger import INFO, DEBUG, WARN, ERROR, FATAL


class Llamaupdate_cacheOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Llamaupdate_cacheOp, self).__init__(graph, attr_dict)
        self.update_attributes(Llamaupdate_cacheOp, attr_dict)
        assert self.check_required(), 'Llamaupdate_cacheOp is missing a required parameter.'

    def infer_shape(self):
        super(Llamaupdate_cacheOp, self).infer_shape()
        inputs = self.get_input_tensors()
        output_tensor = np.add(*inputs)
        self.set_out_tensor(output_tensor)

    @property
    def correspond_onnx_op(self):
        return {'type': 'Add', 'version': 11}


class Llamacustom_sdpaOp(LayoutUnawareOp, OpHasOneOutPort, AtenOp):
    @classmethod
    def attributes(cls):
        return {
            1:
                {}
        }

    def __init__(self, graph, attr_dict=None):
        super(Llamacustom_sdpaOp, self).__init__(graph, attr_dict)
        self.update_attributes(Llamacustom_sdpaOp, attr_dict)
        assert self.check_required(), 'Llamacustom_sdpaOp is missing a required parameter.'

    def infer_shape(self):
        super(Llamacustom_sdpaOp, self).infer_shape()
        inputs = self.get_input_tensors()
        q = inputs[0]   # batch, seq_len, num_heads, head_dim
        k = inputs[1]
        v = inputs[2]
        start_pos = int(inputs[3].item())
        attn_mask = inputs[4]  # 2D, q_seq, kv_seq
        drpout_p = inputs[5] if inputs[5] else 0.
        is_causal = bool(inputs[6].item()) if inputs[6] else False
        scale = inputs[7]

        if self.is_all_inputs_const():

            def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
                x_max = np.max(x, axis=axis, keepdims=True)
                tmp = np.exp(x - x_max)
                s = np.sum(tmp, axis=axis, keepdims=True)
                return tmp / s

            # Calculate Scaling Factor if not provided
            if scale is None:
                q_head_size = q.shape[3]
                scale = (1 / np.sqrt(q_head_size)).astype(q.dtype)

            q_sequence_length = q.shape[1]
            kv_sequence_length = k.shape[1]
            q_num_heads = q.shape[2]
            k_num_heads = k.shape[2]
            v_num_heads = v.shape[2]
            if ((q_num_heads != k_num_heads)
                    and (q_num_heads % k_num_heads == 0)
                    and (k_num_heads == v_num_heads)):
                seq_reps = q_num_heads // k_num_heads
                k = np.repeat(k, seq_reps, axis=2)
                v = np.repeat(v, seq_reps, axis=2)

            q_reshaped = q.transpose(0, 2, 1, 3)  # [B, H, T, D]
            k_reshaped = k.transpose(0, 2, 3, 1)  # [B, H, D, T]
            v_reshaped = v.transpose(0, 2, 1, 3)  # [B, H, T, D]

            # 计算注意力分数: [B, H, T, T]
            scores = np.matmul(q_reshaped, k_reshaped) * scale

            if is_causal:
                causal_mask = np.tril(np.ones((q_sequence_length, kv_sequence_length)), k=start_pos)
                scores = scores + (1.0 - causal_mask) * (-np.inf)

            if attn_mask is not None:
                scores = scores + (1.0 - attn_mask) * (-np.inf)

            qk_softmax = _softmax(scores)
            output = np.matmul(qk_softmax, v_reshaped).astype(q.dtype)
            # [B, H, T, D] -> [B, T, H, D]
            output = output.transpose(0, 2, 1, 3)
        else:
            output_shape = q.shape[:-1] + v.shape[-1:]
            output = np.random.ranf(output_shape).astype(q.dtype)
        self.set_out_tensor(output)
