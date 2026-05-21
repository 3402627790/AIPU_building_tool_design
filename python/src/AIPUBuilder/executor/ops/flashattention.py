import torch
import numpy as np
from ..engine import register_gt
from AIPUBuilder.core import OpType, Dtype
from ..engine import register_gt, _OP_METRICS, register_op_metric
from .gt_util.metric import relative_diff, ulp_diff, cos_distance, check_fp_data
from .gt_util.dtype_utils import torch_to_numpy

@register_gt(OpType.FlashAttention)
def flashattention(self, *args, **kwargs):
    """
        Q, K, V shape=N*d, Q@K.T shape=N*N, (Q@K.T)@V shape=N*d
        FlashAttention only support float16 data type, no quantization
    """
    Q = self.inputs[0].betensor.squeeze()
    K = self.inputs[1].betensor.squeeze()
    V = self.inputs[2].betensor.squeeze()
    assert Q.dtype == torch.float16
    assert K.dtype == torch.float16
    assert V.dtype == torch.float16
    Q = Q.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)

    mask = self.constants['mask'].betensor
    scale_d = self.params['scale_d']
    trans_a = self.params['trans_a']
    trans_b = self.params['trans_b']
    trans_c = self.params['trans_c']
    Br = self.params['block_r']
    Bc = self.params['block_c']

    if trans_b:
        K = K.T

    d = Q.shape[-1]  # column
    N = Q.shape[-2]  # row
    assert N % Bc == 0 and N % Br == 0

    O = torch.zeros((N, d), dtype=torch.float32)  # intermediate result fp32
    for i in range(N // Br):
        lbi = i * Br
        lbi_1 = (i + 1) * Br

        sum_last = torch.zeros((Br, 1)).to(O.dtype)
        max_last = torch.tensor([-torch.inf] * Br).reshape((Br, 1)).to(O.dtype)
        O_last = O[lbi: lbi_1, :]
        for j in range(N // Bc):
            tbj = j * Bc
            tbj_1 = (j + 1) * Bc

            # S.shape = (Br, Bc)
            # 1 AIFF: MTP matmul + ITP per-layer mul + ITP per-data add
            S = Q[lbi: lbi_1, :] @ K[:, tbj: tbj_1]
            S = S * scale_d
            S = S + mask[lbi: lbi_1, tbj: tbj_1]

            # max_last.shape = (Br, 1)
            # max_current.shape = (Br, 1)
            # 2 AIFF: ITP rdc surf max
            # 3 AIFF: ITP alu per-surf max
            max_current = torch.max(torch.cat((S, max_last), dim=-1), dim=-1, keepdim=True).values

            # override S
            # P.shape = (Br, Bc)
            # 4 AIFF: ITP per-surface sub + exp
            P = torch.exp(S - max_current).to(O.dtype)

            # override max_last
            # tmp_exp.shape = (Br, 1)
            # 5 AIFF: ITP per-channel sub + exp
            tmp_exp = torch.exp(max_last - max_current).to(O.dtype)

            # sum_current.shape = (Br, 1)
            # 6 AIFF: ITP rdc channel sum
            # 7 AIFF: ITP per-channel mul + ITP per-channel add
            sum_current = sum_last * tmp_exp + torch.sum(P, dim=-1, keepdim=True)
            # # O_current.shape = (Br, d)
            # 8 AIFF: ITP per-surface mul
            # 9 AIFF: MTP matmul + ITP per-data add
            O_current = O_last * tmp_exp + P @ V[tbj: tbj_1, :]

            sum_last = sum_current
            max_last = max_current
            O_last = O_current
        # 10 AIFF: ITP reciprocal
        # 11 AIFF: ITP per-surface mul
        O[lbi: lbi_1, :] = O_last / sum_current
        # O = O.to(torch.float16) # convert to fp16
    O = O.to(self.outputs[0].betensor_type)
    self.outputs[0].betensor = O
    return O

@register_op_metric(OpType.FullyConnected)
@check_fp_data
def metric(self, outputs, index=None):
    def call(r, u, c, t):
        if "float" in t:
            if np.all(u <= 1) or np.all(r <= 1e-4):
                return True
            else:
                return np.all(c > 0.999)
        return np.all(r == 0)

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    layer_dtypes = [str(o.dtype) for o in self.outputs] if index is None else [str(self.outputs[index].dtype)]

    rel_diffs = relative_diff(outputs, groundtruths)
    ulp_diffs = ulp_diff(outputs, groundtruths)
    cos_dists = cos_distance(outputs, groundtruths)
    if self.outputs[0].dtype.is_float:
        return all(call(r, u, c, t) for r, u, c, t in zip(rel_diffs, ulp_diffs, cos_dists, layer_dtypes))

    return _OP_METRICS['_all'](self, outputs, index)