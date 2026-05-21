import torch
import numpy as np
from AIPUBuilder.core import OpType, SubGraphNode
from ...engine import register_op_metric
from ..gt_util.dtype_utils import torch_to_numpy
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2


def check_fp_data(func):
    """
    Perform checks and pre-processing(inf/nan) for fp data
    :param func: metric func
    :return:
    """

    def wrapper(self, outputs, index=None, *args, **kwargs):
        outputs = outputs if index is None or isinstance(outputs, list) else [outputs]
        outputs = [o.flatten() for o in outputs]
        if index is None:
            groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
        else:
            groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

        assert len(outputs) == len(groundtruths), "if check all, the number of outputs must equal to node's output "

        if self.type == OpType.LayoutConvertor:
            layer_type = get_parent_node(self).type
        else:
            layer_type = self.type
        layer_type = str(layer_type).split('OpType.')[-1]

        for o, gt, gt_ori in zip(outputs, groundtruths, self.outputs):  # todo len(self.outputs) maybe != len(outputs)
            if not gt_ori.dtype.is_float:
                continue
            if gt.size == 0:
                continue

            max_value = 0
            if gt_ori.dtype.np == np.float16:
                max_value = 65504
            elif gt_ori.dtype.np == bfloat16:
                max_value = 3.389531389e38
            elif gt_ori.dtype.np == float8_e5m2:
                max_value = 57344
            elif gt_ori.dtype.np == float8_e4m3fn:
                max_value = 448

            # inf
            o_pos = np.isposinf(o)
            gt_pos = np.isposinf(gt)
            same_posinf = np.all(o_pos == gt_pos)
            if not same_posinf:
                print("pos inf is not same as gt: ", np.where(o_pos.flatten() != gt_pos.flatten()))
                if layer_type not in ['GroupNorm', 'Convolution', 'DetectionOutput']:
                    return False
            o[o_pos] = max_value
            gt[gt_pos] = max_value
            gt[(gt == 0.0) & (o == 0)] = 0
            o[(gt == 0.0) & (o == 0)] = 0

            # -inf
            o_neg = np.isneginf(o)
            gt_neg = np.isneginf(gt)
            same_neginf = np.all(o_neg == gt_neg)
            if not same_neginf:
                print("neg inf is not same as gt: ", np.where(o_neg.flatten() != gt_neg.flatten()))
                if layer_type not in ['GroupNorm', 'Convolution', 'DetectionOutput']:
                    return False
            o[o_neg] = -max_value
            gt[gt_neg] = -max_value

            # nan
            o_nan = np.isnan(o)
            gt_nan = np.isnan(gt)
            same_nan = np.all(o_nan == gt_nan)
            if not same_nan:
                print("same_nan is not same as gt: ", np.where(o_nan.flatten() != gt_nan.flatten()))
                if layer_type not in ['DetectionOutput']:  # pad invalid value
                    return False
            o[o_nan] = max_value
            gt[o_nan] = max_value

            if index is None:
                gt_ori.betensor = gt.reshape(gt_ori.betensor.shape)
            else:
                self.outputs[index].betensor = gt.reshape(self.outputs[index].betensor.shape)
        if kwargs:
            return func(self, outputs, index, **kwargs)
        return func(self, outputs, index)
    return wrapper


@register_op_metric('_all')
@check_fp_data
def metric(self, outputs, index=None):
    def call(idx, r):
        condition = r != 0
        if np.any(condition):
            err_idx = np.where(condition)

            err = r[condition]
            gt = groundtruths[idx][condition]
            out = outputs[idx][condition]

            max_idx = np.argmax(err)
            print(f"========= relative error at {idx} ===========")
            print(f"result shape: {r.shape}")
            print(f"=============================================")
            print(f"first error:  {err[0]}")
            print(f"first gt:     {gt[0]}")
            print(f"first output: {out[0]}")
            print(f"first err_idx: \n{(np.asarray(err_idx).T)[0]}")
            print(f"=============================================")
            print(f"max error:  {err[max_idx]}")
            print(f"max gt:     {gt[max_idx]}")
            print(f"max output: {out[max_idx]}")
            print(f"max err_idx: \n{(np.asarray(err_idx).T)[max_idx]}")
            print(f"=============================================")
            return False
        return True

    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    rel_diffs = relative_diff(outputs, groundtruths)
    return all(call(idx, r) for idx, r in enumerate(rel_diffs))


def cos_distance(outputs, groundtruths):
    def handle_unexpected_fp(data, orignal_dtype):
        '''dealing with inf/nan'''
        max_value = 0
        if orignal_dtype == np.float16:
            max_value = 65504
        elif orignal_dtype == bfloat16:
            max_value = 3.3895313892515354759047080037148786688e38
        elif orignal_dtype == float8_e5m2:
            max_value = 57344
        elif orignal_dtype == float8_e4m3fn:
            max_value = 448
        min_value = -max_value

        data[np.where(np.isneginf(data))] = min_value
        data[np.where(np.isposinf(data))] = max_value
        data[np.where(np.isnan(data))] = max_value

    def compute_cos_distance(x, y):
        """Get cosine similarity."""
        x_dtype = x.dtype
        y_dtype = y.dtype
        x = x.astype("float32")
        handle_unexpected_fp(x, x_dtype)
        y = y.astype("float32")
        handle_unexpected_fp(y, y_dtype)
        similarity = torch.cosine_similarity(torch.FloatTensor(x), torch.FloatTensor(y), dim=0).cpu().numpy()
        similarity = 1.0 if (x == y).all() else similarity  # all 0
        return similarity

    cos = []
    for o, gt in zip(outputs, groundtruths):
        cos_dis = compute_cos_distance(o.flatten(), gt.flatten())
        cos.append(cos_dis)
        print("cosine similarity: {}".format(cos_dis))
    return cos


def relative_diff(outputs, groundtruths):
    diffs = []
    for o, gt in zip(outputs, groundtruths):
        if 'float' in str(o.dtype):
            rel = np.abs(o - gt) / (np.abs(gt) + 0.000001)
            rel_diff_max = np.max(rel)
            rel_diff_idx = np.argmax(rel)
            print("max relative error: {}, gt={}, out={}, at {}/2".format(
                rel_diff_max, gt.flatten()[rel_diff_idx], o.flatten()[rel_diff_idx], rel_diff_idx * 2))
        else:
            o_ = o.astype(np.int64)
            gt_ = gt.astype(np.int64)
            rel = np.abs(o_ - gt_)
        diffs.append(rel)
    return diffs


def ulp_diff(outputs, groundtruths):
    ulp_type_map = {
        float8_e4m3fn: np.uint8,
        float8_e5m2: np.uint8,
        np.float16: np.uint16,
        bfloat16: np.uint16,
        np.float32: np.uint32,
    }  # TODO: Maybe static member var.
    diffs = []
    for o, gt in zip(outputs, groundtruths):
        o_type = o.dtype.type
        gt = gt.astype(o_type)
        if "float" in str(o.dtype):
            view_type = ulp_type_map[o_type]

            # # -0(0x80) -> 0(0x00), todo, 0x81
            # o[o == np.float16(-0.0)] = np.float16(0.0)
            # gt[gt == np.float16(-0.0)] = np.float16(0.0)

            o_view = o.view(view_type).astype(np.int64)
            gt_view = gt.view(view_type).astype(np.int64)
        else:
            o_view = o.astype("int64")
            gt_view = gt.astype("int64")
        ulp = np.abs(o_view - gt_view)
        diffs.append(ulp)

        ulp_diff_max = np.max(ulp)
        ulp_diff_idx = np.argmax(ulp)
        print("max ulp error: {} at {}/2".format(ulp_diff_max, ulp_diff_idx * 2))
    return diffs


def get_parent_node(node):
    '''
    get parent node from nodes.real_nodes, which not contain SubGraphOp
    '''
    for n in node.parents:
        if SubGraphNode.is_subgraph_node(n):
            for j in n.real_nodes:
                return j
        else:
            return n
