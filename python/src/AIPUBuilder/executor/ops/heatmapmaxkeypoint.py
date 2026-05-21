from .gt_util import *
from ..engine import register_gt, register_gen_weight
from AIPUBuilder.core import OpType


@register_gt(OpType.HeatmapMaxKeypoint)
def HeatmapMaxKeypoint(self, *args, **kwargs):
    srcs = self.inputs[0].betensor.cpu().numpy().astype(np.int32)
    boxes = self.inputs[1].betensor.cpu().numpy().astype(np.int32)  # uint16 could be overflow after * scale

    srcs = srcs + self.inputs[0].quantization.offset
    boxes = boxes + self.inputs[1].quantization.offset
    numBoxes = srcs.shape[0]
    heatmapSize = srcs.shape[1]
    numKeypoints = srcs.shape[3]

    outputScore = []
    outputData = []

    for k in range(0, numBoxes):
        heatmap = srcs[k].transpose(2, 0, 1)  # [numKeypoints, heatmapSize, heatmapSize]
        # x1, y1, x2, y2 = (boxes[k, :] * self.params['box_scale_value']) // (2 ** (self.params['box_shift_value']))
        x1, y1, x2, y2 = boxes[k, 0], boxes[k, 1], boxes[k, 2], boxes[k, 3]

        xlen = max(x2 - x1, 1)
        ylen = max(y2 - y1, 1)

        for kk in range(0, numKeypoints):
            maxScore = -32768
            maxIndex = 0
            data = heatmap[kk, :].reshape(-1)
            for i in range(0, heatmapSize * heatmapSize):
                val = data[i]
                if (maxScore < val):
                    maxScore = val
                    maxIndex = i

            maxIndexHeight = maxIndex // heatmapSize
            maxIndexWidth = maxIndex - maxIndexHeight * heatmapSize

            localGrid = np.zeros([3, 3])
            for dh in range(-1, 2):
                h = int(dh - 2 * (dh + maxIndexHeight >= heatmapSize) + 2 * (dh + maxIndexHeight < 0))
                for dw in range(-1, 2):
                    w = int(dw - 2 * (dw + maxIndexWidth >= heatmapSize) + 2 * (dw + maxIndexWidth < 0))
                    heatmapIndex = int((h + maxIndexHeight) * heatmapSize + (w + maxIndexWidth))
                    localGrid[dh + 1][dw + 1] = data[heatmapIndex]

            delta, deltaScore = solveForDelta(localGrid, maxScore)
            deltaScore = np.round(deltaScore * self.params['score_scale_value'] /
                                  (2 ** (self.params['score_shift_value'] + 16))).astype('int32')

            outputScore.append(deltaScore)

            # wRelativePos = ((maxIndexWidth + 0.5) * 256 + delta[0])
            # hRelativePos = ((maxIndexHeight + 0.5) * 256 + delta[1])
            wRelativePos = maxIndexWidth * 256 + 128 + delta[0]
            hRelativePos = maxIndexHeight * 256 + 128 + delta[1]

            xout = x1 * heatmapSize * 256 + wRelativePos * xlen
            yout = y1 * heatmapSize * 256 + hRelativePos * ylen

            xout = np.round(
                xout * self.params['box_scale_value'] / (2 ** (self.params['box_shift_value'] + 8)))
            yout = np.round(
                yout * self.params['box_scale_value'] / (2 ** (self.params['box_shift_value'] + 8)))
            outputData.append([xout, yout])

    out_type0 = self.outputs[0].dtype.np
    out_type1 = self.outputs[1].dtype.np
    out_type2 = self.outputs[2].dtype.np
    out_shape0 = list(self.outputs[0].shape)
    out_shape1 = list(self.outputs[1].shape)
    out_shape2 = list(self.outputs[2].shape)

    outputScore = np.asarray(outputScore) - self.outputs[0].quantization.offset
    outputData = np.asarray(outputData) - self.outputs[1].quantization.offset
    score = np.clip(outputScore, np.iinfo(out_type0).min, np.iinfo(out_type0).max).astype(out_type0).reshape(out_shape0)
    keypoints = np.clip(outputData, np.iinfo(out_type1).min, np.iinfo(out_type1).max).astype(out_type1).reshape(
        out_shape1)
    score_softmax = np.zeros(out_shape2, dtype=out_type2)

    self.outputs[0].betensor = score
    self.outputs[1].betensor = keypoints
    self.outputs[2].betensor = score_softmax
    return [o.betensor for o in self.outputs]


def solveForDelta(grid, deltaScore):
    b0 = -2 * (grid[1][2] - grid[1][0])
    b1 = -2 * (grid[2][1] - grid[0][1])
    A00 = 4 * (grid[1][0] - 2.0 * grid[1][1] + grid[1][2])
    A01 = (grid[2][2] - grid[2][0] - grid[0][2] + grid[0][0])
    A10 = A01
    A11 = 4 * (grid[0][1] - 2.0 * grid[1][1] + grid[2][1])

    crossProd1 = A00 * A11
    crossProd2 = A01 * A10
    detA = crossProd1 - crossProd2

    if (abs(detA) == 0):
        return [0, 0], deltaScore << 16

    delta0 = int((A11 * b0 - A01 * b1) * (2 ** 8) / detA)
    delta1 = int((A00 * b1 - A10 * b0) * ((2 ** 8)) / detA)

    if ((abs(delta0) > (1.5 * 256)) | (abs(delta1) > (1.5 * 256))):
        scale = (1.5 * 256) * 256 // max(abs(delta0), abs(delta1))
        delta0 *= scale
        delta1 *= scale
        delta0 = int(delta0) >> 8
        delta1 = int(delta1) >> 8

    s1 = grid[1][1] * (2 ** 16)
    s2 = b0 * delta0 * (2 ** 6)
    s3 = b1 * delta1 * (2 ** 6)
    s4 = int(((A00 * delta0 + A01 * delta1) * delta0 + (A10 * delta0 + A11 * delta1) * delta1) / 8)
    s = int((s1 - s2 - s3 + s4))  # / ((2 ** 16)))
    deltaScore = s
    return [delta0, delta1], deltaScore


@register_gen_weight(OpType.HeatmapMaxKeypoint)
def gen_random_weight(self):
    # softmax lut
    if 'lut' in self.constants:
        scale = 0.5
        nnnn = int(np.log(1 << 20) / scale)
        lut = [np.round(np.exp(scale * i)) for i in range(-256 + nnnn, nnnn)]
        self.constants['lut'].betensor = np.asarray(lut, dtype=self.constants['lut'].dtype.np)


@register_op_metric(OpType.HeatmapMaxKeypoint)
@check_fp_data
def metric(self, outputs, index=None):
    # only compare first 2 outputs
    if index is None:
        groundtruths = [torch_to_numpy(g.betensor, g.dtype.np).flatten() for g in self.outputs]
    else:
        groundtruths = [torch_to_numpy(self.outputs[index].betensor, self.outputs[index].dtype.np).flatten()]

    gts = groundtruths
    if index is None:
        diff_gt0 = np.mean(np.abs(gts[0] - outputs[0]))
        diff_gt1 = np.mean(np.abs(gts[1] - outputs[1]))
        diff_gt = np.asarray([diff_gt0, diff_gt1])
        if (diff_gt != 0).any():
            print("avg error per-output:", diff_gt)
            return False
    else:
        if index in [0, 1]:
            diff_gt = np.mean(np.abs(gts[index] - outputs))
        if diff_gt != 0:
            return False
    return True
