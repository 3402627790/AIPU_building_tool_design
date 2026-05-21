from .gt_util import *
from ..engine import register_gt, register_gen_input
from AIPUBuilder.core import OpType


@register_gt(OpType.SegmentReduce)
def segmentreduce(self, *args, **kwargs):
    scale = self.params['scale_value']
    shift = self.params['shift_value']
    in_zp = self.inputs[0].quantization.offset
    out_zp = self.outputs[0].quantization.offset
    input_type = str(self.inputs[0].dtype)
    output_type = str(self.outputs[0].dtype)

    # overflow clamp type value
    has_zp_in_out = True if in_zp != 0 or out_zp != 0 else False
    is_compute_signed = True if input_type.startswith('int') or has_zp_in_out else False
    min_32bit = np.iinfo(np.int32).min if is_compute_signed else np.iinfo(np.uint32).min
    max_32bit = np.iinfo(np.int32).max if is_compute_signed else np.iinfo(np.uint32).max
    min_16bit = np.iinfo(np.int16).min if is_compute_signed else np.iinfo(np.uint16).min
    max_16bit = np.iinfo(np.int16).max if is_compute_signed else np.iinfo(np.uint16).max
    np_32bit = np.int32 if is_compute_signed else np.uint32
    np_16bit = np.int16 if is_compute_signed else np.uint16

    data = self.inputs[0].betensor.cpu().numpy().astype('int64')
    data = data + in_zp

    segment_ids = self.inputs[1].betensor.cpu().numpy().astype('int32')
    segment_ids = segment_ids + self.inputs[1].quantization.offset
    max_ids = np.max(segment_ids)
    output = []
    for i in range(max_ids+1):
        if self.params['method'].lower() == 'sum':
            segment_data = data[segment_ids == i]
            output.append(np.sum(segment_data, axis=0))
    output = np.asarray(output).astype('int64')
    t = output
    # hi
    hi = np.floor(t * (0.5 ** 16)) * scale
    hi = np.int64(hi)
    hi = np.round(hi * (0.5 ** (shift - 16)))
    hi = np.clip(hi, min_32bit, max_32bit)
    # lo
    lo = t & 0x0000ffff
    lo = lo * scale
    lo = np.round(lo * (0.5 ** shift))
    lo = np.clip(lo, min_32bit, max_32bit)
    # merge hi lo
    t = np.clip(hi + lo, min_32bit, max_32bit)
    t = np.int64(t)
    output = t

    output -= out_zp
    output = np.clip(output, np.iinfo(output_type).min, np.iinfo(output_type).max)
    self.outputs[0].betensor = output
    return output


@register_gen_input(OpType.SegmentReduce)
def gen_input_data(self):
    in_shape0 = list(self.inputs[0].shape)
    in_shape1 = list(self.inputs[1].shape)
    in_type0 = str(self.inputs[0].dtype)
    in_type1 = str(self.inputs[1].dtype)

    input0 = np.random.randint(np.iinfo(in_type0).min, np.iinfo(in_type0).max, size=in_shape0, dtype=in_type0)
    max_ids = self.outputs[0].shape[0] - 1
    input1 = np.random.randint(0, max_ids + 1, size=in_shape1, dtype=in_type1)

    input1.sort()
    input1[-1] = max_ids

    input_data = [input0, input1]
    self.inputs[0].betensor = input_data[0]
    self.inputs[1].betensor = input_data[1]
    return input_data
