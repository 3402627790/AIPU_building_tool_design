import numpy as np
import torch

layout_transfer_dict = {
    'NHWC_NCHWC32': lambda x: torch.permute(torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2], x.shape[-1]//32, 32]), [0, 3, 1, 2, 4]),
    'NCHWC32_NHWC': lambda x: torch.reshape(torch.permute(x, [0, 2, 3, 1, 4]), [x.shape[0], x.shape[2], x.shape[3], x.shape[1] * x.shape[-1]]),
    'NHWC_NCHWC16': lambda x: torch.permute(torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2], x.shape[-1]//16, 16]), [0, 3, 1, 2, 4]),
    'NCHWC16_NHWC': lambda x: torch.reshape(torch.permute(x, [0, 2, 3, 1, 4]), [x.shape[0], x.shape[2], x.shape[3], x.shape[1] * x.shape[-1]]),
    'NDHWC_NDCHWC32': lambda x: torch.permute(torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[-1]//32, 32]), [0, 1, 4, 2, 3, 5]),
    'NDCHWC32_NDHWC': lambda x: torch.reshape(torch.permute(x, [0, 1, 3, 4, 2, 5]), [x.shape[0], x.shape[1], x.shape[3], x.shape[4], x.shape[2] * x.shape[-1]]),
    'NDHWC_NDCHWC16': lambda x: torch.permute(torch.reshape(x, [x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[-1]//16, 16]), [0, 1, 4, 2, 3, 5]),
    'NDCHWC16_NDHWC': lambda x: torch.reshape(torch.permute(x, [0, 1, 3, 4, 2, 5]), [x.shape[0], x.shape[1], x.shape[3], x.shape[4], x.shape[2] * x.shape[-1]]),
    'NHWC_NCHWC32T8': lambda x: NHWC_NCHWCalignT8(x, 32),
    'NCHWC32T8_NHWC': lambda x: NCHWCalignT8_NHWC(x, 32),
    'NHWC_NCHWC16T8': lambda x: NHWC_NCHWCalignT8(x, 16),
    'NCHWC16T8_NHWC': lambda x: NCHWCalignT8_NHWC(x, 16),
    'NDHWC_NDCHWC32T8': lambda x: NDHWC_NDCHWCalignT8(x, 32),
    'NDCHWC32T8_NDHWC': lambda x: NDCHWCalignT8_NDHWC(x, 32),
    'NDHWC_NDCHWC16T8': lambda x: NDHWC_NDCHWCalignT8(x, 16),
    'NDCHWC16T8_NDHWC': lambda x: NDCHWCalignT8_NDHWC(x, 16),
    'NCHWC32_NCHWC16': lambda x: NCHWC32_NCHWC16(x),
    'NCHWC16_NCHWC32': lambda x: NCHWC16_NCHWC32(x),
    'NDCHWC32_NDCHWC16': lambda x: NDCHWC32_NDCHWC16(x),
    'NDCHWC16_NDCHWC32': lambda x: NDCHWC16_NDCHWC32(x),
    'NCHWC32_NCHW': lambda x: NCHWC32_NCHW(x),
    'NCHW_NCHWC32': lambda x: NCHW_NCHWC32(x),
    'NCHWC16_NCHW': lambda x: NCHWC16_NCHW(x),
    'NCHW_NCHWC16': lambda x: NCHW_NCHWC16(x),
    'NHWC_NCHW': lambda x: torch.permute(x, [0, 3, 1, 2]),
    'NCHW_NHWC': lambda x: torch.permute(x, [0, 2, 3, 1]),
    'Flat_NHWC': lambda x: x,
    'NHWC_Flat': lambda x: x,
    'NHWC_NHWC': lambda x: x,
    'NDHWC_NDHWC': lambda x: x,
    'NDHWC_Flat': lambda x: x,
    'Flat_NDHWC': lambda x: x,
    'Flat_Flat': lambda x: x,
    'NDHWC_NHWC': lambda x: x,
    'NHWC_NDHWC': lambda x: x,
    'NCHWC32_NCHWC32T8': lambda x: torch.reshape(x, [x.shape[0], x.shape[2], x.shape[3], x.shape[1] * x.shape[-1]]),
    'NCHWC32T8_NCHWC32': lambda x: torch.reshape(x, [x.shape[0], x.shape[-1]//32, x.shape[1], x.shape[2], 32]),
    "NCHWC16_NCHWC16T8": lambda x: torch.reshape(x, [x.shape[0], x.shape[2], x.shape[3], x.shape[1] * x.shape[-1]]),
    "NCHWC16T8_NCHWC16": lambda x: torch.reshape(x, [x.shape[0], x.shape[-1]//16, x.shape[1], x.shape[2], 16]),
    'NHWC_NCHW': lambda x: torch.permute(x, [0, 3, 1, 2]),
    'NCWH_NHWC': lambda x: torch.permute(x, [0, 2, 3, 1]),
    'NDHWC_NCDHW': lambda x: torch.permute(x, [0, 4, 1, 2, 3]),
    'NCDHW_NDHWC': lambda x: torch.permute(x, [0, 2, 3, 4, 1]),
    'NHWC_NCHWC32T1': lambda x: NHWC_NCHWCalignT8(x, 32),
    'NCHWC32T1_NHWC': lambda x: NCHWCalignT8_NHWC(x, 32),
    'NHWC_NCHWC16T1': lambda x: NHWC_NCHWCalignT8(x, 16),
    'NCHWC16T1_NHWC': lambda x: NCHWCalignT8_NHWC(x, 16),
    'NDHWC_NDCHWC32T1': lambda x: NDHWC_NDCHWCalignT8(x, 32),
    'NDCHWC32T1_NDHWC': lambda x: NDCHWCalignT8_NDHWC(x, 32),
    'NDHWC_NDCHWC16T1': lambda x: NDHWC_NDCHWCalignT8(x, 16),
    'NDCHWC16T1_NDHWC': lambda x: NDCHWCalignT8_NDHWC(x, 16),

    # 'NDCHWC16_NHWC': lambda x: NDCHWC16_NHWC(x)

}


def NDCHWC16_NHWC(x):
    x = layout_transfer_dict['NDCHWC16_NDHWC'](x)
    x = torch.reshape(x, [x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
    return x


def NCHWC32_NCHWC16(x):
    return layout_transfer_dict['NHWC_NCHWC16'](layout_transfer_dict['NCHWC32_NHWC'](x))


def NCHWC16_NCHWC32(x):
    return layout_transfer_dict['NHWC_NCHWC32'](layout_transfer_dict['NCHWC16_NHWC'](x))


def NDCHWC32_NDCHWC16(x):
    return layout_transfer_dict['NDHWC_NDCHWC16'](layout_transfer_dict['NDCHWC32_NDHWC'](x))


def NDCHWC16_NDCHWC32(x):
    return layout_transfer_dict['NDHWC_NDCHWC32'](layout_transfer_dict['NDCHWC16_NDHWC'](x))


def NCHWC32_NCHW(x):
    return layout_transfer_dict['NHWC_NCHW'](layout_transfer_dict['NCHWC32_NHWC'](x))


def NCHW_NCHWC32(x):
    return layout_transfer_dict['NHWC_NCHWC32'](layout_transfer_dict['NCHW_NHWC'](x))


def NCHWC16_NCHW(x):
    return layout_transfer_dict['NHWC_NCHW'](layout_transfer_dict['NCHWC16_NHWC'](x))


def NCHW_NCHWC16(x):
    return layout_transfer_dict['NHWC_NCHWC16'](layout_transfer_dict['NCHW_NHWC'](x))


def NHWC_NCHWCalignT8(x: np.ndarray, align: int):
    n, h, w, c = x.shape
    output = []
    for i in range(n):
        input_a_batch = x[i][np.newaxis, :]
        c_num = x.shape[-1] // align
        input1, input2 = torch.split(input_a_batch, [c_num * align, x.shape[-1] - c_num * align], dim=-1)
        if align == 32:
            output1 = layout_transfer_dict['NHWC_NCHWC32'](input1)
        elif align == 16:
            output1 = layout_transfer_dict['NHWC_NCHWC16'](input1)
        output1 = output1.reshape([1, h, w, c_num*align])
        output1 = output1.flatten()
        input2 = input2.flatten()
        output_a_batch = torch.cat([output1, input2], dim=-1)
        output.append(output_a_batch)
    output = torch.cat(output, dim=0).reshape([n, h, w, c])
    return output


def NCHWCalignT8_NHWC(x: np.ndarray, align: int):
    n, h, w, c = x.shape
    output = []
    for i in range(n):
        input_a_batch = x[i][np.newaxis, :]
        c_num = x.shape[-1] // align
        input1_size = 1 * w * h * c_num * align
        input1, input2 = torch.split(input_a_batch.flatten(), [input1_size, h * w * c - input1_size], dim=-1)
        input1 = input1.reshape(1, c_num, h, w, align)
        input2 = input2.reshape(1, h, w, c - c_num * align)
        if align == 32:
            ouput1 = layout_transfer_dict['NCHWC32_NHWC'](input1)
        elif align == 16:
            ouput1 = layout_transfer_dict['NCHWC16_NHWC'](input1)
        output_a_batch = torch.cat([ouput1, input2], dim=-1)
        output.append(output_a_batch)
    output = torch.cat(output, dim=0).reshape([n, h, w, c])
    return output


def NDHWC_NDCHWCalignT8(x: np.ndarray, align: int):
    n, d, h, w, c = x.shape
    x = x.reshape(n * d, h, w, c)
    output = NHWC_NCHWCalignT8(x, align)
    output = output.reshape([n, d, h, w, c])
    return output


def NDCHWCalignT8_NDHWC(x: np.ndarray, align: int):
    n, d, h, w, c = x.shape
    x = x.reshape(n * d, h, w, c)
    output = NCHWCalignT8_NHWC(x, align)
    output = output.reshape([n, d, h, w, c])
    return output


def layout_convert(data, _from, to):
    if _from == to:
        return data
    _data = data
    if isinstance(data, np.ndarray):
        _data = torch.from_numpy(_data.astype('int64')) if str(_data.dtype).endswith(
            ('int32', 'int64')) else torch.from_numpy(_data.astype('float64'))
    return layout_transfer_dict['{}_{}'.format(_from, to)](_data)


def check_in_datalayout(self, input_data, to_datalayout):
    # avoid int32->fp64->fp32, which will cause precision error
    if isinstance(input_data[0], np.ndarray):
        input_data = [torch.from_numpy(i.astype('int64')) if str(i.dtype).endswith(('int32', 'int64'))
                      else torch.from_numpy(i.astype('float64')) for i in input_data]
    for index, input_tensor in enumerate(self.inputs):
        if input_tensor.shape.layout.name != to_datalayout[index]:
            _from = input_tensor.shape.layout.name
            to = to_datalayout[index]
            input_data[index] = layout_convert(input_data[index], _from=_from, to=to)
    return input_data


def check_out_datalayout(self, output_data, from_datalayout):
    # avoid int32->fp64->fp32, which will cause precision error
    if isinstance(output_data[0], np.ndarray):
        output_data = [torch.from_numpy(o.astype('int64')) if str(o.dtype).endswith(('int32', 'int64'))
                       else torch.from_numpy(o.astype('float64')) for o in output_data]
    for index, output_tensor in enumerate(self.outputs):
        if output_tensor.shape.layout.name != from_datalayout[index]:
            _from = from_datalayout[index]
            to = output_tensor.shape.layout.name
            output_data[index] = layout_convert(output_data[index], _from=_from, to=to)
            #
            if self.outputs[index].shape.layout.name == "Flat":
                shape = self.attrs.get("_output_shapes")[
                    index] if "_output_shapes" in self.attrs else list(self.outputs[index].shape)
                output_data[index] = torch.reshape(output_data[index], shape)
    return output_data


def transfer_shape(shape, _from, to):
    shape_transfer_dict = {
        'NCHWC32_NHWC': lambda x: [x[0], x[2], x[3], x[1] * x[4]],
        'NCHWC16_NHWC': lambda x: [x[0], x[2], x[3], x[1] * x[4]],
        'NHWC_NCHWC32': lambda x: [x[0], x[-1] // 32, x[1], x[2], 32],
        'NHWC_NCHWC16': lambda x: [x[0], x[-1] // 16, x[1], x[2], 16],
        'NHWC_NCHWC32T8': lambda x: x,
        'NHWC_NCHWC16T8': lambda x: x,
        'NCHWC32T8_NHWC': lambda x: x,
        'NCHWC16T8_NHWC': lambda x: x,
        'NHWC_NHWC': lambda x: x,
        'NDHWC_NDHWC': lambda x: x,
        'NDHWC_NDCHWC32': lambda x: [x[0], x[1], x[-1] // 32, x[2], x[3], 32],
        'NDHWC_NDCHWC16': lambda x: [x[0], x[1], x[-1] // 16, x[2], x[3], 16],
        'NDCHWC32_NDHWC': lambda x: [x[0], x[1], x[3], x[4], x[2] * x[-1]],
        'NDCHWC16_NDHWC': lambda x: [x[0], x[1], x[3], x[4], x[2] * x[-1]],
        'NDHWC_NDCHWC16T8': lambda x: x,
        'NDHWC_NDCHWC32T8': lambda x: x,
        'NDCHWC16T8_NDHWC': lambda x: x,
        'NDCHWC32T8_NDHWC': lambda x: x,
        'NHWC_NCHWC32T1': lambda x: x,
        'NCHWC32T1_NHWC': lambda x: x,
        'NHWC_NCHWC16T1': lambda x: x,
        'NCHWC16T1_NHWC': lambda x: x,
        'NDHWC_NDCHWC32T1': lambda x: x,
        'NDCHWC32T1_NDHWC': lambda x: x,
        'NDHWC_NDCHWC16T1': lambda x: x,
        'NDCHWC16T1_NDHWC': lambda x: x,
        'NCHWC32_NCHWC16': lambda x: [x[0], x[1] * x[4] // 16, x[2], x[3], 16],
        'NCHWC16_NCHWC32': lambda x: [x[0], x[1] * x[4] // 32, x[2], x[3], 32],
        'NDCHWC32_NDCHWC16': lambda x: [x[0], x[1], x[2] * x[5] // 16, x[3], x[4], 16],
        'NDCHWC16_NDCHWC32': lambda x: [x[0], x[1], x[2] * x[5] // 32, x[3], x[4], 32],
    }
    return shape_transfer_dict['{}_{}'.format(_from, to)](shape)
