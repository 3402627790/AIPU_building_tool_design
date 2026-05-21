import AIPUBuilder._C._ops as _ops
from AIPUBuilder._C._core import Quantization
from AIPUBuilder.core.tensor import *
from AIPUBuilder.ops.common import *
from typing import List


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def crop_and_resize(
    input: Tensor,
    box: Tensor,
    box_indices: Tensor,
    crop_size: List[int],
    method="NearestNeighbor",
    extrapolation: float or int = 0.0,
    is_perf_mode: bool = True,
) -> Tensor:
    """
    Extracts crops from the input image and resize it by bilinear or nearest neighbor sampling to
    a common output size specified by `crop_size` parameter.

    :param input: A Tensor, specified the featuremap.
    :param box: A Tensor in shape [numboxes, 4], specified coordinates with order [y1,x1,y2,x1] in normalized format.
                For example, y1 is normalized in [0,1], corresponding to [0, height-1].
                If box is quantied, quantization of box should describle the scale of coordinates.
    :param box_indices: A Tensor in shape [numboxes], specified batch index of each box.
    :param crop_size: A List of cropped size in height/width order.
    :param method: Resize method, it can be `Bilinear`, `NearestNeighbor` string,
                   or specified by enum ResizeMethod:
                   - ResizeMethod.Bilinear
                   - ResizeMethod.NearestNeighbor
                   Default is `NearestNeighbor` mode.
    :param extrapolation: A float/int scalar, specified for extrapolation. Generally, default is 0.f.
                          If it is a float value, it means that it is not quantized. API will quantize this value.
                          If it is an int value, it means that it is quantized.
    :param is_perf_mode: A bool scalar, specified whether the op is running with high performance mode.
    :return: A Tensor, with new shape [batch, new_height, new_width, channel].
    """
    if not isinstance(crop_size, (list, tuple)):
        raise ValueError(
            "CropAndResize's crop_size should be list or tuple, and size should be 2"
        )

    resize_method = convert_to_resize_method(method)
    ret = _ops.crop_and_resize(
        input, box, box_indices, crop_size, resize_method, extrapolation, is_perf_mode
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def resize(
    input: Tensor,
    size: List[int] = [],
    scale: List[float] = [],
    method: str = "NearestNeighbor",
    mode: str = "HalfPixel",
    nearest_mode: str = "RoundPreferFloor",
    antialias: bool = False,
    exclude_outside: bool = False,
) -> Tensor:
    """
    Resizes the input tensor using the specified method with specified mode.
    Generally, resize mode has follows,

    - HalfPixel: `x_original = (y_resized + 0.5) / scale - 0.5`.
    - HalfPixelSymmetric:

        - adjustment = output_width_int / output_width
        - center = input_width / 2
        - offset = center * (1 - adjustment)
        - x_ori = offset + (x + 0.5) / scale - 0.5.

    - AlignCorners: `x_original = y_resized * (x_original - 1) / (y_resized - 1)`.
    - Asymmetric: `x_original = y_resized / scale`.

    :param input:  A Tensor, specified by [batch, height, width, channel]
    :param size: A int or list. If int, it will be tiled to spatial dimensions.
                 If list, should be [output_height, output_width] or [output_batch, output_height, output_width, output_channel],
                 specifying the output size.
    :param scale: A float or list. If int, it will be tiled to spatial dimensions.
                  If list, should be [scale_y, scale_x] or [scale_batch, scale_height, scale_width, scale_channel],
                  specifying the output=input*scale.
    :param method: Resize method, it can be `Bilinear`, `NearestNeighbor` string, or specified by enum ResizeMethod:

                   - ResizeMethod.Bilinear
                   - ResizeMethod.NearestNeighbor

    :param mode: Resize mode, it can be `HalfPixel`, `HalfPixelSymmetric`, `AlignCorners`, `Asymmetric`, `PytorchHalfPixel`, `TfHalfPixelForNN`,
                 or specified by enum ResizeMode:

                 - ResizeMode.HalfPixel
                 - ResizeMode.HalfPixelSymmetric
                 - ResizeMode.AlignCorners
                 - ResizeMode.Asymmetric
                 - ResizeMode.PytorchHalfPixel
                 - ResizeMode.TfHalfPixelForNN

    :param nearest_mode: Resize nearest mode, when method is not nearest, just ignore this parameter.
                         mode can be `RoundPreferFloor`, `RoundPreferCeil`, `Floor`, `Ceil`, `Simple`,
                         or specified by enum ResizeNearestMode:

                         - ResizeNearestMode.RoundPreferFloor
                         - ResizeNearestMode.RoundPreferCeil
                         - ResizeNearestMode.Floor
                         - ResizeNearestMode.Ceil
                         - ResizeNearestMode.Simple
    :param antialias: If set to true, "linear" and "cubic" interpolation modes will use an antialiasing filter when downscaling.
                    Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale),
                    which means that when downsampling, more input pixels contribute to an output pixel.
    :param exclude_outside: When calculating the original point coordinates,
                            only the pixels inside the bounding box are considered,
                            the pixels must not exceed the original bounding box.
    :return: A Tensor, with new shape [batch, new_height, new_width, channel].
    """
    if len(size) == 0 and len(scale) == 0:
        raise ValueError("Resize  at least one of scale and size must be valid.")
    ret = _ops.resize(
        input=input,
        out_size=size,
        scale=scale,
        method=convert_to_resize_method(method),
        mode=convert_to_resize_mode(mode),
        nearest_mode=convert_to_resize_nearest_mode(nearest_mode),
        antialias=antialias,
        exclude_outside=exclude_outside,
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def rgb_to_yuv(
    input: Tensor,
    format: str = "I420",
    bits: int = 8,
    conversion: str = "BT709",
    coefficient: List[int] = [],
) -> Tensor:
    """
    Conversion of one or more images from `RGB` to `YUV`. Generally, this operation is used for preprocessing.
    Here, the term `YUV` is commonly encoded using `Y'CbCr`; while the `RGB` term is encoded using `R'dG'dB'd`.
    There are three standards which are so-called `BT.709`. For 8 bits color,
    [0, 255] range is referred as Full Range, which is considerred here; while [16, 235] range for Y and
    [16, 240] for `UN` are referred as Narrow Range. Also, all the conversions are based on integral approximate
    (or using fixed-point arithmetic as an alternative formulation) in this operation.

    :param input: A Tensor, specifying `RGB` image, with a shape of [N, H, W, 3].
    :param format: A string or an enum value of ImageFormat specifying the input image format which will be
                   converted. The optional values are [I420, YV12, NV12, NV21] and defaults to `I420`.
    :param bits: An integer, specifying the bits to store a Y, U or V values in memory, such as DDR device.
                 Defaults to 8.
    :param conversion: A string or an enum value of ImageConversion, which means the different coefficient
                       matrix to convert between `YUV` and `RGB` format. Especially, the `SELF` means the conversion
                       coefficients are customized which is from the configuration. The optional values are
                       [BT709] and defaults to `BT709`.
    :param coefficient: A list of color space coefficient (CSC) for conversion, which is in order of
                        [input_offset, output_offset, coefficient].
    :return: A `YUV` image tensor, with a shape of [N, D].
    """
    check_tensor_shape(input, expect_dim=4)
    assert bits > 0, "YuvToRgb expects bits to a positive integer!"
    assert len(coefficient) in (
        0,
        15,
    ), "YuvToRgb expects length of coefficient to be 0 or 15!"
    ret = _ops.rgb_to_yuv(
        input,
        str_to_enum_member(_ops.ImageFormat, format),
        bits,
        str_to_enum_member(_ops.ImageConversion, conversion),
        coefficient,
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def yuv_to_rgb(
    input: Tensor,
    shape: List[int],
    format: str = "I420",
    bits: int = 8,
    conversion: str = "BT709",
    coefficient: List[int] = [],
) -> Tensor:
    """
    Conversion of one or more images from `YUV` to `RGB`. Generally, this operation is used for preprocessing.
    Here, the term `YUV` is commonly encoded using `Y'CbCr`; while the `RGB` term is encoded using `R'dG'dB'd`.
    There are three standards which are so-called `BT.709`. For 8 bits color,
    [0, 255] range is referred as Full Range, which is considerred here; while [16, 235] range for Y and
    [16, 240] for `UN` are referred as Narrow Range. Also, all the conversions are based on integral approximate
    (or using fixed-point arithmetic as an alternative formulation) in this operation.

    :param input: A Tensor, specifying `YUV` image, with a shape of [N, D].
    :param shape: A list of integer, specifying the shape of output tensor.
    :param format: A string or an enum value of ImageFormat specifying the input image format which will be
                   converted. The optional values are [I420, YV12, NV12, NV21] and defaults to `I420`.
    :param bits: An integer, specifying the bits to store a Y, U or V values in memory, such as DDR device.
                 Defaults to 8.
    :param conversion: A string or an enum value of ImageConversion, which means the different coefficient
                       matrix to convert between `YUV` and `RGB` format. Especially, the `SELF` means the conversion
                       coefficients are customized which is from the configuration. The optional values are
                       [BT709] and defaults to `BT709`.
    :param coefficient: A list of color space coefficient (CSC) for conversion, which is in order of
                        [input_offset, output_offset, coefficient].
    :return: A RGB image tensor, with a shape of [N, H, W, 3].
    """
    check_tensor_shape(input, expect_dim=2)
    assert bits > 0, "YuvToRgb expects bits to a positive integer!"
    assert len(coefficient) in (
        0,
        15,
    ), "YuvToRgb expects length of coefficient to be 0 or 15!"
    ret = _ops.yuv_to_rgb(
        input,
        shape,
        str_to_enum_member(_ops.ImageFormat, format),
        bits,
        str_to_enum_member(_ops.ImageConversion, conversion),
        coefficient,
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def filter(inputs: List[Tensor]) -> List[Tensor]:
    """
    Takes one input data tensor (X) and filter tensor (B) to return an output tensor (Y) after filtration.
    Where, the element will be filtered with zero selector accordingly in filter tensor B, otherwise remaining.
    The filter tensor is a vector which is composed of '0' or `1` with the same length
    as the specified axis in data tensor X.
    Similarly, multiple input tensors (X[i]) will return multiple outputs after the same filtration.

    :param inputs: A Tensor list, which are input data tensor X[i] (`i` means the index of input tensors),
                        filter tensor B (a 1-D tensor).
    :return: A Tensor list, which are output tensor Y[i] (`i` means the index of output tensors),
            effective length (a scalar with an empty shape).
    """
    return _ops.filter(inputs)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def filter_boxes(
    boxes: Tensor,
    scores: Tensor,
    number: Tensor,
    label: Tensor,
    classes: Tensor,
    score_threshold: int or float,
    max_box_num: int,
    minsize: list,
    quantization: List[Quantization] = [],
) -> List[Tensor]:
    """
    Filter boxes that meet limited criteria.
    Note that, in the following Inputs or Outputs sections:
    X1 means the coordinates of the maximum output boxes per batch, of which the shape is [batch_size, num_boxes, 4].
    X2 means the scores of all the boxes per batch, of which the shape is [batch_size, num_boxes].
    X3 means the boxes number of every class per batch, of which the shape is [batch_size, num_classes].
    X4 means the boxes label of every class per batch, of which the shape is [batch_size, num_classes].
    X5 means the valid classes of every batch, of which the shape is [batch_size, 1].
    Y1 means the coordinates of the selected boxes per batch, of which the shape is [batch, max_box_num, 4]
    Y2 means the score of every box per batch after selection, of which the shape is [batch, max_box_num].
    Y3 means the boxes number of every class per batch after selection, of which the shape is [batch_size, num_classes].
    Y4 means the boxes label of every class per batch after selection, of which the shape is [batch_size, num_classes].
    Y5 means the valid classes of every batch after selection, of which the shape is [batch_size, 1].

    :param boxes: A tensor, specified the input tensor X1.
                  If box is quantied, quantization of box should describle the scale of coordinates.
                  The tensor shape is [batch_size, num_boxes, 4].
    :param scores: A tensor, specified the input tensor X2.
                   The tensor shape is [batch_size, num_boxes].
    :param number: A tensor, specified the input tensor X3.
                   The tensor shape is [batch_size, num_classes].
    :param label: A tensor, specified the input tensor X4,
                  The tensor shape is [batch_size, num_classes].
    :param classes: A tensor, specified the input tensor X5.
                    The tensor shape is [batch_size, 1].
    :param score_threshold: The of score for valid bounding boxes.
    :param max_box_num: Integer representing the maximum number of boxes to be selected per batch. It is a scalar.
    :param minsize: The minimum size of a valid bounding box. [h,w].
    :param quantization: A "Quantization" vector, specifying the (Y1,Y2,,Y4,Y5) scale and offset.
    :return: A tensor list, which are output tensor Y1,Y2,Y3,Y4,Y5.
    """
    minsizes = convert_to_list(minsize, "minsize", length=2)
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.filter_boxes(
        boxes,
        scores,
        number,
        label,
        classes,
        score_threshold,
        max_box_num,
        minsizes,
        quantizations,
    )
