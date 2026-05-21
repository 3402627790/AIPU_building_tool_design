import AIPUBuilder._C._ops as _ops
from AIPUBuilder.core.tensor import *
from AIPUBuilder.core.node import *
from AIPUBuilder.core.graph import *
from AIPUBuilder.core.builder_plugin import *
from AIPUBuilder.ops.common import *
from typing import Tuple, Union, List


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def batch_to_space(
    input: Tensor, block_shape: int or list, crops: list = [[0, 0], [0, 0]]
) -> Tensor:
    """
    This operation reshapes the batch dimension (dimension 0) into M + 1 dimensions of
    shape block_shape + [batch],interleaves these blocks back into the grid defined
    by the spatial dimensions [1, ..., M],to obtain a result with the same rank as the input.

    :param input: A Tensor,specifying the input tensor, rank must be 4.
    :param block_shape: An int scalar or list.
            this parameter will be active on both height and width.
    :param crops: Must be one of the following types: `int32`, `int64`.
                  2-D with shape `[2, 2]`, all values must be >= 0.
                  `crops[i] = [crop_start, crop_end]` specifies the amount to crop from input
                  dimension `i + 1`, which corresponds to spatial dimension `i`. It isrequired that
                  `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.
                  example: `crops = [[0, 0], [0, 0]]`
    :return: A Tensor has same data type and scale/offset with `input`. Output shape
            can be inferred by block_shape and crops.
    """
    block_shape = convert_block_shape(block_shape)
    check_crops_or_padding(crops)
    ret = _ops.batch_to_space(input, block_shape, crops)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def batch_to_space_nd(
    input: Tensor, block_size: int or list, crops: list = [[0, 0], [0, 0]]
) -> Tensor:
    """
    This operation reshapes the "batch" dimension 0 into M + 1 dimensions of
    shape block_shape + [batch], interleaves these blocks back into the grid
    defined by the spatial dimensions [1, ..., M],
    to obtain a result with the same rank as the input.
    The spatial dimensions of this intermediate result
    are then optionally cropped according to crops to produce the output.
    This is the reverse of SpaceToBatch.

    :param input: A Tensor,specifying the input tensor.
    :param block_size: It is a list that means that blocks are moved,
            where block_size is the block size along with the excepte batch and
            c dimension in data format.
            Generally, the output tensor will be
            [N/prod(block_size), shape * block_size - crop_top - crop_bottom,...,C].
    :param crops: It is a list that means the outputs tensor after cropping
                the dimension with the size of crop_top and crop_bottom.
    :return: A Tensor has same data type and scale/offset with `input`. Output shape
            can be inferred by block_size and crops.
    """
    block_size = convert_block_shape(block_size, len(crops))
    check_crops_or_padding(crops, len(crops))
    ret = _ops.batch_to_space_nd(input, block_size, crops)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bit_shift(
    input: Tensor, shift: Tensor, direction: str, quantization: Quantization = None
) -> Tensor:
    """
    Bitwise shift operator performs element-wise operation.
    For each input element, if the attribute "direction" is "Right",
    this operator moves its binary representation toward the right side so that
    the input value is effectively decreased. If the attribute "direction" is "Left",
    bits of binary representation moves toward the left side,
    which results the increase of its actual value.
    The input X is the tensor to be shifted and another input Y specifies the amounts of shifting.
    For example, if "direction" is "Right", X is [1, 4], and S is [1, 1],
    the corresponding output Z would be [0, 2].
    If "direction" is "Left" with X=[1, 2] and S=[1, 2],
    the corresponding output Y would be [2, 8].

    :param input An n-D Tensor, specifying the tensor needs to be shifted.
    :param shift A Tensor, specifying the shift tensor.
    :param direction Specifying the direction of shift, it can be "Right" or "Left".
    :param quantization A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, specifying shifted tensor.
    """
    return _ops.bit_shift(
        input,
        shift,
        str_to_enum_member(_ops.ShiftDirection, direction),
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def channel_shuffle(input: Tensor, groups: int, split: int = 1) -> List[Tensor]:
    """
    Shuffle the channels of the input tensor.

    :param input: An n-D Tensor, specifying the tensor to be shuffled.
    :param groups: Number of groups to divide channels in.
    :param split: An positive integer. The number of output tensors, that is,
                  how many numbers of output tensors will be split from the
                  inputs after channel shuffle. Default value is 1.
    :return: A list of Tensors, lenght should be equal to `split` number.
    """
    dim_len = input.shape.__len__()
    return _ops.channel_shuffle(input, groups, split)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def compress(input: Tensor, condition: Tensor, axis: int = 0) -> Tensor:
    """
    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress.

    :param input: A Tensor, specifying the input tensor.
    :param condition: A list of Tensor, must have data.The data value should be 0 or 1.
    :param axis: A scalar means the axis to take slices.If not specified, input is flattened before elements being selected.
                 Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
    :return: A Tensor, its shape are described by parameter `axis` and parameter `condition`.
    """
    dim_len = input.shape.__len__()
    check_axis_value(axis, dim_len)
    return _ops.compress(input, condition, axis)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def concat(
    inputs: List[Tensor], axis: int = -1, quantization: Quantization = None
) -> Tensor:
    """
    Concatenate a list of tensors into a single tensor. All input tensors must have the same shape,
    except for the dimension size of the axis to concatenate on.

    :param inputs: List of Tensor for concatenation.
    :param axis: axis Which axis to concat on. A negative value means counting dimensions from the back.
                 Accepted range is [-r, r-1] where `r = rank(inputs)`.
                 Default value is -1.
    :param quantization: A 'Quantization' object, specifying the output data scale and offset.
    :return: A Tensor,resulting from concatenation of the param `input`.
    """
    return _ops.concat(inputs, axis, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def cacheupdate(
    cache: Tensor, position_id: Tensor, value: Tensor, block_index: int, axis: int = 2
) -> List[Tensor]:
    """
    Updates the given cache in inplace mode.

    :param cache: A Tensor, specifying the cache to update. Its shape is 3-D or 4-D,
                  If shape is 3-D, only one cache block is allowed, block_index must be 0,
                  and shape[1] means the max token length. If shape is 4-D,
                  shape[0] means the number of blocks in cache, and shape[2] means the max token length.
    :param position_id: A Tensor, specifying the current token index.
                        Its shape is 2-D, and type can be int32 or uint32. Currently,
                        only continuous indexes are supported in prefill mode.
    :param value: A Tensor, specifying the current value to fill into cache.
                  Its shape is 4-D, and shape[2] must be less than or equal to cache's max token length.
    :param block_index: A int scalar, specifying which block in cache to update.
                        It must be less than cache's block count.
    :param axis: A int scalar, specifying which axis to update. Supported value: 2, 3.
                 Default value is 2.
    :return: A list of Tensors, specifying the output tensors. It has two outputs,
             and output0 is the updated cache block, output1 is the updated whole cache.
    """
    return _ops.cacheupdate(cache, position_id, value, block_index, axis)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_constant_op
def col2im(
    input: Tensor,
    image_shape: Tensor,
    block_shape: Tensor,
    strides: list = [],
    padding: list = [],
    dilations: list = [],
    quantization: Quantization = None,
) -> Tensor:
    """
    The operator rearranges column blocks back into a multidimensional image.

    :param input: A tensor, specifying input tensor.
                  Input data tensor to be rearranged from column blocks back into an image.S
                  This is a 3-dimensional tensor containing [N, C * n-ary- product(block_shape), L], where N is batch dimension,
                  C is image channel dimension and L is number of blocks.The blocks are enumerated in increasing lexicographic-order
                  of their indices.For example, with an image-size 10*20 and block-size 9*18, there would be 2*3 blocks,
                  enumerated in the order block(0, 0), block(0, 1), block(0, 2), block(1, 0), block(1, 1), block(1, 2).
    :param image_shape: The shape of the spatial dimensions of the image after rearranging the column blocks.
                        This is a 1-dimensional tensor with size of at least 2, containing the value [H_img, W_img] for a 2-D image
                        or [dim_i1, dim_i2, ..., dim_iN] for a N-D image.
    :param block_shape: The shape of the block to apply on the input.This is a 1-dimensional tensor of size of at least 2,
                        containing the value [H_block, W_block] for a 2-D image or [dim_b1, dim_b2,..., dim_bN] for a N-D block.
                        This is the block-shape before dilation is applied to it.
    :param strides: 1-dimensional tensor with stride value along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
    :param padding: 1-dimensional tensor with padding value for the beginning and ending along each spatial axis,
                    it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning
                    and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin ... x1_end, x2_end,...],
                    where xi_begin is the number of pixels added at the beginning of axis i and xi_end is the number of pixels
                    added at the end of axis i. If not present, the padding defaults to 0 along start and end of each spatial axis.
    :param dilations: 1-dimensional tensor with dilation value along each spatial axis of the image.
                      If not present, the dilation defaults to 1 along each spatial axis of the image.
    :param quantization: A "Quantization" object, specifying output quantization parameters.
    :return: A tensor specifying the output tensor.
    """
    return _ops.col2im(
        input,
        image_shape,
        block_shape,
        strides,
        _ops.Padding(padding),
        dilations,
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def crop(
    input: Tensor,
    crops: List[tuple],
) -> Tensor:
    """
    Crops the input tensor into the output tensor with the shape computing by the given beginning
    and end index (exclusive the ending indices) and the shape of the output tensor.
    For example, considering an input tensor X. `shape = [1, 4, 5, 3]`, `crops = [[0, 1], [1, 4], [2, 4], [0, -1]]`,
    and the result will return a tensor Y with shape [1, 3, 2, 2].

    :param input: A Tensor, specifying input tensor.
    :param crops: A nested list means cropping from the input tensor (X), such as `crops =
                  [[crop0_begin, crop0_end],[crop1_begin, crop1_end], ..., [crop[n]_begin, crop[n]_end]]`,
                  where [D, 0] means the effective beginning indices and [D, 1] means
                  the effective ending indices (exclusive the ending indices) in the D dimension.
                  Generally, a negative value means cropping backward.
    :return: A Tensor,with the same shape and type as the tensors in `input`.
    """

    if not isinstance(crops, list):
        raise ValueError(
            "Op 'crop' expected argument 'crops' as [[crop0_begin, crop0_end],[crop1_begin, crop1_end], ..., [cropn_begin, cropn_end]]"
        )
    args = [[a if isinstance(a, Expr) else Expr(a) for a in crop] for crop in crops]
    return _ops.crop(input, [], args)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def data_stride(input: Tensor, kernel: int or list, stride: int or list) -> Tensor:
    """
    Performs an input data tensor (X) to produce a output tensor (Y) with a stride combination,
    which means taking kernel size numbers of data after every stride operation and gathering all these data along each axis respectively.
    For example, given an input tensor with the shape as [1, 8, 8, 3] and kernel_x (kernel_y) =3, stride_x (stride_y) = 3,
    then it returns an output tensor Y with the shape as [1, 6, 6, 3].

    :param input: A Tensor, specifying the input tensor, rank must be 4.
    :param kernel: An int32 scalar, specifying the cut size during each stride.
                  It will be active on both height and with.
    :param stride: An int32 scalar, specifying the each stride size.
                  It will be active on both height and with.
    :return: A Tensor, which has same data type and scale/offset with `input`.
            Output shape can be inferred by param 'kernel' and param `stride`.
    """
    kernel = convert_to_kernels(kernel)
    stride = convert_to_strides(stride)
    return _ops.data_stride(input, kernel, stride)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def depth_to_space(input: Tensor, block_size: int, mode: str = "DCR") -> Tensor:
    """
    DepthToSpace permutes data from depth into blocks of spatial data. This is the reverse transformation of SpaceToDepth.
    More specially, this operator outputs a copy of the input tensor where values from the depth dimension
    are moved in spatial blocks to the 'height' and 'width' dimensions.
    In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the following order:
    'height', 'width' and 'channel'.
    The input will be reshape into [N, H , W , block_size_y, block_size_x, C/(block_size_y * block_size_x)] ,
    and transpose to [N, H, block_size_y, W, block_size_x, C/(block_size_y * block_size_x)].
    The output shape will be reshape to [N, H * block_size_y, W * block_size_x, C/(block_size_y * block_size_x)].
    In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the following order:
    'height', 'width' and 'channel'.
    The input will be reshape into [N, H , W , C/(block_size_y * block_size_x), block_size_y, block_size_x] ,
    and transpose to [N, H, block_size_y, W, block_size_x, C/(block_size_y * block_size_x)].
    The output shape will be reshape to [N, H * block_size_y, W * block_size_x, C/(block_size_y * block_size_x)].
    The output shape will be [N, H * block_size_y, W * block_size_x, C/(block_size_y * block_size_x)].

    :param input: specifying the input tensor, rank must be 4.
    :param block_size: An int32 scalar, specifying the size of spatial block.
    :param mode: Indicates in which mode the elements along the depth dimension from the input tensor are rearranged.
                Support "DCR", "CRD". "DCR" is the default mode.
    :return: A Tensor, which has same data type and scale/offset with param `input`.
            Output shape can be inferred by block_size.
    """
    c = input.shape.c
    if c % (block_size * block_size) != 0:
        raise ValueError(" argument 'block_size' invalid: %r " % block_size)

    ret = _ops.depth_to_space(
        input, block_size, str_to_enum_member(_ops.DepthToSpaceMode, mode)
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def embedding_lookup(params: Tensor, indices: Tensor) -> Tensor:
    """
    Split a tensor into a list of sub tensors.

    :param params: A Tensor representing the complete embedding tensor.
    :param indices: A Tensor with type `int32` containing the indices to be looked up in `params`.
    :return: A Tensor with the same type as the tensors in `params`.
    """
    return gather(params, indices, 0)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def embedding_lookup_sparse(
    params: Tensor,
    sp_ids: Tensor,
    sp_value: Tensor,
    weight_value: Tensor,
    dense_shape: list = [],
    combiner: str = "MEAN",
    max_norm: float = 0,
    quantization: Quantization = None,
) -> Tensor:
    """
    This operator assumes that there is at least one ID for each row in the dense tensor represented
    by sp_ids (i.e. there are no rows with empty features), and that all the indices of sp_ids are in canonical row-major order.
    sp_ids and sp_weights (if not None) are SparseTensors or RaggedTensors with rank of 2.
    For SpareTensors with left-aligned non-zero entries which can be described as RaggedTensors,
    use of RaggedTensors can yield higher performance.
    It also assumes that all ID values lie in the range [0, p0), where p0 is the sum of the size of params along dimension 0.
    If len(params) > 1, each element of sp_ids is partitioned between the elements of params according to
    the "div" partition strategy, which means we assign ids to partitions in a contiguous manner.
    For instance, 13 IDs are split across 5 partitions as: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]].
    If the ID space does not evenly divide the number of partitions, each of the first (max_id + 1) % len(params) partitions
    will be assigned one more id.
    Note that, in the following Inputs sections:
    X1: params, a single tensor representing the complete embedding tensor.
    X2: sp_ids, a tensor indicating the output indeces.
    X3: sp_value, a tensor indicating the sp_value , which means how to gather the params.
    X4: weight_value, a tensor means the weight for the gathered params to output.

    :param params:  A tensor representing the complete embedding tensor.
    :param sp_ids: A tensor indicating the output indices.
    :param sp_value: A tensor indicating the sp_value , which means how to gather the params.
    :param weight_value:  A tensor means the weight for the gathered params to output.
    :param dense_shape: [N, M] where N is typically batch size and M is arbitrary or a RaggedTensor with rank 2.
    :param combiner: A string specifying the reduction op.
                        "SUM" computes the weighted sum of the embedding results for each row.
                        "MEAN" is the weighted sum divided by the total weight.
                        "SQRTN" is the weighted sum divided by the square root of the sum of the squares of the weights.
                        Defaults to "MEAN".
    :param max_norm: A float value. If not NONE, each embedding is clipped if its l2-norm is larger than this value, before combining.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A tensor specifying the output tensor.
    """
    if len(dense_shape) == 2:
        m_shape = convert_to_shape(dense_shape)
    else:
        m_shape = TensorShape()
    return _ops.embedding_lookup_sparse(
        params,
        sp_ids,
        sp_value,
        weight_value,
        m_shape,
        combiner=str_to_enum_member(_ops.EmbeddingLookupSparseCombiner, combiner),
        max_norm=max_norm,
        quantization=convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def expand_dims(input: Tensor, axis: int) -> Tensor:
    """Returns a tensor with a length 1 axis inserted at index `axis`..
    Given tensor, this operation returns a new tensor that has the same
    values as tensor in the same order, except with a new shape.

    :param input: A Tensor,specifying the input tensor.
    :param axis: specifying the dimension index at which to expand the shape.
                 For example: input tensor shape is [2,1,4]`,  and axis is 3,
                 then return tensor shape is [2,1,4,1].
    :return: An expanded tensor.
    """
    ret = _ops.expand_dims(input, axis)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def gather(input: Tensor, indices: Tensor, axis: int, batch_dims: int = 0) -> Tensor:
    """
    Given parameters and indices tensor, gathers entries of the parameters tensor indexed by indices
    and concatenates them in an output tensor. The indices are a tensor of integer,
    and there will be an error if any of the index values are out of bounds.
    Generally, a negative value in the indices tensor means counting the index from the back.

    :param input:  A Tensor, which to gather values. Must be at least rank `axis + 1`.
    :param indices: A tensor pointer with type `int32` containing the indices to be looked up in param `input`.
    :param axis: Which axis to gather on. Negative value means counting dimensions from the back.
                 Accepted range is [-r, r-1] where `r = rank(data)`.
    :param batch_dims: A scalar of integer to indicate the number of batch dimensions. Generally, defaults to 0.

    :return: A tensor pointer with the same type as the param `input`.
    """
    dim_len = input.shape.__len__()
    if axis < -dim_len or axis > dim_len - 1:
        raise ValueError(
            "argument 'axis' Accepted range is [-%d, %d]" % (-dim_len, dim_len - 1)
        )

    return _ops.gather(input, indices, axis, batch_dims)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def gather_elements(input: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """
    GatherElements takes two inputs data and indices of the same rank r >= 1 and an optional attribute
    axis that identifies an axis of data (by default, the outer-most axis, that is axis 0).
    It is an indexing operation that produces its output by indexing into the input data tensor at
    index positions determined by elements of the indices tensor.
    Its output shape is the same as the shape of indices and consists of one value
    (gathered from the data) for each element in indices.

    :param input: A Tensor, which to gather values.
    :param indices: A Tensor, specifying fetch index of `input`.
    :param axis: Which axis to gather on. Negative value means counting dimensions from the back.
                 Accepted range is [-r, r-1] where r = rank(data).
    :return: A tensor
    """
    check_axis_value(axis, input.shape.__len__())
    return _ops.gather_elements(input, indices, axis)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def gather_nd(input: Tensor, indices: Tensor, batch_dims: int = 0) -> Tensor:
    """
    Given the input indices tensor, then gather slices from input tensor X into the output tensor Y
    with shape specified by indices. The output tensor has the shape `indices.shape[ :-1] + X.
    shape[indices.shape[-1] + batch_dims: ]` with the left closed and right open interval.
    Generally, `indices[-1] + batch_dims` should be less than or equal to `rank(X)`. Note that
    the negative value in indices tensor means counting the index from the back.

    :param input: input  A Tensor, which to gather values.
    :param indices: A tensor pointer with type `int32` containing the indices to be looked up in `input`.
    :param batch_dims: A scalar of integer to indicate the number of batch dimensions.
                       Generally, defaults to 0 if not present.
                       Currently only support `batch_dims=0`.
    :return: A Tensor with the same type as the `input`.
    """
    return _ops.gather_nd(input, indices, batch_dims)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def left_shift(
    input: Tensor, shift: Tensor, quantization: Quantization = None
) -> Tensor:
    """
    The left shift operator performs element-wise operations. For each input element, this operator moves its binary representation
    toward the left side so that the input value is effectively increased.
    The input X is the tensor to be shifted and another input Y specifies the amounts of shifting.
    For example, X is [1, 4], and Y is [1, 1], the corresponding output Z will be [2, 8]. This operator supports multidirectional broadcasting.

    :param input: A Tensor, specifying the tensor needs to be shifted.
    :param shift: A Tensor, specifying the shift tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, specifying shifted tensor.
    """
    return _ops.left_shift(input, shift, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def meshgrid(
    inputs: List[Tensor],
    indexing: str = "xy",
    sparse: bool = False,
    copy: bool = True,
    quantization: List[Quantization] = [],
) -> List[Tensor]:
    """
    Returns coordinate matrices from coordinate vectors.
    Makes N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields over N-D grids,
    given one-dimensional coordinate arrays x1, x2,..., xn.
    This function supports both indexing conventions through the indexing keyword argument.
    Giving the string 'ij' returns a meshgrid with matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.
    In the 3-D case with inputs of length M, N and P, outputs are of shape (N, M, P) for 'xy' indexing and (M, N, P) for 'ij' indexing.

    :param inputs: A Tensor object list containing the input data tensor X1, X2, ..., XN.
    :param indexing: Cartesian ('xy', default) or matrix ('ij') indexing of output.s
    :param sparse: If it is 'true' the shape of the returned coordinate array for dimension i is reduced from (N1, ..., Ni, ... Nn) to (1, ..., 1, Ni, 1, ..., 1).
               These sparse coordinate grids are intended to be used with broadcasting.
               When all coordinates are used in an expression, broadcasting still leads to a fully-dimensional result array.
               By default is 'false'.
    :param copy: If it is 'false', a view into the original arrays is returned to conserve memory.
               By default it is 'true'.
               Note that sparse=false, copy=false will likely return non-contiguous arrays.
               Furthermore, more than one element of a broadcast array may refer to a single memory location.
               If you need to write to the arrays, make copies first.
               Note: Currently only supports 'true'.
    :param quantization: A "Quantization" list, specifying the (Y1, Y2, ..., YN) scale and offset.
                If it is empty, Y1's quantization parameter is the same as x1, and YN's quantization parameter is the same as xN.
                If it is not empty and the number M is less than N, then the output after M uses the first quantization parameter.
    :return: A Tensor object list containing the output tensors Y1, Y2, ..., YN.
    """
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.meshgrid(inputs, indexing, sparse, copy, quantization=quantizations)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def pad(
    input: Tensor,
    pads: List[Tuple[int | Expr, int | Expr]],
    mode: str = "Constant",
    constant_value: float or int = 0.0,
) -> Tensor:
    """
    Given the input tensor containing the data to be padded,
     a tensor containing the number of the start and end pad values for specified axis and mode.

    :param input: A Tensor, specifying the input tensor.
    :param pads: A nested list means the padding value with shape [n, 2] to a `rank=n` data tensor.
                 Padding values support expressions.
    :param mode: Mode to add padding. The default value is `Constant`.

                .. code-block:: none

                 `Constant` pads with a given value as specified by constant_value.
                 `Reflect` pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
                 `Symmetric` pads with the symmetric value.
                 `Edge` pads with the edge values of array.
                 `Wrap` wrap-around padding as if the data tensor forms a torus.

    :param constant_value: A float or int scalar value to be used if the mode chosen is CONSTANT. The default value is 0..
                           - float, means `constant_value` is non-quantized.
                           - int, means `constant_value` is quantized.
    :return: A Tensor, specifying the output tensor.
             Output has the same data type and scale/offset with `input`.
    """
    inp_len = input.shape.__len__()

    has_expr = any([isinstance(s, (Expr, Symbol)) for pad in pads for s in pad])
    if has_expr:
        return _ops.pad(
            input,
            [],
            convert_to_padmode(mode),
            constant_value,
            [[s if isinstance(s, Expr) else Expr(s) for s in pad] for pad in pads],
        )
    else:
        check_crops_or_padding(pads, inp_len)
        return _ops.pad(input, pads, convert_to_padmode(mode), constant_value, [])


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def query_rebatch(inputs: TensorList) -> Tensor:
    """
    QueryRebatch implements the QueryRebatch operation.
    The input of this operator is [X1, X2~XN], and the output is Y.
    X1 means the input feature map that will be reranged by new indices X2~XN.
    X2~XN means the new indices. Every new indices input will be a new sub-batch of output.
    The following equation describes the QueryRebatch operation:
    for j in output batch
       for i, index in enmuerate(X2~XN)
            indices=Xi
            output[j,i,:len(Xi)] = X1[j,indices]
    :param inputs: A tensor list, specifying the input tensor X1~XN.
    :return: A tensor, specifying the output tensor Y.
    """
    return _ops.query_rebatch(inputs)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def repeat(input: Tensor, repeats: Tensor, shape: list, axis: int = None) -> Tensor:
    """
    Repeat input tensor according axis.

    :param input: A Tensor, specified the input tensor.
    :param repeats: A Tensor, specified 1D tensor, its length must be equal with `input[axis]`, it axis is None,
                    length of repeats should be 1.
    :param axis: Default is None, its range can be [-rank, rank) of input tensor.
    :param shape: List,Tuple or TensorShape object, specified output shape.
    :return: A Tensor, specifying the output tensor. Output has the same data type and scale/offset with `input`.
            Output shape will be inferred by `repeats`.
    """
    check_axis_value(axis, input.shape.__len__())
    output_shape = convert_to_shape(shape)
    if axis is None:
        ret = _ops.repeat(input, repeats, output_shape)
    else:
        ret = _ops.repeat(input, repeats, output_shape, axis)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reshape(
    input: Tensor,
    shape: Union[list, tuple, TensorShape, SymbolShape],
) -> Tensor:
    """
    Reshape a tensor.
    Given tensor, this operation returns a new tensor that has the same
    values in the same order, except with a new shape given by `shape`.

    :param input: A Tensor, specifying input tensor.
    :param shape: Specifying output shape, support List, Tuple or TensorShape
                  for example: [2,3,4,5], TensorShape(2,3,4,5)
    :param exprs: The dynamic shape expression.
    :return: A Tensor with new shape.
    """
    has_expr = isinstance(shape, (list, SymbolShape)) and any(
        [isinstance(s, Expr) and not s.is_constant() for s in shape]
    )
    if (
        not has_expr
        and not input.shape.symbol.is_constant
        and any([s.value == -1 if isinstance(s, Expr) else s == -1 for s in shape])
    ):
        has_expr = True
    if has_expr:
        exprs_formated = [s if isinstance(s, Expr) else Expr(s) for s in shape]
        return _ops.reshape(input, TensorShape(), exprs_formated)
    else:
        m_shape = TensorShape(input.shape)
        m_shape.reshape(
            convert_to_shape(
                [int(s.value) if isinstance(s, Expr) else s for s in shape]
            )
        )
        return _ops.reshape(input, m_shape, [])


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def reverse_sequence(
    input: Tensor, seq_lengths: Tensor = None, seq_axis: int = 1, batch_axis: int = 0
) -> Tensor:
    """
    Reverses batch of sequences having different lengths specified by seq_lengths.
    For each slice `i` iterating on the batch axis, the operator reverses the first `seq_lengths[i]`
    element to `time_axis`, and copies elements whose indexes are beyond `seq_lengths[i]` to the output.
    So, the output slice `i` includes reverse sequences elements in the first `seq_lengths[i]`
    and original elements copied beyond the seq_lengths index. Besides, another important parameter is
    len which indicates the effective length to reverse sequence. The [T, N, C] data format
    means `[time_step, batch_size, channel]`.

    :param input: A Tensor. Supports dimensions: 2-D,3-D.
    :param seq_lengths: A Tensor. 1-D with length `tensor.dims(batch_axis)` and `max(seq_lengths) <=tensor.dims(seq_axis)`.
    :param seq_axis: Specifies which axis is the time axis. Generally, the default value is 0 in [T, N, C] data format.
                     Parameter is temporarily not supported, using default.
    :param batch_axis: batch_axis Specifies which axis is the batch axis. Generally,
                       the default value is 1 in [T, N, C] data format.
                       Parameter is temporarily not supported, using default.
    :return: A Tensor, with the same shape and type as the tensors in `input`.
    """
    return _ops.reverse_sequence(input, seq_lengths, seq_axis, batch_axis)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def right_shift(
    input: Tensor, shift: Tensor, quantization: Quantization = None
) -> Tensor:
    """
    The right shift operator performs element-wise operations. For each input element, this operator moves its binary representation
    toward the right side so that the input value is effectively increased.
    The input X is the tensor to be shifted and another input Y specifies the amounts of shifting.
    For example, X is [1, 4], and Y is [1, 1], the corresponding output Z will be [0, 2]. This operator supports multidirectional broadcasting.

    :param input: A Tensor, specifying the tensor needs to be shifted.
    :param shift: A Tensor, specifying the shift tensor.
    :param quantization: A Quantization object, specifying the output quantization parameters.
    :return: A Tensor, specifying shifted tensor.
    """
    return _ops.right_shift(input, shift, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def scatter_elements(
    input: Tensor,
    indices: Tensor,
    updates: Tensor,
    reduction: str = "None",
    axis: int = 0,
    quantization: Quantization = None,
) -> Tensor:
    """
    ScatterElements takes three inputs data, updates, and indices of the same rank `r >= 1` and
    an optional attribute axis that identifies an axis of data (by default, the outer-most axis, that is axis 0).
    The output of the operation is produced by creating a copy of the input data, and then updating its value
    to values specified by updates at specific index positions specified by indices.

    :param input: A Tensor, specifying the input data tensor.
    :param indices: A Tensor, specifying the indices.
    :param updates: A Tensor, specifying values to scatter into the output tensor.
    :param reduction: A string, specifying the reduction method, default is `none`, bellow are the choice:
                        `None`: no reduction applied.
                        `Add`: reduction using the addition operation.
                        `Mul`: reduction using the multiplication operation.
                        `Min`: reduction using the min operation.
                        `Max`: reduction using the max operation.
    :param axis: A scalar int32, specifying axis scatter on.
    :param quantization: A Quantization object, specifying the output quantization parameters, default to empty.
    :return: A Tensor, which has the same shape as `input`.
    """
    assert (
        input.shape.__len__() == indices.shape.__len__()
    ), "Rank of input and indices must be same!"
    assert (
        input.shape.__len__() == updates.shape.__len__()
    ), "Rank of input and updates must be same!"
    check_axis_value(axis, input.shape.__len__())
    return _ops.scatter_elements(
        input,
        indices,
        updates,
        str_to_enum_member(_ops.ScatterReduction, reduction),
        axis,
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def scatter_nd(
    input: Tensor,
    indices: Tensor,
    updates: Tensor,
    reduction: str,
    quantization: Quantization = None,
) -> Tensor:
    """
    Scatters updates into a tensor of input according to indices.
    ScatterND takes three inputs data tensor of rank `r >= 1`, indices tensor of rank `q >= 1`, and updates tensor of rank `q + r - indices.shape[-1] - 1`.
    The output of the operation is produced by creating a copy of the input data, and then updating its value to values specified by updates at
    specific index positions specified by indices. Its output shape is the same as the shape of input.
    Note that indices should not have duplicate entries. That is, two or more updates for the same index-location is not supported.

    :param input: A Tensor, specifying the input data tensor.
    :param indices: A Tensor, specifying the indices.
    :param updates: A Tensor, specifying values to scatter into the output tensor.
    :param reduction: A string, specifying the reduction method, default is 'none', bellow are the choice:
                      `None`: no reduction applied.
                      `Add`: reduction using the addition operation.
                      `Mul`: reduction using the multiplication operation.
                      `Min`: reduction using the min operation.
                      `Max`: reduction using the max operation.
    :param quantization: A Quantization object, specifying the output quantization parameters, default to empty.
    :return: A Tensor, which has the same shape as `input`.
    """
    input_rank = input.shape.__len__()
    indices_rank = indices.shape.__len__()
    updates_rank = updates.shape.__len__()
    assert (
        indices.shape[-1] <= input_rank
    ), "The third input indices.shape[-1] must be <= input.rank!"
    assert (
        updates_rank == input_rank + indices_rank - indices.shape[-1] - 1
    ), "It must follow: updates.rank==input.rank+indices.rank-indices.shape[-1]-1!"
    return _ops.scatter_nd(
        input,
        indices,
        updates,
        str_to_enum_member(_ops.ScatterReduction, reduction),
        convert_quantization(quantization),
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def slice(
    input: Tensor,
    begin: list,
    end: list,
    strides: list,
    begin_mask: int = 0,
    end_mask: int = 0,
    shrink_axis_mask: int = 0,
) -> Tensor:
    """
    Extracts a strided slice of a tensor.
    Roughly speaking, this op extracts a slice of size (end - begin) / stride from the given input tensor.
    Starting at the location specified by begin the slice continues by adding stride to the index until
    all dimensions are not less than end. Note that a stride can be negative, which causes a reverse slice.

    :param input: A Tensor, specifying the input tensor.
    :param begin: An int32 list, specifying the starts of the dimensions of the input tensor to be sliced.
                 Size must be rank of `input`.
    :param end: An int32 vector, specifying the ends of the dimensions of the input tensor to be sliced.
                 Size must be rank of `input`.
    :param strides: An int32 list, strides of the dimensions of the input tensor to be sliced.
                   Size must be rank of `input`. The value cannot be zero.
    :param begin_mask: An int32 scalar. If the i-th bit of param `begin_mask` is set, param `begin[i]` is ignored and
                       the fullest possible range in that dimension is used instead.
    :param end_mask: An int32 scalar. If the i-th bit of param `end_mask` is set, param `end[i]` is ignored and
                     the fullest possible range in that dimension is used instead.
    :param shrink_axis_mask: An int32 scalar. If the i-th bit of param `shrink_axis_mask` is set,
                             the i-th dimension specification shrinks the dimensionality by 1, taking on the value at index param `begin[i]`.
                             In this case, the i-th specification must define a slice of size 1.
                             e.g. param begin[i] = x, param end[i] = x + 1.
    :return: A Tensor, specifying the output tensor, which has the same data type and scale/offset as `input`.
            Output shape will be inferred by params.
    """
    return _ops.slice(
        input, begin, end, strides, begin_mask, end_mask, shrink_axis_mask
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def dynamic_slice(
    inputs: TensorList,
) -> TensorList:
    """
    Produces a slice of the input tensor along multiple axes.
    Slice uses the starts, ends, axes and steps inputs to select a sub-tensor of its input data tensor.

    :param inputs: A Tensor list, specifying the input tensors.
                 The order is [data, starts, ends, axes, steps].
                 The axes and steps are optional.
    :return: A Tensor list, specifying the output tensor, and output shape tensor.
    """
    return _ops.dynamic_slice(inputs)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def sort(input: Tensor) -> TensorList:
    """
    Sorts the elements of the input tensor along a given dimension in ascending order by value.

    :param input: A tensor, specifying the input tensor.
     * @return A tensor list, specifying the output tensor, which has the same shape with input.
     *         It contains 'result' and 'indices'.
    """
    return _ops.sort(input)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def space_to_batch(
    input: Tensor, block_shape: list, paddings: List[Tuple[int, int]] = [[0, 0], [0, 0]]
) -> Tensor:
    """
    This operation divides spatial dimensions(height/width) of the input into a grid of blocks of shape block_shape,
    and interleaves these blocks with the batch dimension (0) such that in the output, the spatial dimensions
    correspond to the position within the grid, and the batch dimension combines both the position within
    a spatial block and the original batch position.

    :param input: A Tensor, specifying the input tensor, rank must be 4.
    :param block_shape: An int list, size should be 2. The order is height, width.
    :param paddings: A int list,  Must be one of the following types: `int32`, `int64`.
                     the padding of the input with zeros across the spatial dimensions as follows:
                     paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    :return: A Tensor, which has same data type and scale/offset with `input`.
             Output shape can be inferred by param `block_shape` and param `paddings`.
    """
    block_shape = convert_block_shape(block_shape)
    check_crops_or_padding(paddings)
    ret = _ops.space_to_batch(input, block_shape, paddings)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def space_to_depth(input: Tensor, block_size: int) -> Tensor:
    """
    SpaceToDepth rearranges blocks of spatial data into depth. this op outputs a copy of the input tensor
    where values from the height and width dimensions are moved to the depth dimension.

    :param input:A Tensor, specifying the input tensor, rank must be 4.
    :param block_size: An int scalar, specifying the size of spatial block.
    :return: A Tensor, which has same data type and scale/offset with `input`. Output shape can be inferred by `block_size`.
    """
    ret = _ops.space_to_depth(input, block_size)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def split(input: Tensor, splits: list or int, axis: int = -1) -> List[Tensor]:
    """
    Splits a tensor into multiple sub-tensors along the specified axis.

    :param input: The input tensor to be split.
    :param splits: If it is an integer, specifies the number of equal-sized splits along the specified axis.
                   If it is a list or tuple, specifies the size of each split along the specified axis.
    :param axis: The axis along which to split the input tensor.
    :return: A list of tensors that result from splitting the input tensor.
    """
    if not isinstance(splits, (int, list, tuple)):
        raise TypeError(
            "Split param 'splits' needs to be (int,list,tuple), but now: %s"
            % (type(splits))
        )
    if isinstance(splits, (list, tuple)):
        has_expr = any([isinstance(s, (Expr, Symbol)) for s in splits])
        if has_expr:
            return _ops.split(
                input, [], axis, [s if isinstance(s, Expr) else Expr(s) for s in splits]
            )
        else:
            return _ops.split(input, splits, axis)
    else:
        return _ops.split(input, splits, axis)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def squeeze(input: Tensor, axis: list = []) -> Tensor:
    """
    Removes dimensions of size 1 from the shape of a tensor.
    If parameter `axis` is not specified, all dimensions with size 1 will be removed.
    Given a tensor, this operation returns a new tensor that has the same
    values in the same order.

    :param input: A Tensor, specifying input tensor.
    :param axis: A int32 or List, Tuple, specifying squeeze dimensions on.
                 If not specified, it will squeeze all [1] dimensions.
                 For example: input tensor shape is [2,1,4,1],  and axis is [1],
                 then return tensor shape is [2,4,1].
    :return: A squeezed tensor.
    """
    dim_len = input.shape.__len__()
    if isinstance(axis, int):
        axis = [axis]
    elif isinstance(axis, (list, tuple)):
        for ax in axis:
            if ax > dim_len - 1:
                raise ValueError(
                    "Op 'squeeze' expected argument 'axis' %d < %d" % (ax, dim_len)
                )
    else:
        raise TypeError(
            "Op 'squeeze' axis must be list, tuple or int, but now %s" % (type(axis))
        )
    ret = _ops.squeeze(input, axis)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def strided_slice(
    input: Tensor,
    begin: list,
    end: list,
    strides: list,
    begin_mask: int = 0,
    end_mask: int = 0,
    shrink_axis_mask: int = 0,
) -> Tensor:
    """
    Extracts a strided slice of a tensor.
    This operation extracts a slice from the input tensor, starting at the location specified by `begin` and ending before the location specified by `end`, with strides determined by the `strides` parameter. If the stride is positive, the slicing is done in the forward direction; if negative, it is done in the backward direction.

    :param input: A Tensor, specifying the input tensor.
    :param begin: An int32 list or tuple, specifying the starts of the dimensions of the input tensor to be sliced. The size must match the rank of `input`.
    :param end: An int32 list or tuple, specifying the ends of the dimensions of the input tensor to be sliced. The size must match the rank of `input`.
    :param strides: An int32 list or tuple, specifying the strides for each dimension of the input tensor. The value cannot be zero.
    :param begin_mask: An int32 scalar. If the i-th bit of `begin_mask` is set, `begin[i]` is ignored and the fullest possible range in that dimension is used instead.
    :param end_mask: An int32 scalar. If the i-th bit of `end_mask` is set, `end[i]` is ignored and the fullest possible range in that dimension is used instead.
    :param shrink_axis_mask: An int32 scalar. If the i-th bit of `shrink_axis_mask` is set, the i-th dimension specification shrinks the dimensionality by 1, taking on the value at index `begin[i]`. In this case, the i-th specification must define a slice of size 1.
    :return: A Tensor, specifying the output tensor, which has the same data type and scale/offset as `input`. The output shape will be inferred based on the provided parameters.
    """
    return _ops.strided_slice(
        input, begin, end, strides, begin_mask, end_mask, shrink_axis_mask
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def tile(input: Tensor, repeats: list) -> Tensor:
    """
    Constructs a tensor by tiling a given tensor. This operation creates a new tensor by replicating the input tensor `repeats` times.
    The output tensor's i-th dimension has `input.dims(i) * repeats[i]` elements, and the values of input
    are replicated repeats[i] times along the i-th dimension.

    :param input: A Tensor, specifying the input tensor.
    :param repeats: List or Tuple, including numbers of repeated copies along the `input` dimensions.
                    Size must be rank of `input`, that is, it doesn't support broadcast.
    :return: A Tensor, specifying the output tensor. Output has the same data type and scale/offset with `input`.
             Output shape will be inferred by `repeats`.
    """
    rank_inp = input.shape.__len__()
    if rank_inp != len(repeats):
        raise ValueError(
            "tile input rank: %d, repeats len: %d, tile doesn't support broadcast"
            % rank_inp,
            len(repeats),
        )
    repeats = convert_to_list(repeats, "repeats", len(repeats))
    has_expr = any([isinstance(s, (Expr, Symbol)) for s in repeats])
    if has_expr:
        args = [a if isinstance(a, Expr) else Expr(a) for a in repeats]
        return _ops.tile(input, [], args)
    else:
        return _ops.tile(input, repeats, [])


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def transpose(input: Tensor, perm: List[int] = []) -> Tensor:
    """
    Transposes the input tensor, permuting the dimensions according to the `perm`.

    :param input: A Tensor, specifying the input tensor.
    :param perm: An int32 list, permuting the axis according to the value given.
                 Size must be rank of `input`.
    :return: A Tensor, specifying the output input.
             Output has the same data type and scale/offset with param `input`.
             Output shape will be inferred by `perm`.
    """
    dim_len = input.shape.__len__()
    if dim_len != len(perm):
        raise ValueError(
            "Op 'transpose' expected argument 'perm' len is %d,but actually it's %d."
            % (dim_len, len(perm))
        )
    perm_value = perm.copy()
    perm_value.sort()
    perm_exp = list(range(dim_len))
    if perm_value != perm_exp:
        raise ValueError("Op 'transpose' argument 'perm' value is invalid: %r ." % perm)

    ret = _ops.transpose(input, perm)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def where(
    condition: Tensor,
    input1: Tensor = None,
    input2: Tensor = None,
    quantization: Quantization = None,
) -> Tensor:
    """
    `y[i] = condition[i] ? x1[i] : x2[i]`.
    If x1 and x2 are not provided, it will return the indices of condition that are non-zero, in the form of a 2-D
    tensor with shape [n, d], where n is the number of non-zero elements in condition, and d is the number of axes
    of condition.

    :param condition: A Tensor, acting as a mask that chooses, based on the value at each element, whether the
                      corresponding element in the output should be taken from param `input1` (if 1) or param `input1`
                      (if 0). If input1 and input2 are not provided, then it can have a dtype of bool or any numeric
                      dtype.
    :param input1: A Tensor, specifying the first input tensor.
    :param input2: A Tensor, specifying the second input tensor, which has the same shape and data type with param `input1`.
    :param quantization: A Quantization object, specifying the output tensor quantization parameters.
    :return: A Tensor, specifying the output tensor. Output has the same shape and data type with param `input1` if
            `input1` and `input2` are provided. Otherwise, the output's shape is 2-D and data type is int32.
            The output can have different scale and offset.
    """
    return _ops.where(condition, input1, input2, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def shape(
    input: Tensor,
    start: int = 0,
    end: Union[int, None] = None,
):
    """
    Shape implements the shape of the input tensor operation.
    Optional attributes start and end can be used to compute a slice of the input tensor's shape.
        If start axis is omitted, the slice starts from axis 0.
        The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
        If the end axis is omitted, the axes upto the last one will be included. Negative axes indicate counting back from the last axis.
    Note that axes will be clamped to the range [0, r], where r is the rank of the input tensor
    if they are out-of-range (after adding r in the case of negative axis).
    Thus, specifying any end value > r is equivalent to specifying an end value of r,
    and specifying any start value < -r is equivalent to specifying a start value of 0.
    If start > end, the result will be an empty shape.

    :param input: A Tensor, specifying the input tensor.
    :param start: A start axis for slicing the shape, default value is 0.
    :param end: A end axis for slicing the shape, default value is None, if None, will set end = input.dim().
    :return: A Tensor, specifying the output tensor.
    """

    idim = input.shape.dim()
    start = start + idim if start < 0 else start
    if end is None:
        end = idim
    end = end + idim if end < 0 else end

    return _ops.shape(input, end, start)
