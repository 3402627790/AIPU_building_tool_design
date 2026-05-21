import AIPUBuilder._C._ops as _ops
from AIPUBuilder._C._core import Quantization
from AIPUBuilder._C._ops import SpatialScaleQuant
from AIPUBuilder.core.tensor import *
from AIPUBuilder.ops.common import *
from typing import List


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def accidental_hits(true_classes: Tensor, sampled_candidates: Tensor) -> List[Tensor]:
    """
    Computes the position ids in sampled_candidates matching `true_classes`.

    :param true_classes: A Tensor, specifying the input tensor with shape [batch_size, num_true].
    :param sampled_candidates: A Tensor, means the candidates matched by `true_classes` with a shape [num_sampled].
    :return: A Tensor list of size is 3.
             First Tensor records the row number indices of candidates data in `true_classes`, its rank is 1,
             dimension length should be less than `true_classes` shape size.
             Second Tensor records the indices of candidates data in sampled_candidates, its shape should be same with first return Tensor.
             Third Tensor records the effective number in above two tensors, its shape is [1].
    """
    return _ops.accidental_hits(true_classes, sampled_candidates)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def bounding_box(
    input1: Tensor, input2: Tensor, quantization: Quantization = None
) -> Tensor:
    """
    Boundingbox regression.
    Note that, in the following Inputs or Outputs sections:

    - X1 means the input boxes.
    - X2 means the delta of box.
    - Y1 means the output box.

    :param input1: A "TensorPtr", specifying the input tensor(X1).
    :param input2: A "TensorPtr", specifying the input tensor(X2).
    :param quant: A "Quantization" object, specifying the output quantization paramemters.
    :return: A "TensorPtr", specifying the output tensor Y.
    """
    return _ops.bounding_box(input1, input2, convert_quantization(quantization))


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def box_with_nms_limit(
    score: Tensor,
    location: Tensor,
    batch_id: Tensor,
    score_threshold,
    max_num_detections=-1,
    nms_method="Hard",
    iou_threshold=0.1,
    sigma=0.1,
    nms_score_threshold=0.1,
    num_output_rois=100,
) -> List[Tensor]:
    """
    Greedily selects a subset of bounding boxes in descending order of score.
    This op applies NMS algorithm to each class. In each loop of execution,
    the box with maximum score gets selected and removed from the pending set.
    The scores of the rest of boxes are lowered according to the intersection-over-union (IOU)
    overlapping with the previously selected boxes and a specified NMS kernel method.
    Any boxes with score less than a threshold are removed from the pending set.

    :param score: A Tensor, which is a 2-D Tensor of shape [num_rois, num_classes],
                    specifying the score of each bounding box proposal.
    :param location: A Tensor, which is a 2-D Tensor specifying the bounding boxes of shape [num_rois, num_classes * 4],
                        organized in the order [x1,y1,x2,y2]. The boxes are grouped by batches in the first dimension.
                        The sequential order of the boxes corresponds with `score`.
    :param batch_id: A Tensor, which a 1-D int tensor, shape[num_rois], specifying the batch index of each box.
                        Boxes with the same batch index are grouped together.
    :param score_threshold: A float scalar, boxes with scores lower than the threshold are filtered before sending to the NMS algorithm.
    :param max_num_detections: An int scalar, specifying the maximum number of selected bounding boxes for each image.
                                Set to a negative value for unlimited number of output bounding boxes.
    :param nms_method: Supports `Hard`, `Linear`, `Gaussian`, or specified by enum NmsMethod:
                        - NmsMethod.Hard
                        - NmsMethod.Linear
                        - NmsMethod.Gaussian
    :param iou_threshold: A float scalar, specifying the IoU threshold in `Hard` and `Linear` NMS kernel.
                            This field is ignored if `Gaussian` kernel is selected.
    :param sigma: A float scalar, specifying the sigma in `Gaussian` NMS kernel.
                    This field is ignored if `Gaussian` kernel is not selected.
    :param nms_score_threshold: A float scalar, Boxes with scores lower than the threshold are dropped
                                    during the score updating phase in soft NMS.
    :param num_output_rois: An int scalar, specifying the number of output rois, which can be an expect max number output rois.
    :return: List of Tensor, first Tensor with shape [num_output_rois],
                specifying the score of each output box. The boxes are grouped by batches, but the sequential order in each batch is not guaranteed.
                Second Tensor with shape [num_output_rois, 4],
                specifying the coordinates of each output bounding box with the same format as `location`. The sequential order of
                the boxes corresponds with first Tensor.
                Third Tensor with shape [num_output_rois], specifying the class of each output box.
                the boxes corresponds with first Tensor.
                Forth Tensor with shape [num_output_rois], specifying the batch index of each box. Boxes with
                the same batch index are grouped together.
                Fifth Tensor with shape [1], specifying the real number of output rois.
    """
    ret = _ops.box_with_nms_limit(
        score=score,
        location=location,
        batch_id=batch_id,
        score_threshold=score_threshold,
        max_num_detections=max_num_detections,
        nms_method=convert_to_nms_method(nms_method),
        iou_threshold=iou_threshold,
        sigma=sigma,
        nms_score_threshold=nms_score_threshold,
        num_output_rois=num_output_rois,
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_2_input_arg_op
def decode_box(
    scores: Tensor,
    coordinates: Tensor,
    anchors: List[Tensor],
    width: int,
    height: int,
    score_threshold,
    max_output_size: int,
    quantization: List[Quantization] = [],
) -> List[Tensor]:
    """
    Boxes with scores less than Score_Threshold are dropped or discarded in the last output.
    the box need translated into bounding box.
    Note that, in the following Inputs or Outputs sections:
    Input tensor X1, X2.

    Output tensor Y1, Y2, Y3, Y4, Y5.

    - X1 means the scores of all the boxes per batch and per class, of which the shape is [batch_size, num_boxes, class_num + 1].

    - X2 means the coordinates of the maximum output boxes per batch, of which the shape is [batch_size, num_boxes, 4].

    - Y1 means the coordinates of the selected boxes per batch, of which the shape is [batch_size, max_output_size, 4].

    - Y2 means the num of the selected boxes belongs to cluster class_id per batch, of which the shape is [batch_size, max_output_size].

    - Y3 means the num of the all selected boxes belongs to class per batch, of which the shape is [batch_size, 1].

    - Y4 means the scores of all the selected boxes per batch, of which the shape is the same as Y2.
    - Y5 means the class_ids of the selected boxes per batch in input tensor X1, of which the shape is the same as Y2 as well.

    :param input1: A Tensor, specifying input tensor X1.

    :param input2: A Tensor, specifying input tensor X2.

    :param anchors: A Tensor list, specifying the anchors parameter (h, w, ycenter, xcenter).

    :param width: The float IR parameters during normalized coordinate quantization,
            which indicate the width of the input feature map.

    :param height: The float IR parameters during normalized coordinate quantization,
            which indicate the height of the input feature map.

    :param score_threshold: The threshold for deciding when to remove boxes based on the score.

    :param max_output_size: The maximum number of boxes to be output.

    :param quantization: A "Quantization" list, specifying the (Y1,Y4) scale and offset.
            If it is empty, Y1's quantization parameter is the same as X2,
            and Y4's quantization parameter is the same as X1.

    :return: A Tensor list, which are output tensor Y1,Y2,Y3,Y4,Y5.
    """
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.decode_box(
        scores,
        coordinates,
        anchors,
        width,
        height,
        score_threshold,
        max_output_size,
        quantization=quantizations,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def generate_proposals(
    inputs: List[Tensor],
    pre_max_num: int = -1,
    post_max_num: int = -1,
    iou_threshold: float = 0.1,
    min_size: float = 1.0,
    num_output_rois: int = 100,
    image_info: List[int] = [-1, -1],
    stride_info: float or List[float] = 1.0,
) -> List[Tensor]:
    """
    Brief Generate aixs-aligned bounding box proposals.
           Bounding box proposals are generated by applying transformation on a set of predefined anchors with
           the bounding box deltas from bounding box regression.
           A final step of hard NMS is applied to limit the number of returned boxes.

    :param inputs: A Tensor list.
            It has 'scores','deltas','anchors',and optinal input 'image_info'.
            The 'scores' is a 4-D Tensor specifying the score of each anchor at each location.
            With "NHWC" data layout, the tensor shape is [batches, height, width, num_anchors].
            With "NCHW" data layout, the tensor shape is [batches, num_anchors, height, width].
            THe deltas is a 4-D Tensor specifying the bounding box deltas.
            With "NHWC" data layout, the tensor shape is [batches, height, width, num_anchors * 4].
            With "NCHW" data layout, the tensor shape is [batches, num_anchors * 4, height, width].
            The box deltas are encoded in the order of [dx, dy, dw, dh],where dx and dy is the linear-scale
            relative correction factor for the center position of the bounding boxwith respect to
            the width and height,dw and dh is the log-scale relative correction factor for the width and height.
            The last dimensions is the channel dimension.
            The 'anchors' is a 2-D Tensor of shape [num_anchors, 4],
            specifying the shape of each predefined anchor, with format [x1, y1, x2, y2].
            The 'image_info' is a optinal 2-D Tensor,specifying the image info, size [batches, 3],
            format (height, width, scale).
    :param pre_max_num: An int32 scalar,
            specifying the maximum number of boxes before going into the hard NMS algorithm.
            Boxes with the lowest scores are discarded to meet the limit.
            Set to a non-positive value for unlimited number.
    :param post_max_num: An int32 scalar,
            specifying the maximum number of boxes returning from the hard NMS algorithm.
            Boxes with the lowest scores are discarded to meet the limit.
            Set to a non-positive value for unlimited number.
    :param iou_threshold: A float scalar,
            specifying the IoU threshold for hard NMS.
    :param min_size: A float scalar,
            specifying boxes with height or width lower than the absolute threshold are filtered out.
    :param num_output_rois: An int32 scalar,
            specifying the number of output rois, which can be an expect max number output rois.
    :param image_info: specifying the size of each image in the batch,
            with format [image_height, image_width].
    :param stride_info: specifying the ratio from the  original image to the feature map.
            with format [stride_height, stride_width].
    :return: list of Tensor.
            First Tensor of shape [num_output_rois], specifying the score of each output box.
            The boxes are grouped by batches, but the sequential order in each batch is not guaranteed.
            Data type and scale/offset will be same with input scores.
            Second Tensor of shape [num_output_rois, 4], specifying the coordinates of each output bounding box for each class,
            with format [x1,y1,x2,y2]. The sequential order of the boxes corresponds with first Tensor.
            Data type will be same with input anchors. Scale and offset must be same with input anchors.
            Third Tensor of shape [num_output_rois], specifying the batch index of each box. Boxes with the same batch index are grouped together.
            Forth Tensor of shape [1], specifying the valid number of output rois.
    """
    ret = _ops.generate_proposals(
        inputs=inputs,
        pre_max_num=pre_max_num,
        post_max_num=post_max_num,
        iou_threshold=iou_threshold,
        min_size=min_size,
        num_output_rois=num_output_rois,
        image_info=image_info,
        stride_info=convert_to_list(stride_info, "stride_info"),
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def get_valid_counts(
    input: Tensor, score_threshold: float, id_index: int = 0, score_index: int = 1
) -> List[Tensor]:
    """
    Get valid count of bounding boxes given a score threshold. Also moves valid boxes to the top of input data.
    Note that, in the following Inputs or Outputs sections:

    - X1 means input data with shape [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].
    - Y1 means the valid number of boxes.
    - Y2 means the rearranged data tensor.
    - Y3 means the related index in input data.

    :param input: A Tensor, specifying input tensor X1.
    :param score_threshold: The of score for valid bounding boxes.
    :param id_index: Optional index of the class categories, -1 to disable. Default value is 0.
    :param score_index: Optional index of the scores/confidence of boxes. Default value is 1.
    :return: List of Tensor, which are output tensor Y1,Y2,Y3.
    """
    ret = _ops.get_valid_counts(input, score_threshold, id_index, score_index)
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def heatmap_max_keypoint(
    heatmaps: Tensor,
    boxes: Tensor,
    is_softmax=False,
    score_quant: Quantization = None,
    location_quant: Quantization = None,
    softmax_quant: Quantization = None,
) -> List[Tensor]:
    """
    Localize the maximum keypoints from heatmaps, This operation approximates the accurate maximum keypoint scores and
            indices after bicubic upscaling by using Taylor expansion up to the quadratic term.

    :param heatmaps: A Tensor, which is 4-D Tensor of shape [num_boxes, heatmap_size, heatmap_size, num_keypoints],
                    specifying the heatmaps, the height and width of `heatmaps` should be the same, and must be greater than or equal to 2.
    :param boxes: A Tensor, which is 2-D Tensor of shape [num_boxes, 4], specifying the bounding boxes, each with format [x1,y1,x2,y2]
    :param is_softmax: A boolean variable, which means whether there is a softmax after heatmap_max_keypoints compute. Default is False.
    :param score_quant: A Quantization object, should be initialized as (scale,offset), these paramters can be different from `heatmaps`,
                        these quantization params are for first output.
    :param location_quant: A Quantization object, should be initialized as (scale,offset), typically will be (0.125,0).
                           These quantization params are for second output
    :param softmax_quant: A Quantization object, should be initialized as (scale,offset), if `is_softmax` is False, this will be ignored.
    :return: List of Tensor, first is score_of_keypoints. Data type will have the same bitwidth with input heatamps.
            Quantization parameters can be different from `heatmaps`.
            Second is location_of_keypoints, data type and Quantization parameters will be same with input boxes.
            Third is softmax for first output tensor.
    """
    ret = _ops.heatmap_max_keypoint(
        heatmaps,
        boxes,
        is_softmax,
        convert_quantization(score_quant),
        convert_quantization(location_quant),
        convert_quantization(softmax_quant),
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def max_roi_pool(
    input: Tensor,
    rois: Tensor,
    pooled_shape: list or int = 1,
    spatial_scale: list or float = 1.0,
) -> Tensor:
    """
    Select and scale the feature map of each region of interest to a unified output size by max-pooling.

    :param input: A Tensor, specifying the input featuremap tensor in `NHWC`. Rank must be 4.
    :param rois: A Tensor, specifying the roi tensor with shape [num_rois, 5],
                 the order of second dimension should be [batch_id,x1,y1,x2,y2].
    :param pooled_shape: output spatial shape with [height, width]. If not provided, default to 1 along each spatial dimension.
    :param spatial_scale: Spatial scale from featuremap to input image with [height, width].
                        If not provided, default to 1.0 on each spatail dimension. `scale = featuremap / input`.
    :return: A Tensor, specifying the output tensor, which shape will be [num_rois, pooled_shape[0], pooled_shape[1], input[c]] in `NHWC` format.
    """
    pooled_shape = convert_to_list(pooled_shape, "pooled_shape")
    spatial_scale = convert_to_list(spatial_scale, "spatial_scale")
    ret = _ops.max_roi_pool(
        input=input,
        rois=rois,
        batch_id=None,
        pooled_shape=pooled_shape,
        spatial_scale=NodeParamValue(spatial_scale),
        quantized=True,
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def max_un_pool(
    input: Tensor,
    indices: Tensor,
    shape: List[int],
    flatten_dim: str = "HWC",
    storage_order: int = 0,
) -> Tensor:
    """
    MaxUnpool essentially computes the partial inverse of the MaxPool operator.
    The input information to this operator is typically the output information from a MaxPool operator.
    The first input tensor X is the tensor that needs to be unpooled,
    which is typically the pooled tensor (first output) from MaxPool.
    The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
    to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool operator.
    The third input is a vector that specifies the output size of the unpooling operation.

    :param input: A Tensor, specifying the input data tensor.
    :param indices: A Tensor, specifying the indices tensor.
    :param shape: A list of integer, specifying the shape of the output tensor.
    :param flatten_dim:Indicating the flatten mode in the indices tensor. Support value: NHWC, HWC, HW.
                       Default value is `HWC`.
    :param storage_order: An int32 scalar, specifying the storage order of the tensor. 0 is row(h) major, and 1 is column(w) major.
    :return: A Tensor, which has the same shape as param shape and same dtype as input tensor.
    """
    return _ops.max_un_pool(
        input, indices, shape, convert_to_flattendim(flatten_dim), storage_order
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def multibox_transform_loc(
    input: Tensor,
    loc_pred: Tensor,
    anchor: Tensor,
    score_threshold: float = 0.01,
    variance: List[float] = [0.1, 0.1, 0.2, 0.2],
    quantization: Quantization = None,
) -> List[Tensor]:
    """
    Location transformation for multibox detection. Note that, in the following Inputs or Outputs sections:
    - X1 means the input class probabilities.
    - X2 means the input location regression predictions.
    - X3 means the input prior anchor boxes.
    - Y1 means the output box.
    - Y2 means the valid count of output box.

    :param input: A Tensor, specified the input class probabilities(X1).
    :param loc_pred: A Tensor,specified the input location regression predictions(X2).
    :param anchor: A Tensor,specified the input prior anchor boxes(X3).
    :param score_threshold: The threshold for deciding when to remove boxes based on the score. It is a scalar.
    :param variance: A list of variances to be decoded from box regression output with the the order of [y, x, h, w].
    :return: List of Tensor, which are output tensor Y1,Y2.
    """
    ret = _ops.multibox_transform_loc(
        input,
        loc_pred,
        anchor,
        score_threshold,
        variance,
        convert_quantization(quantization),
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def nms(
    box_coor: Tensor,
    box_num_per_class: Tensor,
    class_num_per_batch: Tensor,
    score: Tensor,
    method: str = "Hard",
    max_output_size: int = 5000,
    score_threshold: float or int = 0.6,
    iou_threshold: float or int = 0.5,
    sigma: float = 0.0,
    center_point_box=False,
    image_info: List = [],
) -> List[Tensor]:
    """
    The Non-Maximum Suppression (NMS) filters out boxes that have high Intersection-Over-Union (IOU)
    overlap with previously selected boxes.

    :param box_coor: A Tensor, specified coordinates of boxes in each batch with shape [batch_size, num_boxes, 4].
    :param box_num_per_class: A Tensor, specified number of boxes in each class and each batch with shape [batch_size, num_classes].
    :param class_num_per_batch: A Tensor, specified number of classes in each batch with shape [batch_size, 1].
    :param score: A Tensor, specified scores of boxes in each batch with shape [batch_size, num_boxes].
    :param nms_method: A string, specifying nms method, it can be `Hard`, `Gaussian`.
    :param max_output_size: An uint32 scalar, specifying max output box number. Default is 5000.
    :param score_threshold: A float/int scalar, boxes with scores lower than the threshold are filtered before sending to the NMS algorithm. Default is 0.6.
                            If it is a float value, it means that it is not quantized. API will quantize this value.
                            If it is an int value, it means that it is quantized.
    :param iou_threshold: A float/int scalar, which will be ignored when method is "Gaussian". Default is 0.5.
                          If it is a float value, it means that it is not quantized. API will quantize this value.
                          If it is an int value, it means that it is quantized.
    :param sigma: A float scalar, which will be ignored when method is not "Gaussian". Default is 0.
    :param center_point_box: A boolean scalar, `False` means coordinates format is [y1,x1,y2,x2]. `True` means coordinates format is [y_center,x_center,height,width].
    :param image_info: A List or Tuple, specified input featuremap height/width information, due to coordinates is normalized value, so we need this to back it.
                     This parameter is valid only when 'center_point_box' is enabled.
                     It defaults to empty.
    :return: List of Tensor, first Tensor specified coordinates of selected boxes in each batch. Its shape is same with param `box_coor`.
            Second Tensor specified number of selected boxes in each batch. Its shape is same with param `box_num_per_class`.
            Third Tensor specified scores of boxes in each batch. Its shape is same with param `score`.
            Forth Tensor specified indices of boxes in each batch. Its shape is same with param `score`.
    """
    if not isinstance(image_info, (list, tuple)):
        raise TypeError(
            " image_info type should be List or Tuple, but now %s " % type(image_info)
        )
    if len(image_info) > 0:
        image_info = convert_to_list(image_info, "image_info")
    return _ops.nms(
        box_coor,
        box_num_per_class,
        class_num_per_batch,
        score,
        str_to_enum_member(_ops.NmsMethod, method.upper()),
        max_output_size,
        score_threshold,
        iou_threshold,
        sigma,
        center_point_box,
        image_info,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_2_input_arg_op
def proposal(
    scores: Tensor,
    coordinates: Tensor,
    anchors: List[Tensor],
    image_info: List,
    score_threshold: float,
    output_size: int,
) -> List[Tensor]:
    """
    Boxes with scores less than Score_Threshold are dropped or discarded.
    the box need translated into bounding box.
    Note that, in the following Inputs or Outputs sections:
        X1 means the scores of all the boxes belongs to proposal and backgroud,
           of which the shape is [batch_size, num_boxes, 2].
        X2 means the coordinates of the maximum output boxes per batch,
           of which the shape is [batch_size, num_boxes, 4].
        Y1 means the scores of all the selected boxes,
           of which the shape is [batch_size, max_output_size, 1].
        Y2 means the coordinates of the selected boxes,
           of which the shape is [batch_size, max_output_size, 4].
        Y3 means the num of the all selected boxes belongs to class per batch,
           of which the shape is [batch_size, 1].
        Y4 means the class_ids of the selected boxes per batch in input tensor X1,
           of which the shape is the same as Y3 as well.

    :param scores: A Tensor,  specifying the input X1.
    :param coordinates: A Tensor, specifying the input X2.

    :param anchors: A Tensor list, specifying the anchors parameter (h, w, ycenter, xcenter).

    :param image_info: An uint32 vector, specified input featuremap height/width information,
                        due to coordinates is normalized value, so we need this to back it.
    :param score_threshold: A float scalar, specifying the scroe threshold.
    :param max_output_size: An uint32 scalar, specifying max output box number.
                             Default is num_boxes.
    :return: A list of Tensor, which is [Y1,Y2,Y3,Y4].
    """
    return _ops.proposal(
        scores,
        coordinates,
        anchors,
        image_info,
        score_threshold,
        output_size,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def post_nms1(
    coordinates: Tensor,
    scores: Tensor,
    valid_numbers: Tensor,
    image_info: List,
    output_size: int,
) -> List[Tensor]:
    """
    This operator completes the operation between proposal and NMS,
    extracts and quantized the valid boxes,
    outputs the box coordinates of the quantized and un-quantized boxes,
    so that NMS can perform the operation.

    :param coordinates: A Tensor, specified  the coordinates of the input boxes,
                        of which the shape is [batch_size, num_boxes, 4].
    :param scores: A Tensor, specified the scores of input boxes,
                   of which the shape is [batch_size, num_boxes, 1].
    :param valid_numbers: A Tensor, specified the valid numbers of input boxes,
                          of which the shape is [batch_size, 1].
    :param image_info: An uint32 vector, specified input featuremap height/width information,
                       due to coordinates is normalized value, so we need this to back it.
    :param output_size: An uint32 scalar, specifying max output box number.
                        Default is num_boxes.

    :return: A Tensor list, which is [Y1,Y2].
            Y1 means the coordinates of the valid boxes per batch,
            of which the shape is [batch_size, output_size, 4].
            Y2 means the coordinates of the quantized valid boxes per batch,
            of which the shape is [batch_size, output_size, 4].
    """
    return _ops.post_nms1(
        coordinates,
        scores,
        valid_numbers,
        image_info,
        output_size,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def pyramid_roi_align(
    inputs: List[Tensor],
    resize_shape: list or int,
    image_info: list or int,
    sample_points: list or int = 1,
    spatial_scale: list or float = 1.0,
    coordinate_transformation_mode: str = "HalfPixel",
    proposal_normalized: bool = False,
    quantization: Quantization = None,
) -> Tensor:
    """
    Region of Interest (ROI) align is used for converting all the proposals to the fixed shape
    as required by the next special layers, especially in an object detection network.
    PyramidROI align is proposed to avoid the misalignment by removing quantization
    while converting from the original image into the feature map
    and from the different shape Pyramid feature map into the ROI feature.
    In each ROI bin, the value of the sampled locations is computed directly
    through bilinear interpolation other than quantization.
    Note that, in the following Inputs or Outputs sections:
     X1 means the scores of all the boxes belongs to proposal and backgroud,
        of which the shape is [batch_size, num_boxes, 2].
     X2~X5 mean the feature map with different size.

     Y means the output feature map.

    :param inputs: A list of Tensor, specifying the input tensor X2~X5.
    :param resize_shape: An int32_t vector, specified output featuremap height/width information.
    :param image_info: An int32_t vector, specified input featuremap height/width information.
    :param sample_points: An uint32 vector, specifying the sample points
                             on height and width dimension. Vector size must be 2.
    :param spatial_scale: A quantization scale object,
                 used to describe the scale quantization parameter.
    :param coordinate_transformation_mode: An enum sepcifying the method to shift the input.
                 Allowed values are 'HALF_PIXEL' and 'OUTPUT_HALF_PIXEL'.
                 Use the value 'HALF_PIXEL' to pixel shift the input coordinates
                 by -0.5 (the recommended behavior).
                 Use the value 'OUTPUT_HALF_PIXEL' to omit the pixel shift for the input.
                 Defaults to 'HALF_PIXEL'.
    :param proposal_normalized: It is a value means different quantization method.
    :param quantization: A "Quantization" object, specifying the output scale and offset.
    :return: A Tensor, specifying the output tensor.
    """
    resize_shape = convert_to_list(resize_shape, "pooled_shape")
    image_info = convert_to_list(image_info, "sample_points")
    sample_points = convert_to_list(sample_points, "sample_points")
    spatial_scale = convert_to_list(spatial_scale, "spatial_scale", length=4)
    ret = _ops.pyramid_roi_align(
        inputs=inputs,
        resize_shape=resize_shape,
        image_info=image_info,
        sample_points=sample_points,
        spatial_scale=spatial_scale,
        ct_mode=str_to_enum_member(
            _ops.CTMode,
            coordinate_transformation_mode.upper(),
        ),
        proposal_normalized=proposal_normalized,
        quantization=convert_quantization(quantization),
    )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def region(
    input: Tensor,
    obj_threshold: int or float,
    max_box_num: int,
    box_per_grid: int,
    class_num: int,
    grid_width: int,
    grid_height: int,
    grid_compensate: bool = False,
    anchors: List[float] = [],
    quantization: List[Quantization] = [],
) -> List[Tensor]:
    """
    Yolo network predicts 5 bounding boxes at each cell in the output feature map.
    The network predicts 5 coordinates for each bounding box, tx, ty, tw, th, and to.
    If the cell is offset from the top left corner of the image by (cx, cy) and the bounding box prior
    has width and height pw, ph, then the predictions correspond to:
    bx = O(tx) + cx by = O(ty) + cy bw = pw * exp(tw) bh = ph * exp(th) Pr(object) * IOU(b, object) = O(to)
    Region layer is to process the calculation and output the bounding boxed and confidence for NMS.
    Note that, in the following Inputs or Outputs sections:

    - X1 means input data of which the shape is [batch_size, s, s, anchor_num, num_classes + 5].where s is the grid num of a feature map, num_classes is the classification number of the network.

    - Y1 means the scores of the selected boxes, of which the shape is [batch_size, max_box_num].where max_box_num is the config parameter of the network.

    - Y2 means the coordinates of the filtered boxes, of which the shape is [batch_size, max_box_num, 4].

    - Y3 means the boxes number of every class per batch, of which the shape is [batch_size, num_classes].

    - Y4 means the valid classes label of per class, of which the shape is [batch_size, num_classes].

    - Y5 means the classes number of every class per batch.

    :param input: A Tensor, specifying the input tensor X1.
    :param obj_threshold: The threshold for deciding when to which score is belong to an object. It is a scalar.
    :param max_box_num: Integer representing the maximum number of boxes to be selected per batch. It is a scalar.
    :param box_per_grid: Integer representing the number of boxes to be presented by a grid. It is a scalar.
    :param class_num: Integer representing the number of class for the input. It is a scalar.
    :param grid_width: Integer representing the grid number of feature map width. It is a scalar.
    :param grid_height: Integer representing the grid number of feature map height. It is a scalar.
    :param grid_compensate: For caffe yolo416 which need compensate.
    :param anchors: Anchors are determined by a k-means procedure,
                    looking at all the bounding boxes in your dataset.
                    The k-means routine will figure out a selection of anchors that represent your dataset.
                    k=5 for yolov3, but there are different numbers of anchors for each YOLO version.
    :param quantization: A list of "Quantization" object, specifying the outputs quantization parameters.
    :return: A Tensor list, which are output tensor Y1,Y2,Y3,Y4,Y5.
    """
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.region(
        input,
        obj_threshold,
        max_box_num,
        box_per_grid,
        class_num,
        grid_width,
        grid_height,
        grid_compensate,
        anchors,
        quantizations,
    )


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def region_fuse(
    inputs: List[Tensor], class_num: int, quantization: List[Quantization] = []
) -> List[Tensor]:
    """
    Merge two group input data from Region layer into one output.
    Note that, in the following Inputs or Outputs sections:

    - X1 and X2 mean the scores of all the boxes come from the first and second region layer output, of which the shape is [batch_size, secore_num].

    - X3 and X4 mean the coordinates of boxes come from the first and second region layer output, of which the shape is [batch_size, secore_num, 4].

    - X5 and X6 mean the boxes number of every class per batch come from the first and second region layer output, of which the shape is [batch_size, class_num].

    - X7 and X8 mean the boxes label of every class per batch come from the first and second region layer output, of which the shape is [batch_size, class_num].

    - X9 and X10 mean the classes number of every class per batch come from the first and second region layer output, of which the shape is [batch_size, class_num].

    - Y1 means the merged scores of the selected boxes.
    - Y2 means the merged coordinates of the filtered boxes.
    - Y3 means the total boxes number of every class per batch, of which the shape is [batch_size, num_classes].
    - Y4 means the total valid classes label of per class, of which the shape is [batch_size, num_classes].
    - Y5 means the maxmun class number of every class per batch.

    :param inputs: A Tensor list, specifying the input tensor X1~X10.
    :param class_num: Integer representing the number of class for the input. It is a scalar.
    :param quantization: A list of "Quantization" object, specifying the outputs quantization parameters.
    :return: A Tensor list, which are output tensor Y1,Y2,Y3,Y4,Y5.
    """
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.region_fuse(inputs, class_num, quantizations)


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def roi_align(
    input: Tensor,
    rois: Tensor,
    pooled_shape: list or int = 1,
    sample_points: list or int = 0,
    spatial_scale: list or float = 1.0,
    coordinate_transformation_mode: str = "HalfPixel",
    mode: str = "Avg",
    quantization: Quantization = None,
) -> Tensor:
    """
    Select and scale the feature map of each region of interest to a unified
    output size by average pooling sampling points from bilinear interpolation.

    :param input: A Tensor, specifying the input featuremap tensor in NHWC. Rank of input must be 4.
    :param rois: A Tensor, specifying the roi tensor.
                 Shape is [num_rois, 5], the order of second dimension should be [batch_id,x1,y1,x2,y2].
    :param pooled_shape: Output spatial shape with [height, width]. If not provided, default to 1 along each spatial dimension.
    :param sample_points: Sample points on spatial dimension with [height, width].
                        If not provided, default to 0 on each dimension. sample points: 0 means ceil(roi_width/out_width).
    :param spatial_scale: Spatial scale from featuremap to input image with [height, width].
                        If not provided, default to 1.0 on each spatail dimension. scale = featuremap / input.
    :param coordinate_transformation_mode: A string, sepcifying the method to shift the input.
                                           Allowed values are `HalfPixel` and `OutputHalfPixel`.
                                           Use the value `HalfPixel` to pixel shift the input coordinates by
                                           -0.5 (the recommended behavior). Use the value `OutputHalfPixel` to omit the pixel
                                           shift for the input. Defaults to `HalfPixel`.
    :param mode: A string, specifying the pooling method.
                 Two modes are supported: `Avg` and `Max`.
                 Default is `Avg`.
    :param quantization: Optional Quantization object, specifying the output data scale and offset.
                  If not set, used default.
    :return: A Tensor, specifying the output tensor, which shape will be [num_rois, pooled_shape[0], pooled_shape[1], input[c]] in `NHWC` format.
    """
    pooled_shape = convert_to_list(pooled_shape, "pooled_shape")
    sample_points = convert_to_list(sample_points, "sample_points")
    if not isinstance(spatial_scale, SpatialScaleQuant):
        spatial_scale = convert_to_list(spatial_scale, "spatial_scale")
        ret = _ops.roi_align(
            input=input,
            rois=rois,
            batch_id=None,
            pooled_shape=pooled_shape,
            sample_points=sample_points,
            spatial_scale=NodeParamValue(spatial_scale),
            ct_mode=str_to_enum_member(
                _ops.CTMode,
                coordinate_transformation_mode.upper(),
            ),
            mode=str_to_enum_member(_ops.RoiAlignMode, mode.upper()),
            quantization=convert_quantization(quantization),
        )
    else:
        ret = _ops.roi_align(
            input=input,
            rois=rois,
            spatial_scale=spatial_scale,
            batch_id=None,
            pooled_shape=pooled_shape,
            sample_points=sample_points,
            ct_mode=str_to_enum_member(
                _ops.CTMode,
                coordinate_transformation_mode.upper(),
            ),
            mode=str_to_enum_member(_ops.RoiAlignMode, mode.upper()),
            quantization=convert_quantization(quantization),
        )
    return ret


@eager_mode
@wrap_numpy_to_tensor
@check_rterrors
@check_input_op
def detection_output(
    scores: Tensor,
    coordinates: Tensor,
    anchor: Tensor,
    width: int,
    height: int,
    score_threshold: int or float,
    max_output_size: int,
    variance: List[float] = [],
    quantization: List[Quantization] = [],
) -> List[Tensor]:
    """
    Boxes with scores less than Score_Threshold are dropped or discarded in the last output.
    the box need translated into bounding box.
    Note that, in the following Inputs or Outputs sections:

    - X1 means the scores of all the boxes per batch and per class, of which the shape is [batch_size, anchor_num, class_num + 1].

    - X2 means the coordinates of the input boxes per batch, of which the shape is [batch_size, anchor_num, class_num + 1, 4].

    - X3 means the coordinates of the anchor per batch, of which the shape is [batch_size, anchor_num, 4].

    - Y1 means the scores of all the selected boxes per batch, of which the shape is the same as [batch_size, max_output_size].

    - Y2 means the coordinates of the selected boxes per batch, of which the shape is [batch_size, max_output_size, 4].

    - Y3 means the num of the selected boxes belongs to cluster class_id per batch, of which the shape is [batch_size, max_output_size].

    - Y4 means the class_ids of the selected boxes per batch in input tensor X1, of which the shape is the same as Y3 as well.

    - Y5 means the num of the all selected boxes belongs to class per batch, of which the shape is [batch_size, 1].

    :param scores: A Tensor, specified the input tensor X1,
                   The tensor shape is [batch_size, anchor_num, class_num + 1].

    :param coordinates: A Tensor, specified the input tensor X2.
                        The tensor shape is [batch_size, anchor_num, class_num + 1, 4].

    :param anchor: A Tensor, specified the input tensor X3,
                    The tensor shape is [batch_size, anchor_num, 4].

    :param width: The float IR parameters during normalized coordinate quantization,
                    which indicate the width of the input feature map.

    :param height: The float IR parameters during normalized coordinate quantization,
                    which indicate the height of the input feature map.

    :param score_threshold: The of score for valid bounding boxes.

    :param max_output_size: Integer representing the maximum number
                            of boxes to be selected per batch. It is a scalar.

    :param quantization: A "Quantization" list, specifying the (Y1,Y2,Y3,Y4,Y5) scale and offset.

    :return: A Tensor list, which are output tensor Y1,Y2,Y3,Y4,Y5.
    """
    quantizations = [convert_quantization(quant) for quant in quantization]
    return _ops.detection_output(
        scores,
        coordinates,
        anchor,
        width,
        height,
        score_threshold,
        max_output_size,
        variance,
        quantizations,
    )
