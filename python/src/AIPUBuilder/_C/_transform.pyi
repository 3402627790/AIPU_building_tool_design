from typing import List
from enum import Enum
from AIPUBuilder.core import Graph
from AIPUBuilder.core.graph import Graph


def fuse_transpose(g: Graph) -> bool:
    """!a pass fuse transpose + transpose to transpose

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_reshape(g: Graph) -> bool:
    """!a pass fuse reshape + reshape to reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def slice_to_crop(g: Graph) -> bool:
    """!a pass convert slice to crop if available

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def split_to_single_crop(g: Graph) -> bool:
    """!a pass convert split to one crop if only one output of split is used

    argument 'g' a Graph to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def eliminate_resize(g: Graph) -> bool:
    """!a pass convert resize to reshape if available

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def eliminate_transpose(g: Graph) -> bool:
    """!a pass convert transpose to reshape if available

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def eliminate_reshape(g: Graph) -> bool:
    """!a pass remove reshape if available

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def eliminate_crop(g: Graph) -> bool:
    """!a pass remove useless crop if available

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_transpose_with_reshape(g: Graph) -> bool:
    """!a pass fuse transpose + reshape + transpose to reshape + transpose + reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_pad(g: Graph) -> bool:
    """!a pass fuse pad + pad to pad

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_crop(g: Graph) -> bool:
    """!a pass fuse crop + crop to crop

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_cast(g: Graph) -> bool:
    """!a pass fuse cast + cast to cast

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def eliminate_cast(g: Graph) -> bool:
    """!a pass convert a case to reshape if available

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_activation(g: Graph) -> bool:
    """!a pass fuse [Conv-like] + activation to [Conv-like]

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_pad_crop(g: Graph) -> bool:
    """!a pass fuse pad + crop to crop

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_split(g: Graph) -> bool:
    """!a pass fuse split + split to split

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_concat(g: Graph) -> bool:
    """!a pass fuse concat + concat to concat

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_concat_split(g: Graph) -> bool:
    """!a pass fuse concat + split  to concat/split/[concat+split]

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_trans_trans_eltwise(g: Graph) -> bool:
    """!a pass fuse transpose,transpose + eltwise to etlwise+transpose

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def rotate_trans(g: Graph) -> bool:
    """!a pass rotate a transpose as below:

                  /----trans1             /----trans2(0+1)
    ----trans0---/             ==>   ----/
                 \----- Any              \-----trans0----Any

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def swap_trans_eltwise(g: Graph) -> bool:
    """!a pass swap trans to etlwise other side if possible as below:

    0----------etlwise-----------3       0-----trans1----5-----eltwise-----trans2----3
                /   \                ==>                       /  \
               /     \                        /---trans0--2   /    \
    1-trans0--/2      \---trans1--4       1--/---------------/      \----------------4

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def concat_split_with_arithmetic(g: Graph) -> bool:
    """!a pass if all inputs of concat op(or all outputs of split op) has the same arithmetic op(like sine/
    tanh/cosine...), will change to concat + arithmetic(or arithmetic + split) as below shows:

    -0-> sine --|                      -0-|
    -1-> sine --|-> concat -3->   ==>  -1-|-> concat --> sine -3->
    -2-> sine --|                      -2-|

               --> sine -1->                              -1->
    -0-> split --> sine -2->      ==> -0-> sine --> split -2->
               --> sine -3->                              -3->

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def CSE_subnode(g: Graph) -> bool:
    """!a pass for subexpression elimination

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def sqrt_div_to_rsqrt_mul(g: Graph) -> bool:
    """!a pass for perfoming a conversion: div(x, sqrt(y)) => mul(x, rsqrt(y))

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_concat_constant_to_pad(g: Graph) -> bool:
    """!a pass for fusing the constant + concat to the single pad.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_pad_conv_pool(g: Graph) -> bool:
    """!a pass for fusing the pad + pooling/pooling3d/conv/conv3d to the single pooling/pooling3d/conv/conv3d.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_crop_conv_pool(g: Graph) -> bool:
    """!a pass for fusing the crop + pooling/pooling3d/conv/conv3d to the single pooling/pooling3d/conv/conv3d.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_tile_eltwise(g: Graph) -> bool:
    """!a pass for fusing the tile + eltwise to the single arithmetical op(Add/Sub/Mul/Div).

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_mul_add_to_bn(g: Graph) -> bool:
    """!a pass for fusing the mul + add to the single BatchNorm op.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_add_mul_to_bn(g: Graph) -> bool:
    """!a pass for fusing the add + mul to the single BatchNorm op.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_transpose(g: Graph) -> bool:
    """!a pass for lowering tranpose: transpose+other -> other+transpose .

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def swap_pad_crop(g: Graph) -> bool:
    """!a pass for swaping the pad + crop --> crop + pad when meets the condition,
    and then using the fuse_pad_conv_pool pass to fuse the pad to pooling/conv op.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def eliminate_split(g: Graph) -> bool:
    """!a pass for eliminating the split which input_shape == output_shape or output.size == 1.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_quantize_and_dequantize(g: Graph) -> bool:
    """!a pass for fuse quantize + dequantize op to transparent the input of quantize to the output of dequantize.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_fp8_dequantize(g: Graph) -> bool:
    """!a pass fuse fp8 operator + dequantize_fp16/bfp16 to fp8 operator output fp16/bfp16 directly

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def eliminate_concat_crop(g: Graph) -> bool:
    """!a pass for eliminate the pattern concat+n*crop, where the crop's output is the concat input.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_pooling(g: Graph) -> bool:
    """!a pass fuse n * avg pooling to ONE big kernel global avg pooling

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_clamp(g: Graph) -> bool:
    """!a pass fuse constant+eltwise_min(with relu) to clip

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lift_activation(g: Graph) -> bool:
    """!a pass lift activation for a chance that it can be merged with conv/fc/etlwise etc.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def merge_conv_output_crop(g: Graph) -> bool:
    """!a pass merge conv + n * crop to n * conv when children of conv are all crop channel

    conv --- crop_0 channel ---
          |  ...
          |- crop_n channel ---
    ===>
    -------- conv_0 ---
          |- ...
          |- conv_n ---

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_conv_outc(g: Graph) -> bool:
    """!a pass merge conv *n + concat to a big conv  when concat on channel

         /-----> conv --|
    inp--------> conv --|-> concat---out    ==>   inp--->Conv--->out
         \-----> conv --|

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_conv1x1_conv(g: Graph) -> bool:
    """!a pass merge conv1x1 + conv to a big conv
    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_split_concat(g: Graph) -> bool:
    """!a pass to fuse split+concat to equivalent transpose/reshape
    1. Input [N, H, C] -> Split [1, H, C] * N -> Concat [1, H, N*C]
    => Input [N, H, C] -> Reshape [1, N, H, C] -> Transpose [1, H, N, C] -> Reshape [1, H, N*C]
    2. Input [1, H, N*C] -> Split [1, H, C] * N -> Concat [N, H, C]
    => Input [1, H, N*C] -> Reshape [1, H, N, C] -> Transpose [1, N, H, C] -> Reshape [N, H, C]
    When H=1, we can further remove the split+concat node
    1. Input [N, 1, C] -> Split [1, 1, C] * N -> Concat [1, 1, N*C]
    => Input [N, 1, C] -> Reshape [1, 1, N*C]
    2. Input [1, 1, N*C] -> Split [1, 1, C] * N -> Concat [N, 1, C]
    => Input [1, 1, N*C] -> Reshape [N, 1, C]

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def reduce_const_eltwise(g: Graph) -> bool:
    """!a pass to reduce const+eltwise if the constant data can be reduced
    """
    ...


def eliminate_channel_shuffle(g: Graph) -> bool:
    """!a pass for eliminating the channel shuffle which params group == splits == 1.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_s2d_conv(g: Graph) -> bool:
    """!a pass for eliminating the s2d node for s2d+conv pattern.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def eliminate_global_transpose(g: Graph) -> bool:
    """!a pass for eliminating the global layout transpose.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def conv_transpose_to_d2s(g: Graph) -> bool:
    """!a pass for converting conv--reshape--transpose  to conv--depthtospace--reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_constant_eltwise_to_leakyrelu(g: Graph) -> bool:
    """!a pass for fuse the pooling + constant + eltwise to pooling + leakyrelu pattern.
    pooling  --|
               |--> eltwise ===> pooling --> leakyrelu
    constant --|

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_constant_cast(g: Graph) -> bool:
    """!a pass fuse constant + cast to constant

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_sigmoid_mul_to_silu(g: Graph) -> bool:
    """!a pass try to fuse simoid + mul to silu

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def remove_batch(g: Graph) -> bool:
    """!a pass try to remove the graph global batch if the batch!=1

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_slice(g: Graph) -> bool:
    """!a pass to fuse slice+slice --> slice

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_slice_concat(g: Graph) -> bool:
    """!a pass to fuse slice(*n)+concat

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_reshape(g: Graph) -> bool:
    """!a pass try to lower reshape: reshape+other -> other+reshape .

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def duplicate_reshape(g: Graph) -> bool:
    """!a pass try to duplicate reshape .

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def duplicate_transpose(g: Graph) -> bool:
    """!a pass try to duplicate transpose .

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_reshape_reshape_eltwise(g: Graph) -> bool:
    """!a pass fuse reshape,reshape + eltwise to etlwise+reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_reshape_eltwise(g: Graph) -> bool:
    """!a pass fuse reshape + eltwise to etlwise+reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_reshape_internal_constant_eltwise(g: Graph) -> bool:
    """!a pass fuse reshape,insternal_constant + eltwise to etlwise+reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_reshape_reshape_add(g: Graph) -> bool:
    """!a pass fuse reshape + single arithmetical op(Add/Sub/Mul) to single arithmetical op(Add/Sub/Mul)+reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def lower_reshape_add(g: Graph) -> bool:
    """!a pass fuse reshape + single arithmetical op(Add/Sub/Mul) to single arithmetical op(Add/Sub/Mul)+reshape

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
    ...


def fuse_crop_pad(g: Graph) -> bool:
    """!a pass fuse crop + pad where crop is a adaption node generated by bt.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def div_const_to_mul(g: Graph) -> bool:
    """!a pass convert a x/constant to x* (1/constant).

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def fuse_crop_pad_to_crop(g: Graph) -> bool:
    """!a pass fuse crop + pad -> crop  where crop is a adaption node generated by bt.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def eliminate_eltwise_constant(g: Graph) -> bool:
    """!a pass eliminate reshape + constant -> add.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def fuse_fc_crop(g: Graph) -> bool:
    """!a pass tranform fc+crop+crop -> fc0 + fc1.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def lift_crop(g: Graph) -> bool:
    """!a pass lift crop.

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def fuse_transpose_matmul(g: Graph) -> bool:
    """!a pass eliminate transpose + matmul -> matmul(trans_b=true)

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def fuse_quantize_cast(g: Graph) -> bool:
    """!a pass fuse quantize + cast -> quantize, and the output of quantize node is substituted to cast's output

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def fuse_matmul_cast(g: Graph) -> bool:
    """!a pass fuse matmul + (no need quantization ops) + cast -> matmul + (no need quantization ops)

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def full_shape_add_sub_mul_to_eltwise(g: Graph) -> bool:
    """!a pass convert add/sub/mul with two same inputs shape to eltwise

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def fuse_reshape_transpose_conv(g: Graph) -> bool:
    """!a pass to fuse reshape+transpose+reshape+conv -> conv

    argument 'g' a Graph to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def transpose_to_d2s(g: Graph) -> bool:
    """!a pass to convert special perm transpose to depth_to_space which has good performance
    argument 'g' a Graph to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def gather_to_slice(g: Graph) -> bool:
    """!a pass convert gather to slice

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def fuse_equal_not(g: Graph) -> bool:
    """!a pass fuse equal and not

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def lift_rmsnorm_reshape(g: Graph) -> bool:
    """!a pass lift reshape from a node after rmsnorm to one before rmsnorm

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def lower_quantize_reshape_matmul(g: Graph) -> bool:
    """!a pass lift reshape from a node after quantize to one before quantize

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def gatherelements_to_gather(g: Graph) -> bool:
    """!a pass convert gather_elements to gather

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """


def global_avg_pool_3d_to_2d(g: Graph) -> bool:
    """!a pass converts GlobalAvgPool 3D node to multiple GlobalAvgPool 2D nodes

    argument 'g' a Graph  to be passed
    return(bool): true if the Graph 'g' changed, vice versa.
    """
