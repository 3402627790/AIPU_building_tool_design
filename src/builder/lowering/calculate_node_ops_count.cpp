//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yuchou.Gan
 * @Date: 2020-02-25 17:58:05
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-11-15 17:31:24
 */
#include "core/core.h"
#include "builder/lowering/lowering.h"

#define ALIGN(x, y) ((x + y - 1).floordiv(y) * y)

namespace aipubt
{
void calculate_node_ops_count(GraphPtr g)
{
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (n->inputs().size() == 0 || n->outputs().size() == 0)
        {
            n->set_attr("ops_count", Expr(0));
            continue;
        }
        SymbolShape in_shape = n->inputs()[0]->shape().symbol();
        SymbolShape out_shape = n->outputs()[0]->shape().symbol();
        Expr count = 0;
        switch (n->type())
        {
        case OpType_Convolution:
        case OpType_DepthwiseConv:
        case OpType_ConvTranspose:
        {
            bool with_winograd = false;
            if (n->params().count("with_winograd"))
            {
                with_winograd = n->params()["with_winograd"];
            }
            std::vector<NodeParamValue> Fmnrs = {0, 0};
            if (n->params().count("Fmnrs"))
            {
                Fmnrs = n->params()["Fmnrs"].arr;
            }
            // output is N, H, W, C
            int kernel_x = n->params()["kernel_x"];
            if (with_winograd && Fmnrs[1].v.i == 3)
            {
                kernel_x = 3;
            }
            int kernel_y = n->params()["kernel_y"];
            if (with_winograd && Fmnrs[0].v.i == 3)
            {
                kernel_y = 3;
            }
            TensorShape weights = n->constants()["weights"]->shape();
            // mac count as 2 ops
            Expr ic = in_shape.c();
            if (in_shape.dim() > 4)
            {
                ic = in_shape[1] * in_shape[4];
            }
            count = out_shape.size() * kernel_x * kernel_y * ic * 2;

            int group = 1;
            if (n->params().count("group"))
            {
                group = n->params()["group"];
            }
            count = count / group;
            if (n->params().count("with_activation") && (std::string)n->params()["with_activation"] != Constants::act_none)
            {
                count += out_shape.size();
            }
            if (n->constants()["biases"])
            {
                count += out_shape.size();
            }
            break;
        }
        case OpType_Convolution3D:
        case OpType_ConvTranspose3D:
        {
            // output is N, D, H, W, C
            int kernel_x = n->params()["kernel_x"];
            int kernel_y = n->params()["kernel_y"];
            int kernel_z = n->params()["kernel_z"];
            TensorShape weights = n->constants()["weights"]->shape();
            // mac count as 2 ops
            Expr ic = in_shape.c();
            if (in_shape.dim() > 5)
            {
                ic = in_shape[1] * in_shape[5];
            }
            count = out_shape.size() * kernel_x * kernel_y * kernel_z * ic * 2;

            Expr group = ic / weights[4];
            count = count / group;
            if (n->params().count("with_activation") && (std::string)n->params()["with_activation"] != Constants::act_none)
            {
                count += out_shape.size();
            }
            if (n->constants()["biases"])
            {
                count += out_shape.size();
            }
            break;
        }
        case OpType_FullyConnected:
        {
            Expr batch = in_shape[0];
            count = batch * in_shape[1] * out_shape[1] * 2;
            Expr bias_add_op = out_shape.size() / batch;
            if (n->params().count("with_activation") && (std::string)n->params()["with_activation"] != Constants::act_none)
            {
                count += bias_add_op;
            }
            if (n->constants().count("biases") && n->constants()["biases"])
            {
                count += bias_add_op;
            }
            break;
        }
        case OpType_Gemm:
        {
            Expr batch = in_shape[0];
            count = batch * in_shape[1] * out_shape[1] * 2;
            count += out_shape.size() * 2;
            break;
        }
        case OpType_MatMul:
        {
            std::vector<std::vector<Expr>> origin_out_shapes = n->params()["ds_origin_top_shapes"];
            out_shape = origin_out_shapes[0];
            auto out_matrix_size = out_shape[2] * out_shape[3];
            auto out_batch = out_shape[0] * out_shape[1];

            std::vector<std::vector<Expr>> origin_in_shapes = n->params()["ds_origin_bottom_shapes"];
            Expr in_size = 1;
            for (size_t i = 0; i < origin_in_shapes.size(); i++)
            {
                auto in_sym_shape = origin_in_shapes[i];
                in_size *= in_sym_shape[2] * in_sym_shape[3];
            }
            auto k = (in_size / out_matrix_size).sqrt().floor();

            count = out_batch * out_matrix_size * k * 2;
            break;
        }
        case OpType_Activation:
        case OpType_Eltwise:
        case OpType_RELU:
        case OpType_CRELU:
        case OpType_Div:
        case OpType_Logical:
        case OpType_BiasAdd:
        case OpType_Abs:
        case OpType_Mod:
        case OpType_Pow:
        case OpType_Cast:
        case OpType_Unique:
        case OpType_Log:
        case OpType_Exp:
        case OpType_Sine:
        case OpType_Cosine:
        case OpType_Sqrt:
        case OpType_Square:
        case OpType_Elu:
        case OpType_Ceil:
        case OpType_Floor:
        case OpType_Acos:
        case OpType_Acosh:
        case OpType_Asin:
        case OpType_Asinh:
        case OpType_Cosh:
        case OpType_Sinh:
        case OpType_Tan:
        case OpType_Round:
        case OpType_Mul:
        case OpType_Add:
        case OpType_Sub:
        case OpType_Bitwise:
        case OpType_DivMod:
        case OpType_Atan:
        case OpType_Where:
        case OpType_Erf:
        case OpType_BitShift:
        case OpType_Sign:
        case OpType_Region2:
        case OpType_Region:
        case OpType_RegionFuse:
        case OpType_Meshgrid:
        case OpType_GridSample:
        case OpType_Proposal:
        case OpType_OverlapAdd:
        case OpType_BNLL:
        case OpType_Negative:
        {
            count = out_shape.size();
            break;
        }
        case OpType_ArgMinMax:
        case OpType_Reduce:
        case OpType_SegmentReduce:
        case OpType_GetValidCount:
        case OpType_Cumulate:
        case OpType_Rank:
        case OpType_TopKV2:
        case OpType_TopK:
        case OpType_Histogram:
        case OpType_SufficientStatistics:
        case OpType_Filter:
        case OpType_FilterBoxes:
        {
            count = in_shape.size();
            break;
        }
        case OpType_Hardmax:
        case OpType_Trunc:
        case OpType_GELU:
        case OpType_Silu:
        case OpType_HardSwish:
        case OpType_ThresholdedRelu:
        case OpType_SquaredDifference:
        case OpType_MVN:
        {
            count = 2 * out_shape.size();
            break;
        }
        case OpType_Softmax:
        case OpType_LogSoftmax:
        case OpType_FakeQuantWithMinMaxVars:
        case OpType_ProposalWithoutSoftmax:
        {
            // Softmax (input, axis) = Exp(input) / Reduce_Sum (Exp(input))
            //                         1 Exp, 2 Reduce_Sum, 3 Div
            count = 3 * out_shape.size();
            break;
        }
        case OpType_Dilation:
        case OpType_Erosion:
        {
            count = out_shape.size() * 3;
            break;
        }
        case OpType_Sort:
        {
            auto size = in_shape.size();
            count = size * size;
            break;
        }
        case OpType_GruSeq:
        case OpType_GRUv1:
        case OpType_GRUv3:
        {
            auto N = in_shape.n();
            int64_t T = (int64_t)n->params()["time_steps"];
            int64_t X = (int64_t)n->params()["input_size"];
            int64_t H = (int64_t)n->params()["cell_size"];
            count = N * (H * (X + H) * 2 * T);
            count *= 3;
            break;
        }
        case OpType_BasicLSTM:
        {
            // N* h * (h+x)*2*t*4
            auto N = in_shape.n();
            int64_t T = (int64_t)n->params()["time_steps"];
            int64_t X = (int64_t)n->params()["input_size"];
            int64_t H = (int64_t)n->params()["cell_size"];
            count = N * (H * (X + H) * 2 * T);
            count *= 4;
            break;
        }
        case OpType_RoiAlign:
        {
            auto b = out_shape.n();
            auto c = out_shape.c();
            auto pooled_h = out_shape.h();
            auto pooled_w = out_shape.w();
            int64_t sample_h = n->params()["sample"][1];
            sample_h = std::max(sample_h, int64_t(1));
            int64_t sample_w = n->params()["sample"][0];
            sample_w = std::max(sample_w, int64_t(1));

            // step 1 cal roi start end
            // ignored

            // step 2 cal 4 featuremap offset
            // offset ~= int((y * y_step + x) * x_step + z_step)
            count = (b * pooled_h * pooled_w * sample_h * sample_w * 5 * 4)  ;

            // step 3 cal linear weight
            count += (b * pooled_h * pooled_w * sample_h * sample_w * 6 * 4)  ;

            // step 4 cal pooling for each 4 point value
            if(std::string(n->params()["method"]) == "AVG") {
                count += (b * pooled_h * pooled_w * sample_h * sample_w * 8 * c);
            } else { // MAX
                count += (b * pooled_h * pooled_w * (sample_h * sample_w - 1) * 7 * c);
            }
            break;
        }
        case OpType_PyramidROIAlign:
        case OpType_MaxRoiPool:
        case OpType_RgbToYuv:
        case OpType_YuvToRgb:
        {
            count = out_shape.size() * 5;
            break;
        }
        case OpType_BatchNorm:
        case OpType_RELU1:
        case OpType_RELU6:
        case OpType_PRELU:
        case OpType_Rsqrt:
        case OpType_LeakyRELU:
        case OpType_Clip:
        case OpType_Quantize:
        case OpType_DeQuantize:
        case OpType_HardSigmoid:
        {
            count = out_shape.size() * 2;
            break;
        }
        case OpType_LayerNorm:
        case OpType_InstanceNorm:
        case OpType_Normalization:
        case OpType_RMSNorm:
        {
            std::vector<int> axis = n->params()["axis"];
            Expr size = 1;
            for (auto dim : axis)
            {
                size *= in_shape[dim];
            }
            // sum for mean and square sum for var
            count = size * 2;
            // sub mean and div var
            // also mul gamma and add beta
            count += in_shape.size() * 4;
            break;
        }
        case OpType_LRN:
        {
            Expr size = in_shape[in_shape.dim() - 1];
            int depth = n->params()["size"];
            count = size * depth;
            count += in_shape.size() * 4;
            break;
        }
        case OpType_GroupNorm:
        {
            int axis = n->params()["axis"];
            Expr size = in_shape[axis];
            int g = n->params()["group"];
            count = (size / g).floor() * 2;
            count += in_shape.size() * 4;
            break;
        }
        case OpType_RnnSeq:
        {
            count = out_shape[0] * out_shape[2] * (out_shape[2] + in_shape[2]) * 2 * in_shape[1];
            break;
        }
        case OpType_Pooling:
        {
            count = in_shape[0];
            for (int i = 1; i < out_shape.dim(); i++)
            {
                count *= out_shape[i];
            }  
            int kernel_x = n->params()["kernel_x"];
            int kernel_y = n->params()["kernel_y"];
            std::string method = std::string(n->params()["method"]);    
            if (method == "MAX")
            {
                // In a kernel, max: k * k - 1
                count *= kernel_x * kernel_y - 1;
            }
            if (method == "AVG")
            {
                // In a kernel, sum: k * k - 1; div: 1
                count *= kernel_x * kernel_y;
            }
            if (method == "L1")
            {
                // In a kernel, abs: k * k; sum: k * k - 1 
                count *= 2 * kernel_x * kernel_y - 1;
            }
            if (method == "L2")
            {
                // In a kernel, square: k * k; sum: k * k - 1; root: 1
                count *= 2 * kernel_x * kernel_y;
            }
            break;
        }
        case OpType_ROIPooling:
        case OpType_Resample:
        case OpType_Interp:
        case OpType_Upsample:
        case OpType_Resize:
        case OpType_Pooling3D:
        case OpType_MaxPooling3D:
        case OpType_MaxPoolingWithArgMax:
        case OpType_CropAndResize:
        case OpType_FractionalPool:
        {
            count = in_shape[0];
            for (int i = 1; i < in_shape.dim(); i++)
            {
                count *= ALIGN(in_shape[i], out_shape[i]);
            }
            break;
        }
        case OpType_SubGraph:
        {
            SubGraphNodePtr subnode = std::static_pointer_cast<SubGraphNode>(n);
            auto subgraph = subnode->subgraph();
            calculate_node_ops_count(subgraph);
            for (auto n : subgraph->nodes())
            {
                count += n->get_attr<Expr>("ops_count");
            }
            break;
        }
        default:
            break;
        }
        count->simplify();
        n->set_attr("ops_count", count);
    }
}
} // namespace aipubt
