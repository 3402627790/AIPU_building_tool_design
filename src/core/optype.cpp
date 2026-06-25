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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:28:17
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-07-19 17:59:48
 */
#include "core.h"

#include <algorithm>
#include <map>
namespace aipubt
{

std::map<std::string, OpType> &get_optype_dict()
{
    static std::map<std::string, OpType> OpTypeDict;
    return OpTypeDict;
}
static std::map<std::string, OpType> &get_optype_upper_dict()
{
    static std::map<std::string, OpType> OpTypeDict;
    return OpTypeDict;
}

std::map<OpType, std::string> &get_optype_name_dict()
{
    static std::map<OpType, std::string> OpTypeNameDict;
    return OpTypeNameDict;
}
OpType string_to_optype(const std::string &s)
{
    std::string str = s;
    return get_optype_dict()[str];
}
OpType string_upper_to_optype(const std::string &s)
{
    std::string upper = s;
    transform(s.begin(), s.end(), upper.begin(), ::toupper);
    return get_optype_upper_dict()[upper];
}
std::ostream &operator<<(std::ostream &os, const OpType opt)
{
    return os << optype_to_string(opt);
}

const std::string &optype_to_string(const OpType opt)
{
    return get_optype_name_dict()[opt];
}

static void register_new_op_type(const std::string &s, const OpType &opt)
{
    if (get_optype_upper_dict().count(s))
    {
        return;
    }
    if (!get_optype_name_dict().count(opt))
    {
        // we use the first registrant as the default name.registran
        get_optype_name_dict()[opt] = s;
    }
    std::string upper = s;
    get_optype_dict()[s] = opt;
    transform(s.begin(), s.end(), upper.begin(), ::toupper);
    get_optype_upper_dict()[upper] = opt;
}
OPTypeRegister::OPTypeRegister(const std::string &name, OpType type)
{
    register_new_op_type(name, type);
}

OPTYPE_REGISTER(OpType_Input);
OPTYPE_REGISTER(OpType_Convolution);
OPTYPE_REGISTER(OpType_DepthwiseConv);
OPTYPE_REGISTER(OpType_FullyConnected);
OPTYPE_REGISTER(OpType_Eltwise);
OPTYPE_REGISTER(OpType_Softmax);
OPTYPE_REGISTER(OpType_Pooling);
OPTYPE_REGISTER(OpType_Reshape);
OPTYPE_REGISTER(OpType_Concat);
OPTYPE_REGISTER(OpType_Activation);
OPTYPE_REGISTER(OpType_Resize);
OPTYPE_REGISTER(OpType_Interp);
OPTYPE_REGISTER(OpType_Upsample);
OPTYPE_REGISTER(OpType_GruSeq);
OPTYPE_REGISTER(OpType_BasicLSTM);
OPTYPE_REGISTER(OpType_Crop);
OPTYPE_REGISTER(OpType_BatchNorm);
OPTYPE_REGISTER(OpType_Pad);
OPTYPE_REGISTER(OpType_ConvTranspose);
OPTYPE_REGISTER(OpType_Deconvolution);
OPTYPE_REGISTER(OpType_Transpose4D);
OPTYPE_REGISTER(OpType_Permute);
OPTYPE_REGISTER(OpType_RnnSeq);
OPTYPE_REGISTER(OpType_Slice);
OPTYPE_REGISTER(OpType_Repeat);
OPTYPE_REGISTER(OpType_L2Norm);
OPTYPE_REGISTER(OpType_Split);
OPTYPE_REGISTER(OpType_RegionReshape);
OPTYPE_REGISTER(OpType_Region2);
OPTYPE_REGISTER(OpType_FcnPost1);
OPTYPE_REGISTER(OpType_FcnPost2);
OPTYPE_REGISTER(OpType_DataStride);
OPTYPE_REGISTER(OpType_ReverseSequence);
OPTYPE_REGISTER(OpType_RELU);
OPTYPE_REGISTER(OpType_Mfcc);
OPTYPE_REGISTER(OpType_PostNMS1);
OPTYPE_REGISTER(OpType_PostNMS2);
OPTYPE_REGISTER(OpType_Post_NMS1);
OPTYPE_REGISTER(OpType_Post_NMS2);
OPTYPE_REGISTER(OpType_CropAndResize);
OPTYPE_REGISTER(OpType_ROIPooling);
OPTYPE_REGISTER(OpType_Mask);
OPTYPE_REGISTER(OpType_Gbb);
OPTYPE_REGISTER(OpType_BoxReg);
OPTYPE_REGISTER(OpType_ReRec);
OPTYPE_REGISTER(OpType_MtPad);
OPTYPE_REGISTER(OpType_Resample);
OPTYPE_REGISTER(OpType_SecondBoxMv);
OPTYPE_REGISTER(OpType_BBreg);
OPTYPE_REGISTER(OpType_ThirdBoxPtmv);
OPTYPE_REGISTER(OpType_BoxScore);
OPTYPE_REGISTER(OpType_BoxScore2);
OPTYPE_REGISTER(OpType_Pick);
OPTYPE_REGISTER(OpType_NMS);
OPTYPE_REGISTER(OpType_NMSSSD);
OPTYPE_REGISTER(OpType_DecodeBox);
OPTYPE_REGISTER(OpType_CTCGreedySearch);
OPTYPE_REGISTER(OpType_FCNNMS);
OPTYPE_REGISTER(OpType_Region);
OPTYPE_REGISTER(OpType_Reorg);
OPTYPE_REGISTER(OpType_SSDNMS);
OPTYPE_REGISTER(OpType_Proposal);
OPTYPE_REGISTER(OpType_DecodeBoxSoftmax);
OPTYPE_REGISTER(OpType_DetectionOutput);
OPTYPE_REGISTER(OpType_CTCGreedyDecoder);
OPTYPE_REGISTER(OpType_YoloNMS);
OPTYPE_REGISTER(OpType_FasterRcnnNMS);
OPTYPE_REGISTER(OpType_Gru);
OPTYPE_REGISTER(OpType_Tile);
OPTYPE_REGISTER(OpType_ChannelShuffle);
OPTYPE_REGISTER(OpType_Moments);
OPTYPE_REGISTER(OpType_LayerNorm);
OPTYPE_REGISTER(OpType_Negative);
OPTYPE_REGISTER(OpType_Pooling3D);
OPTYPE_REGISTER(OpType_GatherV2);
OPTYPE_REGISTER(OpType_InTopK);
OPTYPE_REGISTER(OpType_MaxPoolingWithArgMax);
OPTYPE_REGISTER(OpType_HeatmapMaxKeypoint);
OPTYPE_REGISTER(OpType_Transpose);
OPTYPE_REGISTER(OpType_BiasAdd);
OPTYPE_REGISTER(OpType_Logical);
OPTYPE_REGISTER(OpType_Div);
OPTYPE_REGISTER(OpType_RoiAlign);
OPTYPE_REGISTER(OpType_AccidentalHits);
OPTYPE_REGISTER(OpType_ComputeAccidentalHits);
OPTYPE_REGISTER(OpType_LogSoftmax);
OPTYPE_REGISTER(OpType_Where);
// OPTYPE_REGISTER(OpType_Select);
OPTYPE_REGISTER(OpType_StridedSlice);
OPTYPE_REGISTER(OpType_ArgMinMax);
OPTYPE_REGISTER(OpType_Reduce);
OPTYPE_REGISTER(OpType_ZeroFraction);
OPTYPE_REGISTER(OpType_LRN);
OPTYPE_REGISTER(OpType_FractionalPool);
OPTYPE_REGISTER(OpType_RELU1);
OPTYPE_REGISTER(OpType_RELU6);
OPTYPE_REGISTER(OpType_PRELU);
OPTYPE_REGISTER(OpType_LeakyRELU);
OPTYPE_REGISTER(OpType_HardSwish);
OPTYPE_REGISTER(OpType_Sort);
OPTYPE_REGISTER(OpType_Gather);
OPTYPE_REGISTER(OpType_EmbeddingLookup);
OPTYPE_REGISTER(OpType_Squeeze);
OPTYPE_REGISTER(OpType_HashLookupTable);
OPTYPE_REGISTER(OpType_Constant);
// OPTYPE_REGISTER(OpType_Const);
OPTYPE_REGISTER(OpType_Normalization);
OPTYPE_REGISTER(OpType_Abs);
OPTYPE_REGISTER(OpType_LUT);
OPTYPE_REGISTER(OpType_InstanceNorm);
OPTYPE_REGISTER(OpType_NormalizedMoments);
OPTYPE_REGISTER(OpType_TopKV2);
OPTYPE_REGISTER(OpType_MatMul);
OPTYPE_REGISTER(OpType_EmbeddingLookupSparse);
OPTYPE_REGISTER(OpType_Clip);
OPTYPE_REGISTER(OpType_MaxRoiPool);
OPTYPE_REGISTER(OpType_Mod);
OPTYPE_REGISTER(OpType_Pow);
OPTYPE_REGISTER(OpType_Reciprocal);
OPTYPE_REGISTER(OpType_Det);
OPTYPE_REGISTER(OpType_ThresholdedRelu);
OPTYPE_REGISTER(OpType_Cast);
OPTYPE_REGISTER(OpType_Unique);
OPTYPE_REGISTER(OpType_ProposalWithoutSoftmax);
OPTYPE_REGISTER(OpType_BoxWithNmsLimit);
OPTYPE_REGISTER(OpType_Dilation);
OPTYPE_REGISTER(OpType_Erosion);
OPTYPE_REGISTER(OpType_ScoreGaussUpdate);
OPTYPE_REGISTER(OpType_TopK);
OPTYPE_REGISTER(OpType_CollapseRepeated);
OPTYPE_REGISTER(OpType_OverlapAdd);
OPTYPE_REGISTER(OpType_SufficientStatistics);
OPTYPE_REGISTER(OpType_GenerateProposals);
OPTYPE_REGISTER(OpType_RegionFuse);
OPTYPE_REGISTER(OpType_RoiAlign29);
OPTYPE_REGISTER(OpType_BatchReindex);
OPTYPE_REGISTER(OpType_MVN);
OPTYPE_REGISTER(OpType_Log);
OPTYPE_REGISTER(OpType_Exp);
OPTYPE_REGISTER(OpType_Sine);
OPTYPE_REGISTER(OpType_Cosine);
OPTYPE_REGISTER(OpType_Rsqrt);
OPTYPE_REGISTER(OpType_Sqrt);
OPTYPE_REGISTER(OpType_Square);
OPTYPE_REGISTER(OpType_GRUv3);
OPTYPE_REGISTER(OpType_GRUv1);
OPTYPE_REGISTER(OpType_BNLL);
OPTYPE_REGISTER(OpType_Intersection);
OPTYPE_REGISTER(OpType_Filter);
OPTYPE_REGISTER(OpType_Histogram);
OPTYPE_REGISTER(OpType_UpsampleByIndex);
OPTYPE_REGISTER(OpType_Sign);
OPTYPE_REGISTER(OpType_OneHot);
OPTYPE_REGISTER(OpType_BoundingBox);
OPTYPE_REGISTER(OpType_GatherND);
OPTYPE_REGISTER(OpType_PyramidROIAlign);
OPTYPE_REGISTER(OpType_Convolution3D);
OPTYPE_REGISTER(OpType_FilterBoxes);
OPTYPE_REGISTER(OpType_HardSigmoid);
OPTYPE_REGISTER(OpType_Ceil);
OPTYPE_REGISTER(OpType_ConvTranspose3D);
OPTYPE_REGISTER(OpType_Floor);
OPTYPE_REGISTER(OpType_Gemm);
OPTYPE_REGISTER(OpType_MaxPooling3D);
OPTYPE_REGISTER(OpType_RgbToYuv);
OPTYPE_REGISTER(OpType_Silu);
OPTYPE_REGISTER(OpType_YuvToRgb);
OPTYPE_REGISTER(OpType_I420ToYUV444pkd);
OPTYPE_REGISTER(OpType_YUV444pkdToI420);
OPTYPE_REGISTER(OpType_Count);
OPTYPE_REGISTER(OpType_ScatterND);
OPTYPE_REGISTER(OpType_SegmentReduce);
OPTYPE_REGISTER(OpType_Rank);
OPTYPE_REGISTER(OpType_BitShift);
OPTYPE_REGISTER(OpType_Erf);
OPTYPE_REGISTER(OpType_RefineDetDetection);
OPTYPE_REGISTER(OpType_SquaredDifference);
OPTYPE_REGISTER(OpType_Acos);
OPTYPE_REGISTER(OpType_Acosh);
OPTYPE_REGISTER(OpType_Asin);
OPTYPE_REGISTER(OpType_Asinh);
OPTYPE_REGISTER(OpType_Cosh);
OPTYPE_REGISTER(OpType_Sinh);
OPTYPE_REGISTER(OpType_Tan);
OPTYPE_REGISTER(OpType_Round);
OPTYPE_REGISTER(OpType_MatMulInteger);
OPTYPE_REGISTER(OpType_ConvInteger);
OPTYPE_REGISTER(OpType_ScatterElements);
OPTYPE_REGISTER(OpType_GatherElements);
OPTYPE_REGISTER(OpType_Compress);
OPTYPE_REGISTER(OpType_MaxUnpool);
OPTYPE_REGISTER(OpType_GridSample);
OPTYPE_REGISTER(OpType_CRELU);
OPTYPE_REGISTER(OpType_Hardmax);
OPTYPE_REGISTER(OpType_Add);
OPTYPE_REGISTER(OpType_Sub);
OPTYPE_REGISTER(OpType_Mul);
OPTYPE_REGISTER(OpType_GELU);
OPTYPE_REGISTER(OpType_Elu);
OPTYPE_REGISTER(OpType_GroupNorm);
OPTYPE_REGISTER(OpType_RNN);
OPTYPE_REGISTER(OpType_Meshgrid);
OPTYPE_REGISTER(OpType_Quantize);
OPTYPE_REGISTER(OpType_DeQuantize);
OPTYPE_REGISTER(OpType_Cumulate);
OPTYPE_REGISTER(OpType_Bitwise);
OPTYPE_REGISTER(OpType_MultiboxTransformLoc);
OPTYPE_REGISTER(OpType_Trunc);
OPTYPE_REGISTER(OpType_GetValidCount);
OPTYPE_REGISTER(OpType_RMSNorm);
OPTYPE_REGISTER(OpType_DivMod);
OPTYPE_REGISTER(OpType_Col2Im);
OPTYPE_REGISTER(OpType_Atan);
OPTYPE_REGISTER(OpType_AdaptivePool);
OPTYPE_REGISTER(OpType_GlobalPool);
OPTYPE_REGISTER(OpType_Atanh);
OPTYPE_REGISTER(OpType_AffineGrid);
OPTYPE_REGISTER(OpType_NonZero);
OPTYPE_REGISTER(OpType_QueryRebatch);
OPTYPE_REGISTER(OpType_SlotUpdate);
OPTYPE_REGISTER(OpType_IsInf);
OPTYPE_REGISTER(OpType_IsNaN);
OPTYPE_REGISTER(OpType_CacheUpdate);
OPTYPE_REGISTER(OpType_Dummy);
OPTYPE_REGISTER(OpType_Shape);
OPTYPE_REGISTER(OpType_ConstantOfShape);
OPTYPE_REGISTER(OpType_RotaryEmbedding);
OPTYPE_REGISTER(OpType_FlashAttention);
OPTYPE_REGISTER(OpType_Switch);
OPTYPE_REGISTER(OpType_Empty);
OPTYPE_REGISTER(OpType_FlashDecoding);
OPTYPE_REGISTER(OpType_ScaledDotProductAttention);
OPTYPE_REGISTER(OpType_Trilu);
OPTYPE_REGISTER(OpType_ParamUpdate);
OPTYPE_REGISTER(OpType_FusedMoELoop);

OPTYPE_REGISTER(OpType_BatchToDepth);
OPTYPE_REGISTER(OpType_DepthToBatch);

OPTYPE_REGISTER(OpType_BatchToSpace);
OPTYPE_REGISTER(OpType_SpaceToBatch);

OPTYPE_REGISTER(OpType_BatchToSpaceND);
OPTYPE_REGISTER(OpType_SpaceToBatchND);

OPTYPE_REGISTER(OpType_DepthToSpace);
OPTYPE_REGISTER(OpType_SpaceToDepth);

OPTYPE_REGISTER(OpType_ChannelToSpace);
OPTYPE_REGISTER(OpType_SpaceToChannel);
OPTYPE_REGISTER(OpType_LayoutConvertor);

OPTYPE_REGISTER(OpType_FakeQuantWithMinMaxVars);

OPTYPE_REGISTER(OpType_InternalConcatenated);
OPTYPE_REGISTER(OpType_InternalCrop);
OPTYPE_REGISTER(OpType_InternalSpliter);
OPTYPE_REGISTER(OpType_ForRangeHeader);
OPTYPE_REGISTER(OpType_ForRangeEnder);
OPTYPE_REGISTER(OpType_Call);
OPTYPE_REGISTER(OpType_InternalSlice);
OPTYPE_REGISTER(OpType_InternalConstant);
OPTYPE_REGISTER(OpType_InternalPad);
OPTYPE_REGISTER(OpType_IfHeader);
OPTYPE_REGISTER(OpType_IfEnder);
OPTYPE_REGISTER(OpType_InternalSwitch);

OPTYPE_REGISTER(OpType_ConvPre);
OPTYPE_REGISTER(OpType_ConvMerge);

OPTYPE_REGISTER(OpType_SubGraphInput);

OPTYPE_REGISTER(OpType_SubGraph);
OPTYPE_REGISTER(OpType_Customize);
OPTYPE_REGISTER(OpType_ForRange);
OPTYPE_REGISTER(OpType_Loop);
OPTYPE_REGISTER(OpType_If);

OPTYPE_REGISTER(OpType_Testing);
} // namespace aipubt
