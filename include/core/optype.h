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
 * @Date: 2020-02-09 12:15:34
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 14:40:48
 */
#ifndef CORE_OPTYPE_H
#define CORE_OPTYPE_H

#include <cstdint>
#include <ostream>
#include <string>

namespace aipubt
{
enum OpType
{
    OpType_NoOp = 0x0,
    OpType_MAX_COUNT = (1 << 16) - 1,
    OpType_Input = 1,
    OpType_Convolution = 2,
    OpType_DepthwiseConv = 3,
    OpType_FullyConnected = 4,
    OpType_Eltwise = 5,
    OpType_Softmax = 6,
    OpType_Pooling = 7,
    OpType_Reshape = 8,
    OpType_Concat = 9,
    OpType_Activation = 10,
    OpType_Interp = 11,   //TODO!! remove later
    OpType_Upsample = 12, //TODO!! remove later
    OpType_GruSeq = 13,   //TODO!! remove later
    OpType_BasicLSTM = 14,
    OpType_Crop = 15,
    OpType_BatchNorm = 16,
    OpType_Pad = 17,
    OpType_ConvTranspose = 18,
    OpType_Deconvolution = 18, //TODO: it is a temp name, and may be renamed later.
    OpType_Transpose4D = 19,   //TODO!! remove later
    OpType_Permute = 20,
    OpType_RnnSeq = 21,
    OpType_Slice = 22,
    OpType_Repeat = 23,
    OpType_L2Norm = 24,
    OpType_Split = 25,
    OpType_RegionReshape = 26,
    OpType_Region2 = 27,
    OpType_FcnPost1 = 28, //TODO!! remove later
    OpType_FcnPost2 = 29, //TODO!! remove later
    OpType_DataStride = 30,
    OpType_ReverseSequence = 31,
    OpType_RELU = 32, //TODO!! remove later
    OpType_Mfcc = 33,
    OpType_PostNMS1 = 34,  //TODO!! remove later
    OpType_PostNMS2 = 35,  //TODO!! remove later
    OpType_Post_NMS1 = 36, //TODO!! remove later
    OpType_Post_NMS2 = 37, //TODO!! remove later
    OpType_CropAndResize = 38,
    OpType_ROIPooling = 39,
    OpType_Mask = 40,
    OpType_Gbb = 41,
    OpType_BoxReg = 42,
    OpType_ReRec = 43,
    OpType_MtPad = 44,
    OpType_Resample = 45,
    OpType_SecondBoxMv = 46,
    OpType_BBreg = 47,
    OpType_ThirdBoxPtmv = 48,
    OpType_BoxScore = 49,
    OpType_BoxScore2 = 50,
    OpType_Pick = 51,
    OpType_NMS = 52,
    OpType_NMSSSD = 53,
    OpType_DecodeBox = 54,
    OpType_CTCGreedySearch = 55,
    OpType_FCNNMS = 56,
    OpType_Region = 57,
    OpType_Reorg = 58,
    OpType_SSDNMS = 59,
    OpType_Proposal = 60,
    OpType_DecodeBoxSoftmax = 61,
    OpType_DetectionOutput = 62,
    OpType_CTCGreedyDecoder = 63,
    OpType_YoloNMS = 64,
    OpType_FasterRcnnNMS = 65,
    OpType_Gru = 66,
    OpType_Tile = 67,
    OpType_ChannelShuffle = 68,
    OpType_Moments = 69,
    OpType_LayerNorm = 70,
    OpType_Negative = 71,
    OpType_Pooling3D = 72,
    OpType_GatherV2 = 73,
    OpType_InTopK = 74,
    OpType_MaxPoolingWithArgMax = 75,
    OpType_HeatmapMaxKeypoint = 76,
    OpType_Transpose = 77,
    OpType_BiasAdd = 78,
    OpType_Logical = 79,
    OpType_Div = 80,
    OpType_Resize = 81,
    OpType_RoiAlign = 82,
    OpType_AccidentalHits = 83,
    OpType_ComputeAccidentalHits = 83,
    OpType_LogSoftmax = 84,
    // OpType_Select = 85,
    OpType_Where = 85,
    OpType_StridedSlice = 86,
    OpType_ArgMinMax = 87,
    OpType_Reduce = 88,
    OpType_ZeroFraction = 89,
    OpType_LRN = 90,
    OpType_FractionalPool = 91,
    OpType_RELU1 = 92,
    OpType_RELU6 = 93,
    OpType_PRELU = 94,
    OpType_Gather = 95,
    OpType_EmbeddingLookup = 96,
    OpType_Squeeze = 97,
    OpType_HashLookupTable = 98,
    // OpType_Const = 99,
    OpType_Constant = 99,
    OpType_Normalization = 100,
    OpType_Abs = 101,
    OpType_LUT = 102,
    OpType_InstanceNorm = 103,
    OpType_NormalizedMoments = 104,
    OpType_TopKV2 = 105,
    OpType_MatMul = 106,
    OpType_EmbeddingLookupSparse = 107,
    OpType_Clip = 108,
    OpType_MaxRoiPool = 109,
    OpType_Mod = 110,
    OpType_Pow = 111,
    OpType_Reciprocal = 112,
    OpType_Det = 113,
    OpType_ThresholdedRelu = 114,
    OpType_Cast = 115,
    OpType_Unique = 116,
    OpType_ProposalWithoutSoftmax = 117,
    OpType_BoxWithNmsLimit = 118,
    OpType_Dilation = 119,
    OpType_Erosion = 120,
    OpType_ScoreGaussUpdate = 121,
    OpType_TopK = 122,
    OpType_CollapseRepeated = 123,
    OpType_OverlapAdd = 124,
    OpType_SufficientStatistics = 125,
    OpType_GenerateProposals = 126,
    OpType_RegionFuse = 127,
    OpType_RoiAlign29 = 128,
    OpType_BatchReindex = 129,
    OpType_MVN = 130,
    OpType_Log = 131,
    OpType_Exp = 132,
    OpType_Sine = 133,
    OpType_Cosine = 134,
    OpType_Rsqrt = 135,
    OpType_Sqrt = 136,
    OpType_Square = 137,
    OpType_GRUv3 = 138,
    OpType_GRUv1 = 139,
    OpType_BNLL = 140,
    OpType_Intersection = 141,
    OpType_Filter = 142,
    OpType_Histogram = 143,
    OpType_UpsampleByIndex = 144,
    OpType_Sign = 145,
    OpType_OneHot = 146,
    OpType_BoundingBox = 147,
    OpType_GatherND = 148,
    OpType_PyramidROIAlign = 149,
    OpType_LeakyRELU = 150,
    OpType_HardSwish = 151,
    OpType_Sort = 152,
    OpType_Elu = 153,
    OpType_Convolution3D = 154,
    OpType_FilterBoxes = 155,
    OpType_HardSigmoid = 156,
    OpType_Ceil = 157,
    OpType_ConvTranspose3D = 158,
    OpType_Floor = 159,
    OpType_Gemm = 160,
    OpType_MaxPooling3D = 161,
    OpType_RgbToYuv = 162,
    OpType_Silu = 163,
    OpType_YuvToRgb = 164,
    OpType_I420ToYUV444pkd = 165,
    OpType_YUV444pkdToI420 = 166,
    OpType_Count = 167,
    OpType_ScatterND = 168,
    OpType_SegmentReduce = 169,
    OpType_Rank = 170,
    OpType_BitShift = 171,
    OpType_Erf = 172,
    OpType_RefineDetDetection = 173,
    OpType_SquaredDifference = 174,
    OpType_Acos = 175,
    OpType_Acosh = 176,
    OpType_Asin = 177,
    OpType_Asinh = 178,
    OpType_Cosh = 179,
    OpType_Sinh = 180,
    OpType_Tan = 181,
    OpType_Round = 182,
    OpType_MatMulInteger = 183,
    OpType_ConvInteger = 184,
    OpType_ScatterElements = 185,
    OpType_GatherElements = 186,
    OpType_Compress = 187,
    OpType_MaxUnpool = 188,
    OpType_GridSample = 189,
    OpType_CRELU = 190,
    OpType_Hardmax = 191,
    OpType_Add = 192,
    OpType_Sub = 193,
    OpType_Mul = 194,
    OpType_GELU = 195,
    OpType_GroupNorm = 196,
    OpType_RNN = 197,
    OpType_Meshgrid = 198,
    OpType_Quantize = 199,
    OpType_DeQuantize = 200,
    OpType_Cumulate = 201,
    OpType_Bitwise = 202,
    OpType_MultiboxTransformLoc = 203,
    OpType_Trunc = 204,
    OpType_GetValidCount = 205,
    OpType_RMSNorm = 206,
    OpType_DivMod = 207,
    OpType_Col2Im = 208,
    OpType_Atan = 209,
    OpType_AdaptivePool = 210,
    OpType_GlobalPool = 211,
    OpType_Atanh = 212,
    OpType_AffineGrid = 213,
    OpType_NonZero = 214,
    OpType_QueryRebatch = 215,
    OpType_SlotUpdate = 216,
    OpType_IsInf = 217,
    OpType_IsNaN = 218,
    OpType_CacheUpdate = 219,
    OpType_Dummy = 220,
    OpType_Shape = 221,
    OpType_ConstantOfShape = 222,
    OpType_RotaryEmbedding = 223,
    OpType_FlashAttention = 224,
    OpType_Switch = 225,
    OpType_Empty = 226,

    OpType_BatchToDepth = 1000,
    OpType_DepthToBatch = 1001,
    OpType_BatchToSpace = 1020,
    OpType_SpaceToBatch = 1021,
    OpType_DepthToSpace = 1022,
    OpType_SpaceToDepth = 1023,
    OpType_ChannelToSpace = 1024,
    OpType_SpaceToChannel = 1025,
    OpType_LayoutConvertor = 1026,

    OpType_BatchToSpaceND = 1120,
    OpType_SpaceToBatchND = 1121,

    OpType_FakeQuantWithMinMaxVars = 1500,

    OpType_InternalSpliter = 2000,
    OpType_InternalConcatenated = 2001,
    OpType_InternalCrop = 2002,
    OpType_ForRangeHeader = 2003,
    OpType_ForRangeEnder = 2004,
    OpType_Call = 2005,
    OpType_InternalSlice = 2006,
    OpType_InternalConstant = 2007,
    OpType_InternalPad = 2008,
    OpType_IfHeader = 2009,
    OpType_IfEnder = 2010,
    OpType_InternalSwitch = 2011,

    OpType_ConvPre = 3000,
    OpType_ConvMerge = 3001,

    OpType_SubGraphInput = 4095,

    //OpType >= OpType_subGraph && OpType <= OpType_SubGraphMaxCount will be considered subgraphnode
    OpType_SubGraph = 4096,
    OpType_ForRange = 4098,
    OpType_Loop = 4099,
    OpType_If = 4100,
    OpType_SubGraphMaxCount,

    OpType_Testing = 8192,
    OpType_Customize = 8193,
};

class OPTypeRegister
{
public:
    OPTypeRegister(const std::string &name, OpType type);
};
std::ostream &operator<<(std::ostream &os, const OpType opt);
OpType string_to_optype(const std::string &s);
OpType string_upper_to_optype(const std::string &s);
const std::string &optype_to_string(const OpType opt);

} // namespace aipubt

#define OPTYPE_REGISTER(name) \
    static aipubt::OPTypeRegister _register##name((#name) + 7, (OpType)name);

#if __cplusplus >= 201402L
#define HASH_SIZE (1 << 24)
constexpr int typename2int(const char *p, int len)
{
    uint32_t ret = 0;
    for (int i = 0; i < len; ++i)
    {
        ret = ((int)(p[i])) + 31 * ret;
    }
    ret = ret % HASH_SIZE;
    ret = ret + (1 << 28);
    return ret;
}

#define CUSTOMIZETYPE(name) OpType_##name = typename2int(#name, sizeof(#name))
#define REGISTER_CUSTOMIZE_OPTYPE(name) \
    enum                                \
    {                                   \
        CUSTOMIZETYPE(name),            \
    };                                  \
    OPTYPE_REGISTER(OpType_##name)
#endif // c++14
#endif //! CORE_OPTYPE_H
