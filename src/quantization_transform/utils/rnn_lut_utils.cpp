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
 * @Author: Samuel Han
 * @Date: 2020-05-08 14:56:54
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-03-22 16:49:50
 */

#include "rnn_lut_utils.h"
namespace qtlibn
{

typedef std::pair<std::string, std::string> METHOD;
typedef std::pair<std::string, aipubt::Tensor::Dtype> DTYPE;
static std::map<METHOD, std::string> G_RNN_LUT_SUPPORTED = {
    {std::make_pair("BasicLSTM", "SIGMOID"), "rnn_sigm_lut_all_v3"},
    {std::make_pair("BasicLSTM", "TANH"), "rnn_tanh_lut_all_v3"},
    {std::make_pair("GRUv3", "SIGMOID"), "rnn_sigm_lut_all_v3"},
    {std::make_pair("GRUv3", "TANH"), "rnn_tanh_lut_all_v3"},
    {std::make_pair("GRUv1", "SIGMOID"), "rnn_sigm_lut_all_v1"},
    {std::make_pair("GRUv1", "TANH"), "rnn_tanh_lut_all_v1"},
};
//get scales
static std::map<DTYPE, std::vector<float>> bit8_G_RNN_SCALES_MAP = {
    {std::make_pair("rnn_tanh_lut_all_v3", aipubt::Tensor::INT8),
     {12364.9056604, 127.0}},
    {std::make_pair("rnn_sigm_lut_all_v3", aipubt::Tensor::INT8),
     {7123.2608695, 127.0}},
    {std::make_pair("rnn_tanh_lut_all_v1", aipubt::Tensor::INT8),
     {128 / 3.1 * 256, 127.0}},
    {std::make_pair("rnn_sigm_lut_all_v1", aipubt::Tensor::INT8),
     {128.0 / 6.2 * 256.0, 127.0}},
};
static std::map<DTYPE, std::vector<float>> bit16_G_RNN_SCALES_MAP = {
    {std::make_pair("rnn_tanh_lut_all_v3", aipubt::Tensor::INT16),
     {32768 / 3.1, 32768.0}},
    {std::make_pair("rnn_sigm_lut_all_v3", aipubt::Tensor::INT16),
     {32768 / 6.2, 32768.0}},
};
//get lut
static std::map<DTYPE, std::vector<int8_t>> bit8_G_RNN_LUT_MAP = {
    {std::make_pair("rnn_tanh_lut_all_v3", aipubt::Tensor::INT8),
     {-127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126,
      -126, -126, -126, -125, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124,
      -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121,
      -121, -121, -120, -120, -120, -119, -119, -119, -118, -118, -117, -117, -117, -116,
      -116, -115, -115, -114, -114, -113, -112, -112, -111, -110, -110, -109, -108, -108,
      -107, -106, -105, -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -92,
      -91, -90, -88, -87, -86, -84, -83, -81, -79, -78, -76, -74, -72, -71, -69, -67, -65,
      -63, -61, -59, -57, -55, -52, -50, -48, -46, -43, -41, -39, -36, -34, -31, -29, -26,
      -24, -21, -18, -16, -13, -11, -8, -5, -3, 0, 3, 5, 8, 11, 13, 16, 18, 21, 24, 26, 29,
      31, 34, 36, 39, 41, 43, 46, 48, 50, 52, 55, 57, 59, 61, 63, 65, 67, 69, 71, 72, 74, 76,
      78, 79, 81, 83, 84, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
      104, 105, 106, 107, 108, 108, 109, 110, 110, 111, 112, 112, 113, 114, 114, 115, 115, 116,
      116, 117, 117, 117, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 121, 122, 122,
      122, 122, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125,
      125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127,
      127}},
    {std::make_pair("rnn_sigm_lut_all_v3", aipubt::Tensor::INT8),
     {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9,
      9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
      18, 18, 19, 19, 20, 21, 21, 22, 22, 23, 24, 25, 25, 26, 27, 28, 28, 29, 30, 31,
      32, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
      52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73,
      74, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
      95, 96, 96, 97, 98, 99, 100, 100, 101, 102, 103, 103, 104, 105, 106, 106, 107,
      107, 108, 109, 109, 110, 110, 111, 111, 112, 112, 113, 113, 114, 114, 115, 115,
      116, 116, 116, 117, 117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 120,
      121, 121, 121, 121, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 124,
      124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127,
      127, 127}},
    {std::make_pair("rnn_tanh_lut_all_v1", aipubt::Tensor::INT8),
     {-127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126,
      -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125,
      -125, -125, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -124,
      -123, -123, -123, -123, -123, -123, -122, -122, -122, -122, -121, -121, -121, -120,
      -120, -120, -119, -119, -119, -118, -118, -117, -117, -116, -116, -115, -115, -114,
      -113, -113, -112, -111, -111, -110, -109, -108, -107, -106, -106, -105, -103, -102,
      -101, -100, -99, -98, -96, -95, -94, -92, -91, -89, -87, -86, -84, -82, -80, -78, -76,
      -74, -72, -70, -68, -66, -63, -61, -58, -56, -53, -51, -48, -45, -42, -40, -37, -34,
      -31, -28, -25, -22, -19, -16, -13, -9, -6, -3, 0, 3, 6, 9, 13, 16, 19, 22, 25, 28,
      31, 34, 37, 40, 42, 45, 48, 51, 53, 56, 58, 61, 63, 66, 68, 70, 72, 74, 76, 78, 80,
      82, 84, 86, 87, 89, 91, 92, 94, 95, 96, 98, 99, 100, 101, 102, 103, 105, 106, 106,
      107, 108, 109, 110, 111, 111, 112, 113, 113, 114, 115, 115, 116, 116, 117, 117, 118,
      118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122, 122, 122, 123, 123, 123,
      123, 123, 123, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125,
      125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 127, 127}},
    {std::make_pair("rnn_sigm_lut_all_v1", aipubt::Tensor::INT8),
     {1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4,
      5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11,
      11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 22, 22,
      23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42,
      44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 62, 63, 64, 65, 66, 68,
      69, 71, 72, 74, 75, 76, 78, 79, 80, 82, 83, 84, 86, 87, 88, 89, 91, 92, 93,
      94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 106, 107, 108,
      109, 109, 110, 111, 111, 112, 113, 113, 114, 114, 115, 115, 116, 116, 117,
      117, 117, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 121, 122,
      122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124,
      124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126,
      126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127}}};
static std::map<DTYPE, std::vector<int16_t>> bit16_G_RNN_LUT_MAP = {
    {std::make_pair("rnn_tanh_lut_all_v3", aipubt::Tensor::INT16),
     {-32635, -32629, -32622, -32615, -32607, -32599, -32591, -32582, -32573, -32563,
      -32553, -32542, -32531, -32519, -32507, -32494, -32481, -32466, -32451, -32436,
      -32419, -32402, -32384, -32365, -32345, -32325, -32303, -32280, -32256, -32231,
      -32204, -32176, -32147, -32117, -32085, -32051, -32016, -31979, -31941, -31900,
      -31858, -31813, -31766, -31718, -31666, -31613, -31556, -31497, -31435, -31371,
      -31303, -31232, -31158, -31080, -30998, -30913, -30824, -30730, -30632, -30530,
      -30423, -30311, -30194, -30071, -29943, -29809, -29669, -29523, -29370, -29211,
      -29044, -28870, -28689, -28500, -28302, -28097, -27882, -27658, -27426, -27183,
      -26930, -26668, -26394, -26110, -25815, -25508, -25189, -24859, -24516, -24160,
      -23791, -23409, -23014, -22605, -22182, -21745, -21294, -20828, -20348, -19853,
      -19343, -18818, -18279, -17725, -17156, -16573, -15975, -15363, -14737, -14097,
      -13444, -12777, -12098, -11407, -10703, -9989, -9264, -8529, -7784, -7031,
      -6271, -5503, -4728, -3949, -3165, -2377, -1586, -793, 0, 793,
      1586, 2377, 3165, 3949, 4728, 5503, 6271, 7031, 7784, 8529,
      9264, 9989, 10703, 11407, 12098, 12777, 13444, 14097, 14737, 15363,
      15975, 16573, 17156, 17725, 18279, 18818, 19343, 19853, 20348, 20828,
      21294, 21745, 22182, 22605, 23014, 23409, 23791, 24160, 24516, 24859,
      25189, 25508, 25815, 26110, 26394, 26668, 26930, 27183, 27426, 27658,
      27882, 28097, 28302, 28500, 28689, 28870, 29044, 29211, 29370, 29523,
      29669, 29809, 29943, 30071, 30194, 30311, 30423, 30530, 30632, 30730,
      30824, 30913, 30998, 31080, 31158, 31232, 31303, 31371, 31435, 31497,
      31556, 31613, 31666, 31718, 31766, 31813, 31858, 31900, 31941, 31979,
      32016, 32051, 32085, 32117, 32147, 32176, 32204, 32231, 32256, 32280,
      32303, 32325, 32345, 32365, 32384, 32402, 32419, 32436, 32451, 32466,
      32481, 32494, 32507, 32519, 32531, 32542, 32553, 32563, 32573, 32582,
      32591, 32599, 32607, 32615, 32622, 32629, 32635}},
    {std::make_pair("rnn_sigm_lut_all_v3", aipubt::Tensor::INT16),
     {66, 70, 73, 77, 81, 85, 89, 93, 98, 103, 108, 113,
      118, 124, 130, 137, 144, 151, 158, 166, 174, 183, 192, 201,
      211, 222, 233, 244, 256, 269, 282, 296, 310, 326, 342, 358,
      376, 394, 414, 434, 455, 477, 501, 525, 551, 578, 606, 635,
      666, 699, 733, 768, 805, 844, 885, 928, 972, 1019, 1068, 1119,
      1173, 1229, 1287, 1348, 1413, 1479, 1549, 1623, 1699, 1779, 1862, 1949,
      2039, 2134, 2233, 2336, 2443, 2555, 2671, 2793, 2919, 3050, 3187, 3329,
      3477, 3630, 3789, 3955, 4126, 4304, 4488, 4679, 4877, 5082, 5293, 5512,
      5737, 5970, 6210, 6458, 6713, 6975, 7244, 7521, 7806, 8097, 8396, 8702,
      9015, 9335, 9662, 9995, 10335, 10681, 11032, 11390, 11752, 12120, 12492, 12868,
      13249, 13633, 14020, 14410, 14802, 15196, 15591, 15987, 16384, 16781, 17177, 17572,
      17966, 18358, 18748, 19135, 19519, 19900, 20276, 20648, 21016, 21378, 21736, 22087,
      22433, 22773, 23106, 23433, 23753, 24066, 24372, 24671, 24962, 25247, 25524, 25793,
      26055, 26310, 26558, 26798, 27031, 27256, 27475, 27686, 27891, 28089, 28280, 28464,
      28642, 28813, 28979, 29138, 29291, 29439, 29581, 29718, 29849, 29975, 30097, 30213,
      30325, 30432, 30535, 30634, 30729, 30819, 30906, 30989, 31069, 31145, 31219, 31289,
      31355, 31420, 31481, 31539, 31595, 31649, 31700, 31749, 31796, 31840, 31883, 31924,
      31963, 32000, 32035, 32069, 32102, 32133, 32162, 32190, 32217, 32243, 32267, 32291,
      32313, 32334, 32354, 32374, 32392, 32410, 32426, 32442, 32458, 32472, 32486, 32499,
      32512, 32524, 32535, 32546, 32557, 32567, 32576, 32585, 32594, 32602, 32610, 32617,
      32624, 32631, 32638, 32644, 32650, 32655, 32660, 32665, 32670, 32675, 32679, 32683,
      32687, 32691, 32695, 32698, 32702}}};
std::vector<float> get_activation_scale(std::string op_type, std::vector<std::string> activations, aipubt::Tensor::Dtype dtype)
{
    std::vector<float> activation_scales = {};
    //default basicLSTM activation list is sigmoid, tanh, tanh
    std::vector<float> scale_pair;
    if (dtype == aipubt::Tensor::INT8)
    {
        for (std::string s : activations)
        {
            std::transform(s.begin(), s.end(), s.begin(), ::toupper);
            std::string activation_method = G_RNN_LUT_SUPPORTED[std::make_pair(op_type, s)];
            scale_pair = bit8_G_RNN_SCALES_MAP[std::make_pair(activation_method, dtype)];
            activation_scales.insert(activation_scales.end(), scale_pair.begin(), scale_pair.end());
        }
    }
    else //dtype is int16
    {
        for (std::string s : activations)
        {
            std::string activation_method = G_RNN_LUT_SUPPORTED[std::make_pair(op_type, s)];
            scale_pair = bit16_G_RNN_SCALES_MAP[std::make_pair(activation_method, dtype)];
            activation_scales.insert(activation_scales.end(), scale_pair.begin(), scale_pair.end());
        }
    }
    return activation_scales;
}

void generate_activation_lut(void *&lut, uint16_t &len, std::string op_type, std::string method, aipubt::Tensor::Dtype dtype)
{
    // std::string op_type = aipubt::optype_to_string(node->type());
    std::transform(method.begin(), method.end(), method.begin(), ::toupper);
    std::string activation_method = G_RNN_LUT_SUPPORTED[std::make_pair(op_type, method)];
    if (dtype == aipubt::Tensor::INT8)
    {
        len = bit8_G_RNN_LUT_MAP[std::make_pair(activation_method, dtype)].size();
        int8_t *lut_ = new int8_t[len];
        memcpy(lut_, &(bit8_G_RNN_LUT_MAP[std::make_pair(activation_method, dtype)][0]), len * sizeof(int8_t));
        lut = (void *)lut_;
        DEBUG() << "len = " << (int32_t)len << ", lut[0:4] " << int32_t(((int8_t *)lut)[0]) << ", " << int32_t(((int8_t *)lut)[255]);
    }
    if (dtype == aipubt::Tensor::INT16)
    {
        len = bit16_G_RNN_LUT_MAP[std::make_pair(activation_method, dtype)].size() - 1; // only get the first 256 items from fixed lut
        int16_t *lut_ = new int16_t[len];
        memcpy(lut_, &(bit16_G_RNN_LUT_MAP[std::make_pair(activation_method, dtype)][0]), len * sizeof(int16_t));
        lut = (void *)lut_;
    }
}

// template <typename WT, typename BT>
// static void absorb_input_zp_to_bias(aipubt::TensorPtr weight_t, aipubt::TensorPtr bias_t, const int32_t aipu_input_zp)
// {
//     auto w_datas = weight_t->data<WT>();
//     auto b_datas = bias_t->data<BT>();
//     DEBUG() << "wshape = " << weight_t->shape()[0] << ", " << weight_t->shape()[1];

//     auto w_size = weight_t->mem_size() / sizeof(WT);
//     auto out_c_num = weight_t->shape()[1]; // 2048
//     auto w_step = out_c_num;               // 2048
//     int32_t *w_by_zp = new int32_t[out_c_num];
//     memset(w_by_zp, 0, out_c_num * sizeof(int32_t));
//     for (uint32_t w = 0; w < w_size; w++)
//     {
//         auto idx = w - (w / w_step) * w_step;
//         w_by_zp[idx] += w_datas[w] * aipu_input_zp;
//     }

//     auto b_size = bias_t->mem_size() / sizeof(BT);
//     auto b_shape = bias_t->shape();
//     if ((int32_t)b_shape[0] != out_c_num)
//     {
//         ERROR() << "weight out_c " << out_c_num << " != bias out_c " << b_size << ", please check.";
//         return;
//     }

//     auto bias_dtype = bias_t->type();
//     auto bias_qmin = dtype_utils::dtype_to_range_min(bias_dtype);
//     auto bias_qmax = dtype_utils::dtype_to_range_max(bias_dtype);
//     for (int32_t bc = 0; bc < out_c_num; bc++)
//     {
//         float tmp = w_by_zp[bc] + b_datas[bc];
//         tmp = quant_utils::clip<float, float>(tmp, bias_qmin, bias_qmax);
//         b_datas[bc] = static_cast<BT>(tmp);
//     }
//     // bias_t->set_data(b_datas, b_size * sizeof(BT));
// }

// static std::map<std::vector<aipubt::Tensor::Dtype>, std::function<void(aipubt::TensorPtr, aipubt::TensorPtr, const int32_t)>> _DTYPE_TO_FUNC = {
//     {{aipubt::Tensor::INT8, aipubt::Tensor::INT32}, absorb_input_zp_to_bias<int8_t, int32_t>},
//     {{aipubt::Tensor::UINT8, aipubt::Tensor::INT32}, absorb_input_zp_to_bias<uint8_t, int32_t>},
//     {{aipubt::Tensor::INT16, aipubt::Tensor::INT32}, absorb_input_zp_to_bias<int16_t, int32_t>},
//     {{aipubt::Tensor::INT16, aipubt::Tensor::INT64}, absorb_input_zp_to_bias<int16_t, int64_t>},
//     {{aipubt::Tensor::UINT16, aipubt::Tensor::INT64}, absorb_input_zp_to_bias<uint16_t, int64_t>},
// };

namespace RNNUtils
{

    template <typename T>
    void *_concate_bias(std::vector<aipubt::TensorPtr> bias_parts)
    {
        aipu_assert(bias_parts.size() >= 2 && "error in concate bias function.");
        aipu_assert(bias_parts[0]->type() == bias_parts[1]->type() && "data type error in bias concat.");

        size_t size = 0;
        for (auto ptr : bias_parts)
        {
            size += ptr->size();
        }
        T *new_b_mem = new T[size];

        T *start_addr = new_b_mem;
        for (size_t i = 0; i < bias_parts.size(); ++i)
        {
            memcpy(start_addr, bias_parts[i]->data<T>(), bias_parts[i]->size() * aipubt::Tensor::dtype_size(bias_parts[i]->type()));
            start_addr += bias_parts[i]->size();
        }

        return static_cast<void *>(new_b_mem);
    }

    static std::map<aipubt::Tensor::Dtype, std::function<void *(std::vector<aipubt::TensorPtr>)>>
        CONCATE_BIAS =
            {
                {aipubt::Tensor::INT32, _concate_bias<int32_t>},
                {aipubt::Tensor::UINT32, _concate_bias<uint32_t>},
                {aipubt::Tensor::INT64, _concate_bias<int64_t>},
                {aipubt::Tensor::UINT64, _concate_bias<uint64_t>},
    };

    void *concat_bias(std::vector<aipubt::TensorPtr> biases)
    {
        return CONCATE_BIAS[biases[0]->type()](biases);
    }

    // stack quantized weights.
    template <typename T>
    void *creat_mem(size_t size, std::vector<aipubt::TensorPtr> Ts)
    {
        // wx_gw_t = Ts[0]; shape=[2H, X]
        // wh_gw_t = Ts[1]; shape=[2H, H]
        // wx_cw_t = Ts[2];
        // wh_cw_t = Ts[3];
        size_t X_size = Ts[0]->shape()[1];
        size_t H_size = Ts[1]->shape()[1];

        auto p_wx_g = Ts[0]->data<T>();
        auto p_wh_g = Ts[1]->data<T>();
        auto p_wx_c = Ts[2]->data<T>();
        auto p_wh_c = Ts[3]->data<T>();

        T *m = new T[size * sizeof(T)];

        for (size_t h = 0; h < 2 * H_size; ++h)
        {
            for (size_t w = 0; w < X_size; ++w)
            {
                m[h * (X_size + H_size) + w] = p_wx_g[h * X_size + w];
            }

            for (size_t w = 0; w < H_size; ++w)
            {
                m[h * (X_size + H_size) + (w + X_size)] = p_wh_g[h * H_size + w];
            }
        }

        for (size_t h = 0; h < H_size; ++h)
        {
            size_t p_wx_c_start = 2 * H_size * (X_size + H_size);
            size_t p_wh_c_start = p_wx_c_start + X_size;
            for (size_t w = 0; w < X_size; ++w)
            {
                m[p_wx_c_start + h * (X_size + H_size) + w] = p_wx_c[h * H_size + w];
            }

            for (size_t w = 0; w < H_size; ++w)
            {
                m[p_wh_c_start + h * (X_size + H_size) + w] = p_wh_c[h * H_size + w];
            }
        }
        return static_cast<void *>(m);
    }

    static std::map<aipubt::Tensor::Dtype, std::function<void *(size_t, std::vector<aipubt::TensorPtr>)>>
        CONCAT_MEM =
            {
                {aipubt::Tensor::INT8, creat_mem<int8_t>},
                {aipubt::Tensor::UINT8, creat_mem<uint8_t>},
                {aipubt::Tensor::INT16, creat_mem<int16_t>},
                {aipubt::Tensor::UINT16, creat_mem<uint16_t>},
                {aipubt::Tensor::INT32, creat_mem<int32_t>},
                {aipubt::Tensor::UINT32, creat_mem<uint32_t>},
    };

    void *stack_gru_weights(aipubt::Tensor::Dtype dtype, std::vector<aipubt::TensorPtr> weights, size_t size)
    {
        //wx_gw_t, wh_gw_t, wx_cw_t, wh_cw_t
        return CONCAT_MEM[dtype](size, weights);
    }
    // mem copy
    template <typename T>
    static aipubt::TensorPtr m_copy(std::string name, aipubt::TensorPtr t, int64_t start, int64_t end, aipubt::TensorShape &ts)
    {
        T *new_tensor_mem = new T[end - start];
        T *ori_tensor_mem = t->data<T>();
        int8_t bsize = sizeof(T);
        uint32_t mem_size = bsize * (end - start);
        // memset(new_tensor_mem, 0, mem_size);
        // memcpy(new_tensor_mem, ori_tensor_mem + start * bsize, mem_size);

        for (size_t i = 0; i < (size_t)(end - start); ++i)
        {
            new_tensor_mem[i] = ori_tensor_mem[(size_t)start + i];
            // DEBUG() << new_tensor_mem[i] << "," << ori_tensor_mem[(size_t)start + i];
        }
        aipubt::TensorPtr new_tensor_ptr = std::make_shared<aipubt::Tensor>(name + "_copyed", new_tensor_mem, mem_size);
        new_tensor_ptr->shape() = ts;
        new_tensor_ptr->set_type(t->type());
        // DEBUG() << "new tensor " << new_tensor_ptr->name() << "," << new_tensor_ptr->shape().to_string();
        return new_tensor_ptr;
    }

    static std::map<aipubt::Tensor::Dtype, std::function<aipubt::TensorPtr(std::string, aipubt::TensorPtr, int64_t, int64_t, aipubt::TensorShape &)>>
        TENSOR_COPY =
            {
                {aipubt::Tensor::INT8, m_copy<int8_t>},
                {aipubt::Tensor::UINT8, m_copy<uint8_t>},
                {aipubt::Tensor::INT16, m_copy<int16_t>},
                {aipubt::Tensor::UINT16, m_copy<uint16_t>},
                {aipubt::Tensor::INT32, m_copy<int32_t>},
                {aipubt::Tensor::UINT32, m_copy<uint32_t>},
                {aipubt::Tensor::FP32, m_copy<float>},
                {aipubt::Tensor::FP64, m_copy<double_t>},
    };

    aipubt::TensorPtr tensor_copy(std::string name, aipubt::TensorPtr t, int64_t start, int64_t end, aipubt::TensorShape ts)
    {
        return TENSOR_COPY[t->type()](name, t, start, end, ts);
    }

    void absorb_I_H_zp_to_bias(aipubt::NodePtr node)
    {
        auto aipu_input_zp = node->inputs()[0]->quantization().offset();
        auto aipu_h_zp = node->inputs()[1]->quantization().offset();
        aipubt::TensorPtr w_t = node->constants()["weights"];
        aipubt::TensorPtr inp_t = node->inputs()[0];

        auto bk_bias_quantization = node->constants()["biases"]->quantization();
        auto bias_type = node->constants()["biases"]->type();
        auto bias_shape = node->constants()["biases"]->shape();

        // auto bias_size = w_t->shape()[1];
        auto input_size = node->params()["input_size"];
        auto cell_size = node->params()["cell_size"];

        aipubt::ComputeTensor c_w_t = aipubt::ComputeTensor(node->constants()["weights"]);
        c_w_t = c_w_t.permute({1, 0});

        auto c_w_shape1 = c_w_t->shape()[1];
        aipubt::ComputeTensor c_b_t = aipubt::ComputeTensor(node->constants()["biases"]);
        auto c_bias = c_b_t[{0, c_w_shape1}];
        aipubt::ComputeTensor c_x_w = c_w_t[{{0, input_size}}];
        aipubt::ComputeTensor c_h_w = c_w_t[{{input_size, -1}}];

        if (aipu_input_zp)
        {
            aipubt::ComputeTensor c_inp_zerop = aipubt::ComputeTensor::full({1, input_size}, aipu_input_zp);
            auto x_weights_sum = c_inp_zerop.astype(aipubt::Tensor::FP32).matmul(c_x_w.astype(aipubt::Tensor::FP32));
            aipubt::TensorPtr x_weights_sum_t = x_weights_sum;
            c_b_t = c_b_t.astype(aipubt::Tensor::FP32) + x_weights_sum;
        }

        if (aipu_h_zp)
        {
            aipubt::ComputeTensor h_inp_zerop = aipubt::ComputeTensor::full({1, cell_size}, aipu_h_zp);
            auto h_weights_sum = h_inp_zerop.astype(aipubt::Tensor::FP32).matmul(c_h_w.astype(aipubt::Tensor::FP32));
            c_b_t = c_b_t.astype(aipubt::Tensor::FP32) + h_weights_sum;
        }

        node->constants()["biases"] = c_b_t.astype(bias_type).reshape(bias_shape);
        // c_b_t has not quantization structure, so we copy it
        node->constants()["biases"]->quantization() = bk_bias_quantization;
    }

    bool is_per_channel(const std::string &tensor_mode)
    {
        std::string::size_type idx;
        idx = tensor_mode.find("per_channel");
        return !(idx == std::string::npos);
    }

    bool is_full_range(const std::string &quant_mode)
    {
        std::string::size_type idx;
        idx = quant_mode.find("full_range");
        return !(idx == std::string::npos);
    }

    bool is_asymmetric(const std::string &quant_mode)
    {
        std::string::size_type idx;
        idx = quant_mode.find("asymmetric");
        return !(idx == std::string::npos);
    }

    template <typename T>
    static void statistic_max_min(aipubt::TensorPtr tensor,
                                  std::vector<float> &statistic_max,
                                  std::vector<float> &statistic_min)
    {
        auto scales = tensor->quantization().scales();
        if (scales.size() == 1)
        {

            T min_v = 0;
            T max_v = 0;
            if (tensor->data<T>() == nullptr)
            {
                if (tensor->quantization().mins().size() > 0)
                    min_v = tensor->quantization().mins()[0];
                if (tensor->quantization().maxs().size() > 0)
                    max_v = tensor->quantization().maxs()[0];
            }
            else
            {
                min_v = tensor->min<T>();
                max_v = tensor->max<T>();
            }

            // DEBUG() << tensor->name() <<", minv and maxv:" << min_v << "," << max_v;
            auto scale = tensor->quantization().scale();
            auto zerop = tensor->quantization().offset();
            float f_min = typeid(T) == typeid(float) ? min_v : DEQUANTIZE(min_v, scale, zerop);
            float f_max = typeid(T) == typeid(float) ? max_v : DEQUANTIZE(max_v, scale, zerop);
            statistic_min.push_back(f_min);
            statistic_max.push_back(f_max);
        }
        else if (scales.size() > 1)
        {
            auto scales = tensor->quantization().scales();
            auto zerops = tensor->quantization().offsets();
            auto group = tensor->quantization().group();
            auto permuted_tensor = permute_utils::permute(tensor);
            auto datas = permuted_tensor->data<T>();
            // per-group or per-channel
            auto step = tensor->mem_size() / sizeof(T) / group;
            // DEBUG() << "statis_max_min: mem_size/sizeof(T)= " << tensor->mem_size() / sizeof(T) << ", and group = " << group << ", so step = " << step;
            T *tmp_datas = new T[step];
            for (auto i = 0; i < group; i++)
            {
                memset(tmp_datas, 0, sizeof(T) * step);
                memcpy(tmp_datas, datas + i * step, step * sizeof(T));
                T min_v = *std::min_element(tmp_datas, tmp_datas + step);
                T max_v = *std::max_element(tmp_datas, tmp_datas + step);
                float f_min = typeid(T) == typeid(float) ? min_v : DEQUANTIZE(min_v, scales[i], zerops[i]);
                float f_max = typeid(T) == typeid(float) ? max_v : DEQUANTIZE(max_v, scales[i], zerops[i]);

                statistic_min.push_back(f_min);
                statistic_max.push_back(f_max);
            }
            delete[] tmp_datas;
        }
        else
        {
            ERROR() << "unsupported granularity, we only support [Per_Tensor, Per_Channel, Per_Group]";
        }
    }

    static std::map<aipubt::Tensor::Dtype, std::function<void(aipubt::TensorPtr, std::vector<float> &, std::vector<float> &)>>
        STATISTIC_MAX_MIN = {
            {aipubt::Tensor::INT8, statistic_max_min<int8_t>},
            {aipubt::Tensor::UINT8, statistic_max_min<uint8_t>},
            {aipubt::Tensor::INT16, statistic_max_min<int16_t>},
            {aipubt::Tensor::UINT16, statistic_max_min<uint16_t>},
            {aipubt::Tensor::INT32, statistic_max_min<int32_t>},
            {aipubt::Tensor::UINT32, statistic_max_min<uint32_t>},
            {aipubt::Tensor::FP32, statistic_max_min<float>},
    };

    void statistic_max_min(aipubt::TensorPtr tensor,
                           std::vector<float> &statistic_max,
                           std::vector<float> &statistic_min)
    {
        auto dtype = tensor->type();
        STATISTIC_MAX_MIN[dtype](tensor, statistic_max, statistic_min);
    }

    void get_linear_quant_params_from_tensor(
        aipubt::TensorPtr t, std::string quant_mode, int8_t bits, bool is_signed,
        std::vector<float> &scales, std::vector<int32_t> &zeros)
    {
        aipu_assert(bits > 0 && "tensor bits should > 0;");
        float QUANTIZATION_ZERO_BAND = 1.1920928955078125e-07;

        int16_t q_max = (1 << bits) - 1;
        int16_t q_min = 0;
        if (is_signed)
        {
            if (is_full_range(quant_mode))
            {
                q_max = (1 << (bits - 1)) - 1;
                q_min = -1 * q_max - 1;
            }
            else
            {
                q_max = (1 << (bits - 1)) - 1;
                q_min = -1 * q_max;
            }
        }
        auto q_range = q_max - q_min;

        std::vector<float> f_maxs;
        std::vector<float> f_mins;
        statistic_max_min(t, f_maxs, f_mins);

        if (f_maxs.size() > 1) // per-channel
        {
            std::vector<float> f_ranges(f_maxs.size(), 0.0);
            std::vector<float> f_zfactor(f_ranges);
            std::vector<float> f_zoffset(f_ranges);
            if (is_asymmetric(quant_mode))
            {
                for (size_t i = 0; i < f_maxs.size(); ++i)
                {
                    f_ranges[i] = f_maxs[i] - f_mins[i];
                    f_zfactor[i] = f_mins[i];
                    f_zoffset[i] = q_min;
                }
            }
            else
            {
                for (size_t i = 0; i < f_maxs.size(); ++i)
                {
                    f_ranges[i] = std::max(std::abs(f_maxs[i]), std::abs(f_mins[i]));
                    f_ranges[i] = is_signed ? f_ranges[i] * 2 : f_ranges[i];
                    // f_zfactor[i] = 0.0; alreay initied.
                    // f_zoffset[i] = q_min;
                }
            }

            for (size_t i = 0; i < f_ranges.size(); ++i)
            {
                f_ranges[i] = f_ranges[i] < QUANTIZATION_ZERO_BAND ? q_range : f_ranges[i];
                scales[i] = q_range / f_ranges[i];
                zeros[i] = std::round(scales[i] * f_zfactor[i]) - f_zoffset[i];
                zeros[i] = std::min(std::max(zeros[i], 1 << (bits - 1)), -(1 << (bits - 1)));
            }
        }
        else //per-tensor
        {
            float f_range;
            float f_zfactor;
            float f_zoffset;
            if (is_asymmetric(quant_mode))
            {
                f_range = f_maxs[0] - f_mins[0];
                f_zfactor = f_mins[0];
                f_zoffset = q_min * f_zfactor;
            }
            else
            {
                f_range = std::max(std::abs(f_maxs[0]), std::abs(f_mins[0]));
                f_range = is_signed ? f_range * 2 : f_range;
                f_zfactor = 0.0;
                f_zoffset = 0.0;
            }

            f_range = f_range < QUANTIZATION_ZERO_BAND ? static_cast<float>(q_range) : f_range;
            scales.push_back(static_cast<float>(q_range) / f_range);
            zeros.push_back(std::round(scales[0] * f_zfactor) - f_zoffset);
            zeros[0] = std::min(std::max(zeros[0], 1 << (bits - 1)), -(1 << (bits - 1)));
        }
    }

    std::string q_mode_string(aipubt::TensorPtr t)
    {
        std::string q_mode_str = "";
        switch (t->quantization().granularity())
        {
        case aipubt::Quantization::Granularity::Per_Tensor:
            q_mode_str += "per_tensor";
            break;
        case aipubt::Quantization::Granularity::Per_Group:
            q_mode_str += "per_channel";
            break;
        case aipubt::Quantization::Granularity::Per_Channel:
            q_mode_str += "per_channel";
            break;
        default:
            q_mode_str += "per_tensor";
        }

        if (t->quantization().scheme() == aipubt::Quantization::Scheme::Asymmetric)
        {
            q_mode_str += "_asymmetric";
        }
        else
        {
            q_mode_str += "_symmetric";
        }

        if (t->quantization().quantized_range() == aipubt::Quantization::QuantizedRange::Restricted_Range)
        {
            q_mode_str += "_restricted_range";
        }
        else
        {
            q_mode_str += "_full_range";
        }

        return q_mode_str;
    }

    template <typename T>
    static aipubt::TensorPtr _permute2d(std::string name, aipubt::TensorPtr t)
    {
        auto t_shape = t->shape();
        int32_t _W = t_shape[1];
        int32_t _H = t_shape[0];
        t_shape[0] = _W;
        t_shape[1] = _H;
        auto t_dtype = t->type();

        aipubt::TensorPtr permuted_tensor = std::make_shared<aipubt::Tensor>(name + "_permute2d", t_shape, t_dtype);
        permuted_tensor->quantization() = aipubt::Quantization(1.0, 0);
        int dim = t->shape().dim();
        if ((uint32_t)dim != 2)
        {
            ERROR() << "please check permute perms." << dim;
            return nullptr;
        }

        uint32_t data_size = t->mem_size() / sizeof(T);
        T *constant_data = t->data<T>();
        T *permute_data = new T[data_size];
        size_t idx = 0;
        if (dim == 2)
        {
            // DEBUG() << _H << "," <<_W;
            for (int w = 0; w < _W; w++)
            {
                for (int h = 0; h < _H; h++)
                {
                    int is = h * _W + w;
                    permute_data[idx++] = constant_data[is];
                    // DEBUG() << idx-1 << "<--" << is << ", " << permute_data[idx-1] << ":" << constant_data[is];
                }
            }
        }
        else
        {
            DEBUG() << "dim = " << dim << " not supported.";
            return t;
        }

        permuted_tensor->set_data(permute_data, data_size * sizeof(T));

        return permuted_tensor;
    }

    static std::map<aipubt::Tensor::Dtype, std::function<aipubt::TensorPtr(std::string, aipubt::TensorPtr)>> PERMUTE2D_MAP = {
        {aipubt::Tensor::INT8, _permute2d<int8_t>},
        {aipubt::Tensor::UINT8, _permute2d<uint8_t>},
        {aipubt::Tensor::INT16, _permute2d<int16_t>},
        {aipubt::Tensor::UINT16, _permute2d<uint16_t>},
        {aipubt::Tensor::INT32, _permute2d<int32_t>},
        {aipubt::Tensor::UINT32, _permute2d<uint32_t>},
        {aipubt::Tensor::INT64, _permute2d<int64_t>},
        {aipubt::Tensor::UINT64, _permute2d<uint64_t>},
        {aipubt::Tensor::FP32, _permute2d<float>},
        {aipubt::Tensor::FP64, _permute2d<double>},
    };

    aipubt::TensorPtr permute2D(std::string name, aipubt::TensorPtr t)
    {
        auto dtype = t->type();
        return PERMUTE2D_MAP[dtype](name, t);
    }
}; // namespace RNNUtils

}; // namespace qtlibn
