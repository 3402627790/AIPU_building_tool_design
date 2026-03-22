
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
 * @Author: ALex Zheng
 * @Date: 2021-07-09 14:38:57
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-07-10 17:23:10
 */
#ifndef CORE_OPS_OPS_UTILS_H
#define CORE_OPS_OPS_UTILS_H

#include "ops.h"
#include <core/quantization.h>
#include <core/tensor.h>

namespace aipubt
{

namespace ops
{
    using NodeParamDict = std::map<std::string, NodeParamValue>;
    /**
     * @brief extract valid stride from strides.
     * @param[in] strides An unsiged int or list of `unsiged ints` that has length `1`, `3` or `5`. The
     *                    stride of the sliding window for each dimension of the input tensor.
     * @param[in] valid_strides Must have the correct size.
     * @return  0: Success,other:Failed.
     */
    int extract_valid_stride(std::vector<int32_t> strides,
                             std::vector<int32_t> &valid_strides);

    void cal_explicit_padding(const int32_t &in_size,
                              const int32_t &stride,
                              const int32_t &filter_size,
                              const Padding &padding,
                              int32_t &padding_head,
                              int32_t &padding_tail,
                              const int32_t &dilation_factor = 1,
                              const bool &is_transpose_conv = false);

    void cal_explicit_padding(const int32_t &in_size,
                              const int32_t &stride,
                              const int32_t &filter_size,
                              const std::string padding,
                              int32_t &padding_head,
                              int32_t &padding_tail,
                              const int32_t &dilation_factor = 1,
                              const bool &is_transpose_conv = false);

    inline int32_t compute_out_size(int32_t image_size,
                                    int32_t filter_size,
                                    int32_t stride,
                                    int32_t padding_head,
                                    int32_t padding_tail)
    {
        return (image_size - filter_size + stride + padding_head + padding_tail) / stride;
    }

    inline int32_t compute_out_size(int32_t image_size,
                                    int32_t filter_size,
                                    int32_t stride,
                                    int32_t dilation_rate,
                                    int32_t padding_head,
                                    int32_t padding_tail,
                                    bool ceil_mode = false)
    {
        int32_t output = 0;
        if (ceil_mode)
        {
            double sum = image_size - filter_size * dilation_rate + dilation_rate + padding_head + padding_tail - 1;
            output = ::ceil(sum / stride) + 1;
            if ((output - 1) * stride >= (image_size + padding_head + padding_tail))
            {
                --output;
            }
        }
        else
        {
            output = (image_size - filter_size * dilation_rate + dilation_rate + padding_head + padding_tail - 1) / stride + 1;
        }
        return output;
    }

    /**
     * @brief Through insert reshape and transpose, move the target axis to last axis.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] orig_axis An int32 scalar, specifying the target axis.
     * @return A "TensorPtr", specifying the transposed "TensorPtr".
     */
    TensorPtr transpose_to_last_axis(TensorPtr input, int32_t orig_axis);

    /**
     * @brief Through insert reshape and transpose, move the last axis back to original axis.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] orig_shape A "TensorShape" specifying the original "TensorShape".
     * @param[in] orig_axis An int32 scalar, specifying the original axis.
     * @return A "TensorPtr", specifying the transpose back "TensorPtr".
     */
    TensorPtr transpose_last_axis_back(TensorPtr input, TensorShape orig_shape, int32_t orig_axis);

    /**
     * @brief Convert axis to positive, if it exceeds [-rank, rank) of input, it will return false.
     * @param[in,out] axis Origin axis in, convert to positive out.
     * @param[in] rank A scalar int32, specifying the input tensor's rank.
     * @param[in] op_name Print operator name on failure.
     * @return Axis is valid, return true, otherwise, return false.
     */
    bool axis_to_positive(int32_t &axis, int32_t rank, std::string op_name = "");
    /**
     * @brief Check whether input rank is the expected rank.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] expect_rank Expect rank of input.
     * @return Rank is valid, return true, otherwise, return false.
     */
    bool check_rank(TensorPtr input, int32_t expect_rank);

    /**
     * @brief Set default layout according rank size
     * @param[in, out] shape A "TensorShape":
     *                       Rank = 4, NHWC
     *                       Rank = 5, NDHWC
     * @return None.
     */
    void set_default_layout(TensorShape &shape);

    /**
     * @brief Check inputs whether support unibroadcast.
     * @param[in] input1_dim An int32 vector, specifying the first dimension of input1.
     * @param[in] input2_dim An int32 vector, specifying the second dimension of input2.
     * @return true, support unibroadcast.
     *         false, not support unibroadcast.
     */
    bool is_unibroadcastable(const std::vector<int32_t> &input1_dim,
                             const std::vector<int32_t> &input2_dim);

    /**
     * @brief This API will make inputs support multibroadcast through insert tile and reshape.
     * @param[in] inputs A vector of "TensorPtr", specifying the "TensorPtr" whose "TensorShape" needs to do multibroadcast.
     * @return A vector of "TensorPtr", if @p inputs shape doesn't support multibroadcast,
     *         it will return an empty vector. Otherwise, "TensorPtr" will include multibroadcasted
     *         "TensorShape". Vector size is the same with inputs vector size.
     */
    std::vector<TensorPtr> do_multibroadcast(std::vector<TensorPtr> inputs);

    /**
     * @brief Reshape dimensions to 4-D dimension reperents by "TensorShpae"
     *        1.Doesn't support rank > 4.
     *        2.Insert dimension 1 at high dimension, for example, [3,4]->[1,1,3,4].
     * @param[in] orig_shape A "TensorShape" object, which dimension will be reshaped.
     * @return A "TensorShape", which dimension has be reshaped.
     */
    TensorShape reshape_to_4d(const TensorShape &orig_shape);

    /**
     * @brief Reshape dimensions to 4-D dimension
     *        1.Doesn't support rank > 4.
     *        2.Insert dimension 1 at high dimension, for example, [3,4]->[1,1,3,4].
     * @param[in] method An int32 vector, specifying the dimension needs to reshape.
     * @return dimensions of 4-D.
     */
    std::vector<int32_t> reshape_to_4d(const std::vector<int32_t> &orig_dims);

    /**
     * @brief Convert enumerate object to string
     * @param[in] map_table A map table containing enumerate object and coresponding string
     * @param[in] enum_value An enumerate object of class T
     * @return string format of @p enum_value
     */
    template <class T>
    std::string enum_to_string(std::map<T, std::string> &map_table, T enum_value)
    {
        if (map_table.count(enum_value))
            return map_table[enum_value];
        return {};
    }

    /**
     * @brief Convert enumerate object to string
     * @param[in] flattendim An enumerate object of FlattenDim.
     * @return string format of @p flattendim
     */
    std::string flattendim_to_string(FlattenDim flattendim);

    /**
     * @brief Convert enumerate object to string
     * @param[in] padding An enumerate object of Padding.
     * @return string format of @p padding
     */
    std::string padding_to_string(Padding padding);

    /**
     * @brief Convert enumerate object to string
     * @param[in] method An enumerate object of ReduceMethod.
     * @return String format of @p method
     */
    std::string method_to_string(ReduceMethod method);

    /**
     * @brief convert enumerate object to string
     * @param[in] method An enumerate object of ResizeMethod.
     * @return String format of @p method
     */
    std::string method_to_string(ResizeMethod method);

    /**
     * @brief Convert enumerate object to string
     * @param[in] method An enumerate object of TopkSelectMethod.
     * @return String format of @p method
     */
    std::string method_to_string(TopkSelectMethod method);

    /**
     * @brief Convert enumerate to string
     * @param[in] mode An enumerate object of ResizeMode.
     * @return String format of @p mode
     */
    std::string mode_to_string(ResizeMode mode);

    /**
     * @brief Convert enumerate to string
     * @param[in] mode An enumerate object of ResizeNearestMode.
     * @return String format of @p mode
     */
    std::string mode_to_string(ResizeNearestMode nearest_mode);

    /**
     * @brief Convert enumerate to string
     * @param[in] mode An enumerate object of ClipMode.
     * @return String format of @p mode
     */
    std::string mode_to_string(ClipMode mode);

    /**
     * @brief Convert enumerate to string
     * @param[in] format An enumerate object of ImageFormat.
     * @return String format of @p format
     */
    std::string image_format_to_string(ImageFormat format);

    /**
     * @brief Convert Quantize round mode to string
     * @param[in] round_mode An round mode object of Quantize.
     * @return String format of @p format
     */
    std::string quantize_round_mode_to_stirng(QuantizeRoundMode round_mode);

    /**
     * @brief Convert ScatterReducation enumerate to string.
     * @param[in] reduction An enumerate object of ScatterReduction.
     * @return IR spec defines string format of @p reduction.
     */
    std::string to_string(ScatterReduction reduction);

    /**
     * @brief According @p input and @p quant information to initialize @p output quantization.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] output A "TensorPtr", specifying the output tensor.
     * @param[in] quant A "Quantization", specifying the output tensor's quantization paramters.
     * @param[in] need_set_type A bool scalar, if true, output tensor type will be set same with input tensor type.
     * @return None
     */
    void add_tensor_info(const TensorPtr input, TensorPtr output, const Quantization &quant, bool need_set_type = true);

    /**
     * @brief According @p input and @p quant information to initialize @p output quantization.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] output A "TensorPtr", specifying the output tensor.
     * @param[in] need_set_type A bool scalar, if true, output tensor type will be set same with input tensor type.
     * @param[in] inherit_aipu_dtype A bool scalar, if false, ops will clear aipu_dtype, and move it to qtlib.
     * @return None
     */
    void add_tensor_info(const TensorPtr input, TensorPtr output, bool need_set_type = true, bool inherit_aipu_dtype = true);

    void append_activation(NodePtr n, Activation act);

    /**
     * @brief for BasicLSTM/GRUv1/GRUv3
     *        Store parameters such as alpha/beta into the NodeParamValue object in the order of the formula.
     * @param[in] param Parameter storage object.
     * @param[in] act Activation object.
     */
    void append_activation_params(NodePtr n, NodeParamValue &param, Activation act);

    /**
     * @brief Quantize for node, it will call qtlib to quantize for current node.
     *        If we try to call qtlib, all tensors must have quantization params(except int32 tensor), because it will impact following layers.
     *        If int32 tensor has no quantization params, before call qtlib, we will set it default to (1.0, 0).
     * @param[in] node A "NodePtr", specifying the node to quantize.
     * @param[in] output_index A int32 scalar, specifying output index, whose scheme() and bits() will be quantize paramemters.
     */
    bool quantize(NodePtr node, int32_t output_index = 0);

#define AXIS_CHECK(axis, input)                                      \
    if (!axis_to_positive(axis, input->shape().dim(), __FUNCTION__)) \
    {                                                                \
        return {};                                                   \
    }

    GraphPtr get_root_graph(NodePtr n);
} // namespace ops
} // namespace aipubt

#endif //! CORE_OPS_OPS_UTILS_H
