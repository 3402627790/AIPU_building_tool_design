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
 * @Author: Neo Wang
 * @Date: 2021-04-14 11:00:45
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-02-14 18:13:37
 */
#ifndef CORE_OPS_OPS_H
#define CORE_OPS_OPS_H

#include <core/quantization.h>
#include <core/tensor.h>

namespace aipubt
{

namespace ops
{
    enum class ClipMode : int32_t
    {
        Saturation = 0x00,
        Truncation,
    };

    enum class FlattenDim : int32_t
    {
        HW = 0x0,
        HWC,
        NHWC,
        NCHW,
    };

    enum class GridSampleMethod : int32_t
    {
        Bilinear = 0x0,
        Nearest,
        Bicubic,
        Linear,
    };

    enum class GridSamplePaddingMode : int32_t
    {
        Zeros = 0x0,
        Border,
        Reflection,
    };

    enum class GruDirection : int32_t
    {
        Forward = 0x0,
        Reverse,
    };

    enum class GruOutSequence : int32_t
    {
        H = 0x0,
        Hn,
        H_Hn,
    };

    enum class ImageConversion : int32_t
    {
        BT601 = 0x0,
        BT709,
        BT2020,
        SELF,
    };

    enum class ImageFormat : int32_t
    {
        I420 = 0x0,
        YV12,
        NV12,
        NV21,
    };

    enum class LrnMethod : int32_t
    {
        AcrossChannels = 0x0,
        WithinChannel,
    };

    enum class LSTMOutSequence : int32_t
    {
        Y = 0x0,
        H,
        C,
        Y_H,
        Y_C,
        H_C,
        Y_H_C,
    };

    enum class NmsMethod : int32_t
    {
        Hard = 0x0,
        Linear,
        Gaussian,
    };

    enum class PadMode : int32_t
    {
        Constant = 0x0,
        Reflect,
        Symmetric,
        Edge,
        Wrap,
    };

    enum class ReduceMethod : int32_t
    {
        All = 0x0,
        Any,
        Max,
        Min,
        Mean,
        Sum,
        L1,
        L2,
        Variance,
        UnbiasedVariance,
        Prod,
    };

    enum class ResizeMethod : int32_t
    {
        Bilinear = (int32_t)GridSampleMethod::Bilinear,
        NearestNeighbor,
    };

    enum class ResizeMode : int32_t
    {
        HalfPixel = 0x0,
        HalfPixelSymmetric,
        AlignCorners,
        Asymmetric,
        PytorchHalfPixel,
        TfHalfPixelForNN,
    };

    enum class ResizeNearestMode : int32_t
    {
        RoundPreferFloor = 0x00,
        RoundPreferCeil,
        Floor,
        Ceil,
        Simple,
    };

    enum class CTMode : int32_t
    {
        HalfPixel = (int32_t)ResizeMode::HalfPixel,
        OutputHalfPixel,
    };

    enum class RoiAlignMode : int32_t
    {
        Avg = 0x0,
        Max,
    };

    enum class ScatterReduction : int32_t
    {
        None = 0x0,
        Add,
        Mul,
        Min,
        Max,
    };

    enum class SegmentReduceMethod : int32_t
    {
        Max = (int32_t)ReduceMethod::Max,
        Mean = (int32_t)ReduceMethod::Mean,
        Min = (int32_t)ReduceMethod::Min,
        Sum = (int32_t)ReduceMethod::Sum,
        Prod,
    };

    enum class ShiftDirection : int32_t
    {
        LEFT = 0x0,
        RIGHT,
    };

    enum class BitwiseMethod : int32_t
    {
        AND = 0x0,
        NOT,
        OR,
        XOR,
    };

    enum class TopkSelectMethod : int32_t
    {
        First = 0x0,
        Last,
        Random,
    };

    enum class CumulateMethod : int32_t
    {
        SUM = 0x0,
        PROD,
    };

    enum class DepthToSpaceMode : int32_t
    {
        DCR = 0x0,
        CRD,
    };

    enum class DivmodMethod : int32_t
    {
        TRUNC = 0x0,
        FLOOR,
    };
    enum class EmbeddingLookupSparseCombiner : int32_t
    {
        MEAN = 0x0,
        SUM,
        SQRTN,
    };

    enum class PoolMethod : int32_t
    {
        Avg = 0x0,
        Max,
    };

    enum class QuantizeRoundMode : int32_t
    {
        RoundToEven = 0x0,
        Ceil,
        Floor,
        Trunc,
    };

    struct Activation
    {
        Activation() {}
        Activation(const std::string &t) : type(t)
        {
            std::transform(type.begin(), type.end(), type.begin(), ::toupper);
        }
        Activation(const char *t) : type(t)
        {
            std::transform(type.begin(), type.end(), type.begin(), ::toupper);
        }

        Activation(const std::string &t, float p0) : type(t)
        {
            std::transform(type.begin(), type.end(), type.begin(), ::toupper);
            params_f.push_back(p0);
        }
        Activation(const std::string &t, int32_t p0) : type(t), params_quantized(true)
        {
            std::transform(type.begin(), type.end(), type.begin(), ::toupper);
            params_i.push_back(p0);
        }

        Activation(const std::string &t, float p0, float p1) : type(t)
        {
            std::transform(type.begin(), type.end(), type.begin(), ::toupper);
            params_f.push_back(p0);
            params_f.push_back(p1);
        }
        Activation(const std::string &t, int32_t p0, int32_t p1) : type(t), params_quantized(true)
        {
            std::transform(type.begin(), type.end(), type.begin(), ::toupper);
            params_i.push_back(p0);
            params_i.push_back(p1);
        }

        Activation(const std::string &t, TensorPtr p0) : type(t)
        {
            std::transform(type.begin(), type.end(), type.begin(), ::toupper);
            if (p0)
                tensors.push_back(p0);
        }

        std::string type = "NONE";
        std::vector<float> params_f;
        std::vector<int32_t> params_i;
        std::vector<aipubt::TensorPtr> tensors;
        bool params_quantized = false; // private? we allow user to modify it in special situation.
    };

    struct Padding
    {
        Padding() {}
        Padding(std::vector<int32_t> v) : values(v) {}
        Padding(const std::string &s) : name(s) { transform(name.begin(), name.end(), name.begin(), ::toupper); }
        Padding(const char *s) : name(s) { transform(name.begin(), name.end(), name.begin(), ::toupper); }
        Padding(const Padding &other)
        {
            name = other.name;
            values = other.values;
        }
        Padding(int32_t l)
        {
            values.push_back(l);
        }
        Padding(std::initializer_list<int32_t> l)
        {
            for (auto &&i : l)
            {
                values.push_back(i);
            }
        }
        size_t size()
        {
            return values.size();
        }
        Padding &operator=(const Padding &other)
        {
            name = other.name;
            values = other.values;
            return *this;
        }
        int32_t &operator[](int i)
        {
            aipu_assert(i < static_cast<int>(values.size()));
            return values[i];
        }

        operator std::string() const
        {
            return name;
        }
        operator std::vector<int32_t>() const
        {
            return values;
        }
        bool operator==(const Padding &other)
        {
            return ((name == other.name) && (values == other.values));
        }
        bool operator==(const char *s)
        {
            return (name == std::string(s));
        }
        bool operator==(const std::string &str)
        {
            return (name == str);
        }
        std::string name = "VALID";
        std::vector<int32_t> values;
    };

    struct SpatialScaleQuant
    {
        SpatialScaleQuant() {}
        SpatialScaleQuant(std::vector<int32_t> scale_value, std::vector<Tensor::Dtype> scale_type,
                          int32_t shift_value, Tensor::Dtype shift_type)
            : spatial_scale_value(scale_value),
              spatial_scale_type(scale_type),
              spatial_shift_value(shift_value),
              spatial_shift_type(shift_type) {}

        SpatialScaleQuant &operator=(const SpatialScaleQuant &other)
        {
            spatial_scale_value = other.spatial_scale_value;
            spatial_scale_type = other.spatial_scale_type;
            spatial_shift_value = other.spatial_shift_value;
            spatial_shift_type = other.spatial_shift_type;
            return *this;
        }
        std::vector<int32_t> spatial_scale_value;
        std::vector<Tensor::Dtype> spatial_scale_type;
        int32_t spatial_shift_value;
        Tensor::Dtype spatial_shift_type;
    };

    /**
     * @brief Computes the absolute value of a tensor.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A Quantization object ,including scale and offset.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr abs(TensorPtr input,
                  const Quantization &quant = {});

    /**
     * @brief Computes the position ids in sampled_candidates matching true_classes.
     * @param[in] true_classes A "TensorPtr", specifying the input tensor with shape [batch_size, num_true].
     * @param[in] sampled_candidates A "TensorPtr", means the candidates matched by true_classes with a shape [num_sampled].
     * @return A "TensorPtr" vector of size is 3.
     *         First "TensorPtr" records the row number indices of candidates data in true_classes, its rank is 1,
     *         dimension length should be less than @p true_classes shape size.
     *         Second "TensorPtr" records the indices of candidates data in sampled_candidates, its shape should be same with firt return "TensorPtr".
     *         Third "TensorPtr" records the effective number in above two tensors, its shape is [1].
     */
    std::vector<TensorPtr> accidental_hits(TensorPtr true_classes, TensorPtr sampled_candidates);

    /**
     * @brief Calculates the arccosine (inverse of cosine) of the given input tensor, element-wise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr acos(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Calculates the hyperbolic arccosine of the given input tensor element-wise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr acosh(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Applies a 2D adaptive average pooling over an input signal composed of several input planes.
              The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] output_size  the target output size of the image of the form H x W.
     *                  Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int,
     *                  or None which means the size will be the same as that of the input.
     * @return A "TensorPtr", specifying the output tensor.
    */
    TensorPtr adaptive_avg_pool(TensorPtr input, const std::vector<int32_t> &output_size);

    /**
     * @brief Applies a 2D adaptive max pooling over an input signal composed of several input planes.
              The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] output_size  the target output size of the image of the form H x W.
     *                  Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int,
     *                  or None which means the size will be the same as that of the input.
     * @return A "TensorPtr", specifying the output tensor.
    */
    TensorPtr adaptive_max_pool(TensorPtr input, const std::vector<int32_t> &output_size);

    /**
     * @brief Applies a 2D adaptive pooling over an input signal composed of several input planes.
              The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] output_size  the target output size of the image of the form H x W.
     *                  Can be a tuple (H, W) or a single H for a square image H x H. H and W can be either a int,
     *                  or None which means the size will be the same as that of the input.
     * @param[in] method The method of pooling.
     *                   AVG/MAX.
     * @return A "TensorPtr", specifying the output tensor.
    */
    TensorPtr adaptive_pool(TensorPtr input, const std::vector<int32_t> &output_size, const PoolMethod method = PoolMethod::Avg);

    /**
     * @brief Performs elementwise addition of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr add(TensorPtr input1,
                  TensorPtr input2,
                  const Activation &with_activation = {},
                  const Quantization &quant = {});

    /**
     * @brief Generate 2D or 3D flow field (sampling grid), given a batch of affine matrices theta.
     * @param[in] theta input batch of affine matrices with shape (Nx2x3) for 2D or (Nx3x4) for 3D.
     * @param[in] size  the target output image size. (NxHxWxC for 2D or NxCxDxHxW for 3D).
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.The output tensor of size is (NxHxWx2).
     */
    TensorPtr affine_grid(TensorPtr theta,
                          TensorPtr size,
                          const bool align_corners = false,
                          const Quantization &quant = {});

    /**
     * @brief Computes the indices of the max elements of the input tensor's element along the provided axis.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] axis An int32 scalar, specifying the axis to compute arg indices on. Default -1.
     * @param[in] keep_dim A boolean scalar, if keep_dim is true, resulting tensor has the same rank as the input.
     *                     If keep_dim is false, then the resulting tensor have the reduced dimension pruned. Default true.
     * @param[in] select_last_index A boolean scalar, if select_last_index is True (default false),
     *                              the index of the last occurrence of the max is selected if the max appears more than once in the input.
     *                              Otherwise the index of the first occurrence is selected.
     * @return A "TensorPtr", which has the same data type and scale/offset with @p input.
     *         Output tensor shape will be inffered by @p keep_dim.
     */
    TensorPtr argmax(TensorPtr input,
                     int32_t axis = -1,
                     bool keep_dim = true,
                     bool select_last_index = false);

    /**
     * @brief Computes the indices of the min elements of the input tensor's element along the provided axis.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] axis An int32 scalar, specifying the axis to compute arg indices on. Default -1.
     * @param[in] keep_dim A boolean scalar, if keep_dim is true, resulting tensor has the same rank as the input.
     *                     If keep_dim is false, then the resulting tensor have the reduced dimension pruned. Default true.
     * @param[in] select_last_index A boolean scalar, if select_last_index is True (default false),
     *                              the index of the last occurrence of the min is selected if the min appears more than once in the input.
     *                              Otherwise the index of the first occurrence is selected.
     * @return A "TensorPtr", which has the same data type and scale/offset with @p input.
     *         Output tensor shape will be inffered by @p keep_dim.
     */
    TensorPtr argmin(TensorPtr input,
                     int32_t axis = -1,
                     bool keep_dim = true,
                     bool select_last_index = false);

    /**
     * @brief Calculates the arcsine (inverse of sine) of the given input tensor, element-wise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr asin(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Calculates the hyperbolic arcsine of the given input tensor element-wise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr asinh(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Calculates the arc tangent of the given input tensor, element-wise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr atan(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Calculates the hyperbolic arctangent of the given input tensor element-wise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr atanh(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Avg_pool takes an input tensor and applies average pooling across the tensor by the kernel size,
     *        stride size, and padding. Avg_pool consists of computing the average on all values of
     *        a subset of the input tensor according to the kernel size and down sampling the data into
     *        the output tensor for further processing. If no padding is present, the 'pad' group parameters default to zero.
     * @param[in] input  A "TensorPtr".
     *                   Support the following shape:
     *                   [batch,height,channel],
     *                   [batch,height,width,channel],
     *                   [batch,depth,height,width,channel].
     * @param[in] kernel  The size of the kernel along each axis. The 'kernel' parameter format should be as follows:
     *                    1d:[kernel_x],
     *                    2d:[kernel_y,kernel_x],
     *                    3d:[kernel_z,kernel_y,kernel_x].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    1d:[stride_x],
     *                    2d:[stride_y,stride_x],
     *                    3d:[stride_z,stride_y,stride_x].
     * @param[in] dilations Dilation of filter along each spatial dimension. If not present, defaults to 1 along each spatial dimensions.
     *                      The 'dilations' parameter format should be as follows:
     *                      1d:[dilation_x],
     *                      2d:[dilation_y,dilation_x],
     *                      3d:[dilatioin_z,dilation_y,dilation_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    1d:[pad_left, pad_right],
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right],
     *                    3d:[pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end].
     * @param[in] ceil_mode Means using Ceil or Floor function to compute the output shape.
     *                      Generally, the default value is `false` if not specified, which means using floor function.
     * @param[in] count_include_pad Means whether include pad pixels when calculating values for the edges.
     *                              `false` doesn't count include pad; while `true` need count the paddings.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs.
     */
    TensorPtr avg_pool(TensorPtr input,
                       const std::vector<int32_t> &kernel,
                       const std::vector<int32_t> &strides,
                       const std::vector<int32_t> &dilations,
                       Padding padding = Padding("VALID"),
                       bool ceil_mode = false,
                       bool count_include_pad = false,
                       const Quantization &quant = {});

    /** @brief Performs a single time step in a Long Short-Term Memory (LSTM) layer.
     *
     * @param[in] input             A "TensorPtr", which is a 3D tensor with shape [batch_size, time_step, input_size],
     *                              or a 2D tensor with shape [batch_size, input_size](time_step defaults to 1).
     * @param[in] hidden_state     A "TensorPtr", which means the initial hidden state for each element
     *                             in the batch with the shape as [batch size, hidden size].
     *                              Data type same as @p input.
     * @param[in] cell_gate        A "TensorPtr", which means the initial cell gate with the shape as [batch_size, hidden_size].
     *                              Data type same as @p input.
     * @param[in] weights           A "TensorPtr".The weight tensor that will be used in the computation and has weights_type,
     *                              weights_offset, weights_size and weights_shape which mean the weights data format,
     *                              weights address offset, total weights size (counted in bytes) and weights shape of
     *                              the current layer. The corresponding equation is, 'weights_size = data_type * h * (x + h)'
     *                              and 'weights_shape = 4 * h, x + h'. Where, data type = 1 (or 2, 4) if weight_type = int8 (or int16, int32).
     *                              Note that 'wight_shape' is stacked in order of [[Wi0 , Ri0], [Wc0 , Rc0], [Wf0 , Rf0], [Wo0 , Ro0]],
     *                              ..., [[Wi(h-1) , Ri(h-1]], [Wc(h-1) , Rc(h-1)], [Wf(h-1] , Rf(h-1)], [Wo(h-1) , Ro(h-1)]]
     *                              (that is, weights_shape = [4*h, x+h]). Where, 'x' and 'h' mean the input size and hidden size
     *                              as described in the preceding section.
     * @param[in] biases            A "TensorPtr", which is an 1D weights tensor with concatenation of `B[icfo]`,
     *                              and shape is [4 * output_size].
     * @param[in] activations       A list of activation functions for input (or output), cell and output gates in order of (f, g, h) in BasicLSTM.
     *                              The activation must be one of the activation functions defined in the preceding section.
     *                              - Activation("RELU")
     *                              - Activation("TANH")
     *                              - Activation("SIGMOID")
     *                              - Activation("LEAKYRELU", negative_slope_value)
     *                              - Activation("THRESHOLDEDRELU", alpha)
     *                              - Activation("HARDSIGHMOID", alpha, beta)
     *                              - Activation("ELU", alpha)
     *                              - Activation("SOFTSIGN")
     *                              - Activation("SOFTPLUS")
     *                              Quantization parameter of "TensorPtr" should have 4 pair of scale/offset.
     * @param[in] cell_quant        Optional "Quantization" object, specifying scale and offset for cell_state_in.
     *                              If not set, used default.
     * @param[in] out_quant         Optional "Quantization" object, specifying scale and offset for output_state_in.
     *                              If not set, used default.
     * @param[in] out_sequence      The output sequence is one of enumerated lists and in a fixed order. Default
     *                              sequence is [Y,H,C].
     *                              Here, enumerated list includes [Y], [H], [C], [Y,H], [Y,C], [H,C], [Y,H,C].
     *                              Y concats the intermediate output values of the hidden state with shape
     *                              [batch_size, seq_length, output_size]. H means the last output value of the
     *                              hidden state with shape [batch_size, output_size]. C means the last output
     *                              value of the LSTM cell state with shape [batch_size, output_size].
     * @return vector of "TensorPtr"
     *         First "TensorPtr", specifying hidden output state of cell. It is a 2D tensor with shape [output_size, batch_size].
     *         Data type will be same as @p cell_state_in. Scale and offset can be different from @p cell_state_in.
     *         Second "TensorPtr", specifying hidden output state of output. It is a 2D tensor with shape [output_size, batch_size].
     *         Data type will be same as @p output_state_in. Scale and ofset can be different from @p output_state_in.
     */
    std::vector<TensorPtr> basic_lstm(TensorPtr input,
                                      TensorPtr hidden_state,
                                      TensorPtr cell_gate,
                                      TensorPtr weights,
                                      TensorPtr biases,
                                      std::vector<Activation> activations = {Activation("SIGMOID"), Activation("TANH"), Activation("TANH")},
                                      const Quantization &cell_quant = {},
                                      const Quantization &out_quant = {},
                                      const LSTMOutSequence &out_sequence = LSTMOutSequence::Y_H_C);

    /**
     * @brief Computes the batch normalization according to the model's running.
     *        This op works for inferred mode instead of training mode, so input tensors are constant tensor except for data tensor X.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] weights A "TensorPtr", specifying the weight tensor, which is a constant tensor.
     *                  Generally, the shape of weight is an 1-D shape equals to the @p input[axis] dimension.
     *                  weight = (gamma / sqrt(var + epsilon)), the 'epsilon' is a parameter during float computation.
     * @param[in] biases A "TensorPtr", specifying the bias tensor, which is a constant tensor.
     *                 Generally, the shape of bias is an 1-D shape equals to the @p input[axis] dimension.
     *                 bias = beta - mean * (gamma / sqrt(var + epsilon), the 'epsilon' is a parameter during float computation.
     * @param[in] axis  An int32 scalar,  indicating that the statistics are continuously computed for the specified dimension over
     *                  N in [N, H, W, C] or [N, D1, D2, Di, ..., C] data format.
     *                  For example, axis = 3 means that statistics are computed for every channel of C over N, H and W dimensions in NHWC format.
     *                  That is, all 'N', 'H', 'W' dimensions share the same mean and variance during batch normalization when axis = 3.
     * @param[in] epsilon A float IR parameter. The epsilon value is used to avoid division by zero.
     *                    Generally, the default value is less than 10-5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input. Data type is inferred by @p quant.
     *         Output scale and offset are described in @p quant.
     */
    TensorPtr batch_normalization(TensorPtr input,
                                  TensorPtr weights,
                                  TensorPtr biases,
                                  int32_t axis = -1,
                                  float epsilon = 1e-5f,
                                  const Quantization &quant = {});

    /**
     * @brief This operation reshapes the batch dimension (dimension 0) into M + 1 dimensions of shape block_shape + [batch],
     *        interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., M],
     *        to obtain a result with the same rank as the input.
     * @param[in] input A "TensorPtr", specifying the input tensor, rank must be 4.
     * @param[in] block_size An uint32 vector, size should be 2. The order is height, width.
     * @param[in] crops A vector of uint32 pair, vector size should be same as M(2). The order is height, width.
     * @return A "TensorPtr", which has same data type and scale/offset with @p input. Output shape can be inferred by
     *         @p block_size and @p crops.
     */
    TensorPtr batch_to_space(TensorPtr input,
                             const std::vector<uint32_t> &block_size,
                             const std::vector<std::pair<uint32_t, uint32_t>> &crops);

    /**
     * @brief This operation reshapes the "batch" dimension 0 into M + 1 dimensions of
     *        shape block_shape + [batch], interleaves these blocks back into the grid
     *        defined by the spatial dimensions [1, ..., M],
     *        to obtain a result with the same rank as the input.
     *        The spatial dimensions of this intermediate result
     *        are then optionally cropped according to crops to produce the output.
     *        This is the reverse of SpaceToBatch.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] block_size It is a list that means that blocks are moved,
     *            where block_size is the block size along with the excepte batch and
     *            c dimension in data format.
     *            Generally, the output tensor will be
     *            [N/prod(block_size), shape * block_size - crop_top - crop_bottom,...,C].
     * @param[in] crops It is a list that means the outputs tensor after cropping
     *                  the dimension with the size of crop_top and crop_bottom.
     * @return A "TensorPtr", which has same data type and scale/offset with @p input. Output shape can be inferred by
     *         @p block_size and @p crops.
     */
    TensorPtr batch_to_space_nd(TensorPtr input,
                                const std::vector<int32_t> &block_size,
                                const std::vector<std::vector<int32_t>> &crops);

    /**
     * @brief Adds "bias to input.
     * @param[in] input A "TensorPtr".
     * @param[in] bias  A 1-D `Tensor` with size matching the channel dimension of "input",and with the
     *                  type of "input".
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr bias_add(TensorPtr input,
                       TensorPtr bias,
                       const Quantization &quant = {});

    /**
     * @brief Returns the tensor resulting from performing the bitwise and operation elementwise on the input tensors.
     *        This operator supports multidirectional (i.e., Numpy-style) broadcasting..
     * @param[in] inputs A Tensor pointer list.
                         When the method is "AND", "OR", "XOR", it contains tensor A and tensor B.
                         When method is "NOT", it contains tensor A.
     * @param[in] method Bitwise operation method, it can be:
     *                   - BitwiseMethod.AND : returns the element-wise truth value of 'x AND y'.
     *                   - BitwiseMethod.NOT : returns the element-wise value of 'NOT x'.
     *                   - BitwiseMethod.OR : returns the element-wise truth value of 'x OR y'.
     *                   - BitwiseMethod.XOR : returns the element-wise truth value of 'x XOR y'.
     * @param[in] quant A "Quantization" object, specifying the output quantization paramemters.
     * @return A "TensorPtr", with the same shape as the tensors in "input1".
    */
    TensorPtr bitwise(std::vector<TensorPtr> inputs,
                      BitwiseMethod method,
                      const Quantization &quant);

    /**
     * @brief Bitwise shift operator performs element-wise operation.
     * @param[in] input A "TensorPtr", specifying the tensor needs to be shifted.
     * @param[in] shift A "TensorPtr", specifying the shift tensor.
     * @param[in] direction Specifying the direction of shift, it can be:
     *                      - ShiftDirection.Left
     *                      - ShiftDirection.Right
     * @param[in] quant A "Quantization" object, specifying the output quantization paramemters.
     * @return A "TensorPtr", specifying shifted tensor.
     */
    TensorPtr bit_shift(TensorPtr input,
                        TensorPtr shift,
                        ShiftDirection direction,
                        const Quantization &quant = {});

    /**
     * @brief BNLL(binomial normal log likelihood)
     *        Takes one input tensor and produces one output tensor,
     *        where the sigmoid function 'y = x + log(1 + exp(-x))' if x > 0,
     *        otherwise y = 'log(1 + exp(x))', is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output quantization paramemters.
     * @return A "TensorPtr", which has same shape and scale/offset with @p input.
     */
    TensorPtr bnll(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Boundingbox regression.
     *        Note that, in the following Inputs or Outputs sections:
     *        X1 means the input boxes.
     *        X2 means the delta of box.
     *        Y1 means the output box.
     * @param[in] input1 A "TensorPtr", specifying the input tensor(X1).
     * @param[in] input2 A "TensorPtr", specifying the input tensor(X2).
     * @param[in] quant A "Quantization" object, specifying the output quantization paramemters.
     * @return A "TensorPtr", specifying the output tensor Y.
     */
    TensorPtr bounding_box(TensorPtr input1, TensorPtr input2, const Quantization &quant = {});

    /**
     * @brief Greedily selects a subset of bounding boxes in descending order of score.
     *        This op applies NMS algorithm to each class. In each loop of execution,
     *        the box with maximum score gets selected and removed from the pending set.
     *        The scores of the rest of boxes are lowered according to the intersection-over-union (IOU)
     *        overlapping with the previously selected boxes and a specified NMS kernel method.
     *        Any boxes with score less than a threshold are removed from the pending set.
     * @param[in] score A "TensorPtr", which is a 2-D Tensor of shape [num_rois, num_classes],
     *                  specifying the score of each bounding box proposal.
     * @param[in] location A "TensorPtr", which is a 2-D Tensor specifying the bounding boxes of shape [num_rois, num_classes * 4],
     *                     organized in the order [x1, y1, x2, y2]. The boxes are grouped by batches in the first dimension.
     *                     The sequential order of the boxes corresponds with "score".
     * @param[in] batch_id A "TensorPtr", which a 1-D int32 tensor, shape[num_rois], specifying the batch index of each box.
     *                     Boxes with the same batch index are grouped together.
     * @param[in] score_threshold A float scalar, boxes with scores lower than the threshold are filtered before sending to the NMS algorithm.
     * @param[in] max_num_detections An int32 scalar, specifying the maximum number of selected bounding boxes for each image.
     *                               Set to a negative value for unlimited number of output bounding boxes.
     * @param[in] method Supports "HARD", "LINEAR", "GAUSSIAN"
     * @param[in] iou_threshold A float scalar, specifying the IoU threshold in "HARD" and "LINEAR" NMS kernel.
     *                          This field is ignored if "GAUSSIAN" kernel is selected.
     * @param[in] sigma A float scalar, specifying the sigma in "GAUSSIAN" NMS kernel.
     *                  This field is ignored if "GAUSSIAN" kernel is not selected.
     * @param[in] nms_score_threshold A float scalar, Boxes with scores lower than the threshold are dropped
     *                                during the score updating phase in soft NMS.
     * @param[in] num_output_rois An int32 scalar, specifying the number of output rois, which can be an expect max number output rois.
     * @return vector of "TensorPtr", first "TensorPtr" with same data type and scale/offset with @p score, with shape [num_output_rois],
     *         specifying the score of each output box. The boxes are grouped by batches, but the sequential order in each batch is not guaranteed.
     *         Second "TensorPtr" with same data type and scale/offset with @p location, with shape [num_output_rois, 4],
     *         specifying the coordinates of each output bounding box with the same format as "location". The sequential order of
     *         the boxes corresponds with first "TensorPtr".
     *         Third "TensorPtr" with shape [num_output_rois], specifying the class of each output box.
     *         the boxes corresponds with first "TensorPtr".
     *         Forth "TensorPtr" with shape [num_output_rois], specifying the batch index of each box. Boxes with
     *         the same batch index are grouped together.
     *         Fifth "TensorPtr" with shape [1], specifying the real number of output rois.
     */
    std::vector<TensorPtr> box_with_nms_limit(TensorPtr score,
                                              TensorPtr location,
                                              TensorPtr batch_id,
                                              float score_threshold,
                                              int32_t max_num_detections = -1,
                                              NmsMethod method = NmsMethod::Hard,
                                              float iou_threshold = 0.1,
                                              float sigma = 0.1,
                                              float nms_score_threshold = 0.1,
                                              int32_t num_output_rois = 100);

    /**
     * @brief The operator call will create the call op and its subgraph information. This subgraph is existed in another file.
     * @param[in] inputs A "TensorPtrList", specifying the tensors of subgraph.
     * @param[in] call_subgraph  A "std::string", specifying the subgraph name.
     * @param[in] txt_path  A "std::string", specifying the subgraph IR serialize path.
     * @return A "TensorPtrList", specifying the subgraph's output tensors.
     */
    TensorPtrList call(TensorPtrList inputs, const std::string &call_subgraph = "", const std::string &txt_path = "");

    /**
     * @brief The operator call will create the call op and its subgraph information. This subgraph is existed in another file.
     * @param[in] inputs A "TensorPtrList", specifying the input tensors of graph.
     * @param[in] call_subgraph  A "GraphPtr", specifying the subgraph.
     * @param[in] txt_path  A "std::string", specifying the subgraph IR serialize path.
     * @return A "TensorPtrList", specifying the subgraph's output tensors.
     */
    TensorPtrList call(TensorPtrList inputs, GraphPtr call_subgraph, const std::string &txt_path = "");

    /**
     * @brief The operator casts the elements of a given input tensor to a data type specified by the 'to' argument.
     *        The 'to' argument must be one of the data types specified in the 'Tensor::Dtype' enum.
     *        If input data overflows thetarget  data type, it will be cut-off.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] to_dtype A "Tensor::Dtype", specifying the target data type.
     * @param[in] ignore_scale_zp A "bool" type, default false.
     * @param[in] clip_mode An "Enumeration" of ClipMode, specifying clip with saturation or truncation.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters. Take an example:
     *                  If input dtype is `UINT8` with scale and zero_point, cast to `INT32`, quantization should be (1.0,0).
     * @return A "TensorPtr", which has same shape and scale/offset with @p input.
     */
    TensorPtr cast(TensorPtr input,
                   Tensor::Dtype to_dtype,
                   bool ignore_scale_zp = false,
                   ClipMode mode = ClipMode::Saturation,
                   const Quantization &quant = {});

    /**
     * @brief The ceil operation takes one input tensor (X) and produces one output tensor (Y).
     *        Where the ceil function 'y = ceil(x)'s is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset, default is empty.
     * @return A "TensorPtr", which has same shape and scale/offset with @p input.
     */
    TensorPtr ceil(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Takes an input tensor (X) and produces one output tensor (Y), where, the function
     *        'f(x) = max(0,x) + min(0,alpha*(exp(x/alpha)-1))', is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] alpha A float scalar, specifying the alpha in the formula, default is 1.0.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset, default is empty.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr celu(TensorPtr input,
                   float alpha = 1.0,
                   const Quantization &quant = {});

    /**
     * @brief Shuffle the channels of the input tensor.
     * @param[in] input   An n-D "TensorPtr", specifying the tensor to be shuffled.
     * @param[in] groups  number of groups to divide channels in.
     * @param[in] split  An positive integer. The number of output tensors, that is,
     *                   how many numbers of output tensors will be split from the
     *                   inputs after channel shuffle. Default value is 1.
     * @return A vector of "TensorPtr". Each TensorPtr points to tensor which has the
     *         same type as the tensors in "input".
     */
    std::vector<TensorPtr> channel_shuffle(TensorPtr input,
                                           int32_t groups,
                                           int32_t split = 1);

    /**
     * @brief Clip will clip output into [clip_min, clip_max].
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] clip_min A float/int scalar, specifying down boundary.
     * @param[in] clip_max A float/int scalar, specifying up boundary.
     * @param[in] quant A Quantization class, include scale and zeropoint,
     *                  If it is same between input & output, just ignore this parametr as default.
     * @return A "TensorPtr", which has the same data type, shapewith the @p input.
     */
    TensorPtr clip(TensorPtr input,
                   const NodeParamValue &clip_min,
                   const NodeParamValue &clip_max,
                   const Quantization &quant = {});

    /**
     * @brief Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
     *        In case axis is not provided, input is flattened before elements are selected.
     *        Compress behaves like numpy.compress.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] condition  A "TensorPtr", must have data.The data value should be 0 or 1.
     * @param[in] axis A scalar means the axis to take slices.If not specified, input is flattened before elements being selected.
     *                 Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
     * @return A "TensorPtr", its shape are described by @p axis and @p condition.
     */
    TensorPtr compress(TensorPtr input,
                       TensorPtr condition,
                       int32_t axis = -1);

    /**
     * @brief Concatenate a list of tensors into a single tensor. All input tensors must have the same shape,
     *        except for the dimension size of the axis to concatenate on.
     * @param[in] inputs List of "TensorPtr" for concatenation.
     * @param[in] axis Which axis to concat on. A negative value means counting dimensions from the back.
     *            Accepted range is [-r, r-1] where r = rank(inputs).
     *            Default value is -1.
     * @param[in] quant A "Quantization" object, specifying the output data scale and offset.
     * @return A "TensorPtr",resulting from concatenation of the @p input tensors.
     */
    TensorPtr concat(const std::vector<TensorPtr> &inputs,
                     int32_t axis = -1,
                     const Quantization &quant = {});

    /**
     * @brief Creates a constant tensor from a specified memory.
     * @param[in] g A "Graph" pointer, specifying the graph that Constant layer will be added.
     * @param[in] data A void pointer points to a constant data memory.
     * @param[in] shape A "TensorShape", specifying the shape of this layer.
     * @param[in] primitive_dtype A "Tensor::Dtype", specifying the primitive data type. Default is "UINT8".
     * @param[in] quant A "Quantization" object, specifying the constant data scale and offset.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr constant(GraphPtr g,
                       const void *data,
                       const TensorShape &shape,
                       Tensor::Dtype primitive_dtype = Tensor::UINT8,
                       const Quantization &quant = {});

    /**
     * @brief Creates a constant tensor from a specified memory.
     * @param[in] data A void pointer points to a constant data memory.
     * @param[in] shape A "TensorShape", specifying the shape of this layer.
     * @param[in] primitive_dtype A "Tensor::Dtype", specifying the primitive data type. Default is "UINT8".
     * @param[in] quant A "Quantization" object, specifying the constant data scale and offset.
     * @param[in] g A "Graph" pointer, specifying the graph that Constant layer will be added.
     *              Default is nullptr, system will bind to the specified graph if there only one graph.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr constant(const void *data,
                       const TensorShape &shape,
                       Tensor::Dtype primitive_dtype = Tensor::UINT8,
                       const Quantization &quant = {},
                       GraphPtr g = nullptr);

    /**
     * @brief Creates a constant tensor from a specified memory.
     * @param[in] c A tensor with data.
     * @param[in] quant A "Quantization" object, specifying the constant data scale and offset.
     * @param[in] g A "Graph" pointer, specifying the graph that Constant layer will be added.
     *              Default is nullptr, system will bind to the specified graph if there only one graph.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr constant(TensorPtr c,
                       const Quantization &quant = {},
                       GraphPtr g = nullptr);

    /**
     * @brief The conv_group operator consumes an input tensor and a filter, and computes the output.
     *        Number of groups that input channels and output channels are divided into. 'group' is greater than 1.
     * @param[in] input  A "TensorPtr",Support the following shape:
     *                   [batch,height,width,channel].
     * @param[in] filter  A "TensorPtr", specifying the size of the filter along each axis.
     *                    The 'filter' parameter format should be as follows:
     *                   [filter_y,filter_x].
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the convolution.
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
     *                      the height (width) axis of the filter in the NHWC data format.
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr conv_group(TensorPtr input,
                         TensorPtr filter,
                         TensorPtr bias,
                         const std::vector<int32_t> &strides,
                         Padding padding,
                         const Activation &with_activation = {},
                         const std::vector<int32_t> &dilations = {},
                         const Quantization &quant = {},
                         const TensorShape &output_shape = {});

    /**
     * @brief The conv1d operator consumes an input tensor and a filter, and computes the output.
     * @param[in] input  A "TensorPtr".
     *                   Support the following shape:
     *                   [batch,width,channel].
     * @param[in] filter  A "TensorPtr", specifying the size of the filter along each axis.
     *                    The 'filter' parameter format should be as follows:
     *                    [out_channel, filter_x, input_channel].
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the convolution.
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    1d:[pad_left, pad_right].
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis. Where dilation_x means
     *                      the width axis of the filter in the NHWC data format.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr conv1d(TensorPtr input,
                     TensorPtr filter,
                     TensorPtr bias,
                     const std::vector<int32_t> &strides,
                     Padding padding,
                     const Activation &with_activation = {},
                     const std::vector<int32_t> &dilations = {},
                     const Quantization &quant = {},
                     const TensorShape &output_shape = {});

    /**
     * @brief The The conv1d_transpose operator consumes an input tensor and a filter, and computes the output.
     * @param[in] input  A "TensorPtr".
     *                   Support the following shape:
     *                   [batch,width,channel].
     * @param[in] filter  A "TensorPtr", specifying the size of the filter along each axis.
     *                    The 'filter' parameter format should be as follows:
     *                    [out_channel, filter_x, input_channel].
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the conv2d_transpose.
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    1d:[pad_left, pad_right].
     * @param[in] group Number of groups that input channels and output channels are divided into.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr conv1d_transpose(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               int32_t group = 1,
                               const Activation &with_activation = {},
                               const std::vector<int32_t> &dilations = {},
                               const Quantization &quant = {},
                               const TensorShape &output_shape = {});

    /**
     * @brief The conv_2d operator consumes an input tensor and a filter, and computes the output.
     * @param[in] input  A "TensorPtr". Default shape [batch,height,width,channel].
     * @param[in] filter  A "TensorPtr", specifying the filter with shape [out_channel,filter_y,filter_x,input_channel].
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the convolution.
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
     *                      the height (width) axis of the filter in the NHWC data format.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr conv2d(TensorPtr input,
                     TensorPtr filter,
                     TensorPtr bias,
                     const std::vector<int32_t> &strides,
                     Padding padding,
                     const Activation &with_activation = {},
                     const std::vector<int32_t> &dilations = {},
                     const Quantization &quant = {},
                     const TensorShape &output_shape = {});

    /**
     * @brief The conv_2d operator consumes an input tensor and a filter, and computes the output.
     * @param[in] input  A "TensorPtr". Default shape [batch,height,width,channel].
     * @param[in] filter  A "TensorPtr", specifying the filter with shape [out_channel,filter_y,filter_x,input_channel].
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the convolution.
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
     *                      the height (width) axis of the filter in the NHWC data format.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @param[in] with_winograd An algorithm optimization parameter in an int IR.
     *                          'true means computing convolution with Winograd algorithm mode.
     *                          'false' means computing the convolution as general matrix multiplication.
     * @param[in] Fmnrs A list of integers in an int IR to indicate a 1-D or 2-D algorithm for F(m, r) or F(m x n, r x s).
     *                  For example, [2, 3] means the 1-D algorithm F(2, 3). [2, 2, 3, 3] means the 2-D algorithm F(2x2, 3x3).
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr conv2d_winograd(TensorPtr input,
                              TensorPtr filter,
                              TensorPtr bias,
                              const std::vector<int32_t> &strides,
                              Padding padding,
                              const Activation &with_activation = {},
                              const std::vector<int32_t> &dilations = {},
                              const Quantization &quant = {},
                              const TensorShape &output_shape = {},
                              const bool with_winograd = false,
                              const std::vector<int32_t> Fmnrs = {});

    /**
     * @brief Performs a filter on the input tensor (X) and produces the output tensor (Y). If bias is necessary,
     *        a bias vector will be added to the output tensor. Similarly, if the activation function is necessary,
     *        it is applied to the output tensor as well. By default, the input or output data format is NDHWC.
     * @param[in] input  A "TensorPtr", default is the following shape: [batch,depth,height,width,channel].
     * @param[in] filter  A "TensorPtr", specifying filter. Default shape [out_channel,kernel_y,kernel_x,kernel_z,input_channel].
     *                    kernel_x,kernel_y,kernel_z corresponds to width,height,depth.
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the convolution, generally, its shape should be [out_channel].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows: [stride_z,stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    3d:[pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end].
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis.
     *                      Format as following: [dilations_z, dilations_y, dilations_x].
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] act_params vector of float, specifying activation parameters, for example, if activation is CLIP,
     *                       it shoulbe be {clip_min, clip_max}.
     * @param[in] act_tensor vector of "TensorPtr", specifying the constant data of activation, for example,
     *                       negative_slop(alpha) data when activation is PRELU. If not, it can be empty.
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default empty.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape, layout should be same with @p input.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and quantization parameters are described by above parameters.
     */
    TensorPtr conv3d(TensorPtr input,
                     TensorPtr filter,
                     TensorPtr bias,
                     const std::vector<int32_t> &strides,
                     Padding padding,
                     const Activation &with_activation = {},
                     const std::vector<int32_t> &dilations = {},
                     const Quantization &quant = {},
                     const TensorShape &output_shape = {});

    /**
     * @brief The The conv2d_transpose operator consumes an input tensor and a filter, and computes the output.
     * @param[in] input  A "TensorPtr".
     *                   Support the following shape:
     *                   [batch,height,width,channel].
     * @param[in] filter  A "TensorPtr", specifying the size of the filter along each axis.
     *                    The 'filter' parameter format should be as follows:
     *                    [out_channel, filter_y, filter_x, input_channel].
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the conv2d_transpose.
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] group Number of groups that input channels and output channels are divided into.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr conv2d_transpose(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               int32_t group = 1,
                               const Activation &with_activation = {},
                               const std::vector<int32_t> &dilations = {},
                               const Quantization &quant = {},
                               const TensorShape &output_shape = {});

    /**
     * @brief ConvTranspose3D applies a 3D transposed convolution operator over an input image composed of several input planes.
     *        That is, it performs a 3D-filter on the input tensor (X) and produces the output tensor (Y).
     *        Generally, the output shape is calculated through the following equation:
     *          output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_size[i] -1) * dilation[i] + 1) - pad_i_start - pad_i_end
     *        Where output_padding is the padding added to the output.
     * @param[in] input  A "TensorPtr", default is the following shape: [batch,depth,height,width,channel].
     * @param[in] filter  A "TensorPtr", specifying filter. Default shape [out_channel,kernel_y,kernel_x,kernel_z,input_channel].
     *                    kernel_x,kernel_y,kernel_z corresponds to width,height,depth.
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the convolution, generally, its shape should be [out_channel].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows: [stride_z,stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    3d:[pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end].
     * @param[in] group Number of groups that input channels and output channels are divided into.
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis.
     *                      Format as following: [dilations_z, dilations_y, dilations_x].
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] act_params initializer_list of float, specifying activation parameters, for example, if activation is CLIP,
     *                       it shoulbe be {clip_min, clip_max}.
     * @param[in] act_tensor initializer_list of "TensorPtr", specifying the constant data of activation, for example,
     *                       negative_slop(alpha) data when activation is PRELU. If not, it can be empty.
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default empty.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape, layout should be same with @p input.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and quantization parameters are described by above parameters.
     */
    TensorPtr conv3d_transpose(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               int32_t group = 1,
                               const Activation &with_activation = {},
                               const std::vector<int32_t> &dilations = {},
                               const Quantization &quant = {},
                               const TensorShape &output_shape = {});

    /**
     * @brief Performs  a layout convertion for the tensor original layout to the expected layout.
     * @param[in] input          A "TensorPtr" which perform the convertion.
     * @param[in] convert_to     A "DataLayout", which is the expected layout. if the convertion is impossible, a nullptr will return.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr convert_layout(TensorPtr input, DataLayout convert_to);

    /**
     * @brief Merge repeated labels into single labels.
     * @param[in] labels A "TensorPtr" which shape is [batch, max value in seq_length].
     * @param[in] seq_length A "TensorPtr" which shape is [batch], sequence length of each batch element.
     * @return A "TensorPtr" list, specifying the output tensors [collapsed_labels, new_seq_length].
     */
    std::vector<TensorPtr> collapse_repeated(TensorPtr labels, TensorPtr seq_length);

    /**
     * @brief The operator rearranges column blocks back into a multidimensional image.
     * @param[in] input A "TensorPtr", specifying input tensor.
     *                  Input data tensor to be rearranged from column blocks back into an image.S
     *                  This is a 3-dimensional tensor containing [N, C * n-ary- product(block_shape), L], where N is batch dimension,
     *                  C is image channel dimension and L is number of blocks.The blocks are enumerated in increasing lexicographic-order
     *                  of their indices.For example, with an image-size 10*20 and block-size 9*18, there would be 2*3 blocks,
     *                  enumerated in the order block(0, 0), block(0, 1), block(0, 2), block(1, 0), block(1, 1), block(1, 2).
     * @param[in] image_shape The shape of the spatial dimensions of the image after rearranging the column blocks.
     *                        This is a 1-dimensional tensor with size of at least 2, containing the value [H_img, W_img] for a 2-D image
     *                        or [dim_i1, dim_i2, ..., dim_iN] for a N-D image.
     * @param[in] block_shape The shape of the block to apply on the input.This is a 1-dimensional tensor of size of at least 2,
     *                        containing the value [H_block, W_block] for a 2-D image or [dim_b1, dim_b2,..., dim_bN] for a N-D block.
     *                        This is the block-shape before dilation is applied to it.
     * @param[in] strides 1-dimensional tensor with stride value along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     * @param[in] padding 1-dimensional tensor with padding value for the beginning and ending along each spatial axis,
     *                    it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning
     *                    and end part of the corresponding axis. pads format should be as follow [x1_begin, x2_begin ... x1_end, x2_end,...],
     *                    where xi_begin is the number of pixels added at the beginning of axis i and xi_end is the number of pixels
     *                    added at the end of axis i. If not present, the padding defaults to 0 along start and end of each spatial axis.
     * @param[in] dilations 1-dimensional tensor with dilation value along each spatial axis of the image.
     *                      If not present, the dilation defaults to 1 along each spatial axis of the image.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr col2im(TensorPtr input,
                     TensorPtr image_shape,
                     TensorPtr block_shape,
                     const std::vector<int32_t> &strides = {},
                     Padding padding = Padding({0, 0, 0, 0}),
                     const std::vector<int32_t> &dilations = {},
                     const Quantization &quant = {});

    /**
     * @brief Calculates the hyperbolic cosine of the given input tensor element-wise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr cosh(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Performs a Relu operation on the input tensor (X) which selects the positive part of the activation and negative part of the activation respectively,
     *        then concatenates them as an output tensor (Y) along with the specified axis. That is, Crelu(x) = Concat [Relu(x), Relu(-x)].
     * @param[in] input A "TensorPtr", specified input tensor.
     * @param[in] aixs A int32 scalar, specified the axis that the output values are concatenated along, default is -1.
     * @param[in] quant A "Quantization", specifying output quantization params, default is empty.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr crelu(TensorPtr input, int32_t axis = -1, const Quantization &quant = {});

    /**
     * @brief Crop a tensor by the gived crop parameters.
     * @param[in] input A "TensorPtr", which needs to be cropped.
     * @param[in] crops A vector describles crop parameters,
     *                  vector size should be same with spatial dimensions,
     *                  each element of vector should
     *                  reperesent crop begin and crop end in pair.
     * @param[in] ds_crops The dynamic crop parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr crop(TensorPtr input,
                   const std::vector<std::pair<int32_t, int32_t>> &crops = {},
                   const std::vector<std::pair<tongue::Expr, tongue::Expr>> &ds_crops = {});

    /**
     * @brief Extracts crops from the input image and resize it by bilinear or nearest neighbor sampling to a common output size specified by `crop_size` parameter.
     * @param[in] input A "TensorPtr", specified input image.
     * @param[in] box A "TensorPtr", specified box cooridinates in shape [numboxes, 4] with order [y1,x1,y2,x2], it is normalized tensor in [0,1],
     *                corresponding to [0, height-1]. If box is quantied, quantization of box should describle the scale of coordinates.
     * @param[in] box_indices A "TensorPtr", specified the each box batch index in order. It shape should be [numboxes].
     * @param[in] crop_size A vector to specified cropped height and width.
     * @param[in] method An enum in "ResizeMethod" scalar, specified sampling mode.
     * @param[in] extrapolation_value A float/int scalar, tspecified for extrapolation. Generally, default is 0.
     * @param[in] is_perf_mode A bool scalar, specified whether the op is running with high performance mode.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr crop_and_resize(TensorPtr input,
                              TensorPtr box,
                              TensorPtr box_indices,
                              const std::vector<uint32_t> &crop_size,
                              ResizeMethod method = ResizeMethod::NearestNeighbor,
                              NodeParamValue extrapolation_value = 0,
                              bool is_perf_mode = true);

    /**
     * @brief Computes cosine of x element-wise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr cosine(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Performs greedy decoding on the logits given in the input tensor, including the score tensor and
     *        sequence length tensor. Where the sequence length's shape equals the batch size of input tensor X.
     *        By default, mege_repeated is `true`, and the operator merges repeated classes in output results.
     *        This means that if the maximum indices of consecutive logits are the same, only the first of these is emitted.
     * @param[in] input          A "TensorPtr", Shape size should be 3 in (n,h,w) format.
     * @param[in] seq_len        A "TensorPtr", Shape size should be 1, and this dimension equals to input[n].
     * @param[in] merge_repeated Defaults to `true`, which means merging repeated classes in output results.
     *                           That is, if consecutive logits' maximum indices are the same, then only the first one is emitted. For example,
     *                           giving an input sequence X = [A, B, B, C, B, D, B], then returns the Y = [A, B, C, B, D, B] when `merge_repeated` is `true`;
     *                           or Y = [A, B, B, C, B, D, B] when `merge_repeated` is `false`.
     * @param[in] output_shape   A "TensorShape", describes each dimension of output tensor
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr ctc_greedy_decoder(TensorPtr input,
                                 TensorPtr seq_len,
                                 bool merge_repeated = true,
                                 const TensorShape &output_shape = {});

    /**
     * @brief Applies to input numeric tensor X and returns a 2-D histogram to count the number of entries
     *        in the numeric tensor that fell into every bin. Generally,
     *        the bins are equal width and determined by the parameters 'min', 'max' and 'nbins'.
     *        Note that elements in input numeric tensor (X) lower than 'min' and higher than 'max' are ignored.
     *        By default, the numeric data is counted over batch dimension. For example,
     *        numeric_X = [[-1, 1, 0, 1, 2, 3, 12]], min = 0, max = 4, nbins = 5 and 'discrete = true',
     *        then returns a 2-D tensor Y =[[1, 2, 1, 1, 0]].
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] min A float/int lower end of the range (inclusive).
     * @param[in] max A float/int upper end of the range (inclusive).
     * @param[in] nbins A scalar of integer to indicate the number of histogram bins.
     *                  Generally, every bin is a left closed and right open interval.
     * @param[in] discrete A statistic mode. 'true' means treating the bins as discrete values
     *                    from minimum to maximum when 'min' and 'max' are integers.
     *                    'false' means treating the bins as an interval.
     * @return A "TensorPtr", its shape is [input.batch, nbins], and data type is uint16.
     */
    TensorPtr count(TensorPtr input, NodeParamValue min, NodeParamValue max, uint32_t nbins, bool discrete = true);

    /**
     * @brief Compute the cumulative value of the tensor x along axis.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] axis, A int32 (default: 0). Must be in the range [-rank(x), rank(x)).
     * @param[in] exclusive, If True, perform exclusive cumsum.
     * @param[in] reverse, A bool (default: False).
     * @return A "TensorPtr", which has same data type and scale/offset with @p input.
     */
    TensorPtr cumulate(TensorPtr input,
                       CumulateMethod method = CumulateMethod::SUM,
                       int32_t axis = 0,
                       bool exclusive = false,
                       bool reverse = false);

    /**
     * @brief Performs an input data tensor (X) to produce a output tensor (Y) with a stride combination,
     *        which means taking kernel size numbers of data after every stride operation and gathering all these data along each axis respectively.
     *        For example, given an input tensor with the shape as [1, 8, 8, 3] and kernel_x (kernel_y) =3, stride_x (stride_y) = 3,
     *        then it returns an output tensor Y with the shape as [1, 6, 6, 3].
     * @param[in] input A "TensorPtr", specifying the input tensor, rank must be 4.
     * @param[in] kernel, An uint32 vector, size is 2. It specifies the cut size during each stride, and
     *                         order is kernel_height, kernel_width.
     * @param[in] stride, An uint32 vector, size is 2. It specifies the each stride size, and order is
     *                         stride_height, stride_width.
     * @return A "TensorPtr", which has same data type and scale/offset with @p input. Output shape can be inferred by
     *         @p kernel and @p stride.
     */
    TensorPtr data_stride(TensorPtr input,
                          const std::vector<uint32_t> &kernel,
                          const std::vector<uint32_t> &stride);

    /**
     * @brief Boxes with scores less than Score_Threshold are dropped or discarded in the last output.
     *        the box need translated into bounding box.
     *        Note that, in the following Inputs or Outputs sections:
     *        X1 means the scores of all the boxes per batch and per class, of which the shape is [batch_size, num_boxes, class_num + 1].
     *        X2 means the coordinates of the maximum output boxes per batch, of which the shape is [batch_size, num_boxes, 4].
     *        Y1 means the coordinates of the selected boxes per batch, of which the shape is [batch_size, max_output_size, 4].
     *        Y2 means the num of the selected boxes belongs to cluster class_id per batch, of which the shape is [batch_size, max_output_size].
     *        Y3 means the num of the all selected boxes belongs to class per batch, of which the shape is [batch_size, 1].
     *        Y4 means the scores of all the selected boxes per batch, of which the shape is the same as Y2.
     *        Y5 means the class_ids of the selected boxes per batch in input tensor X1, of which the shape is the same as Y2 as well.
     * @param[in] input1 A "TensorPtr", specifying input tensor X1.
     * @param[in] input2 A "TensorPtr", specifying input tensor X2.
     * @param[in] anchors A "TensorPtr" list, specifying the anchors parameter (h, w, ycenter, xcenter).
     * @param[in] width The float IR parameters during normalized coordinate quantization,
     *                  which indicate the width of the input feature map.
     * @param[in] height The float IR parameters during normalized coordinate quantization,
     *                   which indicate the height of the input feature map.
     * @param[in] score_threshold The threshold for deciding when to remove boxes based on the score. It is a scalar.
     * @param[in] max_output_size The maximum number of boxes to be output.
     * @param[in] quants A "Quantization" vector, specifying the (Y1,Y4) scale and offset.
     *                   If it is empty, Y1's quantization parameter is the same as X2,
     *                   and Y4's quantization parameter is the same as X1.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2,Y3,Y4,Y5.
     */
    std::vector<TensorPtr> decode_box(TensorPtr input1,
                                      TensorPtr input2,
                                      std::vector<TensorPtr> anchors,
                                      const int32_t width,
                                      const int32_t height,
                                      const NodeParamValue score_threshold,
                                      const int32_t max_output_size,
                                      const std::vector<Quantization> &quants = {});

    /**
     * @brief DepthToSpace permutes data from depth into blocks of spatial data. This is the reverse transformation of SpaceToDepth.
     *        More specially, this operator outputs a copy of the input tensor where values from the depth dimension
     *        are moved in spatial blocks to the 'height' and 'width' dimensions.
     *        In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the following order:
     *        'height', 'width' and 'channel'.
     *        The input will be reshape into [N, H , W , block_size_y, block_size_x, C/(block_size_y * block_size_x)] ,
     *        and transpose to [N, H, block_size_y, W, block_size_x, C/(block_size_y * block_size_x)].
     *        The output shape will be reshape to [N, H * block_size_y, W * block_size_x, C/(block_size_y * block_size_x)].
     *        In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the following order:
     *        'height', 'width' and 'channel'.
     *        The input will be reshape into [N, H , W , C/(block_size_y * block_size_x), block_size_y, block_size_x] ,
     *        and transpose to [N, H, block_size_y, W, block_size_x, C/(block_size_y * block_size_x)].
     *        The output shape will be reshape to [N, H * block_size_y, W * block_size_x, C/(block_size_y * block_size_x)].
     * @param[in] input A "TensorPtr", specifying the input tensor, rank must be 4.
     * @param[in] block_size An uint32 scalar, specifying the size of spatial block.
     * @param[in] mode Indicates in which mode the elements along the depth dimension from the input tensor are rearranged.
     *                 DCR is the default mode.
     * @return A "TensorPtr", which has same data type and scale/offset with @p input. Output shape can be inferred by
     *         @p block_size.
     */
    TensorPtr depth_to_space(TensorPtr input, uint32_t block_size, DepthToSpaceMode mode = DepthToSpaceMode::DCR);

    /**
     * @brief Performs a different filter to each channel of input tensor (X) and then concatenates
     *        the results of convolution together. Generally, the channel number of the output tensor
     *        will be out_channel = in_channels * multiplier.
     * @param[in] input  A "TensorPtr".
     *                   Support the following shape:
     *                   [batch,height,width,channel].
     * @param[in] filter  A "TensorPtr", specifying the size of the filter along each axis.
     *                    The 'filter' parameter format should be as follows:
     *                    [out_channel, kernel_y, kernel_x, 1].
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the convolution.
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
     *                      the height (width) axis of the filter in the NHWC data format.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr depthwise_conv2d(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               const Activation &with_activation = {},
                               const std::vector<int32_t> &dilations = {},
                               const Quantization &quant = {},
                               int32_t quantization_dim = 0,
                               const TensorShape &output_shape = {});

    /**
     * @brief Deuantize the input tensor
     *         - quant -> float
     *         - quant -> quant (tflite)
     * @param[in] input A "TensorPtr", specifying the input tensor
     * @param[in] output_type A "Dtype", specifying output data type
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Its data type will be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr dequantize(TensorPtr input,
                         Tensor::Dtype output_type = Tensor::Dtype::FP32,
                         const Quantization &quant = {});

    /**
     * @brief Computes the grayscale dilation of 4-D input and 3-D filters tensors.
     * @param[in] input  A "TensorPtr". Default shape [batch,height,width,channel].
     * @param[in] weights A "TensorPtr", specifying the weights with shape [input_channel, kernel_y, kernel_x, 1].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
     *                      the height (width) axis of the filter in the NHWC data format.
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr dilation(TensorPtr input,
                       TensorPtr weights,
                       const std::vector<int32_t> &strides,
                       Padding padding,
                       const std::vector<int32_t> &dilations = {},
                       const Quantization &quant = {});

    /**
     * @brief Performs element-wise binary division (with Numpy-style broadcasting support).
     *        This operator supports multidirectional (i.e., Numpy-style) broadcasting;
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor,
     *                   which must have the same data type bitwith with @p input1.
     *                   But it can have different scale and zp with @p input1.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", has same data type with @p inputs.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr div(TensorPtr input1, TensorPtr input2, const Quantization &quant = {});

    /**
     * @brief Performs integer lement-wise division and modular(with NumPy-style broadcasting support).
     *        This operator also supports multidirectional (that is, NumPy-style) broadcasting.
     * @param[in] input1 A "TensorPtr", specifying the first tensor(X1).
     * @param[in] input2 A "TensorPtr", specifying the second tensor(X2).
     *                   which must have the same data type bitwith with @p input1.
     *                   But it can have different scale and zp with @p input1.
     * @param[in] mod  Support TRUNC/FLOOR.
                        The parameter defines how the conversion behaves if an output value is out of range of the destination type.
                        When "clip_mode=SATUETION", the output data will saturtion into the output type.
                        When "clip_mode=TRUNCATION", the output data will be truncated into output bit width.
     * @param[in] quants A "Quantization" vector, specifying the (Y1,Y2) scale and offset.
     *                   If it is empty, Y1's quantization parameter is the same as x1, and Y2's quantization parameter is the same as x2.
     *                   If the number of parameters is 1, Y1 and Y2 will use the same quantization.
     * @return A "TensorPtr" vector, which are division tensor Y1,modular tensor Y2.
     *         The output tensors Y1,Y2 can have the different scale and offset from `input1` & `input2`.
    */
    std::vector<TensorPtr> divmod(TensorPtr input1,
                                  TensorPtr input2,
                                  const DivmodMethod mod = DivmodMethod::TRUNC,
                                  const std::vector<Quantization> &quants = {});

    /**
     * @brief Boxes with scores less than Score_Threshold are dropped or discarded in the last output.
     * the box need translated into bounding box.
     * Note that, in the following Inputs or Outputs sections:
     *  X1 means the scores of all the boxes per batch and per class,
     *     of which the shape is [batch_size, anchor_num, class_num + 1].
     *  X2 means the coordinates of the input boxes per batch,
     *     of which the shape is [batch_size, anchor_num, class_num + 1, 4].
     *  X3 means the coordinates of the anchor per batch,
     *     of which the shape is [batch_size, anchor_num, 4].
     *  Y1 means the scores of all the selected boxes per batch,
     *     of which the shape is the same as [batch_size, max_output_size].
     *  Y2 means the coordinates of the selected boxes per batch,
     *     of which the shape is [batch_size, max_output_size, 4].
     *  Y3 means the num of the selected boxes belongs to cluster class_id per batch,
     *     of which the shape is [batch_size, max_output_size].
     *  Y4 means the class_ids of the selected boxes per batch in input tensor X1,
     *     of which the shape is the same as Y3 as well.
     *  Y5 means the num of the all selected boxes belongs to class per batch,
     *     of which the shape is [batch_size, 1].
     * @param[in] scores A "TensorPtr", specified the input tensor X1,
     *                   The tensor shape is [batch_size, anchor_num, class_num + 1].
     * @param[in] coordinates A "TensorPtr", specified the input tensor X2.
     *                        The tensor shape is [batch_size, anchor_num, class_num + 1, 4].
     * @param[in] anchor A "TensorPtr", specified the input tensor X3,
     *                   The tensor shape is [batch_size, anchor_num, 4].
     * @param[in] width The float IR parameters during normalized coordinate quantization,
                        which indicate the width of the input feature map.
     * @param[in] height The float IR parameters during normalized coordinate quantization,
                         which indicate the height of the input feature map
     * @param[in] score_threshold The of score for valid bounding boxes.
     * @param[in] max_output_size Integer representing the maximum number
     *                            of boxes to be selected per batch. It is a scalar.
     * @param[in] quants A "Quantization" vector, specifying the (Y1,Y2,Y3,Y4,Y5) scale and offset.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2,Y3,Y4,Y5.
     */
    std::vector<TensorPtr> detection_output(TensorPtr scores,
                                            TensorPtr coordinates,
                                            TensorPtr anchor,
                                            const int32_t width,
                                            const int32_t height,
                                            const NodeParamValue score_threshold,
                                            const int32_t max_output_size,
                                            std::vector<float> variance = {1.0, 1.0, 1.0, 1.0},
                                            const std::vector<Quantization> &quants = {});

    /**
     * @brief Connect two tensors together without doing any actual operations.
     *        Used to solve the problem of disconnected graphs.
     * @param[in] inputs A "TensorPtr" list, specifying input tensors.
     * @return A "TensorPtr" list, specifying output tensors,
     *         which has the same type and shape as the inputs.
     */
    TensorPtrList dummy(TensorPtrList inputs);

    /**
     * @brief Performs elementwise addition of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr elementwise_add(TensorPtr input1,
                              TensorPtr input2,
                              const Activation &with_activation = {},
                              const Quantization &quant = {});

    /**
     * @brief Performs elementwise maximum of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                   f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                   This operator supports unidirectional broadcasting (tensor slope should be
     *                   unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr elementwise_max(TensorPtr input1,
                              TensorPtr input2,
                              const Activation &with_activation = {},
                              const Quantization &quant = {});

    /**
     * @brief Performs elementwise minimum of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr elementwise_min(TensorPtr input1,
                              TensorPtr input2,
                              const Activation &with_activation = {},
                              const Quantization &quant = {});

    /**
     * @brief Performs element-wise multiplication of the input tensors.
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr elementwise_mul(TensorPtr input1,
                              TensorPtr input2,
                              const Activation &with_activation = {},
                              const Quantization &quant = {});

    /**
     * @brief Performs elementwise subtraction of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr elementwise_sub(TensorPtr input1,
                              TensorPtr input2,
                              const Activation &with_activation = {},
                              const Quantization &quant = {});

    /**
     * @brief split a tensor into a list of sub tensors.
     * @param[in] params  A tensor pointer representing the complete embedding tensor.
     * @param[in] indices A tensor pointer with type "int32" containing the indices to be looked
     *                    up in "params".
     * @return  A tensor pointer with the same type as the tensors in "params".
     */
    TensorPtr embedding_lookup(TensorPtr params,
                               TensorPtr indices);

    /**
     * @brief This operator assumes that there is at least one ID for each row in the dense tensor represented
     *        by sp_ids (i.e. there are no rows with empty features), and that all the indices of sp_ids are in canonical row-major order.
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
              X3: sp_value, a tensor indicating the sp_value, which means how to gather the params.
              X4: weight_value, a tensor means the weight for the gathered params to output.

     * @param[in] params  A tensor pointer representing the complete embedding tensor.
     * @param[in] sp_ids A tensor pointer indicating the output indeces.
     * @param[in] sp_value A tensor pointer indicating the sp_value , which means how to gather the params.
     * @param[in] weight_value  A tensor pointer means the weight for the gathered params to output.
     * @param[in] dense_shape  [N, M] where N is typically batch size and M is arbitrary or a RaggedTensor with rank 2.
     * @param[in] combiner A enum type specifying the reduction op.
     *                    "SUM" computes the weighted sum of the embedding results for each row.
     *                    "MEAN" is the weighted sum divided by the total weight.
     *                    "SQRTN" is the weighted sum divided by the square root of the sum of the squares of the weights.
     *                    Defaults to "MEAN".
     * @param[in] max_norm A float value. If not NONE, each embedding is clipped if its l2-norm is larger than this value, before combining.
     * @param[in] quant  A Quantization class for output,include scale and zeropoint.
     * @return A "TensorPtr", specifying the output tensor.
    */
    TensorPtr embedding_lookup_sparse(TensorPtr params,
                                      TensorPtr sp_ids,
                                      TensorPtr sp_value,
                                      TensorPtr weight_value,
                                      const TensorShape dense_shape = {},
                                      const EmbeddingLookupSparseCombiner combiner = EmbeddingLookupSparseCombiner::MEAN,
                                      const float max_norm = 0,
                                      const Quantization &quant = {});

    /**
     * @brief Create an empty node with random output data.
     * @param[in] shape "TensorShape", specifying the static shape of the output tensor.
     * @param[in] dtype "Dtype", specifying the data type of the output tensor.
     * @param[in] exprs Optional "Expr" list, specifying the symbolic shape of the output tensor.
     * @param[in] g Optional "GraphPtr", specifying the graph to which the node belongs.
     * @return "TensorPtr", specifying the output tensor.
     */
    TensorPtr empty(const TensorShape &shape,
                    Tensor::Dtype dtype,
                    const std::vector<tongue::Expr> &exprs = {},
                    GraphPtr g = nullptr);

    /**
     * @brief Takes an input tensor (X) and produces one output tensor (Y).
     *        Where, the function f(x) = alpha * (exp(x) -1) for x < 0, f(x) = x for x >= 0,
     *        is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr".
     * @param[in] alpha Coefficient parameter in ELU. It defaults to 1.
     * @param[in] quant  A Quantization class for output,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr elu(TensorPtr input,
                  float alpha = 1.0f,
                  const Quantization &quant = {});

    /**
     * @brief Computes the error function of the given input tensor element-wise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr erf(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Computes the grayscale erosion of 4-D input and 3-D filters tensors.
     * @param[in] input  A "TensorPtr". Default shape [batch,height,width,channel].
     * @param[in] weights  A "TensorPtr", specifying the weights with shape [input_channel, kernel_y, kernel_x, 1].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    [stride_y,stride_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] dilations The dilation value along the spatial axis of the filter. If not present,
     *                      the dilation default value is 1 along each spatial axis. Where dilation_x (dilation_y) means
     *                      the height (width) axis of the filter in the NHWC data format.
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr erosion(TensorPtr input,
                      TensorPtr weights,
                      const std::vector<int32_t> &strides,
                      Padding padding,
                      const std::vector<int32_t> &dilations = {},
                      const Quantization &quant = {});

    /**
     * @brief Returns a tensor with a length 1 axis inserted at index "axis".
     *        The output tensor has the same values as input tensor.
     * @param[in] input A "TensorPtr".
     * @param[in] axis  Integer specifying the dimension index at which to expand the shape of @p input.
     * @return A "TensorPtr", with type as the tensors in @p input, with an additional dimension
               inserted at the index specified by "axis".
    */
    TensorPtr expand_dims(TensorPtr input,
                          int32_t axis);

    /**
     * @brief Returns the truth value of (input1 == input2) element-wise.
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "BOOL".
     */
    TensorPtr equal(TensorPtr input1,
                    TensorPtr input2,
                    const Quantization &quant = {});

    /**
     * @brief Computes exponential of input element-wise.
     * @param[in] input A "TensorPtr".
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr exp(TensorPtr input,
                  const Quantization &quant = {});

    /**
     * @brief Takes one input data tensor (X) and filter tensor (B) to return an output tensor (Y) after filtration.
     *        Where, the element will be filtered with zero selector accordingly in filter tensor B, otherwise remaining.
     *        The filter tensor is a vector which is composed of '0' or '1' with the same length
     *        as the specified axis in data tensor X.
     *        Similarly, multiple input tensors (X[i]) will return multiple outputs after the same filtration.
     * @param[in] inputs A "TensorPtr" list, which are input data tensor X[i] ('i' means the index of input tensors),
     *                   filter tensor B (a 1-D tensor).
     * @return A "TensorPtr" list, which are output tensor Y[i] ('i' means the index of output tensors),
     *         effective length (a scalar with an empty shape).
     */
    std::vector<TensorPtr> filter(std::vector<TensorPtr> inputs);

    /**
     * @brief Filter boxes that meet limited criteria.
     *        Note that, in the following Inputs or Outputs sections:
     *        X1 means box cooridinates in shape [batch_size, num_boxes, 4] with order [y1,x1,y2,x2].
     *        X2 means the scores of the boxes, of which the shape is [batch_size, num_boxes].
     *        X3 means the boxes number of every class per batch,
     *           of which the shape is [batch_size, num_classes].
     *        X4 means the boxes label of every class per batch, of which the shape is
     *           [batch_size, num_classes].
     *        X5 means the valid classes of every batch, of which the shape is [batch_size, 1].
     *        Y1 means the coordinates of the filtered boxes, of which the shape is
     *           [batch_size, max_box_num, 4].
     *        Y2 means the scores of the selected boxes, of which the shape is [batch_size, max_box_num].
     *        Y3 means the boxes number of every class per batch after selection,
     *           of which the shape is [batch_size, num_classes].
     *        Y4 means the boxes label of every class per batch after selection,
     *           of which the shape is [batch_size, num_classes].
     *        Y5 means the valid classes of every batch after selection,
     *           of which the shape is [batch_size, 1].
     * @param[in] boxes A "TensorPtr", specified the input tensor X1.
     *                  If box is quantied, quantization of box should describle the scale of coordinates.
     *                  The tensor shape is [batch_size, num_boxes, 4].
     * @param[in] scores A "TensorPtr", specified the input tensor X2,
     *                   The tensor shape is [batch_size, num_boxes].
     * @param[in] number A "TensorPtr", specified the input tensor X3.
     *                  The tensor shape is [batch_size, num_classes].
     * @param[in] label A "TensorPtr", specified the input tensor X4,
     *                   The tensor shape is [batch_size, num_classes].
     * @param[in] classes A "TensorPtr", specified the input tensor X5.
     *                    The tensor shape is [batch_size, 1].
     * @param[in] score_threshold The of score for valid bounding boxes.
     * @param[in] max_box_num Integer representing the maximum number of boxes to be selected per batch. It is a scalar.
     * @param[in] minsize The minimum size of a valid bounding box. [h,w].
     * @param[in] quants A "Quantization" vector, specifying the (Y1,Y2,Y3,Y4,Y5) scale and offset.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2,Y3,Y4,Y5.
     */
    TensorPtrList filter_boxes(TensorPtr boxes,
                               TensorPtr scores,
                               TensorPtr number,
                               TensorPtr label,
                               TensorPtr classes,
                               const NodeParamValue &score_threshold,
                               const int32_t max_box_num,
                               const std::vector<int32_t> &minsize,
                               const std::vector<Quantization> &quants = {});

    /**
     * @brief The floor operation takes one input tensor (X) and produces one output tensor (Y).
     *        Where the floor function 'y = floor(x)' is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type, quantization parameters as the tensors in "input".
     */
    TensorPtr floor(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Fractional average pooling is similar to Fractional max pooling in the pooling region generation step.
     *        The only difference is that after pooling regions are generated, a mean operation is performed instead of a max operation in each pooling region.
     *        Note that, in the following Inputs or Outputs sections:
     *        X means the input data.
     *        Y1 means output data after fractional avgerage pooling.
     *        Y2 means row_pooling_sequence.
     *        Y3 means col_pooling_sequence.
     * @param[in] input A "TensorPtr", specifying input tensor(X).
     * @param[in] pooling_ratio A list of floats that has length <= 4.
     *                          Pooling ratio for each dimension of value, currently only supports row and col dimension and should be >= 1.0.
     *                          For example, a valid pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements must be 1.0
     *                          because we don't allow pooling on batch and channels dimensions. 1.44 and 1.73
     *                          are pooling ratio on height and width dimensions respectively.
     * @param[in] method The method of pooling.
     *                   AVG/MAX.
     * @param[in] pseudo_random When set to true, generates the pooling sequence in a pseudorandom fashion, otherwise, in a random fashion.
     * @param[in] overlap When set to true, it means when pooling, the values at the boundary of adjacent pooling cells are used by both cells.
     * @param[in] seed An optional int.Defaults to 0. If set to be non-zero, the random number generator is seeded by the given seed.
     *                 Otherwise it is seeded by a random seed.
     * @return A "TensorPtr" vector of size is 3, containing Y1,Y2,Y3.
     */
    std::vector<TensorPtr> fractional_pool(TensorPtr input,
                                           std::vector<double> pooling_ratio,
                                           PoolMethod method,
                                           bool pseudo_random = false,
                                           bool overlap = false,
                                           int32_t seed = 0);

    /**
     * @brief Takes input tensor (X) and computes the class scores and outputs the 1-D array of size equal
     *         to the number of classes. In other words, it performs a 'dot(input_tensor, kernel)' operation
     *         with a bias addition, then outputs the tensor after the activation function.
     * @param[in] input  A "TensorPtr".
     *                   Only support the following shape:
     *                   [batch_size, input_size].
     * @param[in] filter  A "TensorPtr", specifying the filters, of shape [num_units, input_size],
     *                    where "num_units" corresponds to the number of output tensor channel.
     * @param[in] bias   A "TensorPtr", specifying bias to be added to the fully_connected.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *            - Activation("CLIP", clip_min, clip_max)
     *            - Activation("PRELU", negative_slope)
     *            - ...
     * @param[in] quant Optional "Quantization" object, specifying the output data scale and offset.
     *                  If not set, used default.
     * @param[in] output_shape  Optional "TensorShape" object, specifying the output shape.
     *                          If not present, should be inferred.
     * @return A "TensorPtr", its shape and scale/offset are described by above parameters.
     */
    TensorPtr fully_connected(TensorPtr input,
                              TensorPtr filter,
                              TensorPtr bias,
                              const Activation &with_activation = {},
                              const Quantization &quant = {},
                              const TensorShape &output_shape = {});
    /**
     * @brief Given parameters and indices tensor, gathers entries of the parameters tensor indexed by indices
     *        and concatenates them in an output tensor. The indices are a tensor of integer,
     *        and there will be an error if any of the index values are out of bounds.
     *        Generally, a negative value in the indices tensor means counting the index from the back.
     * @param[in] input  A "TensorPtr", which to gather values. Must be at least rank "axis + 1".
     * @param[in] indices A tensor pointer with type "int32" containing the indices to be looked  up in @p input.
     * @param[in] axis Which axis to gather on. Negative value means counting dimensions from the back.
     *                 Accepted range is [-r, r-1] where r = rank(data).
     * @param[in] batch_dims A scalar of integer to indicate the number of batch dimensions.
     *                       Generally, defaults to 0.
     * @return  A tensor pointer with the same type as the tensors @p input.
     */
    TensorPtr gather(TensorPtr input,
                     TensorPtr indices,
                     int32_t axis = 0,
                     int32_t batch_dims = 0);

    /**
     * @brief GatherElements takes two inputs data and indices of the same rank r >= 1 and an optional attribute
     *        axis that identifies an axis of data (by default, the outer-most axis, that is axis 0).
     *        It is an indexing operation that produces its output by indexing into the input data tensor at
     *        index positions determined by elements of the indices tensor.
     *        Its output shape is the same as the shape of indices and consists of one value
     *        (gathered from the data) for each element in indices.
     * @param[in] input  A "TensorPtr", which to gather values.
     * @param[in] indices A TensorPtr, specifying fetch index of @p input.
     * @param[in] axis Which axis to gather on. Negative value means counting dimensions from the back.
     *                 Accepted range is [-r, r-1] where r = rank(data).
     * @return  A tensor pointer with the same type as the tensors @p input.
     */
    TensorPtr gather_elements(TensorPtr input, TensorPtr indices, int32_t axis = 0);

    /**
     * @brief Given the input indices tensor, then gather slices from input tensor X into the output tensor Y
     *         with shape specified by indices. The output tensor has the shape `indices.shape[ :-1] + X.
     *         shape[indices.shape[-1] + batch_dims: ]` with the left closed and right open interval.
     *         Generally, `indices[-1] + batch_dims` should be less than or equal to `rank(X)`. Note that
     *         the negative value in indices tensor means counting the index from the back.
     * @param[in] input  A "TensorPtr", which to gather values.
     * @param[in] indices A tensor pointer with type "int32" containing the indices to be looked up in @p input.
     * @param[in] batch_dims A scalar of integer to indicate the number of batch dimensions.
     *                       Generally, defaults to 0 if not present.
     *                       Currently only support batch_dims=0.
     * @return  A tensor pointer with the same type as the tensors @p input.
     */
    TensorPtr gather_nd(TensorPtr input,
                        TensorPtr indices,
                        int32_t batch_dims = 0);

    /**
     * @brief Computes the Gaussian Error Linear Unit (GELU) activation function.
     *        GELU computes x * P(X <= x), where P(X) ~ N(0, 1). The (GELU) nonlinearity weights inputs by their values,
     *        rather than gates inputs by their signs as in ReLU.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] approximate A bool, whether to enable approximation.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr gelu(TensorPtr input, bool approximate = false, const Quantization &quant = {});

    /**
     * @brief General matrix multiplication. Here,
     *        A' = transpose(A) if 'trans_a' else A.
     *        B' = transpose(B) if 'trans_b' else B.
     *        Y = alpha * (A') * (B') + beta * C.
     *        Where A (or B) should be transposed before performing the computation if 'trans_a' (or 'trans_b') is 'true'.
     *        The input tensor C is optional, and can be broadcast to the shape (M, N).
     *        Note that, in the following Inputs or Outputs sections:
     *            A means the first tensor with a shape (M, K) if 'trans_a' is 'false',
     *              or a shape (K, M) if 'trans_a' is 'true'.
     *            B means the first tensor with a shape (K, N) if 'trans_b' is 'false',
     *              or a shape (N, K) if 'trans_b' is 'true'.
     *            C means the optional tensor whose shape should be unidirectional broadcastable to (M, N).
     *               If not specified, the computation is performed after matrix multiplication.
     *            Y means the output tensor with a shape (M, N).
     * @param[in] A A "TensorPtr", which is input data tensor A.
     * @param[in] B A "TensorPtr", which is input data tensor B.
     * @param[in] C A "TensorPtr", which is optional data tensor C.
     * @param[in] alpha The scalar multiplier for the product of input tensors (A'*B').
     * @param[in] trans_a Indicates whether input tensor A should be transposed.
     * @param[in] trans_b Indicates whether input tensor B should be transposed.
     * @param[in] beta The scalar multiplier for the optional input tensors C.
     * @param[in] quant  A "Quantization" object, specifying the output quantization parameters.
     * @return  A tensor pointer with the shape [M,N].
     */
    TensorPtr gemm(TensorPtr A,
                   TensorPtr B,
                   TensorPtr C = nullptr,
                   const float alpha = 1.0,
                   const bool trans_a = false,
                   const bool trans_b = false,
                   const float beta = 1.0,
                   const Quantization &quant = {});

    /**
     * @brief Generate aixs-aligned bounding box proposals.
     *        Bounding box proposals are generated by applying transformation on a set of predefined anchors with
     *        the bounding box deltas from bounding box regression.
     *        A final step of hard NMS is applied to limit the number of returned boxes.
     * @param[in] inputs A "TensorPtr" list.
     *            It has 'scores','deltas','anchors',and optinal input 'image_info'.
     *            The 'scores' is a 4-D Tensor specifying the score of each anchor at each location.
     *            With "NHWC" data layout, the tensor shape is [batches, height, width, num_anchors].
     *            With "NCHW" data layout, the tensor shape is [batches, num_anchors, height, width].
     *            THe 'deltas'is a 4-D Tensor specifying the bounding box deltas.
     *            With "NHWC" data layout, the tensor shape is [batches, height, width, num_anchors * 4].
     *            With "NCHW" data layout, the tensor shape is [batches, num_anchors * 4, height, width].
     *            The box deltas are encoded in the order of [dx, dy, dw, dh],where dx and dy is the linear-scale
     *            relative correction factor for the center position of the bounding boxwith respect to
     *            the width and height,dw and dh is the log-scale relative correction factor for the width and height.
     *            The last dimensions is the channel dimension.
     *            The 'anchors' is a 2-D Tensor of shape [num_anchors, 4],
     *            specifying the shape of each predefined anchor, with format [x1, y1, x2, y2].
     *            The 'image_info' is a optinal 2-D Tensor,specifying the image info, size [batches, 3],
     *            format (height, width, scale).
     * @param[in] pre_max_num An int32 scalar,
     *            specifying the maximum number of boxes before going into the hard NMS algorithm.
     *            Boxes with the lowest scores are discarded to meet the limit.
     *            Set to a non-positive value for unlimited number.
     * @param[in] post_max_num An int32 scalar,
     *            specifying the maximum number of boxes returning from the hard NMS algorithm.
     *            Boxes with the lowest scores are discarded to meet the limit.
     *            Set to a non-positive value for unlimited number.
     * @param[in] iou_threshold A float scalar, specifying the IoU threshold for hard NMS.
     * @param[in] min_size A float scalar,
     *            specifying boxes with height or width lower than the absolute threshold are filtered out.
     * @param[in] num_output_rois An int32 scalar,
     *            specifying the number of output rois, which can be an expect max number output rois.
     * @param[in] image_info  specifying the size of each image in the batch,
     *            with format [image_height, image_width].
     * @param[in] stride_info specifying the ratio from the  original image to the feature map.
     *            with format [stride_height, stride_width].
     * @return A vector of "TensorPtr".
     *         First "TensorPtr" of shape [batches, num_output_rois], specifying the score of each output box.
     *         The boxes are grouped by batches, but the sequential order in each batch is not guaranteed.
     *         Data type and scale/offset will be same with @p scores.
     *         Second "TensorPtr" of shape [batches, num_output_rois, 4], specifying the coordinates of each output bounding box for each class,
     *         with format [x1, y1, x2, y2]. The sequential order of the boxes corresponds with first "TensorPtr".
     *         Data type will be same with @p anchors. Scale and offset must be same with @p anchors.
     *         Third "TensorPtr" of shape [batches, num_output_rois], specifying the batch index of each box. Boxes with the same batch index are grouped together.
     *         Forth "TensorPtr" of shape [1], specifying the valid number of output rois.
     */
    TensorPtrList generate_proposals(TensorPtrList inputs,
                                     int32_t pre_max_num = -1,
                                     int32_t post_max_num = -1,
                                     float iou_threshold = 0.1,
                                     float min_size = 1.0,
                                     int32_t num_output_rois = 100,
                                     std::vector<int32_t> image_info = {},
                                     std::vector<float> stride_info = {});
    /**
     * @brief Get valid count of bounding boxes given a score threshold. Also moves valid boxes to the top of input data.
     *        Note that, in the following Inputs or Outputs sections:
     *        X1 means input data with shape [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].
     *        Y1 means the valid number of boxes.
     *        Y2 means the rearranged data tensor.
     *        Y3 means the related index in input data.
     * @param[in] input A "TensorPtr", specifying input tensor X1.
     * @param[in] score_threshold The of score for valid bounding boxes.
     * @param[in] id_index Optional index of the class categories, -1 to disable.
     * @param[in] score_index Optional index of the scores/confidence of boxes.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2,Y3.
     */
    std::vector<TensorPtr> get_valid_counts(TensorPtr input,
                                            float score_threshold,
                                            int32_t id_index = 0,
                                            int32_t score_index = 1);

    /**
     * @brief Returns the truth value of (input1 > input2) element-wise.
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "BOOL".
     */
    TensorPtr greater(TensorPtr input1,
                      TensorPtr input2,
                      const Quantization &quant = {});

    /**
     * @brief Returns the truth value of (input1 >= input2) element-wise.
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "BOOL".
     */
    TensorPtr greater_equal(TensorPtr input1,
                            TensorPtr input2,
                            const Quantization &quant = {});

    /**
     * @brief Given an input and a flow-field grid, computes the output using input values and
     *        pixel locations from grid.
     * @param[in] input A "TensorPtr", specifying input data tensor with shape (N, H, W, C) or (N, D, H, W, C).
     * @param[in] grid A "TensorPtr", specifying grid data tensor with shape (N, H_out, W_out, 2) or (N, D_out, H_out, W_out, 3).
     * @param[in] align_corners If align corners=True, the extrema (-1 and 1) are considered as
     *                          referring to the center points of The input's corner pixels.
     *                          If align corners=False, they are instead considered as referring
     *                          to the corner points of the input's corner pixels, making the
     *                          sampling more resolution agnostic. Defaults to False.
     * @param[in] method An enum value of GridSampleMethod specifying interpolation mode. Four
     *                   interpolation modes are supported: bilinear, linear, nearest, and bicubic.
     *                   The mode 'bilinear' only for 4-D input.
     *                   The mode 'linear' only for 5-D input.
     *                   Defaults to bilinear.
     * @param[in] padding_mode A enum value of GridSamplePaddingMode indicating outside grid values.
     *                         Supported the following three padding modes.
     *                          zeros: Use O for out-of-bound grid locations. Default mode.
     *                          border: Use border values for out-of-bound grid locations.
     *                          reflection: Use values at locations reflected by the border for out-of-bound grid locations.
     * @param[in] quantization Optional "Quantization" object, specifying the output data scale and offset.
     *                         If not set, used default.
     * @return A "TensorPtr", with shape (N, H_out, W_out, C) or (N, D_out, H_out, W_out, C).
     */
    TensorPtr grid_sample(TensorPtr input,
                          TensorPtr grid,
                          bool align_corners,
                          GridSampleMethod method,
                          GridSamplePaddingMode padding_mode,
                          const Quantization &quantization = {});

    /**
     * @brief Computes the activation normalization of the previous layer for the given axis in a batch and a group independently.
     *        That is, y = gamma * (x -mean) / sqrt(variance + epsilon) + beta,
     *        where mean and variance are computed under all effective axis within each group.
     *        By default, the shape of 'gamma' and 'beta' is the batch size in NHWC format.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] gamma A "TensorPtr", specifying the gamma constant tensor.
     *                  Generally, the shape of gamma is the shape mutilation of the input channel dimension.
     * @param[in] beta A "TensorPtr", specifying the beta constant tensor.
     *                 Generally, the shape of beta is the shape mulation of the input channel dimension.
     * @param[in] axis A integer scalar to indicate the axis to normalize across.
     *                 Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
     *                 Generally, the value is -1 which represents the last dimension of the input tensor (X).
     * @param[in] group Integer, the number of groups for Group Normalization.
     *                  It can be in the range [1, N] where N is the input dimension.
     *                  The input dimension must be divisible by the number of groups. The default value is 32.
     * @param[in] epsilon A float IR parameter. The epsilon value is used to avoid division by zero.
     *                    Generally, the default value is less than 10-5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input. Data type is inferred by @p quant.
     *         Output scale and offset are described in @p quant.
     * @todo runtime cannot provide mean and var.
     */
    TensorPtr group_normalization(TensorPtr input,
                                  TensorPtr gamma,
                                  TensorPtr beta,
                                  int32_t axis = -1,
                                  int32_t group = 32,
                                  float epsilon = 1e-5f,
                                  const Quantization &quant = {});

    /**
     * @brief Computes a one-layer GRU(gated recurrent unit).
     *        Difference with GRUv3, biases contains two hidden gates, When calculating `ht`, a bn in biases will be added.
     *      Notations:
     *          x: Input tensor
     *          z: Update gate
     *          r: Reset gate
     *          h: Hidden gate
     *          t: Time step (t-1 means the previous time step)
     *          W[r, z, c]: Parameter weight matrix for reset, update and hidden gates
     *          R[r, z, c]: Recurrence weight matrix for reset, update and hidden gates
     *          b[r, z, c, n]: Bias vectors for reset, update, and two hidden gates
     *          H: Hidden state
     *          f: Activation function
     *          g: Activation function
     *      Activation functions:
     *          Relu: - max(0, x)
     *          Tanh: - (1 – e^{-2x}) / (1 + e^{-2x})
     *          Sigmoid: -1/(1 + e^{-x})
     *          Affine: - alpha * x + beta
     *          LeakyRelu: -x if x >= 0 else alpha * x
     *          ThresholdedRelu: -x if x >= alpha else 0
     *          HardSigmoid: - min(max (alpha * x + beta, 0), 1)
     *          Elu: -x if x >= 0 else alpha * (e^x -1)
     *          Softsign: -x / (1 + |x|)
     *          Softplus: - log(1 + e^x)
     *          Clip: clip_max if x >= clip_max, x if clip_min < x < clip_max, clip_min if x <= clip_min
     *      Equations (Default: f = SIGMOID, g = TANH):
     *          rt = f(Wr * xt + Rr * H(t-1) + br);
     *          zt = f(Wz * xt + Rz * H(t-1) + bz);
     *          ht = g(Wc * xt + (Rc * H(t-1) + bn) ʘ rt + bc);
     *          Ht = zt ʘ Ht-1 + (1 - zt) ʘ ht.
     *         Where ʘ means element-wise multiplication (or Hadamard product) and 1 (in bold) means
     *         all the j-th element in a vector is one. During computation, this operator has some optional inputs.
     *         An empty string means using the default value or not specified arguments.
     * @param[in] input A "TensorPtr",with the shape as [batch size, time step, input size].
     * @param[in] h0   A "TensorPtr",means the initial hidden state for each element in the batch
     *                 with the shape as [batch size, cell size].
     * @param[in] weights The weight tensor that will be used in the computation. Where, weights_type, weights_offset, weights_size,
     *                    and weights_shape mean the weights data format, weights address offset, total weights size (counted in bytes),
     *                    and weights shape of the current layer. The corresponding equation is weights_size = data_type * 3 * h * (x + h),
     *                    weights_shape = [3 * h, x + h]. Here, data type = 1 (or 2, 4) if weight_type = int8 (or int16, int32).
     *                    Note that 'wight_shape' is stacked in order of
     *                     [[Wr0 , Rr0], [Wz0 , Rz0], [Wc0 , Wc0]], ..., [[Wr(h-1) , Rr(h-1]], [Wz(h-1] , Rz(h-1)], [Wc(h-1) , Wc(h-1)]]
     *                    (that is, weights_shape = [3*h, x+h]). Where, 'x' and 'h' mean the input size and cell size as described in the preceding section.
     * @param[in] biases   A "TensorPtr",Used to describe parameter "b[r, z, c, n]",
     *                     with the shape as [4*cell size].
     *                     b[r, z, c, n]: Bias vectors for reset, update, and two hidden gates.
     * @param[in] out_sequence The output sequence is one of enumerated lists and in a fixed order.
     *                          Here, enumerated list includes [H], [Hn], [H, Hn].
     *                          Hn means the hidden state for t= time_step with the shape as [batch size, cell size].
     *	                        H means all hidden state with the shape as [batch size, time step, cell_size].
     * @param[in] activations A list of activation functions for update, reset and hidden gates in an order (f, g) in GRU.
     *                        The activation must be one of the activation functions defined above.
     * @param[in] direction Indicates that the RNN is forward or reverse. It must be one of forward as default, or reverse.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters.
     * @return A vector of "TensorPtr",Depending on parameter "out_sequence".
     *         If the parameter "out_sequence" is empty, return value is [Hn].
     */
    std::vector<TensorPtr> gru_v1(TensorPtr input,
                                  TensorPtr h0,
                                  TensorPtr weights,
                                  TensorPtr biases,
                                  const GruOutSequence &out_sequence = GruOutSequence::Hn,
                                  const std::vector<Activation> &activations = {Activation("Sigmoid"), Activation("Tanh")},
                                  const GruDirection &direction = GruDirection::Forward,
                                  const Quantization &quant = {});

    /**
     * @brief Computes a one-layer GRU(gated recurrent unit).
     *        Difference with GRUv1, biases contains one hidden gates, it don't contains bn.
     *      Notations:
     *          x: Input tensor
     *          z: Update gate
     *          r: Reset gate
     *          h: Hidden gate
     *          t: Time step (t-1 means the previous time step)
     *          W[r, z, c]: Parameter weight matrix for reset, update and hidden gates
     *          R[r, z, c]: Recurrence weight matrix for reset, update and hidden gates
     *          b[r, z, c]: Bias vectors for reset, update and hidden gates
     *          H: Hidden state
     *          f: Activation function
     *          g: Activation function
     *      Activation functions:
     *          Relu: - max(0, x)
     *          Tanh: - (1 – e^{-2x}) / (1 + e^{-2x})
     *          Sigmoid: -1/(1 + e^{-x})
     *          Affine: - alpha * x + beta
     *          LeakyRelu: -x if x >= 0 else alpha * x
     *          ThresholdedRelu: -x if x >= alpha else 0
     *          HardSigmoid: - min(max (alpha * x + beta, 0), 1)
     *          Elu: -x if x >= 0 else alpha * (e^x -1)
     *          Softsign: -x / (1 + |x|)
     *          Softplus: - log(1 + e^x)
     *          Clip: clip_max if x >= clip_max, x if clip_min < x < clip_max, clip_min if x <= clip_min
     *      Equations (Default: f = SIGMOID, g = TANH):
     *          rt = f(Wr * xt + Rr * H(t-1) + br);
     *          zt = f(Wz * xt + Rz * H(t-1) + bz);
     *          ht = g(Wc * xt + Rc * (H(t-1) ʘ rt) + bc);
     *          Ht = zt ʘ H(t-1) + (1 - zt) ʘ ht
     *         Where ʘ means element-wise multiplication (or Hadamard product) and 1 (in bold) means
     *         all the j-th element in a vector is one. During computation, this operator has some optional inputs.
     *         An empty string means using the default value or not specified arguments.
     * @param[in] input A "TensorPtr",with the shape as [batch size, time step, input size].
     * @param[in] h0   A "TensorPtr",means the initial hidden state for each element in the batch
     *                 with the shape as [batch size, cell size].
     * @param[in] weights The weight tensor that will be used in the computation. Where, weights_type, weights_offset, weights_size,
     *                    and weights_shape mean the weights data format, weights address offset, total weights size (counted in bytes),
     *                    and weights shape of the currentlayer. The corresponding equation is weights_size = data_type * 3 * h * (x + h),
     *                    weights_shape = [3 * h, x + h]. Here, data type = 1 (or 2, 4) if weight_type = int8 (or int16, int32).
     *                    Note that 'wight_shape' is stacked in order of
     *                    [[Wr0 , Rr0], [Wz0 , Rz0], [Wc0 , Wc0]], ..., [[Wr(h-1) , Rr(h-1]], [Wz(h-1] , Rz(h-1)], [Wc(h-1) , Wc(h-1)]]
     *                    (that is, weights_shape = [3*h, x+h]). Where, 'x' and 'h' mean the input size and cell size as above.
     * @param[in] biases   A "TensorPtr",Used to describe parameter "b[r, z, c]",
     *                     with the shape as [3*cell size].
     *                     b[r, z, c]: Bias vectors for reset, update and hidden gates.
     * @param[in] out_sequence The output sequence is one of enumerated lists and in a fixed order.
     *                          Here, enumerated list includes [H], [Hn], [H, Hn].
     *                          Hn means the hidden state for t= time_step with the shape as [batch size, cell size].
     *	                        H means all hidden state with the shape as [batch size, time step, cell_size].
     * @param[in] activations A list of activation functions for update, reset and hidden gates in an order (f, g) in GRU.
     *                        The activation must be one of the activation functions defined above.
     * @param[in] direction Indicates that the RNN is forward or reverse. It must be one of forward as default, or reverse.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters.
     * @return A vector of "TensorPtr",Depending on parameter "out_sequence".
     *         If the parameter "out_sequence" is empty, return value is [Hn].
     */
    std::vector<TensorPtr> gru_v3(TensorPtr input,
                                  TensorPtr h0,
                                  TensorPtr weights,
                                  TensorPtr biases,
                                  const GruOutSequence &out_sequence = GruOutSequence::Hn,
                                  const std::vector<Activation> &activations = {Activation("Sigmoid"), Activation("Tanh")},
                                  const GruDirection &direction = GruDirection::Forward,
                                  const Quantization &quant = {});

    /**
     * @brief Alias of gru_v3.
     */
    std::vector<TensorPtr> gru(TensorPtr input,
                               TensorPtr h0,
                               TensorPtr weights,
                               TensorPtr biases,
                               const GruOutSequence &out_sequence = GruOutSequence::Hn,
                               const std::vector<Activation> &activations = {Activation("Sigmoid"), Activation("Tanh")},
                               const GruDirection &direction = GruDirection::Forward,
                               const Quantization &quant = {});

    /**
     * @brief Applies the hardsigmoid function element-wise.
     *        The formula is, HardSigmoid(x) = max(0, min(1, alpha * X + beta)).
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] alpha A float scalar, specifying the alpha in the formula, default is 0.1667.
     * @param[in] beta A float scalar, specifiying the beta in the formula, defulat is 0.5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset, default is empty.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr hard_sigmoid(TensorPtr input,
                           float alpha = 0.1667,
                           float beta = 0.5,
                           const Quantization &quant = {});

    /**
     * @brief Computes hard-swish activation on the input tensor element-wise, formula:
     *        h-swish(x) = x * max(0, min(6, (x + 3))) / 6.
     * @param[in] input A "TensorPtr", specifying the input.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters.
     * @return A TensorPtr, specifying the output tensor.
     */
    TensorPtr hard_swish(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Looks up sub-tensors in the input tensor using a key-value map.
     * @param[in] value  A "TensorPtr", specifying the featuremap.
     * @param[in] key    A "TensorPtr", which is a one-dimensional tensor that maps these values to
     *                   value indexes.type:INT32;
     * @param[in] lookup A "TensorPtr", which is a one-dimensional tensor pointer of selection values.
     *                   type:INT32;
     * @return A vector of type TensorPtr,include output and hits.
     *         output: quantization must be the same as value.
     *         hits:A bool "TensorPtr" indicates whether the lookup hits (True) or not (False).
     */
    std::vector<TensorPtr> hashtable_lookup(TensorPtr value,
                                            TensorPtr key,
                                            TensorPtr lookup);

    /**
     * @brief Localize the maximum keypoints from heatmaps, This operation approximates the accurate maximum keypoint scores and
     *        indices after bicubic upscaling by using Taylor expansion up to the quadratic term.
     * @param[in] heatmaps A "TensorPtr", which is 4-D Tensor of shape [num_boxes, heatmap_size, heatmap_size, num_keypoints],
     *                     specifying the heatmaps, the height and width of heatmaps should be the same, and must be greater than or equal to 2.
     * @param[in] bounding_boxes A "TensorPtr", which is 2-D Tensor of shape [num_boxes, 4], specifying the bounding boxes, each with format [x1, y1, x2, y2]
     * @param[in] is_softmax A boolean variable, which means whether there is a softmax after heatmap_max_keypoints compute
     * @param[in] score_quant A "Quantization", should be initialized as {scale, offset}, these paramters can be different from heatmaps,
     *                        these quantization params are for first output
     * @param[in] location_quant A "Quantization", should be initialized as {scale, offset}, shoule be fixed as {0.125, 0}
     *                           these quantization params are for second output
     * @param[in] softmax_quant A "Quantization", if @p is_softmax is false, just ignore this paramter.
     * @return Vector of "TensorPtr", first is score_of_keypoints. Data type will have the same bitwidth with @p heatamps.
     *         Scale and offset can be different from @p heatmap.
     *         Second is location_of_keypoints, data type and scale/offset will be same with @p bounding_boxes.
     *         Third tensor is softmax for first output tensor, if @p is_softmax is false, it will be set all zero.
     */
    std::vector<TensorPtr> heatmap_max_keypoint(TensorPtr heatmaps,
                                                TensorPtr bounding_boxes,
                                                bool is_softmax = false,
                                                const Quantization &score_quant = {},
                                                const Quantization &location_quant = {},
                                                const Quantization &softmax_quant = {});

    /**
     * @brief            Create a input tensor pointer.Any graph must create input tensor first.
     * @param[in] shape  The input shape.
     * @param[in] dtype  A "Tensor::Dtype", specifying the primitive data type.
     *                   Optional parameter, the default value is Tensor::Dtype::FP32;
     * @param[in] quant  A "Quantization" object, specifying the weight scale/offset.
     *                   Optional parameter, the default  is empty;
     * @param[in] g      graph pointer to which input belongs.
     *                   If the parameter "g" does not have or is nullptr, the default static graph will be used.
     * @param[in] enable_ds whether enable dynamice shape, if true, will automatically add symbol to generated tensor.
     * @param[in] exprs The dynamic shape expression.
     * @return A "TensorPtr", specifying the weight tensor.
     */
    TensorPtr input(const TensorShape &new_shape,
                    Tensor::Dtype dtype = Tensor::Dtype::FP32,
                    const Quantization &quant = {},
                    GraphPtr g = nullptr,
                    bool enable_ds = false,
                    std::vector<tongue::Expr> exprs = {});

    /**
     * @brief Create input node, return output tensor of input node, bind this node to @p g,
     *        if @p g is not provided, get default global graph.
     * @param[in] t "TensorPtr" of input node.
     * @param[in] g Graph pointer.
     * @param[in] enable_ds whether enable dynamice shape, if true, will automatically add symbol to generated tensor.
     * @param[in] exprs The dynamic shape expression.
     * @return A "TensorPtr", specifying output tensor of input node.
     */
    TensorPtr input(TensorPtr t, GraphPtr g = nullptr, bool enable_ds = false,
                    std::vector<tongue::Expr> exprs = {});

    /**
     * @brief Performs the input tensor (X) with 'batch_size' classes to check whether the targets are in the top k predictions,
     *        and outputs a 'batch_size' bool array.
     *        That is, the output Y[i] returns 'true' if the predication for the target class exists among the top 'k' predications.
     *        Y[i] = 'true' if predications['target_classes_id (Si)'] ϵ {top-k predications}
     *        Note that these classes are in the top 'k' when multiple classes have the same predication value and
     *        straddle the top 'k' boundary. For example,
     *        X1 = [[0.1, 0.8, 0.7, 0.7], [0.1, 0.6, 0.5, 0.4]], X2 = [3, 3], then returns Y = [true, false].
     * @param[in] input1 A "TensorPtr", specifying the input tensor(X1).
     * @param[in] input2 A "TensorPtr", specifying the target(X2) .
     * @param[in] k A scalar of integer to indicate the number of top elements to look at for computing precision.
     * @return A "TensorPtr", specifying output indices tensor.
     */
    TensorPtr in_topk(TensorPtr input1, TensorPtr input2, int32_t k);

    /**
     * @brief Carries out instance normalization. That is, y = gamma * (x -mean) / sqrt(variance + epsilon) + beta,
     *        where mean and variance are computed per instance per channel.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] gamma A "TensorPtr", specifying the gamma constant tensor.
     *                  Generally, the shape of gamma is the shape mutilation of the input channel dimension.
     * @param[in] beta A "TensorPtr", specifying the beta constant tensor.
     *                 Generally, the shape of beta is the shape mulation of the input channel dimension.
     * @param[in] epsilon A float IR parameter. The epsilon value is used to avoid division by zero.
     *                    Generally, the default value is less than 10-5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input. Data type is inferred by @p quant.
     *         Output scale and offset are described in @p quant.
     * @todo runtime cannot provide mean and var.
     */
    TensorPtr instance_normalization(TensorPtr input,
                                     TensorPtr gamma,
                                     TensorPtr beta,
                                     float epsilon = 1e-5f,
                                     const Quantization &quant = {});

    /**
     * @brief If conditional.
     * @param[in] inputs Include condition tensor, 'then' branch inputs, 'else' branch inputs.
     * @param[in] then_branch GraphPtr to run if condition is true.
     *              Has N outputs: values you wish to be live-out to the enclosing scope.
     *              The number of outputs must match the number of outputs in the else_branch.
     * @param[in] else_branch GraphPtr to run if condition is false.
     *            Has N outputs: values you wish to be live-out to the enclosing scope.
     *            The number of outputs must match the number of outputs in the then_branch.
     * @return A list of "TensorPtr", specifying the then_branch output tensor list.
     */
    TensorPtrList If(TensorPtrList inputs, GraphPtr then_branch, GraphPtr else_branch);

    /**
     * @brief L1Pooling2D consumes an input tensor X and applies L1 pooling across the tensor according to kernel sizes,
     *        stride sizes, and pad lengths. L1 pooling consisting of computing the L1 norm on all values of a subset of
     *        the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.
     * @param[in] input  A "TensorPtr".
     *                   Support the shape [batch,height,width,channel].
     * @param[in] kernel  The size of the kernel along each axis. The 'kernel' parameter format [kernel_y,kernel_x].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format [stride_y,stride_x].
     * @param[in] dilations Dilation of filter along each spatial dimension. If not present, defaults to 1 along each spatial dimensions.
     *                      The 'dilations' parameter format [dilation_y,dilation_x].
     * @param[in] padding The "padding" must be either "VALID" or "SAME".
     *                 - SAME: calculate explicit padding.
     *                 - VALID: Fill 0.
     * @param[in] ceil_mode Means using Ceil or Floor function to compute the output shape.
     *                      Generally, the default value is "false" if not specified, which means using floor function.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs.
     */
    TensorPtr l1pooling2d(TensorPtr input,
                          const std::vector<int32_t> &kernel,
                          const std::vector<int32_t> &strides,
                          const std::vector<int32_t> &dilations,
                          Padding padding = Padding("VALID"),
                          bool ceil_mode = false,
                          const Quantization &quant = Quantization());

    /**
     * @brief Given a matrix, apply L1-normalization along the provided axis.
     * @param[in] input A "TensorPtr", specifying the input tensor to be normalized.
     * @param[in] axis An int32 vector, A list of integers, indicates the axis to normalize across. Default is -1.
     * @param[in] epsilon A float IR parameter. The epsilon value is used to avoid division by zero.
     *                    Generally, the default value is less than 10-5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the tensor which has the same data type and shape with @p input.
     *         Scale and offset are described in @p quant.
     */
    TensorPtr l1_normalization(TensorPtr input,
                               const std::vector<int32_t> &axis = {-1},
                               float epsilon = 1e-5f,
                               const Quantization &quant = {});

    /**
     * @brief Normalizes along dimension axis using an L2 norm. Formula can describe as: y = x / sqrt(max(sum(x**2), epsilon))
     * @param[in] input A "TensorPtr", specifying the input tensor to be normalized.
     * @param[in] axis An int32 vector, A list of integers, indicates the axis to normalize across. Default is -1.
     * @param[in] epsilon A float IR parameter. The epsilon value is used to avoid division by zero.
     *                    Generally, the default value is less than 10-5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the tensor which has the same data type and shape with @p input.
     *         Scale and offset are described in @p quant.
     */
    TensorPtr l2_normalization(TensorPtr input,
                               const std::vector<int32_t> &axis = {-1},
                               float epsilon = 1e-5f,
                               const Quantization &quant = {});

    /**
     * @brief Computes the activation normalization of the previous layer for the given axis in a batch independently.
     *        That is, y = gamma * (x - mean) / sqrt(variance + epsilon) + beta, where mean and variance are computed under all effective axis.
     *        The computation required by standardization can be described by the following equations.
     *        Mean = ReduceMean<axes=normalized_axes>(X)
     *        D = Sub(X, Mean)
     *        DD = Mul(D, D)
     *        Var = ReduceMean<axes=normalized_axes>(DD)
     *        VarEps = Add(Var, epsilon)
     *        StdDev = Sqrt(VarEps)
     *        InvStdDev = Reciprocal(StdDev)
     *        Normalized = Mul(D, InvStdDev)
     *        where normalized_axes is [axis, ..., rank of X - 1].
     *        The variables Var and StdDev stand for variance and standard deviation, respectively.
     *        The second output is Mean and the last one is InvStdDev.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] gamma A "TensorPtr", specifying the gamma constant tensor.
     *                  Generally, the shape of gamma is the shape mutilation of the input dimension that 'axis' specifies.
     * @param[in] beta A "TensorPtr", specifying the beta constant tensor.
     *                 Generally, the shape of beta is the shape mulation of the input dimension that 'axis' specifies.
     *                 If there has no gamma and beta, just set them as nullptr.
     * @param[in] axis  An int32 vector, A list of integers, indicates the axis to normalize across.
     *                  Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
     *                  By default, the value is -1 which represents the last dimension of the @p input.
     *                  Note that this parameter should be continuous from the last dimension.
     *                  For example, axis = [1, 2, 3] means statistics are across the Height, Width and Channel dimensions in NHWC format.
     * @param[in] epsilon A float IR parameter. The epsilon value is used to avoid division by zero.
     *                    Generally, the default value is less than 10-5.
     * @param[in] stash_type Type of Mean and InvStdDev. This also specifies stage one's computation precision.default is true.
     * @param[in] out_mean_invstddev Whether to output Mean and InvStdDev.default is false.
     * @param[in] quants A "Quantization" list, specifying the scale and offset of [output, invstddev].
     *                   Its number can be 0~2.
     * @return A "TensorPtr" list, support lenth (1~3), [Y], or [Y, Mean, InvStdDev].
     */
    std::vector<TensorPtr> layer_normalization(TensorPtr input,
                                               TensorPtr gamma = nullptr,
                                               TensorPtr beta = nullptr,
                                               const std::vector<int32_t> &axis = {-1},
                                               float epsilon = 1e-5f,
                                               bool stash_type = true,
                                               bool out_mean_invstddev = false,
                                               const std::vector<Quantization> &quants = {});

    /**
     * @brief LeakyRelu takes @p input and an argument @p negative_slope, and produces one output Tensor
     *        where the function f(x) = negative_slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] negative_slope A scalar, controlling the angle of the non-quantized negative slope. Default is 0.01.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input. Output data type will be
     *         inferred by @p quant, and output scale/offset are described as @p quant.
     */
    TensorPtr leaky_relu(TensorPtr input, NodeParamValue negative_slope = 0.01, const Quantization &quant = {});

    /**
     * @brief The left shift operator performs element-wise operations. For each input element, this operator moves its binary representation
     *        toward the left side so that the input value is effectively increased.
     *        The input X is the tensor to be shifted and another input Y specifies the amounts of shifting.
     *        For example, X is [1, 4], and Y is [1, 1], the corresponding output Z will be [2, 8]. This operator supports multidirectional broadcasting.
     * @param[in] input A "TensorPtr", specifying the tensor needs to be shifted.
     * @param[in] shift A "TensorPtr", specifying the shift tensor.
     * @param[in] quant A "Quantization" object, specifying the output quantization paramemters.
     * @return A "TensorPtr", specifying shifted tensor.
     */
    TensorPtr left_shift(TensorPtr input, TensorPtr shift, const Quantization &quant = {});

    /**
     * @brief Returns the truth value of (input1 < input2) element-wise.
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @param[in] quant A "Quantization" object, specifying the output quantization paramemters.
     * @return A "TensorPtr",with the same shape as the tensors in "input1",is of type "BOOL".
     */
    TensorPtr less(TensorPtr input1,
                   TensorPtr input2,
                   const Quantization &quant = {});

    /**
     * @brief Returns the truth value of (input1 <= input2) element-wise.
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "BOOL".
     */
    TensorPtr less_equal(TensorPtr input1,
                         TensorPtr input2,
                         const Quantization &quant = {});

    /**
     * @brief Computes natural logarithm of x element-wise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr log(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Returns the truth value of (input1 && input2) element-wise.
     *        example: input1 = [False, False, True, True]
     *                 input2 = [False, True, False, True]
     *                 return [False, False, False, True].
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "UINT8".
     */
    TensorPtr logical_and(TensorPtr input1,
                          TensorPtr input2);

    /**
     * @brief Returns the truth value of (input1 || input2) element-wise.
     *        example: input1 = [False, False, True, True]
     *                 input2 = [False, True, False, True]
     *                 return [False, True, True, True].
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "UINT8".
     */
    TensorPtr logical_or(TensorPtr input1,
                         TensorPtr input2);

    /**
     * @brief Returns the truth value of (input1 ^ input2) element-wise.
     *        example: input1 = [False, False, True, True]
     *                 input2 = [False, True, False, True]
     *                 return [False, True, True, False].
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "UINT8".
     */
    TensorPtr logical_xor(TensorPtr input1,
                          TensorPtr input2);

    /**
     * @brief Returns the truth value of (! input) element-wise.
     *        example: input1 = [False, False, True, True]
     *                 return [True, True, False, False].
     * @param[in] input A "TensorPtr".
     * @return A "TensorPtr",with the same shape as the tensors in "input", is of type "UINT8".
     */
    TensorPtr logical_not(TensorPtr input);

    /**
     * @brief Generic Looping construct.
     *        The operation is consistent with onnx 'Loop' behavior:
     *        https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop
     * @param[in] inputs A "TensorPtr" list, specifying the input tensor.
     *                   All input tensors have the same shape as the input tensor of the body.
     * @param[in] body The graph run each iteration.
     *                 It has 2+N inputs: (iteration_num, condition, loop carried dependencies...).
     *                 It has 1+N+K outputs: (condition, loop carried dependencies..., scan_outputs...).
     *                 Each scan_output is created by concatenating the value of the specified output value
     *                 at the end of each iteration of the loop.
     *                 It is an error if the dimensions or data type of these scan_outputs change across
     *                 loop iterations.
     * @param[in] dependency_inputs_num loop carried dependencies size (N).
     * @param[in] scan_outputs_num scan_outputs size (K).
     * @param[in] keep_out_dim Keep the scan output dimension unchanged.
     * @param[in] max_loop_num If iteration_num is not a constant, max_loop_num is used to infer the shape.
     *                         Default value is 100.
     * @return A "TensorPtr" list. Final N loop carried dependency values then K scan_outputs.
     *         Scan outputs must be Tensors.
     */
    TensorPtrList loop(TensorPtrList inputs,
                       GraphPtr body,
                       int32_t dependency_inputs_num,
                       int32_t scan_outputs_num,
                       bool keep_out_dim = false,
                       int32_t max_loop_num = 100);

    /**
     * @brief It normalizes over local input regions.
     *        The normalization function 'Y[i] = X[i]/ (bias + (alpha/size) * square_sum(X[j]))^beta',
     *        is applied to the tensor elementwise. There are two method of 'ACROSS_CHANNELS'
     *        and 'WITHIN_CHANNEL' during local region normalization.
     *        When method is ACROSS_CHANNELS,
     *        The local region of 'square_sum' is defined as
     *        '{max(0, c - depth_radius ) <= i <= min(C - 1, c + depth_radius)}' under NHWC data format.
     *        Also, square_sum[i] = sum(input tensor X[N, H, W, i - depth_radius : i + depth_radius +1]^ 2).
     *        When method is WITHIN_CHANNEL,
     *        The local region is rectangular and defined as 'max{0, h - depth_radius} <= y <= min(H-1, h + depth_radius+1)},
     *        max{0, w - depth_radius} <= x <= min(W-1, w + depth_radius+1)}' under NHWC data format.
     *        Also, square_sum[i] = sum(input tensor X[N, (y - depth_radius) : (y + depth_radius +1), (x - depth_radius) : (x + depth_radius +1), C]^ 2).
     * @param[in] input A "TensorPtr".
     * @param[in] size The number of channels to sum over.
     * @param[in] method Indicates the local normalization region. It defaults to LrnMethod::AcrossChannels.
     *                   values:LrnMethod::AcrossChannels, LrnMethod::WithinChannel.
     * @param[in] alpha Scaling parameter. It defaults to '1' and is usually positive.
     * @param[in] beta The exponent. It defaults to 0.5.
     * @param[in] bias Optional factor. It defaults to 1 and an offset that is usually positive to avoid dividing by 0.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape as the tensors in "input".
     */
    TensorPtr lrn(TensorPtr input,
                  int32_t size,
                  LrnMethod method = LrnMethod::AcrossChannels,
                  float_t alpha = 1.0,
                  float_t beta = 0.5,
                  float_t bias = 1.0,
                  const Quantization &quant = {});

    /**
     * @brief L2Pooling2D consumes an input tensor X and applies L2 pooling across the tensor according to kernel sizes,
     *        stride sizes, and pad lengths. L2 pooling consisting of computing the L2 norm on all values of a subset of
     *        the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.
     * @param[in] input  A "TensorPtr".
     *                   Support the shape [batch,height,width,channel].
     * @param[in] kernel  The size of the kernel along each axis. The 'kernel' parameter format [kernel_y,kernel_x].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format [stride_y,stride_x].
     * @param[in] dilations Dilation of filter along each spatial dimension. If not present, defaults to 1 along each spatial dimensions.
     *                      The 'dilations' parameter format [dilation_y,dilation_x].
     * @param[in] padding The "padding" must be either "VALID" or "SAME".
     *                 - SAME: calculate explicit padding.
     *                 - VALID: Fill 0.
     * @param[in] ceil_mode Means using Ceil or Floor function to compute the output shape.
     *                      Generally, the default value is "false" if not specified, which means using floor function.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs.
     */
    TensorPtr l2pooling2d(TensorPtr input,
                          const std::vector<int32_t> &kernel,
                          const std::vector<int32_t> &strides,
                          const std::vector<int32_t> &dilations,
                          Padding padding = Padding("VALID"),
                          bool ceil_mode = false,
                          const Quantization &quant = Quantization());

    /**
     * @brief Multiplies matrix "a" by matrix "b", producing "a" * "b".
     * @param[in] a A "TensorPtr".
     *              Supports 4-dimensional,Must be {1,1,height,width}.
     *              Supports 3-dimensional,Must be {1,height,width}.
     *              Supports 2-dimensional,Must be {height,width}.
     * @param[in] b A "TensorPtr",with the same type as the tensors @p a.
     * @param[in] adj_a If "true", "a" is conjugated and transposed before multiplication.
     * @param[in] adj_b If "true", "b" is conjugated and transposed before multiplication.
     * @param[in] fused_multiplier The output will multiply 'fused_multiplier'.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same type as the tensors  @p a.
     */
    TensorPtr matmul(TensorPtr a,
                     TensorPtr b,
                     bool adj_a = false,
                     bool adj_b = false,
                     float fused_multiplier = 1.0,
                     const Quantization &quant = {});

    /**
     * @brief Computes log softmax activations.
     *
     * @param[in] input A "TensorPtr".
     * @param[in] axis Describes the dimension log_softmax that will be performed on.
     *                 A negative value means counting dimensions from the back.
     *                 The default value is axis = -1.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape as the tensors in "input".
     */
    TensorPtr log_softmax(TensorPtr input,
                          int32_t axis = -1,
                          const Quantization &quant = {});

    /**
     * @brief Max_pool takes an input tensor and applies average pooling across the tensor by the kernel size,
     *        stride size, and padding. Max_pool consists of computing the max on all values of a subset of
     *        the input tensor according to the kernel size and downsampling the data into the output tensor for further processing.
     *        If no padding is present, the 'pad' group parameters default to zero.
     * @param[in] input  A "TensorPtr".
     *                   Support the following shape:
     *                   [batch,height,channel],
     *                   [batch,height,width,channel],
     *                   [batch,depth,height,width,channel].
     * @param[in] kernel  The size of the kernel along each axis. The 'kernel' parameter format should be as follows:
     *                    1d:[kernel_x],
     *                    2d:[kernel_y,kernel_x],
     *                    3d:[kernel_z,kernel_y,kernel_x].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    1d:[stride_x],
     *                    2d:[stride_y,stride_x],
     *                    3d:[stride_z,stride_y,stride_x].
     * @param[in] dilations Dilation of filter along each spatial dimension. If not present, defaults to 1 along each spatial dimensions.
     *                      The 'dilations' parameter format should be as follows:
     *                      1d:[dilation_x],
     *                      2d:[dilation_y,dilation_x],
     *                      3d:[dilatioin_z,dilation_y,dilation_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    1d:[pad_left, pad_right],
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right],
     *                    3d:[pad_z_begin, pad_z_end, pad_y_begin, pad_y_end, pad_x_begin, pad_x_end].
     * @param[in] ceil_mode Means using Ceil or Floor function to compute the output shape.
     *                      Generally, the default value is `false` if not specified, which means using floor function.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs.
     */
    TensorPtr max_pool(TensorPtr input,
                       const std::vector<int32_t> &kernel,
                       const std::vector<int32_t> &strides,
                       const std::vector<int32_t> &dilations,
                       Padding padding = Padding("Valid"),
                       bool ceil_mode = false);

    /**
     * @brief Max_pool Performs max pooling on the inputs and returns both max values and indices.
     *        Generally, the indices in ArgMax are flattened, so a maximum value at position "[n, h, w, c]" will be flattened as:
     *         "indices = h * width + wC if flatten_dim = HW
     *         "indices = (h * width + w) * channel + c" if flatten_dim = HWC
     *         "indices = ((n * height + h) * width + w) * channel + c" if flatten_dim = NHWC
     *        Where, height (or width, channel) represents the size of corresponding H (or W, C) dimension under NHWC data format.
     *        Generally, the indices are located at ([0, height], [0, width]) originally before flattening even though padding is involved.
     *        Where, the range is a left closed and right open interval.
     * @param[in] input  A "TensorPtr".
     *                   Support the following shape:
     *                   [batch,height,width,channel].
     * @param[in] kernel  The size of the kernel along each axis. The 'kernel' parameter format should be as follows:
     *                    2d:[kernel_y,kernel_x].
     * @param[in] strides Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
     *                    The 'strides' parameter format should be as follows:
     *                    2d:[stride_y,stride_x].
     * @param[in] dilations Dilation of filter along each spatial dimension. If not present, defaults to 1 along each spatial dimensions.
     *                      The 'dilations' parameter format should be as follows:
     *                      2d:[dilation_y,dilation_x].
     * @param[in] padding The "padding" support "VALID" or "SAME".
     *                    SAME: calculate explicit padding.
     *                    VALID: Fill 0.
     *                    Support vector value:
     *                    2d:[pad_top, pad_bottom, pad_left, pad_right].
     * @param[in] flatten_dim A flatten mode to output the indices. NHWC means including batch dimension in the flattened index of ArgMax.
     *                        HWC means including Height and Width dimensions in the flattened index of ArgMax.
     *                        HW means including the Width dimension in the flattened index of ArgMax.
     *                        Generally, it defaults to HWC mode.
     * @param[in] storage_order An int value of storage_order of the output index tensor.
     *                          The default value is 0. 0 is row-major, and 1 is column-major.
     * @param[in] ceil_mode Means using Ceil or Floor function to compute the output shape.
     *                      Generally, the default value is `false` if not specified, which means using floor function.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs.
     */
    std::vector<TensorPtr> max_pool_with_argmax(TensorPtr input,
                                                const std::vector<int32_t> &kernel,
                                                const std::vector<int32_t> &strides,
                                                const std::vector<int32_t> &dilations = {1, 1},
                                                const Padding &padding = Padding("Valid"),
                                                const FlattenDim flatten_dim = FlattenDim::HWC,
                                                const int32_t storage_order = 0,
                                                const bool ceil_mode = false);

    /**
     * @brief Select and scale the feature map of each region of interest to a unified output size by max-pooling.
     * @param[in] input A "TensorPtr", specifying the input tensor. Rank of input must be 4.
     * @param[in] rois A "TensorPtr", specifying the roi tensor.
     *            If rois shape is [num_rois, 4], the coordinates format should be [x1,y1,x2,y2].
     * @param[in] batch_id A "TensorPtr", specifying the batch index of each box. Boxes with the same batch index are grouped together.
     *                     If batch index is put into rois, this parameter should keep default nullptr.
     * @param[in] pooled_shape An uint32 vector, specifying the height and width of output tensor. Vector size must be 2.
     * @param[in] spatial_scale  An float vector, specifying the height and width spatial scale. Vector size must be 2.
     * @return A "TensorPtr", specifying the output tensor, which shape will be [num_rois, pooled_shape[0], pooled_shape[1], input[c]] in NHWC format.
     *         Output data type will be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr max_roi_pool(TensorPtr input,
                           TensorPtr rois,
                           TensorPtr batch_id = nullptr,
                           const std::vector<uint32_t> &pooled_shape = {1, 1},
                           const NodeParamValue &spatial_scale = {1, 1},
                           bool quantized = false);

    /**
     * @brief MaxUnpool essentially computes the partial inverse of the MaxPool operator.
     *        The input information to this operator is typically the output information from a MaxPool operator.
     *        The first input tensor X is the tensor that needs to be unpooled,
     *        which is typically the pooled tensor (first output) from MaxPool.
     *        The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
     *        to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool operator.
     *        The third input is a vector that specifies the output size of the unpooling operation.
     * @param[in] input A Tensor point list, specifying the input tensor x and y. All input must have the same shape.
     * @param[in] flatten_dim A flatten mode to output the indices.
     *                        NHWC means including batch dimension in the flattened index of ArgMax.
     *                        HWC means including Height and Width dimensions in the flattened index of ArgMax.
     *                        HW means including the Width dimension in the flattened index of ArgMax.
     *                        Generally, it defaults to HWC mode.
     * @param[in] storage_order  An int value of storage_order of the output index tensor.
     *                           The default value is 0. 0 is row-major, and 1 is column-major.
     * @return A Tensor, which has the same shape as param shape and same dtype as input tensor.
     */
    TensorPtr max_un_pool(TensorPtr input,
                          TensorPtr indices,
                          const std::vector<uint32_t> &out_shape,
                          FlattenDim flatten_dim = FlattenDim::HWC,
                          int32_t storage_order = 0);

    /**
     * @brief Computes the mean of elements across dimensions of a tensor.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] axis An int32 scalar, specifying the dimensions to reduce, range must be in [-rank, rank), rank is @p input rank.
     * @param[in] keep_dim A boolean scalar, default is false. Keep the reduced dimension or not, if keep_dim is true,
     *            this dimension will be set as 1.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same data type and scale/offset as @p input,
     *         and output shape will be inferred by @p axis and @p keep_dim.
     */
    TensorPtr mean(TensorPtr input,
                   const std::vector<int32_t> &axis = {},
                   bool keep_dim = false,
                   const Quantization &quant = {});

    /**
     * @brief Performs mean variance normalization on the input data tensor (X).
     *        The formula is 'Y=(X-E(X)) / sqrt(Z), where Z = E(X - E(X))^2'.
     * @param[in] input A "TensorPtr", specifying the input tensor to be normalized.
     * @param[in] axis A list of integers, along which to reduce.
     *                  A default value [0, 1, 2] means calculating mean and variance along each channel.
     *                  Two given variables with the same channel-coordinate share the same mean and variance parameters.
     * @param[in] epsilon A float IR parameter. The epsilon value is used to avoid division by zero.
     *                    Generally, the default value is less than 10-5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the tensor which has the same data type and shape with @p input.
     *         Scale and offset are described in @p quant.
     */
    TensorPtr mean_variance_normalization(TensorPtr input,
                                          const std::vector<int32_t> &axis = {0, 1, 2},
                                          float epsilon = 0.01,
                                          const Quantization &quant = {});

    /**
     * @brief Returns coordinate matrices from coordinate vectors.
     *        Makes N-D coordinate arrays for vectorized evaluations of N-D scalar/vector fields over N-D grids,
     *        given one-dimensional coordinate arrays x1, x2,..., xn.
     *        This function supports both indexing conventions through the indexing keyword argument.
     *        Giving the string 'ij' returns a meshgrid with matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
     *        In the 2-D case with inputs of length M and N, the outputs are of shape (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.
     *        In the 3-D case with inputs of length M, N and P, outputs are of shape (N, M, P) for 'xy' indexing and (M, N, P) for 'ij' indexing.
     * @param[in] inputs A 'TensorPtr' vector containing the input data tensor X1, X2, ..., XN.
     * @param[in] indexing Cartesian ('xy', default) or matrix ('ij') indexing of output.s
     * @param[in] sparse If it is 'true' the shape of the returned coordinate array for dimension i is reduced
     *                   from (N1, ..., Ni, ... Nn) to (1, ..., 1, Ni, 1, ..., 1).
     *                   These sparse coordinate grids are intended to be used with broadcasting.
     *                   When all coordinates are used in an expression, broadcasting still leads to a fully-dimensional result array.
     *                   By default is 'false'.
     * @param[in] copy If it is 'false', a view into the original arrays is returned to conserve memory.
     *                 By default it is 'true'.
     *                 Note that sparse=false, copy=false will likely return non-contiguous arrays.
     *                 Furthermore, more than one element of a broadcast array may refer to a single memory location.
     *                 If you need to write to the arrays, make copies first.
     *                 Note: Currently only supports 'true'.
     * @param[in] quants A "Quantization" vector, specifying the (Y1, Y2, ..., YN) scale and offset.
     *                   If it is empty, Y1's quantization parameter is the same as x1, and YN's quantization parameter is the same as xN.
     *                   If it is not empty and the number M is less than N, then the output after M uses the first quantization parameter.
     * @return A 'TensorPtr' vector containing the output tensors Y1, Y2, ..., YN.
     */
    std::vector<TensorPtr> meshgrid(std::vector<TensorPtr> inputs,
                                    const std::string indexing = "xy",
                                    const bool sparse = false,
                                    const bool copy = true,
                                    const std::vector<Quantization> &quants = {});

    /**
     * @brief A Self Regularized Non-Monotonic Neural Activation Function. Computes mish activation:
     *        "mish(x)=x*tanh⁡(softplus(x))"
     * @param[in] input A "TensorPtr", specifying input.
     * @param[in] quant A Quantization class, include scale and zeropoint.
     * @return A "TensorPtr", with the same shape and type as the tensors in "input".
     */
    TensorPtr mish(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Computes the element-wise remainder of division. The sign of the remainder is the same as that of the dividend.
     *        Besides, the mod operation can also behave like the fmod function in C or NumPy,
     *        This operator also supports broadcasting and multidirectional broadcasting as NumPy style.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same data bitwidth
     *                   with @p input1, but can have different quantization parameters with @p input1.
     * @param[in] fmod: Whether the operator should behave like fmod (default=false means that it will perform integer mods).
                         Set this to true to force fmod treatment.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data bitwidth with @p inputs. The shape is after multibroadcast shape.
     *         It can have the different quantization parameters with @p inputs.
    */
    TensorPtr mod(TensorPtr input1, TensorPtr input2, bool fmod = false, const Quantization &qunt = {});

    /**
     * @brief Performs elementwise maximum of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                   f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                   This operator supports unidirectional broadcasting (tensor slope should be
     *                   unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr max(TensorPtr input1,
                  TensorPtr input2,
                  const Activation &with_activation = {},
                  const Quantization &quant = {});

    /**
     * @brief Performs elementwise minimum of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr min(TensorPtr input1,
                  TensorPtr input2,
                  const Activation &with_activation = {},
                  const Quantization &quant = {});

    /**
     * @brief Computes the mean and variance of the input tensor (X) along the specified axis.
     *        Especially, if input tensor (X) is a 1-D format and 'axis=[0,]',
     *        the result will be the mean and variance of a vector.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] axis A list along which dimension to compute the mean and variance,
     *                 the value range is [-rank, rank).
     * @param[in] keepdims A boolean scalar, default is true.
     *                     Produces moments with the same dimensionality as the input tensor (X).
     * @return vector of "TensorPtr",{Y1(mean),Y2(variance)}.
     *         Tensor will have the same data type and scale/offset with @p input.
     */
    std::vector<TensorPtr> moments(TensorPtr input,
                                   const std::vector<int32_t> &axis,
                                   bool keepdims = true);

    /**
     * @brief Performs element-wise multiplication of the input tensors.
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr mul(TensorPtr input1,
                  TensorPtr input2,
                  const Activation &with_activation = {},
                  const Quantization &quant = {});

    /**
     * @brief Location transformation for multibox detection. Note that, in the following Inputs or Outputs sections:
     *          X1 means the input class probabilities.
     *          X2 means the input location regression predictions.
     *          X3 means the input prior anchor boxes.
     *          Y1 means the output box.
     *          Y2 means the valid count of output box.
     * @param[in] input A "TensorPtr", specified the input class probabilities(X1).
     * @param[in] loc_pred A "TensorPtr",specified the input location regression predictions(X2).
     * @param[in] anchor A "TensorPtr",specified the input prior anchor boxes(X3).
     * @param[in] score_threshold The threshold for deciding when to remove boxes based on the score. It is a scalar.
     * @param[in] variance A list of variances to be decoded from box regression output with the the order of [y, x, h, w].
     * @return A vector of "TensorPtr". The first "TensorPtr" specified the output box(Y1).
     *         The second "TensorPtr" specified the valid count of output box(Y2).
     */
    std::vector<TensorPtr> multibox_transform_loc(TensorPtr input,
                                                  TensorPtr loc_pred,
                                                  TensorPtr anchor,
                                                  float score_threshold = 0.01,
                                                  std::vector<float> variance = {0.1, 0.1, 0.2, 0.2},
                                                  const Quantization &quant = {});

    /**
     * @brief Returns the truth value of (input1 != input2) element-wise.
     * @param[in] input1 A "TensorPtr".
     * @param[in] input2 A "TensorPtr",Must have the same type and shape as input1.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape as the tensors in "input1", is of type "BOOL".
     */
    TensorPtr not_equal(TensorPtr input1,
                        TensorPtr input2,
                        const Quantization &quant = {});

    /**
     * @brief Computes numerical negative value element-wise.
     * @param[in] input A "TensorPtr".
     * @return A "TensorPtr",with the same shape as the tensors in "input".
     */
    TensorPtr negative(TensorPtr input);

    /**
     * @brief Returns the indices of the elements that are non-zero (in row-major order - by dimension).
     * Note that, in the following Outputs sections:
     *    Y1 means the whether the input is zero or not.
     *    Y2 means size of output.

     * @param[in] input A "TensorPtr".
     * @return A "TensorPtr" list,include Y1 and Y2.
     */
    TensorPtrList nonzero(TensorPtr input);

    /**
     * TODO: New IR will be coming.
     * @brief The Non-Maximum Suppression (NMS) filters out boxes that have high Intersection-Over-Union (IOU)
     *        overlap with previously selected boxes.
     * @param[in] box_coor A "TensorPtr", specified coordinates of boxes in each batch with shape [batch_size, num_boxes, 4].
     * @param[in] box_num_per_class A "TensorPtr", specified number of boxes in each class and each batch with shape [batch_size, num_classes].
     * @param[in] class_num_per_batch A "TensorPtr", specified number of classes in each batch with shape [batch_size, 1].
     * @param[in] score A "TensorPtr", specified scores of boxes in each batch with shape [batch_size, num_boxes].
     * @param[in] image_info An uint32 vector, specified input featuremap height/width information, due to coordinates is normalized value, so we need this to back it.
     * @param[in] nms_method A "NmsMethod" enumerate, it can be NmsMethod.Hard, NmsMethod.Gaussian. Default is Hard.
     * @param[in] max_output_size An uint32 scalar, specifying max output box number. Default is 5000.
     * @param[in] score_threshold A float/int scalar, boxes with scores lower than the threshold are filtered before sending to the NMS algorithm. Default is 0.6.
     * @param[in] iou_threshold A float/int scalar, which will be ignored when method is "Gaussian". Default is 0.5.
     * @param[in] sigma A float scalar, which will be ignored when method is not "Gaussian". Default is 0.
     * @param[in] center_point_box A bool scalar, "false" means coordinates format is [y1,x1,y2,x2]. "true" means coordinates format is [y_center,x_center,height,width].
     *                             Default is "false".
     * @return vector of "TensorPtr", first "TensorPtr" specified coordinates of selected boxes in each batch. Its shape is same with @p box_coor.
     *         Second "TensorPtr" specified number of selected boxes in each batch. Its shape is same with @p box_num_per_class.
     *         Third "TensorPtr" specified scores of boxes in each batch. Its shape is same with @p score.
     *         Forth "TensorPtr" specified indices of boxes in each batch. Its shape is same with @p score.
     */
    std::vector<TensorPtr> nms(TensorPtr box_coor,
                               TensorPtr box_num_per_class,
                               TensorPtr class_num_per_batch,
                               TensorPtr score,
                               const std::vector<uint32_t> &image_info,
                               NmsMethod nms_method = NmsMethod::Hard,
                               uint32_t max_output_size = 5000,
                               NodeParamValue core_threshold = 0.6,
                               NodeParamValue iou_threshold = 0.5,
                               float sigma = 0.f,
                               bool center_point_box = false);

    /**
     * @brief Calculate the mean and variance of based on the sufficient statistics.
     *        Note that, in the following Inputs or Outputs sections:
     *           X1 means the input sufficient statistics mean.
     *           X2 means the input sufficient statistics variance.
     *           X3 means the input shift value for input mean and variance data.
     *           Y1 means the output mean value.
     *           Y2 means the output variance value.
     * @param[in] mean_ss A "TensorPtr", specifying input tensor X1.
     * @param[in] variance_ss A "TensorPtr", specifying input tensor X2.
     * @param[in] shift A "TensorPtr", specifying input tensor X3.
     * @param[in] counts A parameter containing the total count of the data.
     * @param[in] quant A "Quantization" vector, specifying the (Y1,Y2) scale and offset.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2.
     */
    std::vector<TensorPtr> normalize_moments(TensorPtr mean_ss,
                                             TensorPtr variance_ss,
                                             TensorPtr shift,
                                             const int32_t counts,
                                             const std::vector<Quantization> &quants = {});

    /**
     * @brief Returns a one-hot tensor (Y) based on the input 'indice' tensor and input scalar 'depth'. Generally, the
     *        index value in the 'indices' input tensor means there is a 'on_value' in that location and the other will
     *        have a value of 'off value' in the output tensor. Where, 'on_value' and 'off value' are the parameters of
     *        the two-element value tensor in order of '[off_value, on_value]'. Besides, the dimension of the output
     *        tensor is 'rank(input tensor) + 1' , where the additional dimension is for one-hot representation extension
     *        along the specified 'axis'. The size of the additional dimension is the same as the value of 'depth'. Any
     *        entries in the 'indices' input tensor with values outside the range '[-depth, depth-1]' will result in all
     *        'off_value' values in the output tensor.
     *        Generally, if 'indices' is a scalar, the shape of the output tensor will be a vector of length 'depth'. Or
     *        if the 'indices' is a vector of length 'features', the shape of the output tensor will be:
     *           'features * depth if axis = -1'
     *           'depth * features if axis = O'
     *        Where, * means the multiplication operator. If the 'indices' is a matrix (in other words, with a batch) with
     *        shape [batch, features], the shape of the output tensor will be:
     *           'batch * features * depth if axis = -1'
     *           'depth * batch * features if axis = 0'
     *           'batch * depth * features if axis = 1'
     *
     * @param[in] indices An input "TensorPtr".
     * @param[in] depth A scalar to specify the number of classes in the one-hot tensor. That is, it is the size of the
     *                  one-hot dimension which is specified by 'axis' and added in the output tensor.
     * @param[in] values An optional two-element tensor in an order as [off_value, on_value].
     *                   Where, 'on_value' is the value to fill in the specified locations of the 'indices',
     *                   and 'off_value' means filling in the locations other than the locations specified
     *                   in the 'indices' tensor. For example, values = [0, 1].
     *                   Default value  [0.0, 1.0].
     * @param[in] axis An optional scalar parameter to specify which dimension to insert to the input tensor 'indices'.
     *                 A negative value means inserting dimensions from the back. It defaults to -1, which means the
     *                 insertion is the last dimension.
     * @return A "TensorPtr", of which the dimension is one greater than the input tensor 'indices'.
     */
    TensorPtr one_hot(TensorPtr indices,
                      int32_t depth,
                      const NodeParamValue values = {},
                      int32_t axis = -1);

    /**
     * @brief Given the input tensor containing the data to be padded,
     *        a tensor containing the number of the start and end pad values for specified axis and mode.
     * @param[in] input A "TensorPtr".
     * @param[in] pads A nested list means the padding number with shape [n, 2] to a N dimension data tensor.
     * @param[in] mode Mode to add padding. The default value is CONSTANT.
     *                 CONSTANT pads with a given value as specified by constant_value.
     *                 REFLECT pads with the reflection of the vector mirrored on the first and last values of
     *                         the vector along each axis.
     *                 SYMMETRIC pads with the symmetric value.
     *                 EDGE pads with the edge values of array.
     *                 WRAP wrap-around padding as if the data tensor forms a torus.
     * @param[in] constant_value  A float scalar value to be used if the mode chosen is CONSTANT.
     *                            The default value is 0.f.
     *                            Representing non-quantized constant value.
     * @param[in] ds_pads It describes the dynamic pads parameters.
     * @return A "TensorPtr",with the same tupe as the tensors @p input.
     */
    TensorPtr pad(TensorPtr input,
                  const std::vector<std::pair<uint32_t, uint32_t>> &pads,
                  PadMode mode = PadMode::Constant,
                  NodeParamValue constant_value = 0,
                  std::vector<std::vector<tongue::Expr>> ds_pads = {});

    /**
     * @brief Boxes with scores less than Score_Threshold are dropped or discarded.
     * the box need translated into bounding box.
     * Note that, in the following Inputs or Outputs sections:
     *      X1 means the scores of all the boxes belongs to proposal and backgroud,
     *         of which the shape is [batch_size, num_boxes, 2].
     *      X2 means the coordinates of the maximum output boxes per batch,
     *         of which the shape is [batch_size, num_boxes, 4].
     *      Y1 means the scores of all the selected boxes,
     *         of which the shape is [batch_size, max_output_size, 1].
     *      Y2 means the coordinates of the selected boxes,
     *         of which the shape is [batch_size, max_output_size, 4].
     *      Y3 means the num of the all selected boxes belongs to class per batch,
     *         of which the shape is [batch_size, 1].
     *      Y4 means the class_ids of the selected boxes per batch in input tensor X1,
     *         of which the shape is the same as Y3 as well.
     * @param[in] scores A "TensorPtr",  specifying the input X1.
     * @param[in] coordinates A "TensorPtr", specifying the input X2.
     * @param[in] anchors A "TensorPtr" list, specifying the anchors parameter (h, w, ycenter, xcenter).
     * @param[in] image_info An uint32 vector, specified input featuremap height/width information,
     *                       due to coordinates is normalized value, so we need this to back it.
     * @param[in] score_threshold A float scalar, specifying the scroe threshold.
     * @param[in] max_output_size An uint32 scalar, specifying max output box number.
     *                             Default is num_boxes.
     * @return A vector of "TensorPtr", which is [Y1,Y2,Y3,Y4].
     */
    std::vector<TensorPtr> proposal(TensorPtr scores,
                                    TensorPtr coordinates,
                                    std::vector<TensorPtr> anchors,
                                    const std::vector<uint32_t> &image_info,
                                    const float score_threshold = 0.1,
                                    const uint32_t output_size = 0);

    /**
     * @brief This operator completes the operation between proposal and NMS,
     * extracts and quantized the valid boxes,
     * outputs the box coordinates of the quantized and un-quantized boxes,
     * so that NMS can perform the operation.
     * @param[in] coordinates A "TensorPtr", specified  the coordinates of the input boxes,
     *                        of which the shape is [batch_size, num_boxes, 4].
     * @param[in] scores A "TensorPtr", specified the scores of input boxes,
     *                   of which the shape is [batch_size, num_boxes, 1].
     * @param[in] valid_numbers A "TensorPtr", specified the valid numbers of input boxes,
     *                          of which the shape is [batch_size, 1].
     * @param[in] image_info An uint32 vector, specified input featuremap height/width information,
     *                       due to coordinates is normalized value, so we need this to back it.
     * @param[in] output_size An uint32 scalar, specifying max output box number.
     *                        Default is num_boxes.
     * @return A vector of "TensorPtr", which is [Y1,Y2].
     *         Y1 means the coordinates of the valid boxes per batch,
     *         of which the shape is [batch_size, output_size, 4].
     *         Y2 means the coordinates of the quantized valid boxes per batch,
     *         of which the shape is [batch_size, output_size, 4].
     */
    std::vector<TensorPtr> post_nms1(TensorPtr coordinates,
                                     TensorPtr scores,
                                     TensorPtr valid_numbers,
                                     const std::vector<uint32_t> &image_info,
                                     const uint32_t output_size);

    /**
     * @brief Takes input data tensor (X) and exponent tensor (Y), then produces an output tensor (Z) where the function 'z = x ^exponent'
     *        (that is, exponent is from exponent tensor Y), is applied to the data tensor elementwise.
     *        This operator also supports multidirectional (that is, NumPy-style) broadcasting.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same data bitwidth
     *                   with @p input1, but can have different quantization parameters with @p input1.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data bitwidth with @p inputs. The shape is after multibroadcast shape.
     *         It can have the different quantization parameters with @p inputs.
     */
    TensorPtr pow(TensorPtr input1,
                  TensorPtr input2,
                  const Quantization &quant = {});

    /**
     * @brief Parametric Rectified Linear Unit. It follows: f(x) = alpha * x for x < 0, f(x) = x for x >= 0,
     *        where alpha is a learned array that is a constant tensor described as "TensorPtr". Especially,
     *        this operator supports unidirectional broadcasting (tensor alpha should be unidirectional broadcastable to input tensor X).
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] alpha A "TensorPtr", specifying the alpha tensor, which is a constant tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Its data type will be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr prelu(TensorPtr input, TensorPtr alpha, const Quantization &quant = {});

    /**
     * @brief Region of Interest (ROI) align is used for converting all the proposals to the fixed shape
     * as required by the next special layers, especially in an object detection network.
     * PyramidROI align is proposed to avoid the misalignment by removing quantization
     * while converting from the original image into the feature map
     * and from the different shape Pyramid feature map into the ROI feature.
     * In each ROI bin, the value of the sampled locations is computed directly
     * through bilinear interpolation other than quantization.
     * Note that, in the following Inputs or Outputs sections:
     *  X1 means the scores of all the boxes belongs to proposal and backgroud,
     *  of which the shape is [batch_size, num_boxes, 2].
     *  X2~X5 mean the feature map with different size.
     *  Y means the output feature map.
     * @param[in] inputs A vector of "TensorPtr", specifying the input tensor X2~X5.
     * @param[in] resize_shape An int32_t vector, specified output featuremap height/width information.
     * @param[in] image_info An int32_t vector, specified input featuremap height/width information.
     * @param[in] sample_points An uint32 vector, specifying the sample points
     *                          on height and width dimension. Vector size must be 2.
     * @param[in] spatial_scale A quantization scale object,
     *              used to describe the scale quantization parameter.
     * @param[in] ct_mode An enum sepcifying the method to shift the input.
     *              Allowed values are 'HALF_PIXEL' and 'OUTPUT_HALF_PIXEL'.
     *              Use the value 'HALF_PIXEL' to pixel shift the input coordinates
     *              by -0.5 (the recommended behavior).
     *              Use the value 'OUTPUT_HALF_PIXEL' to omit the pixel shift for the input.
     *              Defaults to 'HALF_PIXEL'.
     * @param[in] proposal_normalized It is a value means different quantization method.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr pyramid_roi_align(TensorPtrList inputs,
                                const std::vector<int32_t> &resize_shape,
                                const std::vector<int32_t> &image_info,
                                const std::vector<int32_t> &sample_points,
                                const std::vector<float> &spatial_scale,
                                const CTMode &ct_mode = CTMode::HalfPixel,
                                const bool proposal_normalized = false,
                                const Quantization &quant = {});

    /**
     * @brief QueryRebatch implements the QueryRebatch operation.
     *        The input of this operator is [X1, X2~XN], and the output is Y.
     *        X1 means the input feature map that will be reranged by new indices X2~XN.
     *        X2~XN means the new indices. Every new indices input will be a new sub-batch of output.
     *        The following equation describes the QueryRebatch operation:
     *        for j in output batch
     *           for i, index in enmuerate(X2~XN)
     *                indices=Xi
     *                output[j,i,:len(Xi)] = X1[j,indices]
     * @param[in] inputs A "TensorPtr" list, specifying the input tensor X1~XN.
     * @return A "TensorPtr", specifying the output tensor Y.
     */
    TensorPtr query_rebatch(TensorPtrList inputs);

    /**
     * @brief Quantize the input tensor. It has two situations:
     *         - float -> quant
     *         - quant -> quant (tflite)
     * @param[in] input A "TensorPtr", specifying the input tensor
     * @param[in] output_type A "Dtype", specifying output data type
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Its data type will be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr quantize(TensorPtr input,
                       Tensor::Dtype output_type = Tensor::Dtype::UINT8,
                       QuantizeRoundMode round_mode = QuantizeRoundMode::RoundToEven,
                       const Quantization &quant = {});

    /**
     * @brief Calculate rank of input tensor.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @return A "TensorPtr", specifying the output tensor, it should be a constant value with shape=[1].
     */
    TensorPtr rank(TensorPtr input);

    /**
     * @brief Reciprocal takes one input data (Tensor) and produces one output data (Tensor) where the reciprocal is,
     *        "y = 1/x", is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr reciprocal(TensorPtr input,
                         const Quantization &quant = {});

    /**
     * @brief Reduces a tensor though the method specified along given dimensions.
     *        The resulted tensor has the same rank as the input if keepdims is true.
     *        If keepdims is false, then the resulted tensor have the reduced dimension pruned.
     *        ReduceMethod::All: logical and
     *        ReduceMethod::Any: logical or
     *        ReduceMethod::Max: maximum
     *        ReduceMethod::Min: minimum
     *        ReduceMethod::Mean: mean
     *        ReduceMethod::Sum: sum
     *        ReduceMethod::L1: L1 norm
     *        ReduceMethod::L2: L2 norm
     *        ReduceMethod::Variance: variance
     *        ReduceMethod::UnbiasedVariance:unbiased variance
     *        ReduceMethod::Prod:multiplication
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] method A enumerate object of ReduceMethod.
     * @param[in] axis A vector of int32, default is {-1}, the value range is [-rank, rank).
     * @param[in] keep_dim A boolean scalar, default is true. Keep the reduced dimension or not,
     *                     default true mean keep reduced dimension.
     * @return A "TensorPtr", specifying the reduced output. After "logical and", data should be "0" or "1".
     *         Shape will be inferred by "keep_dim". Output has the same data type and scale/offset with the @p input.
     */
    TensorPtr reduce(TensorPtr input,
                     ReduceMethod method = ReduceMethod::All,
                     const std::vector<int32_t> &axis = {-1},
                     bool keep_dim = true,
                     const Quantization &quant = {});

    /**
     * @brief Yolo network predicts 5 bounding boxes at each cell in the output feature map.
     *        The network predicts 5 coordinates for each bounding box, tx, ty, tw, th, and to.
     *        If the cell is offset from the top left corner of the image by (cx, cy) and the bounding box prior
     *        has width and height pw, ph, then the predictions correspond to:
     *        bx = O(tx) + cx by = O(ty) + cy bw = pw * exp(tw) bh = ph * exp(th) Pr(object) * IOU(b, object) = O(to)
     *        Region layer is to process the calculation and output the bounding boxed and confidence for NMS.
     *        Note that, in the following Inputs or Outputs sections:
     *        X1 means input data of which the shape is [batch_size, s, s, anchor_num, num_classes + 5].
     *           where s is the grid num of a feature map, num_classes is the classification number of the network.
     *        Y1 means the scores of the selected boxes, of which the shape is [batch_size, max_box_num].
     *           where max_box_num is the config parameter of the network.
     *        Y2 means the coordinates of the filtered boxes, of which the shape is [batch_size, max_box_num, 4].
     *        Y3 means the boxes number of every class per batch, of which the shape is [batch_size, num_classes].
     *        Y4 means the valid classes label of per class, of which the shape is [batch_size, num_classes].
     *        Y5 means the classes number of every class per batch.
     * @param[in] input A "TensorPtr", specifying the input tensor X1.
     * @param[in] obj_threshold The threshold for deciding when to which score is belong to an object. It is a scalar.
     * @param[in] max_box_num Integer representing the maximum number of boxes to be selected per batch. It is a scalar.
     * @param[in] box_per_grid Integer representing the number of boxes to be presented by a grid. It is a scalar.
     * @param[in] class_num Integer representing the number of class for the input. It is a scalar.
     * @param[in] grid_width Integer representing the grid number of feature map width. It is a scalar.
     * @param[in] grid_height Integer representing the grid number of feature map height. It is a scalar.
     * @param[in] grid_compensate For caffe yolo416 which need compensate.
     * @param[in] anchors Anchors are determined by a k-means procedure,
     *            looking at all the bounding boxes in your dataset.
     *            The k-means routine will figure out a selection of anchors that represent your dataset.
     *            k=5 for yolov3, but there are different numbers of anchors for each YOLO version.
     * @param[in] quants A "Quantization" vector, specifying the (Y1,Y2) scale and offset.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2,Y3,Y4,Y5.
     */
    TensorPtrList region(TensorPtr input,
                         const NodeParamValue &obj_threshold,
                         const int32_t max_box_num,
                         const int32_t box_per_grid,
                         const int32_t class_num,
                         const int32_t grid_width,
                         const int32_t grid_height,
                         const bool grid_compensate = false,
                         const std::vector<float> anchors = {},
                         const std::vector<Quantization> &quants = {});

    /**
     * @brief Merge two group input data from Region layer into one output.
     *        Note that, in the following Inputs or Outputs sections:
     *        X1 and X2 mean the scores of all the boxes come from the first and second region layer output,
     *           of which the shape is [batch_size, secore_num].
     *        X3 and X4 mean the coordinates of boxes come from the first and second region layer output,
     *           of which the shape is [batch_size, secore_num, 4].
     *        X5 and X6 mean the boxes number of every class per batch come from the first and second region layer output,
     *           of which the shape is [batch_size, class_num].
     *        X7 and X8 mean the boxes label of every class per batch come from the first and second region layer output,
     *           of which the shape is [batch_size, class_num].
     *        X9 and X10 mean the classes number of every class per batch come from the first and second region layer output,
     *           of which the shape is [batch_size, class_num].
     *        Y1 means the merged scores of the selected boxes.
     *        Y2 means the merged coordinates of the filtered boxes.
     *        Y3 means the total boxes number of every class per batch, of which the shape is [batch_size, num_classes].
     *        Y4 means the total valid classes label of per class, of which the shape is [batch_size, num_classes].
     *        Y5 means the maxmun class number of every class per batch.
     * @param[in] inputs A "TensorPtr" list, specifying the input tensor X1~X10.
     * @param[in] class_num Integer representing the number of class for the input. It is a scalar.
     * @param[in] quants A "Quantization" vector, specifying the (Y1,Y2) scale and offset.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2,Y3,Y4,Y5.
     */
    std::vector<TensorPtr> region_fuse(std::vector<TensorPtr> inputs,
                                       const int32_t class_num,
                                       const std::vector<Quantization> &quants = {});

    /**
     * @brief Relu takes one input Tensor and produces one output Tensor where the rectified linear function,
     *        y = max(0, x), is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", which has the same data type, shape and scale/offset with the @p input.
     */
    TensorPtr relu(TensorPtr input,
                   const Quantization &quant = {});

    /**
     * @brief Relu1 takes one input Tensor and produces one output Tensor where the rectified linear function,
     *        y = min(1.f, max(-1.f, x)), is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", which has the same data type, shape and scale/offset with the @p input.
     */
    TensorPtr relu1(TensorPtr input,
                    const Quantization &quant = {});

    /**
     * @brief Relu6 takes one input Tensor and produces one output Tensor where the rectified linear function,
     *        y = min(6, max(0, x)), is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", which has the same data type, shape and scale/offset with the @p input.
     */
    TensorPtr relu6(TensorPtr input,
                    const Quantization &quant = {});

    /**
     * @brief Repeat input tensor.
     * @param[in] input A "TensorPtr", specified the input tensor.
     * @param[in] repeats A "TensorPtr", specified 1D tensor, flatten the input with repeats.
     * @param[in] output_shape A "TensorShape" object, specified the output shape, due to flatten, it should be 1D shape.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr repeat(TensorPtr input, TensorPtr repeats, const TensorShape &shape);

    /**
     * @brief Repeat input tensor according axis.
     * @param[in] input A "TensorPtr", specified the input tensor.
     * @param[in] axis A int scalar, specified the dimension repeats on.
     * @param[in] repeats A "TensorPtr", specified 1D tensor, its length must be equal with input[axis].
     * @param[in] output_shape A "TensorShape" object, specified the output shape.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr repeat(TensorPtr input, TensorPtr repeats, const TensorShape &shape, int32_t axis);

    /**
     * @brief            Reshape the input tensor to new shape. The output tensor has the same values as
     *                   input tensor.
     * @param[in] input  A "TensorPtr".
     * @param[in] new_shape The new shape.
     * @param[in] exprs The dynamic shape expression.
     * @return A "TensorPtr", with the new shape and with type as the tensors in @p input.
     */
    TensorPtr reshape(TensorPtr input,
                      const TensorShape &new_shape,
                      const std::vector<tongue::Expr> &exprs = {});

    /**
     * @brief Resizes the input tensor using the specified method with specified mode.
     *         Generally, resize mode has follows,
     *         half_pixel: x_original = (y_resized + 0.5) / scale - 0.5.
     *         half_pixel_symmetric:
     *             adjustment = output_width_int / output_width
     *             center = input_width / 2
     *             offset = center * (1 - adjustment)
     *             x_ori = offset + (x + 0.5) / scale - 0.5.
     *         align_corners: x_original = y_resized * (x_original - 1) / (y_resized - 1).
     *         asymmetric: x_original = y_resized / scale.
     *         pytorch_half_pixel: x_original = length_resized >1 ? (x_resized + 0.5)/ scale - 0.5 : 0.
     *         tf_half_pixel_for_nn: (x_resized + 0.5) / scale
     * @param[in] input  A "TensorPtr", specifying the input tensor.
     * @param[in] out_size
     * @param[in] method An enumerate of ResizeMethd object, including Bilinear, NearestNeighbor, default is NearestNeighbor.
     * @param[in] mode An enumerate of ResizeMode object, including
     *                 HalfPixel, HalfPixelSymmetric, AlignCorners, Asymmetric, PytorchHalfPixel, TfHalfPixelForNN, default is HalfPixel.
     * @param[in] nearest_mode An emuerate of ResizeNearestMode object, including
     *                         RoundPreferFloor, RoundPreferCeil, Floor, Ceil, Simple, default is RoundPreferFloor.
     *                         If method is not nearest, this parameter will be ignored.
     * @param[in] antialias If set to true, "linear" and "cubic" interpolation modes will use an antialiasing filter when downscaling.
     *                      Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale),
     *                      which means that when downsampling, more input pixels contribute to an output pixel.
     * @return A "TensorPtr", specifying the output tensor. Output shape refers to @p out_size.
     */

    /**
     * @brief Resizes the input tensor using the specified method with specified mode.
     *         Generally, resize mode has follows,
     *         half_pixel: x_original = (y_resized + 0.5) / scale - 0.5.
     *         half_pixel_symmetric:
     *             adjustment = output_width_int / output_width
     *             center = input_width / 2
     *             offset = center * (1 - adjustment)
     *             x_ori = offset + (x + 0.5) / scale - 0.5.
     *         align_corners: x_original = y_resized * (x_original - 1) / (y_resized - 1).
     *         asymmetric: x_original = y_resized / scale.
     *         pytorch_half_pixel: x_original = length_resized >1 ? (x_resized + 0.5)/ scale - 0.5 : 0.
     *         tf_half_pixel_for_nn: (x_resized + 0.5) / scale
     * @param[in] input A "TensorPtr", specifying the input tensor
     * @param[in] scale A float vector, orderd by [height, width], or [batch, height, width, channel].
     *                  Calculating with output = input * scale.
     * @param[in] out_size An uint32 vector.
     *                     Support [output_height, output_widht],
     *                             [output_batch, output_height, output_widht, output_channel].
     * Note:      At least one of scale and out_size must be valid.
     * @param[in] method An enumerate of ResizeMethd object, including Bilinear, NearestNeighbor.
     * @param[in] mode An enumerate of ResizeMode object, including
     *                 HalfPixel, HalfPixelSymmetric, AlignCorners, Asymmetric, PytorchHalfPixel, TfHalfPixelForNN, default is HalfPixel.
     * @param[in] nearest_mode An emuerate of ResizeNearestMode object, including
     *                         RoundPreferFloor, RoundPreferCeil, Floor, Ceil, Simple, default is RoundPreferFloor.
     *                         If method is not nearest, this parameter will be ignored.
     * @param[in] antialias If set to true, "linear" and "cubic" interpolation modes will use an antialiasing filter when downscaling.
     *                      Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale),
     *                      which means that when downsampling, more input pixels contribute to an output pixel.
     * @param[in] exclude_outside When calculating the original point coordinates,
     *                            only the pixels inside the bounding box are considered,
     *                            the pixels must not exceed the original bounding box.
     * @return A "TensorPtr", specifying the output tensor. Output shape refers to @p scale.
     */
    TensorPtr resize(TensorPtr input,
                     const std::vector<uint32_t> &out_size = {},
                     const std::vector<float> &scale = {},
                     ResizeMethod method = ResizeMethod::NearestNeighbor,
                     ResizeMode mode = ResizeMode::HalfPixel,
                     ResizeNearestMode nearest_mode = ResizeNearestMode::RoundPreferFloor,
                     bool antialias = false,
                     bool exclude_outside = false);

    /**
     * @brief Resizes the input tensor using the specified method with specified mode.
     *         Generally, resize mode has follows,
     *         half_pixel: x_original = (y_resized + 0.5) / scale - 0.5.
     *         half_pixel_symmetric:
     *             adjustment = output_width_int / output_width
     *             center = input_width / 2
     *             offset = center * (1 - adjustment)
     *             x_ori = offset + (x + 0.5) / scale - 0.5.
     *         align_corners: x_original = y_resized * (x_original - 1) / (y_resized - 1).
     *         asymmetric: x_original = y_resized / scale.
     *         pytorch_half_pixel: x_original = length_resized >1 ? (x_resized + 0.5)/ scale - 0.5 : 0.
     *         tf_half_pixel_for_nn: (x_resized + 0.5) / scale
     * @param[in] input A "TensorPtr", specifying the input tensor
     * @param[in] scale A float vector, orderd by [height, width], or [batch, height, width, channel].
     *                  Calculating with output = input * scale.
     * @param[in] method An enumerate of ResizeMethd object, including Bilinear, NearestNeighbor.
     * @param[in] mode An enumerate of ResizeMode object, including
     *                 HalfPixel, HalfPixelSymmetric, AlignCorners, Asymmetric, PytorchHalfPixel, TfHalfPixelForNN, default is HalfPixel.
     * @param[in] nearest_mode An emuerate of ResizeNearestMode object, including
     *                         RoundPreferFloor, RoundPreferCeil, Floor, Ceil, Simple, default is RoundPreferFloor.
     *                         If method is not nearest, this parameter will be ignored.
     * @param[in] antialias If set to true, "linear" and "cubic" interpolation modes will use an antialiasing filter when downscaling.
     *                      Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale),
     *                      which means that when downsampling, more input pixels contribute to an output pixel.
     * @return A "TensorPtr", specifying the output tensor. Output shape refers to @p scale.
     */

    /**
     * @brief Conversion of one or more images from RGB to YUV. Generally, this operation is used for preprocessing.
     *        Here, the term YUV is commonly encoded using Y'CbCr; while the RGB term is encoded using R'dG'dB'd.
     *        There are three standards which are so-called "BT.709". For 8 bits color,
     *        [0, 255] range is referred as Full Range, which is considerred here; while [16, 235] range for Y and
     *        [16, 240] for UN are referred as Narrow Range. Also, all the conversions are based on integral approximate
     *        (or using fixed-point arithmetic as an alternative formulation) in this operation.
     * @param[in] input A "TensorPtr", specifying YUV image, with a shape of [N, H, W, 3].
     * @param[in] format An enum value of ImageFormat specifying the input image format which will be converted.
     *                   The optional values are [I420, YV12, NV12, NV21].
     * @param[in] bits An integer, specifying the bits to store a Y, U or V values in memory, such as DDR device.
     * @param[in] conversion An enum value of ImageConversion, which means the different coefficient matrix to convert
     *                       between YUV and RGB format. Especially, the 'SELF' means the conversion coefficients are
     *                       customized which is from the configuration. The optional values are [BT709].
     * @param[in] coefficient A list of color space coefficient (CSC) for conversion, which is in order of
                              '[input_offset, output_offset, coefficient]'.
     * @return A "TensorPtr", specifying the output RGB image tensor, with a shape of [N, D].
    */
    TensorPtr rgb_to_yuv(TensorPtr input,
                         const ImageFormat format = ImageFormat::I420,
                         const int bits = 8,
                         const ImageConversion conversion = ImageConversion::BT709,
                         const std::vector<float_t> &coefficient = {});

    /**
     * @brief The right shift operator performs element-wise operations. For each input element, this operator moves its binary representation
     *        toward the right side so that the input value is effectively decreased.
     *        The input X is the tensor to be shifted and another input Y specifies the amounts of shifting.
     *        For example, X is [1, 4], and Y is [1, 1], the corresponding output Z will be [0, 2]. This operator supports multidirectional broadcasting.
     * @param[in] input A "TensorPtr", specifying the tensor needs to be shifted.
     * @param[in] shift A "TensorPtr", specifying the shift tensor.
     * @param[in] quant A "Quantization" object, specifying the output quantization paramemters.
     * @return A "TensorPtr", specifying shifted tensor.
     */
    TensorPtr right_shift(TensorPtr input, TensorPtr shift, const Quantization &quant = {});

    /**
     * @brief Select and scale the feature map of each region of interest to a unified
     *        output size by average pooling sampling points from bilinear interpolation.
     * @param[in] input A "TensorPtr", specifying the input featuremap tensor. Rank of input must be 4.
     * @param[in] rois A "TensorPtr", specifying the roi tensor.
     *            Shape is [num_rois, 5], the format of second dimension should be [batch_id,x1,y1,x2,y2].
     * @param[in] pooled_shape An uint32 vector, specifying the height and width of output tensor. Vector size must be 2.
     * @param[in] batch_id Optional. A "TensorPtr", specifying the batch index of each box.
     *            Boxes with the same batch index are grouped together.
     *            If batch index is put into rois, this parameter should keep default nullptr.
     * @param[in] sample_points An uint32 vector, specifying the sample points on height and width dimension. Vector size must be 2.
     *                          If sample points is 0, it will be dynamic computed as ceil(roi_width/out_width)
     * @param[in] spatial_scale A quantization scale object, used to describe the scale quantization parameter.
     * @param[in] ct_mode An enum sepcifying the method to shift the input.
     *                    Allowed values are 'HALF_PIXEL' and 'OUTPUT_HALF_PIXEL'.
     *                    Use the value 'HALF_PIXEL' to pixel shift the input coordinates
     *                    by -0.5 (the recommended behavior).
     *                    Use the value 'OUTPUT_HALF_PIXEL' to omit the pixel shift for the input.
     *                    Defaults to 'HALF_PIXEL'.
     * @param[in] mode An enum value specifying the pooling method. Two modes are supported: 'AVG' and 'MAX'. Default is 'AVG'.
     * @param[in] quant A "Quantization" object, specifying the scale and offset of output tensor.
     * @return A "TensorPtr", specifying the output tensor, which shape will be [num_rois, pooled_shape[0], pooled_shape[1], input[c]] in NHWC format.
     *         Output data type will be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr roi_align(TensorPtr input,
                        TensorPtr rois,
                        const SpatialScaleQuant &spatial_scale,
                        TensorPtr batch_id = nullptr,
                        const std::vector<uint32_t> &pooled_shape = {1, 1},
                        const std::vector<uint32_t> &sample_points = {0, 0},
                        const CTMode &ct_mode = CTMode::HalfPixel,
                        const RoiAlignMode &mode = RoiAlignMode::Avg,
                        const Quantization &quant = {});

    /**
     * @brief Select and scale the feature map of each region of interest to a unified
     *        output size by average pooling sampling points from bilinear interpolation.
     * @param[in] input A "TensorPtr", specifying the input tensor. Rank of input must be 4.
     * @param[in] rois A "TensorPtr", specifying the roi tensor.
     *            If rois shape is [num_rois, 4], the coordinates format should be [x1,y1,x2,y2].
     * @param[in] batch_id Optional. A "TensorPtr", specifying the batch index of each box.
     *            Boxes with the same batch index are grouped together.
     *            If batch index is put into rois, this parameter should keep default nullptr.
     * @param[in] pooled_shape An uint32 vector, specifying the height and width of output tensor. Vector size must be 2.
     * @param[in] sample_points An uint32 vector, specifying the sample points on height and width dimension. Vector size must be 2.
     *                          If sample points is 0, it will be dynamic computed as ceil(roi_width/out_width)
     * @param[in] spatial_scale An float vector, specifying the height and width spatial scale. Vector size must be 2.
     * @param[in] ct_mode An enum sepcifying the method to shift the input.
     *            Allowed values are 'HALF_PIXEL' and 'OUTPUT_HALF_PIXEL'.
     *            Use the value 'HALF_PIXEL' to pixel shift the input coordinates by
     *            -0.5 (the recommended behavior).
     *            Use the value 'OUTPUT_HALF_PIXEL' to omit the pixel shift for the input.
     *            Defaults to 'HALF_PIXEL'.
     * @param[in] mode An enum value specifying the pooling method. Two modes are supported: 'AVG' and 'MAX'. Default is 'AVG'.
     * @param[in] quant A "Quantization" object, specifying the scale and offset of output tensor.
     * @return A "TensorPtr", specifying the output tensor, which shape will be [num_rois, pooled_shape[0], pooled_shape[1], input[c]] in NHWC format.
     *         Output data type will be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr roi_align(TensorPtr input,
                        TensorPtr rois,
                        TensorPtr batch_id = nullptr,
                        const std::vector<uint32_t> &pooled_shape = {1, 1},
                        const std::vector<uint32_t> &sample_points = {0, 0},
                        const NodeParamValue &spatial_scale = {1.0, 1.0},
                        const CTMode &ct_mode = CTMode::HalfPixel,
                        const RoiAlignMode &mode = RoiAlignMode::Avg,
                        const Quantization &quant = {});

    /**
     * @brief Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
     * @param[in] input A "TensorPtr".
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr rsqrt(TensorPtr input,
                    const Quantization &quant = {});

    /**
     * @brief Reverses batch of sequences having different lengths specified by seq_lengths.
     *        For each slice 'i' iterating on the batch axis, the operator reverses the first seq_lengths[i]'
     *        element to 'time_axis', and copies elements whose indexes are beyond seq_lengths[i]' to the output.
     *        So, the output slice 'i' includes reverse sequences elements in the first seq_lengths[i]'
     *        and original elements copied beyond the seq_lengths index. Besides, another important parameter is
     *        len which indicates the effective length to reverse sequence. The [T, N, C] data format
     *        means [time_step, batch_size, channel].
     * @param[in] input A "TensorPtr".
     *                  Supports dimensions: 2-D,3-D.
     * @param[in] seq_lengths  A "TensorPtr". 1-D with length `input.dims(batch_axis)` and `max(seq_lengths) <=input.dims(seq_axis)`.
     * @param[in] seq_axis Specifies which axis is the time axis. Generally, the default value is 0 in [T, N, C] data format.
     *                     Parameter is temporarily not supported, using default.
     * @param[in] batch_axis Specifies which axis is the batch axis. Generally, the default value is 1 in [T, N, C] data format.
     *                     Parameter is temporarily not supported, using default.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr reverse_sequence(TensorPtr input,
                               TensorPtr seq_lengths,
                               int32_t seq_axis = 1,
                               int32_t batch_axis = 0);

    /**
     * @brief Computes the activation normalization of the previous layer for the given axis in a batch independently.
     *        That is, y = gamma * x / RMS(X), where RMS(X) = sqrt(sum(xi^2)/n), 1 <= i <= n.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] gamma A "TensorPtr", specifying the gamma constant tensor.
     *                  Generally, the shape of gamma is the shape mutilation of the input dimension that 'axis' specifies.
     * @param[in] axis  An int32 vector, A list of integers, indicates the axis to normalize across.
     *                  Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
     *                  By default, the value is -1 which represents the last dimension of the @p input.
     *                  Note that this parameter should be continuous from the last dimension.
     *                  For example, axis = [1, 2, 3] means statistics are across the Height, Width and Channel dimensions in NHWC format.
     * @param[in] epsilon  a value added to the denominator for numerical stability. Default: 1e-5.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input. Data type is inferred by @p quant.
     *         Scale and offset are described in @p quant.
     */
    TensorPtr rms_normalization(TensorPtr input,
                                TensorPtr gamma = nullptr,
                                const std::vector<int32_t> &axis = {-1},
                                const double epsilon = 0.00001,
                                const Quantization &quant = {});

    /**
     * @brief Round takes one input Tensor and rounds the values, element-wise, meaning it finds the nearest integer for each value.
     *        In case of halfs, the rule is to round them to the nearest even integer.
     *        The output tensor has the same shape and type as the input.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A "Quantization" object, specifying output quantization parameters.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr round(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief ScatterElements takes three inputs data, updates, and indices of the same rank r >= 1 and
     *        an optional attribute axis that identifies an axis of data (by default, the outer-most axis, that is axis 0).
     *        The output of the operation is produced by creating a copy of the input data, and then updating its value
     *        to values specified by updates at specific index positions specified by indices.
     *        Its output shape is the same as the shape of data.
     * @param[in] input A "TensorPtr", specifying the input data tensor.
     * @param[in] indices A "TensorPtr", specifying the indices.
     * @param[in] updates A "TensorPtr", specifying values to scatter into the output tensor.
     * @param[in] reducion An enumerate object of ScatterReduction, default is 'None'.
     *                     'None': no reduction applied.
     *                     'Add': reduction using the addition operation.
     *                     'Mul': reduction using the multiplication operation.
     *                     'Min': reduction using the min operation.
     *                     'Max': reduction using the max operation.
     * @param[in] axis A scalar int32, specifying the axis scatter on. Range is [-r, r-1), default is 0.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters, default to empty.
     * @return A "TensorPtr", which has the same shape as @p input.
     */
    TensorPtr scatter_elements(TensorPtr input,
                               TensorPtr indices,
                               TensorPtr updates,
                               ScatterReduction reduction = ScatterReduction::None,
                               int32_t axis = 0,
                               const Quantization &quant = {});

    /**
     * @brief Scatters updates into a tensor of input according to indices.
     *        ScatterND takes three inputs data tensor of rank r >= 1, indices tensor of rank q >= 1, and updates tensor of rank q + r - indices.shape[-1] - 1.
     *        The output of the operation is produced by creating a copy of the input data, and then updating its value to values specified by updates at
     *        specific index positions specified by indices. Its output shape is the same as the shape of input.
     *        Note that indices should not have duplicate entries. That is, two or more updates for the same index-location is not supported
     * @param[in] input A "TensorPtr", specifying the input data tensor.
     * @param[in] indices A "TensorPtr", specifying the indices.
     * @param[in] updates A "TensorPtr", specifying values to scatter into the output tensor.
     * @param[in] reducion An enumerate object of ScatterReduction, default is 'None'.
     *                     'None': no reduction applied.
     *                     'Add': reduction using the addition operation.
     *                     'Mul': reduction using the multiplication operation.
     *                     'Min': reduction using the min operation.
     *                     'Max': reduction using the max operation.
     * @param[in] quant A "Quantization" object, specifying the output quantization parameters, default to empty.
     * @return A "TensorPtr", which has the same shape as @p input.
     */
    TensorPtr scatter_nd(TensorPtr input,
                         TensorPtr indices,
                         TensorPtr updates,
                         ScatterReduction reduction = ScatterReduction::None,
                         const Quantization &quant = {});

    /**
     * @brief Reduce a tensor along the segment.
     * @param[in] input A "TensorPtr", specifying the input data tensor.
     * @param[in] segment_ids A int32_t vector, specifying segment ids. Its size is equal to the size
     *                        of data's first dimension. Values should be sorted and can be repeated.
     * @param[in] method The segment reduce method, where Min computes the minimum along segments of
     *                   a tensor. Max computes the maximum along segments of a tensor. Sum computes
     *                   the sum along segments of a tensor. Mean computes the mean along segments of
     *                   a tensor. Prod computes the multiply along segments of a tensor. Currently
     *                   only Sum is supported. Defaults to Sum.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset. Defaults to
     *                  empty.
     * @return A "TensorPtr", which has the same type as input.
     */
    TensorPtr segment_reduce(TensorPtr input,
                             const std::vector<int32_t> &segment_ids,
                             const SegmentReduceMethod method = SegmentReduceMethod::Sum,
                             const Quantization &quant = {});

    /**
     * @brief Takes an input tensor (X) and produces one output tensor (Y), where the scaled exponential linear unit function
     *        'f(x) = gamma * (alpha * (exp(x) -1)) for x <= 0, f(x) = gamma * x for x > 0', is applied to the tensor elementwise
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] alpha A float scalar, specifying the alpha in the formula, default is 1.67326.
     * @param[in] gamma A float scalar, specifiying the gamma in the formula, defulat is 1.0507.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset, default is empty.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr selu(TensorPtr input,
                   float alpha = 1.67326,
                   float gamma = 1.0507,
                   const Quantization &quant = {});

    /**
     * @brief Takes a tensor as input and outputs an 1D int32 tensor containing the shape of the input tensor.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] end A int32_t scalar, specifying the end axis for slicing the shape. negative value means counting dimensions from the back. if omitted, sizes of all axes upto(including) the last one will be included.
     * @param[in] start A int32_t scalar, specifying the start axis for slicing the shape, default is 0. negative value means counting dimensions from the back
     * @return A "TensorPtr", specifying the output tensor, which is the shape of the input tensor.
     */
    TensorPtr shape(TensorPtr input,
                    int32_t end,
                    int32_t start = 0);

    /**
     * @brief Shrink takes one input data (Tensor) and produces one Tensor output, having the same data type
     *        and shape with the input. It has two attributes, lambd and bias. The formula of this operator is:
     *          If x < -lambd, y = x + bias.
     *          If x > lambd, y = x - bias.
     *          Otherwise, y = O.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] lambd A float scalar, specifying the lambd in the formula, default is 0.5.
     * @param[in] bias A float scalar, specifiying the bias in the formula, defulat is 0.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset, default is empty.
     * @return A "TensorPtr", specifying the output tensor, which has the same data type and shape with @p input.
     */
    TensorPtr shrink(TensorPtr input,
                     float lambd = 0.5,
                     float bias = 0.0,
                     const Quantization &quant = {});

    /**
     * @brief Computes sigmoid activation on the input tensor element-wise. Formula describes as y = 1 / (1 + exp(-x))
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr sigmoid(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Computes sin of x element-wise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr sine(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Computes the hyperbolic sine of the given input tensor element-wise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr sinh(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Calculates the sign of the given input tensor (X) element-wise.
     *        That is, f(x) = 1 if x >0, f(x) = 0 if x == 0, and f(x) = -1 if x < 0.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr sign(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Takes one input tensor and produces one output tensor, where the SiLU or Swish function y = x * sigmoid(x),
     *        is applied to the tensor elementwise and 'sigmoid(x)' is the logistic sigmoid function
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr silu(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Extracts a strided slice of a tensor.
     *        Roughly speaking, this op extracts a slice of size (end - begin) / stride from the given input tensor.
     *        Starting at the location specified by begin the slice continues by adding stride to the index until
     *        all dimensions are not less than end. Note that a stride can be negative, which causes a reverse slice.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] begin An int32 vector, specifying the starts of the dimensions of the input tensor to be sliced.
     *            Size must be rank of @p input.
     * @param[in] end An int32 vector, specifying the ends of the dimensions of the input tensor to be sliced.
     *            Size must be rank of @p input.
     * @param[in] strides An int32 vector, strides of the dimensions of the input tensor to be sliced.
     *                    Size must be rank of @p input. The value cannot be zero.
     * @param[in] begin_mask An int32 scalar. If the i-th bit of @p begin_mask is set, @p begin[i] is ignored and
     *                       the fullest possible range in that dimension is used instead.
     * @param[in] end_mask An int32 scalar. If the i-th bit of @p end_mask is set, @p end[i] is ignored and
     *                       the fullest possible range in that dimension is used instead.
     * @param[in] shrink_axis_mask An int32 scalar. If the i-th bit of @p shrink_axis_mask is set, the i-th dimension specification
     *                             shrinks the dimensionality by 1, taking on the value at index @p begin[i].
     *                             In this case, the i-th specification must define a slice of size 1, e.g. @p begin[i] = x, @p end[i] = x + 1.
     * @return A "TensorPtr", specifying the output tensor, which has the same data type and scale/offset as @p input.
     *         Output shape will be inferred by @p begin, @p end, @p strides, @p begin_mask, @p end_mask, and @p shrink_axis_mask.
     */
    TensorPtr slice(TensorPtr input,
                    const std::vector<int32_t> &begin = {},
                    const std::vector<int32_t> &end = {},
                    const std::vector<int32_t> &strides = {},
                    uint32_t begin_mask = 0,
                    uint32_t end_mask = 0,
                    uint32_t shrink_axis_mask = 0);

    /**
     * @brief Produces a slice of the input tensor along multiple axes.
     *        Slice uses the starts, ends, axes and steps inputs to select a sub-tensor of its input data tensor.
     * @param[in] inputs A "TensorPtr", specifying the input tensors.
     *                   The order is [data, starts, ends, axes, steps].
     *                   The axes and steps are optional.
     * @return A "TensorPtr" list, specifying the output tensor, and output shape tensor.
     */
    TensorPtrList dynamic_slice(TensorPtrList inputs);

    /**
     * SlotUpdate implements the SlotUpdate operation.
     * The input of this operator is [X1, X2~XN], and the output is Y.
     * X1 means the input feature map,
     * that will be update by new indices X2~XN and output the last data to output tensor.
     * X2~XN means the indices. Every new indices input will be a new sub-batch of output.
     * The following equation describes the SlotUpdate operation:
     *      for j in output batch
     *          for i, indices in enmuerate(X2~XN)
     *              output[j,indices] += X1[j,i,:len(indices)]
     * Note that, in the following Inputs or Outputs sections:
     * X1 means the input feature.
     * X2~XN means indices that direct how to get input data.
     * @param[in] input A "TensorPtr", specifying the input tensor X1.
     * @param[in] indices A list of "TensorPtr", specifying the indices tensor X2~XN.
     * @param[in] quant A "Quantization" object, specifying the output tensor scale and offset.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr slot_update(TensorPtr input,
                          TensorPtrList indices,
                          const Quantization &quant = {});

    /**
     * @brief Takes one input tensor (X) and produces one output tensor (Y), where the function `y = x / (abs(x) + 1)`,
     *        is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output tensor scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr softsign(TensorPtr, const Quantization &quant = {});

    /**
     * @brief The operator computes the normalized exponential values for the given input:
     *        y = exp((x - max(x)) * beta) / sum(exp((x - max(x)) * beta))
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] axis An int32 scalar, specifying the dimension the activation would be performed on. Range should be
     *                 in [-rank, rank), rank is @p input rank. Default is -1.
     * @param[in] beta A float scalar, specifying the positive scaling factor for the exponent. Default is 1.0.
     * @param[in] quant A "Quantization" object, specifying the output tensor scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr softmax(TensorPtr input,
                      int32_t axis = -1,
                      float beta = 1.0,
                      const Quantization &quant = {});

    /**
     * @brief Softplus takes one input Tensor and produces one output Tensor where the softplus function, y = ln(exp(x) + 1),
     *        is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output tensor scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr softplus(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Sorts the elements of the input tensor along a given dimension in ascending order by value.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @return A "TensorPtr" list, specifying the output tensor, which has the same shape with @p input.
     *         It contains results and indices.
     */
    TensorPtrList sort(TensorPtr input);

    /**
     * @brief This operation divides "spatial" dimensions(height/width) of the input into a grid of blocks of shape block_shape,
     *        and interleaves these blocks with the "batch" dimension (0) such that in the output, the spatial dimensions
     *        correspond to the position within the grid, and the batch dimension combines both the position within a spatial block and
     *        the original batch position.
     * @param[in] input A "TensorPtr", specifying the input tensor, rank must be 4.
     * @param[in] block_shape An uint32 vector, size should be 2. The order is height, width.
     * @param[in] paddings A vector of uint32 pair, vector size should be same as 2. The order is
     *            [[height_start, height_end], [width_start, width_end]].
     * @return A "TensorPtr", which has same data type and scale/offset with @p input. Output shape can be inferred by
     *         @p block_shape and @p paddings.
     */
    TensorPtr space_to_batch(TensorPtr input,
                             const std::vector<uint32_t> &block_shape,
                             const std::vector<std::pair<uint32_t, uint32_t>> &paddings);

    /**
     * @brief SpaceToDepth rearranges blocks of spatial data into depth. this op outputs a copy of the input tensor
     *        where values from the height and width dimensions are moved to the depth dimension.
     * @param[in] input A "TensorPtr", specifying the input tensor, rank must be 4.
     * @param[in] block_size An uint32 scalar, specifying the size of spatial block.
     * @return A "TensorPtr", which has same data type and scale/offset with @p input. Output shape can be inferred by
     *         @p block_size.
     */
    TensorPtr space_to_depth(TensorPtr input, uint32_t block_size);

    /**
     * @brief split a tensor into a list of sub tensors.
     * @param[in] input input tensor
     * @param[in] num An uint32 scalar, specifying the split number. Along the axis, @p input[axis] must be
     *            divided by num.
     * @param[in] axis the dimension along which to split. Range should be in [-rank, rank).
     * @return vector of "TensorPtr". Tensor will have the same data type and scale/offset with @p input.
     *         Output tensors shape will be inferred by @p num_or_splits and @p axis.
     */
    std::vector<TensorPtr> split(TensorPtr input,
                                 uint32_t num,
                                 int32_t axis = -1);

    /**
     * @brief split a tensor into a list of sub tensors.
     * @param[in] input input tensor
     * @param[in] splits this is a vector, input is split into each size along axis
     * @param[in] axis the dimension along which to split. Range should be in [-rank, rank).
     * @param[in] ds_splits the dynamic split size.
     * @return vector of "TensorPtr". Tensor will have the same data type and scale/offset with @p input.
     *         Output tensors shape will be inferred by @p num_or_splits and @p axis.
     */
    std::vector<TensorPtr> split(TensorPtr input,
                                 const std::vector<uint32_t> &splits = {},
                                 int32_t axis = -1,
                                 const std::vector<tongue::Expr> &ds_splits = {});

    /**
     * @brief Removes dimensions of size 1 from the shape of a tensor.
     *        If paramter "axis" is not specified,will remove all size 1 dimensions.
     *        The output tensor has the same values as input tensor.
     * @param[in] input A "TensorPtr".
     * @param[in] axis  An optional list of "int32_t". Defaults to {}.
     *                  If specified, only squeezes the dimensions listed..
     * @return A "TensorPtr", with type as the tensors in @p input, but has one or more dimensions ofsize 1 removed..
     */
    TensorPtr squeeze(TensorPtr input,
                      const std::vector<int32_t> &axis = {});

    /**
     * @brief Returns a new tensor with the square of the elements of input.
     * @param[in] input A "TensorPtr".
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr square(TensorPtr input,
                     const Quantization &quant = {});

    /**
     * @brief Returns (input1-input2)*(input1-input2) element-wise. This operator supports broadcasting.
     * @param[in] input1 A "TensorPtr", specifying the first operand tensor.
     * @param[in] input2 A "TensorPtr", specifying the second operand tensor.
     * @param[in] quant A Quantization class, include scale and zeropoint.
     * @return A "TensorPtr", with the same type as @p input1.
     */
    TensorPtr squared_difference(TensorPtr input1,
                                 TensorPtr input2,
                                 const Quantization &quant = {});

    /**
     * @brief Computes element-wise square root of the input tensor.
     * @param[in] input A "TensorPtr".
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr sqrt(TensorPtr input,
                   const Quantization &quant = {});

    /**
     * @brief Performs elementwise subtraction of each of the input tensors (with NumPy-style broadcasting support).
     *        All inputs and outputs must have the same data type.
     *        Besides, this operator supports multidirectional (that is, NumPy-style) broadcasting
     *        and sums all the input tensors into the output tensor under the OPERATOR LIBRARY support.
     * @param[in] input1 A "TensorPtr", specifying the first tensor.
     * @param[in] input2 A "TensorPtr", specifying the second tensor, which must have the same
     *            data type with @p input1, but can have different scale and offset with @p input1.
     * @param[in] with_activation An Activation object, default empty, represents "NONE", supports bellow:
     *            - Activation("NONE")
     *            - Activation("RELU")
     *            - Activation("RELU1"), will forward to Activation("CLIP", -1., 1.)
     *            - Activation("RELU6"), will forward to Activation("CLIP", 0., 6.)
     *            - Activation("LEAKYRELU", negative_slope_value)
     *                LeakyRelu takes input data (Tensor) and an argument alpha, then produces one output data (Tensor)
     *                where the function f(x) = alpha * x for x <0, f(x) = x for x >=0, is applied to
     *                the data tensor elementwise, where 'alpha' is the slope parameter.
     *            - Activation("CLIP", clip_min, clip_max)
     *                Where, clip_min (or clip_max) means the minimum (or maximum) saturation threshold
     *                during the clip operation to produce the output tensor after convolution.
     *            - Activation("PRELU", negative_slope)
     *                  PRelu takes input data (Tensor) and slope tensor as input,then produces one output data (Tensor) where the function,
     *                  f(x) = slope * x for x < 0, f(x) = x for x >= 0, is applied to the data tensor elementwise.
     *                  This operator supports unidirectional broadcasting (tensor slope should be
     *                  unidirectional broadcastable to the input tensor).
     *            - ...
     * @param[in] quant A "Quantization" object, specifying the output scale/offset.
     * @return A "TensorPtr", specifying the output tensor pointer.
     *         It has the same data type as @p inputs. The shape is after multibroadcast shape.
     *         Output can have the different scale and offset with @p inputs.
     */
    TensorPtr sub(TensorPtr input1,
                  TensorPtr input2,
                  const Activation &with_activation = {},
                  const Quantization &quant = {});

    /**
     * @brief Calculate the sufficient statistics for the mean and variance of x.
     *        Note that, in the following Inputs or Outputs sections:
     *        X1: the input tensor.
     *        X2: A Tensor containing the value by which to shift the data for numerical stability.
     *        Y1: the shifted sum of the elements of the input tensor along the axis.
     *        Y2: the shifted sum of squares of the elements of the input tensor along the axis.
     * @param[in] input1 A "TensorPtr", specifying input tensor X1.
     * @param[in] input2 A "TensorPtr", specifying input tensor X2.
     * @param[in] axis A list of integers to indicate the axis to calcaulate across.
     *                 Typically, this is the feature axis and the leaving dimensions are typically the batch axis.
     *                 Generally, the value is -1 which represents the last dimension of the input tensor (X).
     * @param[in] quant A "Quantization" vector, specifying the (Y1,Y2) scale and offset.
     * @return A "TensorPtr" list, which are output tensor Y1,Y2.
     */
    std::vector<TensorPtr> sufficient_statistics(TensorPtr input1,
                                                 TensorPtr input2,
                                                 const std::vector<int32_t> &axis = {-1},
                                                 const std::vector<Quantization> &quant = {});

    /**
     * @brief Dynamically choose the first non-empty input tensor as the output.
     * @param[in] inputs A "TensorPtr" list, specifying the input tensors.
     *                   The type and shape of each input tensor must be the same.
     * @return A "TensorPtr", specifying the output tensor.
     */
    TensorPtr dynamic_switch(TensorPtrList inputs);

    /**
     * @brief Extracts a strided slice of a tensor.
     *        Roughly speaking, this op extracts a slice of size (end - begin) / stride from the given input tensor.
     *        Starting at the location specified by begin the slice continues by adding stride to the index until
     *        all dimensions are not less than end. Note that a stride can be negative, which causes a reverse slice.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] begin An int32 vector, specifying the starts of the dimensions of the input tensor to be sliced.
     *            Size must be rank of @p input.
     * @param[in] end An int32 vector, specifying the ends of the dimensions of the input tensor to be sliced.
     *            Size must be rank of @p input.
     * @param[in] strides An int32 vector, strides of the dimensions of the input tensor to be sliced.
     *                    Size must be rank of @p input. The value cannot be zero.
     * @param[in] begin_mask An int32 scalar. If the i-th bit of @p begin_mask is set, @p begin[i] is ignored and
     *                       the fullest possible range in that dimension is used instead.
     * @param[in] end_mask An int32 scalar. If the i-th bit of @p end_mask is set, @p end[i] is ignored and
     *                       the fullest possible range in that dimension is used instead.
     * @param[in] shrink_axis_mask An int32 scalar. If the i-th bit of @p shrink_axis_mask is set, the i-th dimension specification
     *                             shrinks the dimensionality by 1, taking on the value at index @p begin[i].
     *                             In this case, the i-th specification must define a slice of size 1, e.g. @p begin[i] = x, @p end[i] = x + 1.
     * @return A "TensorPtr", specifying the output tensor, which has the same data type and scale/offset as @p input.
     *         Output shape will be inferred by @p begin, @p end, @p strides, @p begin_mask, @p end_mask, and @p shrink_axis_mask.
     */
    TensorPtr strided_slice(TensorPtr input,
                            const std::vector<int32_t> &begin = {},
                            const std::vector<int32_t> &end = {},
                            const std::vector<int32_t> &strides = {},
                            uint32_t begin_mask = 0,
                            uint32_t end_mask = 0,
                            uint32_t shrink_axis_mask = 0);

    /**
     * @brief Takes an input tensor (X) and produces one output tensor (Y), where, the function
     *        'f(x) = x / (1 + exp(alpha * -1 * x))', is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] alpha A float scalar, specifying the alpha in the formula, default is 1.0.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset, default is empty.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr swish(TensorPtr input,
                    float alpha = 1.0,
                    const Quantization &quant = {});

    /**
     * @brief Calculates the tangent of the given input tensor, element-wise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     */
    TensorPtr tan(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Computes hyperbolic tangent of input tensor element-wise. The formula: y = tanh(x)
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor, which has the same shape with @p input.
     *         Output data type can be inferred by @p quant, and scale/offset are described as @p quant.
     */
    TensorPtr tanh(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief ThresholdedRelu takes one input data (Tensor) and produces one output data (Tensor) where the rectified
     *        linear function, y = x for x > alpha, y = 0 otherwise, is applied to the tensor elementwise.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output scale and offset.
     * @return A "TensorPtr", specifying the output tensor. Output has the same data type and shape with @p input.
     */
    TensorPtr thresholded_relu(TensorPtr input, float alpha = 1.0, const Quantization &quant = {});

    /**
     * @brief Constructs a tensor by tiling a given tensor. This operation creates a new tensor by replicating input repeates times.
     *        The output tensor's i-th dimension has input.dims(i) * repeats[i] elements, and the values of input
     *        are replicated repeats[i] times along the i-th dimension.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] repeats An uint32 vector, including numbers of repeated copies along the input's dimensions.
     *            Size must be rank of @p input.
     * @param[in] ds_repeats The dynamic repeats parameters.
     * @return A "TensorPtr", specifying the output tensor. Output has the same data type and scale/offset with @p input.
     *         Output shape will be inferred by @p repeats.
     */
    TensorPtr tile(TensorPtr input,
                   const std::vector<uint32_t> &repeats = {},
                   const std::vector<tongue::Expr> &ds_repeats = {});

    /**
     * @brief Constructs a tensor by tiling a given tensor. This operation creates a new tensor by replicating input repeates times.
     *        The output tensor's i-th dimension has input.dims(i) * repeats[i] elements, and the values of input
     *        are replicated repeats[i] times along the i-th dimension.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] repeates An int32 "TensorPtr", specifying the input repeats. Which should be rank=1, dimension=rank(@p input).
     * @param[in] shape An int32 "TensorPtr", specifying the output tensor shape.
     * @return A "TensorPtr", specifying the output tensor. Output has the same data type and scale/offset with @p input.
     *         Output shape will be inferred by @p repeats.
     */
    TensorPtr tile(TensorPtr input, TensorPtr repeats, const TensorShape &shape);

    /**
     * @brief Retrieves the top-K largest or smallest elements along a specified axis.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] k A "TensorPtr", specifying positive value corresponding to the number of top elements to retrieve.
     * @param[in] axis A int32 scalar, meaning the dimension on which to perform the sort. Default is -1,
     *            range should be in [-rank , rank), rank is @p input rank.
     * @param[in] largest A boolean scalar, indicating whether to return the top-K largest or smallest elements. Default is true.
     * @param[in] sorted A boolean scalar, indicating whether to return the elements in sorted descending  order.
     * @param[in] select_index A string variable, selecting the last index or the first index if the maximum element appears many times.
     *                         Three mode provides in enumerate TopkSelectMethod: First, Last, Random.
     * @return A "TensorPtr" vector, the first "TensorPtr" is containing the k largest elements along specified axis.
     *         The shape of this tensor, axis dimension will be replaced by k. Data type and scale/offset are same as @p input.
     *         Second "TensorPtr", containing the indices of values within the last dimension of input. Data type is "UINT32", and shape
     *         is same with first "TensorPtr".
     * @todo: layerlib select_index cannot match IR definition, will modify in the future.
     */
    std::vector<TensorPtr> topk(TensorPtr input,
                                TensorPtr k,
                                int32_t axis = -1,
                                bool largest = true,
                                bool sorted = true,
                                TopkSelectMethod select_index = TopkSelectMethod::Random);

    /**
     * @brief Retrieves the top-K largest or smallest elements along a specified axis.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] k A uint32 scalar, containing a single positive value corresponding to the number of top elements to retrieve.
     * @param[in] axis A int32 scalar, meaning the dimension on which to perform the sort. Default is -1,
     *            range should be in [-rank , rank), rank is @p input rank.
     * @param[in] largest A boolean scalar, indicating whether to return the top-K largest or smallest elements. Default is true.
     * @param[in] sorted A boolean scalar, indicating whether to return the elements in sorted descending  order.
     * @param[in] select_index A string variable, selecting the last index or the first index if the maximum element appears many times.
     *                         Three mode provides in enumerate TopkSelectMethod: First, Last, Random.
     * @return A "TensorPtr" vector, the first "TensorPtr" is containing the k largest elements along specified axis.
     *         The shape of this tensor, axis dimension will be replaced by k. Data type and scale/offset are same as @p input.
     *         Second "TensorPtr", containing the indices of values within the last dimension of input. Data type is "UINT32", and shape
     *         is same with first "TensorPtr".
     * @todo: layerlib select_index cannot match IR definition, will modify in the future.
     */
    std::vector<TensorPtr> topk(TensorPtr input,
                                uint32_t k,
                                int32_t axis = -1,
                                bool largest = true,
                                bool sorted = true,
                                TopkSelectMethod select_index = TopkSelectMethod::Random);
    /**
     * @brief Transposes the input tensor, permuting the dimensions according to the @p perm.
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] perm An uint32 vector, permuting the axis according to the value given.
     *            Size must be rank of @p input.
     * @return A "TensorPtr", specifying the output tensor. Output has the same data type and scale/offset with @p input.
     *         Output shape will be inferred by @p perm.
     */
    TensorPtr transpose(TensorPtr input, const std::vector<uint32_t> &perm = {});

    /**
     * @brief The truncated value of the input x is the nearest integer i which is closer to zero than x, element-wise.
     * @param[in] input A "TensorPtr", specifying input tensor.
     * @param[in] quant A Quantization class ,include scale and zeropoint.
     * @return A "TensorPtr",with the same shape and type as the tensors in "input".
     */
    TensorPtr trunc(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Performs the upsample operation on the inputs according to the index in the corresponding indices tensor
     *        and returns the output tensor Y. Generally, the input indices tensor has three flattened formats.
     *        For example, a maximum value in the original tensor at position '[n, h, w, c]' will be flattened as:
     *        'indices = h * width + w' if flatten_dim = HW
     *        'indices = (h * width + w) * channel + c' if flatten_dim = HWC
     *        'indices = ((n * height + h) * width + w) * channel + c' if flatten_dim = NHWC
     *        Where, height (or width, channel) represents the size of corresponding H (or W, C) dimension under NHWC
     *        data format. Generally, the indices are located at ([0, height], [0, width]) originally before flattening
     *        even though padding is involved. Where the range is a left closed and right open interval.
     * @param[in] input A Tensor, specifying the input data tensor.
     * @param[in] indices A Tensor, specifying the indices tensor.
     * @param[in] shape A list of integer, specifying the shape of the output tensor.
     * @param[in] flatten_dim A string indicating the flatten mode in the indices tensor. Support value: NHWC, HWC, HW.
     *                        Default value is "HWC".
     * @param[in] storage_order An int32 scalar. specifying the storage order of the tensor. 0 is row(h) major, and 1 is column(w) major.
     * @return A Tensor, which has the same shape as param shape and same dtype as input tensor.
     */
    TensorPtr upsample_by_index(TensorPtr input, TensorPtr indices, const std::vector<uint32_t> &shape, FlattenDim flatten_dim = FlattenDim::HWC, int32_t storage_order = 0);

    /**
     * @brief Using a tensor of booleans mask and input tensors x1 and x2 select values elementwise from both input tensors:
     *        y[i] = condition[i] ? x1[i] : x2[i].
     *        If x1 and x2 are not provided, it will return the indices of condition that are non-zero, in the form of a 2-D
     *        tensor with shape [n, d], where n is the number of non-zero elements in condition, and d is the number of axes
     *        of condition.
     * @param[in] condition A "TensorPtr", acting as a mask that chooses, based on the value at each element,
     *                      whether the corresponding element in the output should be taken from @p input1 (if 1) or @p input2 (if 0).
     *                      It has the same shape as @p input1 and @p input2. The value should be 1 or 0.
     *                      If input1 and input2 are not provided, then it can have a dtype of bool or any numeric dtype.
     * @param[in] input1 A "TensorPtr", specifying the first input tensor.
     * @param[in] input2 A "TensorPtr", specifying the second input tensor, which has the same shape and data type with @p input1.
     * @param[in] quant A "Quantization" object, specifying the output tensor quantization parameters.
     * @return A "TensorPtr", specifying the output tensor. Output has the same shape and data type with @p input1 if input1 and
     *         input2 are provided. Otherwise, the output's shape is 2-D and data type is int32.
     *         The output can have different scale and offset.
     */
    TensorPtr where(TensorPtr condition, TensorPtr input1 = nullptr, TensorPtr input2 = nullptr, const Quantization &quant = {});

    /**
     * @brief Conversion of one or more images from YUV to RGB. Generally, this operation is used for preprocessing.
     *        Here, the term YUV is commonly encoded using Y'CbCr; while the RGB term is encoded using R'dG'dB'd.
     *        There are three standards which are so-called "BT.709". For 8 bits color,
     *        [0, 255] range is referred as Full Range, which is considerred here; while [16, 235] range for Y and
     *        [16, 240] for UN are referred as Narrow Range. Also, all the conversions are based on integral approximate
     *        (or using fixed-point arithmetic as an alternative formulation) in this operation.
     * @param[in] input A "TensorPtr", specifying YUV image, with a shape of [N, D].
     * @param[in] shape A vector of interger, specifying the shape of output tensor.
     * @param[in] format An enum value of ImageFormat specifying the input image format which will be converted.
     *                   The optional values are [I420, YV12, NV12, NV21].
     * @param[in] bits An integer, specifying the bits to store a Y, U or V values in memory, such as DDR device.
     * @param[in] conversion An enum value of ImageConversion, which means the different coefficient matrix to convert
     *                       between YUV and RGB format. Especially, the 'SELF' means the conversion coefficients are
     *                       customized which is from the configuration. The optional values are [BT709].
     * @param[in] coefficient A list of color space coefficient (CSC) for conversion, which is in order of
                              '[input_offset, output_offset, coefficient]'.
     * @return A "TensorPtr", specifying the output RGB image tensor, with a shape of [N, H, W, 3].
    */
    TensorPtr yuv_to_rgb(TensorPtr input,
                         const std::vector<uint32_t> &shape,
                         const ImageFormat format = ImageFormat::I420,
                         const int bits = 8,
                         const ImageConversion conversion = ImageConversion::BT709,
                         const std::vector<float_t> &coefficient = {});

    /**
     * @brief           Returns the fraction of zeros in input.
     *
     * @param[in] input A "TensorPtr", specifying the input tensor.
     * @param[in] quant A "Quantization" object, specifying the output tensor quantization parameters.
     * @return A "TensorPtr", specifying the output tensor. The output's shape is 1-D and data type is float32.
     */
    TensorPtr zero_fraction(TensorPtr input, const Quantization &quant = {});

    /**
     * @brief Updates the given cache in inplace mode.
     *
     * @param[in] cache A "TensorPtr", specifying the cache to update. Its shape is 3-D or 4-D.
     *            If shape is 3-D, only one cache block is allowed, block_index must be 0,
     *            and shape[1] means the max token length. If shape is 4-D,
     *            shape[0] means the number of blocks in cache, and shape[2] means the max token length.
     * @param[in] position_id A "TensorPtr", specifying the current token index.
     *            Its shape is 2-D, and type can be int32 or uint32. Currently,
     *            only continuous indexes are supported in prefill mode.
     * @param[in] value A "TensorPtr", specifying the current value to fill into cache.
     *            Its shape is 4-D, and shape[2] must be less than or equal to cache's max token length.
     * @param[in] block_index A int32 scalar, specifying which block in cache to update.
     *            It must be less than cache's block count.
     * @param[in] axis A int32 scalar, specifying which axis to update. Supported value: "2", "3".
     *            Default value is "2".
     * @return A "TensorPtr" vector, specifying the output tensors. It has two outputs,
     *         and output0 is the updated cache block, output1 is the updated whole cache.
     */
    std::vector<TensorPtr> cacheupdate(TensorPtr cache,
                                       TensorPtr position_id,
                                       TensorPtr value,
                                       int block_index,
                                       int axis = 2);

    /**
     * @brief            Create a weight tensor pointer.
     *                   It is used by interfaces with the parameter "weight".
     * @param[in] g      graph pointer to which weight belongs.
     * @param[in] data   The weight data pointer.Generally, the data are stacked in order of shape.
     * @param[in] shape  The weight shape.
     *                   Support shape: conv2d:[out_channel, kernel_y, kernel_x, input_channel]
     *                                  depthwise_conv2d: [out_channel, kernel_y, kernel_x, 1]
     *                                  fully_connected:[out_channel, input_channel]
     *                                  conv2d_transpose:[out_channel, kernel_y, kernel_x, input_channel]
     *                                  conv1d: [out_channel, kernel_x, input_channel]
     *                                  conv1d_transpose:[out_channel, kernel_x, input_channel]
     *                                  conv_group: [out_channel, kernel_y, kernel_x, input_channel].
     * @param[in] dtype  A "Tensor::Dtype", specifying the primitiv data type.
     *                   Optional parameter, the default value is Tensor::FP32;
     * @param[in] quant  A "Quantization" object, specifying the weight scale/offset.
     *                   Optional parameter, the default value is 1.0/0;
     * @return A "TensorPtr", specifying the weight tensor.
     */

    TensorPtr rotary_embedding(
        TensorPtr input,
        TensorPtr cos_cache,
        TensorPtr sin_cache,
        int num_heads,
        int interleaved,
        int rotary_embedding_dim);

    TensorPtr rotary_embedding(
        TensorPtr input,
        TensorPtr cos_cache,
        TensorPtr sin_cache,
        TensorPtr position_ids,
        int num_heads,
        int interleaved,
        int rotary_embedding_dim);

    TensorPtr make_weight(GraphPtr g,
                          const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype = Tensor::FP32,
                          const Quantization &quant = {});

    /**
     * @brief            Create a weight tensor pointer.
     * @param[in] data   The weight data pointer.
     * @param[in] shape  The weight shape.
     *                   Support shape: conv2d:[out_channel, kernel_y, kernel_x, input_channel]
     *                                  depthwise_conv2d: [out_channel, kernel_y, kernel_x, 1]
     *                                  fully_connected:[out_channel, input_channel]
     *                                  conv2d_transpose:[out_channel, kernel_y, kernel_x, input_channel]
     *                                  conv1d: [out_channel, kernel_x, input_channel]
     *                                  conv1d_transpose:[out_channel, kernel_x, input_channel]
     *                                  conv_group: [out_channel, kernel_y, kernel_x, input_channel].
     * @param[in] dtype  A "Tensor::Dtype", specifying the primitiv data type.
     *                   Optional parameter, the default value is Tensor::FP32.
     * @param[in] quant  A "Quantization" object, specifying the weight scale/offset.
     *                   Optional parameter, the default value is 1.0/0;
     * @param[in] g      graph pointer to which weight belongs.
     * @return A "TensorPtr", specifying the weight tensor.
     */
    TensorPtr make_weight(const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype = Tensor::FP32,
                          const Quantization &quant = {},
                          GraphPtr g = nullptr);

    /**
     * @brief            Create a bias tensor pointer.
     *                   It is used by interfaces with the parameter "bias".
     * @param[in] g      graph pointer to which bias belongs.
     *                   Must be consistent with "ops::input".
     * @param[in] data   The bias data pointer.
     * @param[in] data_len The bias data length.
     * @param[in] dtype  A "Tensor::Dtype", specifying the primitiv data type.
     *                   Optional parameter, the default value is Tensor::INT32.
     * @param[in] quant  A "Quantization" object, specifying the bias scale/offset.
     *                   Optional parameter, the default value is 1.0/0;
     * @return A "TensorPtr", specifying the bias tensor.
     */
    TensorPtr make_bias(GraphPtr g,
                        const void *data,
                        uint32_t data_len,
                        Tensor::Dtype dtype = Tensor::INT32,
                        const Quantization &quant = {});

    /**
     * @brief            Create a bias tensor pointer.
     *                   It is used by interfaces with the parameter "bias".
     * @param[in] data   The bias data pointer.
     * @param[in] data_len The bias data length.
     * @param[in] dtype  A "Tensor::Dtype", specifying the primitiv data type.
     *                   Optional parameter, the default value is Tensor::INT32.
     * @param[in] quant  A "Quantization" object, specifying the bias scale/offset.
     *                   Optional parameter, the default value is 1.0/0;
     * @param[in] g      graph pointer to which bias belongs.
     *                   Must be consistent with "ops::input".
     * @return A "TensorPtr", specifying the bias tensor.
     */
    TensorPtr make_bias(const void *data,
                        uint32_t data_len,
                        Tensor::Dtype dtype = Tensor::INT32,
                        const Quantization &quant = {},
                        GraphPtr g = nullptr);

    /**
     * @brief Create a common tensor, and return the pointer of tensor.
     * @param[in] g A "Graph" pointer, specifying the tensor belongs to.
     * @param[in] data A void pointer points to the data memory.
     * @param[in] shape A "TensorShape" object, specifying the input tensor shape.
     * @param[in] dtype A "Tensor::Dtype" enumerate object, specifying the data type of the tensor.
     * @param[in] quant A "Quantization" object, specifying the scale and offset of the tensor.
     * @param[in] layout A "DataLayout" enumerate object, specifying the layout of the tensor.
     * @param[in] name A string, specfiying the tensor name, which can use "const" as default.
     * @return A "TensorPtr", specifying the generated tensor.
     */
    TensorPtr make_tensor(GraphPtr g,
                          const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype = Tensor::FP32,
                          const Quantization &quant = {},
                          DataLayout layout = DataLayout_NHWC,
                          const std::string &name = "const");

    /**
     * @brief Create a common tensor, and return the pointer of tensor.
     * @param[in] data A void pointer points to the data memory.
     * @param[in] shape A "TensorShape" object, specifying the input tensor shape.
     * @param[in] dtype A "Tensor::Dtype" enumerate object, specifying the data type of the tensor.
     * @param[in] quant A "Quantization" object, specifying the scale and offset of the tensor.
     * @param[in] layout A "DataLayout" enumerate object, specifying the layout of the tensor.
     * @param[in] name A string, specfiying the tensor name, which can use "const" as default.
     * @param[in] g A "Graph" pointer, specifying the tensor belongs to. It allows the default nullptr,
     *              system will bind to the specified graph if there only one graph.
     * @return A "TensorPtr", specifying the generated tensor.
     */
    TensorPtr make_tensor(const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype = Tensor::FP32,
                          const Quantization &quant = {},
                          DataLayout layout = DataLayout_NHWC,
                          const std::string &name = "const",
                          GraphPtr g = nullptr);

} // namespace ops
} // namespace aipubt

#endif //! CORE_OPS_OPS_H
