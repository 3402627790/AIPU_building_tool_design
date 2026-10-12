//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2024-11-15 14:36:32
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-06-25 17:16:34
 */
#include "tongue/ffc/tensor.h"

namespace ffc
{
std::ostream &operator<<(std::ostream &oss, TensorDesc &t)
{
    return oss << t.to_string();
}
std::ostream &operator<<(std::ostream &oss, Tensor &t)
{
    return oss << t.to_string();
}
std::ostream &operator<<(std::ostream &oss, DType &dt)
{
    const std::string names[] = {
        "int4",
        "uint4",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "fp4e2m1fn",
        "fp8e4m3fn",
        "fp8e5m2",
        "fp16",
        "bfp16",
        "fp32",
    };
    return oss << names[(uint32_t)dt];
}
std::ostream &operator<<(std::ostream &oss, Layout &lo)
{
    const std::string names[] = {
        "Flat",
        "NHWC",
        "NCHWC32",
        "NCHWC16",
    };
    return oss << names[(uint32_t)lo];
}
std::ostream &operator<<(std::ostream &oss, uOp &op)
{
    const std::string names[] = {
        "None",
        "Argument",
        "MemCpy",
        "MemSet",
        "MatMul",
        "FC",
        "Conv",
        "DwConv",
        "DeConv",
        "Add",
        "Sub",
        "Mul",
        "RShift",
        "Lut",
        "Pool",
        "Resize",
        "Max",
        "Sum",
        "Slice",
        "StrideSlice",
        "LayoutCvt",
        "Cast",
    };
    return oss << names[(uint32_t)op];
}
static Tensor eltwise(uOp op, Tensor input, float value)
{
    Tensor ret(op);
    ret.add_input(input);
    ret->output.shape = input->output.shape;
    ret->output.type = input->output.type;
    ret->output.layout = input->output.layout;
    ret->params["value"] = *((uint32_t *)&value);
    return ret;
}
static Tensor eltwise(uOp op,
                      Tensor a,
                      Tensor b,
                      int real_iw,
                      int real_ihw,
                      int ihw_offset)
{
    auto infer_shape = [](std::vector<uint32_t> a, std::vector<uint32_t> b)
    {
        std::vector<uint32_t> ret;
        auto max_size = std::max(a.size(), b.size());
        if (max_size == 0)
        {
            return ret;
        }
        while (a.size() < max_size)
        {
            a.insert(a.begin(), 1);
        }
        while (b.size() < max_size)
        {
            b.insert(b.begin(), 1);
        }
        for (size_t i = 0; i < max_size; ++i)
        {
            if (a[i] >= b[i] && a[i] % b[i] == 0)
            {
                ret.push_back(a[i]);
            }
            else if (a[i] < b[i] && b[i] % a[i] == 0)
            {
                ret.push_back(b[i]);
            }
        }
        if (ret.size() != max_size)
        {
            ret.clear();
        }
        return ret;
    };
    Tensor ret(op);
    ret.add_input(a);
    ret.add_input(b);
    if ((a->output.layout == Layout::NCHWC16 || a->output.layout == Layout::NCHWC32) &&
        a->output.shape.size() == 5)
    {
        if ((b->output.layout == Layout::NCHWC16 || b->output.layout == Layout::NCHWC32) &&
            b->output.shape.size() == 5)
        {
            ret->output.shape = infer_shape(a->output.shape, b->output.shape);
        }
        else if ((b->output.layout == Layout::NHWC && b->output.shape.size() == 4) ||
                 (b->output.layout == Layout::Flat && b->output.shape.size() == 1))
        {
            std::vector<uint32_t> shape = {
                a->output.shape[0],
                a->output.shape[2],
                a->output.shape[3],
                a->output.shape[1] * a->output.shape[4],
            };
            auto out_shape = infer_shape(shape, b->output.shape);
            uint32_t align = 32 / a->output.element_size();
            ret->output.shape = {
                out_shape[0],
                out_shape[3] / align,
                out_shape[1],
                out_shape[2],
                align,
            };
        }
        ret->output.type = a->output.type;
        ret->output.layout = a->output.layout;
    }
    ret->params["real_iw"] = real_iw;
    ret->params["real_ihw"] = real_ihw;
    ret->params["ihw_offset"] = ihw_offset;
    return ret;
}
static Tensor reduce(uOp op,
                     Tensor input,
                     const std::vector<int> &axis,
                     const std::string &with_sfu,
                     int real_iw,
                     int real_ihw,
                     int ihw_offset)
{
    static const std::map<std::string, int> sfu_types = {
        {"none", 0},
        {"rcp", 1},
        {"sqrt", 2},
        {"rsqrt", 3},
    };
    Tensor ret(op);
    ret.add_input(input);
    if (input->output.type == DType::FP16 &&
        input->output.layout == Layout::NCHWC16 &&
        input->output.shape.size() == 5 &&
        axis.size() == 1 && (axis[0] == 3 || axis[0] == -1))
    {
        ret->output.shape = {
            input->output.shape[0],
            input->output.shape[2],
            input->output.shape[3],
            1,
        };
        ret->output.type = DType::FP32;
        ret->output.layout = Layout::NHWC;
    }
    ret->params["with_sfu"] = sfu_types.at(with_sfu);
    ret->params["real_iw"] = real_iw;
    ret->params["real_ihw"] = real_ihw;
    ret->params["ihw_offset"] = ihw_offset;
    return ret;
}
static Tensor cast(Tensor input, DType out_dtype)
{
    uint32_t align = 32 / TensorDesc::get_element_size(out_dtype);
    Tensor ret(uOp::Cast);
    ret.add_input(input);
    ret->output.shape = input->output.shape;
    ret->output.type = out_dtype;
    ret->output.layout = align == 16 ? Layout::NCHWC16 : Layout::NCHWC32;
    return ret;
}
static Tensor layoutcvt(Tensor input, Layout out_layout)
{
    Tensor ret(uOp::LayoutCvt);
    ret.add_input(input);
    if ((input->output.layout == Layout::NCHWC16 || input->output.layout == Layout::NCHWC32) &&
        out_layout == Layout::NHWC)
    {
        ret->output.shape = {
            input->output.shape[0],
            input->output.shape[2],
            input->output.shape[3],
            input->output.shape[1] * input->output.shape[4],
        };
        ret->output.type = input->output.type;
        ret->output.layout = out_layout;
    }
    return ret;
}
/**
 * @brief matmul
 * matmul supports a matrix A time to a matrix B, the flowing shape is available without considering trans_a,trans_b
 * A: [M,K] B:[K,N]  -> [M,N]
 * A: [1,M,K] B:[1,K,N] -> [1,M,N]
 * A: [1,K0,h,w,16] B:[1,K0,H,W,16] -> [1,1,h*w,H*W] for 16bit data type
 *
 * @param a
 * @param b
 * @param trans_a
 * @param trans_b
 * @return Tensor
 */
Tensor Tensor::matmul(Tensor a,
                      Tensor b,
                      bool trans_a,
                      bool trans_b)
{
    Tensor ret(uOp::MatMul);
    ret.add_input(a);
    ret.add_input(b);
    // compute output shape
    if (a->output.layout == Layout::NCHWC16 && b->output.layout == Layout::NCHWC16 &&
        a->output.type == DType::FP16 && b->output.type == DType::FP16 &&
        a->output.shape.size() == 5 && b->output.shape.size() == 5 &&
        !trans_a)
    {
        // MK * KN -> MN
        if (trans_b)
        {
            ret->output.shape = {
                a->output.shape[0],
                b->output.shape[3] / 16,
                a->output.shape[2],
                a->output.shape[3],
                16,
            };
        }
        else
        {
            ret->output.shape = {
                a->output.shape[0],
                b->output.shape[1],
                a->output.shape[2],
                a->output.shape[3],
                b->output.shape[4],
            };
        }
        ret->output.type = DType::FP16;
        ret->output.layout = Layout::NCHWC16;
    }
    ret->params["trans_a"] = trans_a;
    ret->params["trans_b"] = trans_b;

    return layoutcvt(ret, Layout::NHWC);
}
Tensor Tensor::conv2d(Tensor input,
                      Tensor weight,
                      Tensor bias,
                      const std::array<int, 2> &kernel,
                      const std::array<int, 2> &stride,
                      const std::array<int, 4> &padding,
                      const std::array<int, 2> &dilation,
                      int group,
                      int ic_gsize,
                      int gptq_mode,
                      DType out_dtype,
                      float out_scale,
                      Layout out_layout,
                      int real_iw,
                      int real_ihw,
                      int ihw_offset,
                      int real_oc,
                      int oc_offset)
{
    auto infer_shape = [](int fm, int k, int s, int d, int p0, int p1) -> uint32_t
    {
        auto new_k = k + (k - 1) * (d - 1);
        return (fm - 1) * s - p0 - p1 + new_k;
    };
    auto ih = input->output.shape[2];
    auto iw = input->output.shape[3];
    auto oh = infer_shape(ih, kernel[1], stride[1], dilation[1], padding[0], padding[1]);
    auto ow = infer_shape(iw, kernel[0], stride[0], dilation[0], padding[2], padding[3]);
    auto oc = weight->output.shape[0];

    Tensor conv(uOp::Conv);
    conv.add_input(input);
    if ((input->output.layout == Layout::NCHWC16 || input->output.layout == Layout::NCHWC32) &&
        (input->output.type == DType::FP16 || input->output.type == DType::FP8E4M3FN ||
         input->output.type == DType::FP8E5M2) &&
        input->output.shape.size() == 5 &&
        (weight->output.type == DType::INT4 || weight->output.type == DType::FP16 ||
         weight->output.type == DType::FP4E2M1FN) &&
        weight->output.shape.size() == 4 &&
        (out_layout == Layout::NCHWC32 || out_layout == Layout::NCHWC16 || out_layout == Layout::NHWC) &&
        (out_dtype == DType::FP16 || out_dtype == DType::BFP16 || out_dtype == DType::FP8E4M3FN || out_dtype == DType::FP8E5M2))
    {
        uint32_t align = 32 / input->output.element_size();
        conv->output.shape = {
            input->output.shape[0],
            oc / align,
            oh,
            ow,
            align,
        };
        conv->output.type = input->output.type;
        conv->output.layout = input->output.layout;
    }
    conv->constants["weights"] = weight;
    conv->constants["biases"] = bias;
    conv->params["kernel_x"] = kernel[0];
    conv->params["kernel_y"] = kernel[1];
    conv->params["stride_x"] = stride[0];
    conv->params["stride_y"] = stride[1];
    conv->params["pad_top"] = padding[0];
    conv->params["pad_bottom"] = padding[1];
    conv->params["pad_left"] = padding[2];
    conv->params["pad_right"] = padding[3];
    conv->params["dilation_x"] = dilation[0];
    conv->params["dilation_y"] = dilation[1];
    conv->params["group"] = group;
    conv->params["ic_gsize"] = ic_gsize;
    conv->params["gptq_mode"] = gptq_mode;
    conv->params["real_iw"] = real_iw;
    conv->params["real_ihw"] = real_ihw;
    conv->params["ihw_offset"] = ihw_offset;
    conv->params["real_oc"] = real_oc;
    conv->params["oc_offset"] = oc_offset;

    auto conv_bias = Tensor({{oc}, DType::FP32, Layout::Flat}, "bias");
    auto ret = eltwise(uOp::Add, conv, conv_bias, 1, 1, 0);

    bool is_gptq_quant = gptq_mode >= 0 && gptq_mode <= 3;
    if (input->output.type != DType::FP16 ||
        (!is_gptq_quant && weight->output.type != DType::FP16))
    {
        auto wt_scale = Tensor({{oc}, DType::FP32, Layout::Flat}, "wt_scale");
        ret = eltwise(uOp::Mul, ret, wt_scale, 1, 1, 0);
    }

    if (input->output.type != DType::FP16)
    {
        ret = eltwise(uOp::Mul, ret, out_scale);
    }

    if (out_dtype != ret->output.type)
    {
        ret = cast(ret, out_dtype);
    }

    if (out_layout == Layout::NHWC)
    {
        ret = layoutcvt(ret, out_layout);
    }
    return ret;
}
Tensor Tensor::add(Tensor a,
                   Tensor b,
                   int real_iw,
                   int real_ihw,
                   int ihw_offset)
{
    return eltwise(uOp::Add, a, b, real_iw, real_ihw, ihw_offset);
}
Tensor Tensor::sub(Tensor a,
                   Tensor b,
                   int real_iw,
                   int real_ihw,
                   int ihw_offset)
{
    auto sub = eltwise(uOp::Sub, a, b, real_iw, real_ihw, ihw_offset);

    return eltwise(uOp::Mul, sub, -1.0f);
}
Tensor Tensor::mul(Tensor a,
                   Tensor b,
                   int real_iw,
                   int real_ihw,
                   int ihw_offset)
{
    return eltwise(uOp::Mul, a, b, real_iw, real_ihw, ihw_offset);
}
Tensor Tensor::exp(Tensor input,
                   int real_iw,
                   int real_ihw,
                   int ihw_offset)
{
    float ln2 = 1.0 / std::log(2.0);
    auto mul = eltwise(uOp::Mul, input, ln2);

    Tensor lut(uOp::Lut);
    lut.add_input(mul);
    lut->output.shape = input->output.shape;
    lut->output.type = input->output.type;
    lut->output.layout = input->output.layout;
    lut->params["method"] = 0; // exp
    lut->params["real_iw"] = real_iw;
    lut->params["real_ihw"] = real_ihw;
    lut->params["ihw_offset"] = ihw_offset;
    return lut;
}
Tensor Tensor::silu(Tensor input,
                    float index_scale,
                    float index_offset,
                    float fmax_slope,
                    float fmin_slope,
                    float value_offset,
                    int real_iw,
                    int real_ihw,
                    int ihw_offset)
{
    auto mul = eltwise(uOp::Mul, input, index_scale);

    auto add0 = eltwise(uOp::Add, mul, index_offset);

    Tensor lut(uOp::Lut);
    lut.add_input(add0);
    lut->output.shape = input->output.shape;
    lut->output.type = input->output.type;
    lut->output.layout = input->output.layout;
    lut->params["method"] = 1; // silu
    lut->params["fmax_slope"] = *((uint32_t *)&fmax_slope);
    lut->params["fmin_slope"] = *((uint32_t *)&fmin_slope);
    lut->params["real_iw"] = real_iw;
    lut->params["real_ihw"] = real_ihw;
    lut->params["ihw_offset"] = ihw_offset;

    auto add1 = eltwise(uOp::Add, lut, value_offset);

    Tensor silu_input({input->output.shape, input->output.type, input->output.layout});
    return eltwise(uOp::Mul, add1, silu_input, real_iw, real_ihw, ihw_offset);
}
Tensor Tensor::max(Tensor a,
                   Tensor b,
                   int real_iw,
                   int real_ihw,
                   int ihw_offset)
{
    return eltwise(uOp::Max, a, b, real_iw, real_ihw, ihw_offset);
}
Tensor Tensor::max(Tensor input,
                   const std::vector<int> &axis,
                   const std::string &with_sfu,
                   int real_iw,
                   int real_ihw,
                   int ihw_offset)
{
    return reduce(uOp::Max, input, axis, with_sfu, real_iw, real_ihw, ihw_offset);
}
Tensor Tensor::sum(Tensor input,
                   const std::vector<int> &axis,
                   const std::string &with_sfu,
                   int real_iw,
                   int real_ihw,
                   int ihw_offset)
{
    return reduce(uOp::Sum, input, axis, with_sfu, real_iw, real_ihw, ihw_offset);
}
} // namespace ffc