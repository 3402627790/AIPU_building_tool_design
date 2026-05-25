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
 * @Date: 2022-05-11 11:03:30
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-11 11:11:47
 */

#include "quantization_transform/ops/gridsample.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{

static inline aipubt::ComputeTensor gs_denormalize(aipubt::ComputeTensor grid, const int32_t length, bool align_corners)
{
    if (align_corners)
    {
        return (grid + 1) / 2.0 * (length - 1);
    }
    else
    {
        return ((grid + 1) * length - 1) / 2.0;
    }
}

void GridSampleOp::quantize(aipubt::NodePtr node)
{
    auto grid_clamp = [&](aipubt::ComputeTensor r, float minv, float maxv, float cminv, float cmaxv) {
        r = r.astype(aipubt::Tensor::FP32);
        auto data = r->data<float>();
        auto out_shape = r->shape();
        int32_t out_size = out_shape.size();
        float *new_data = new float[out_size];
        for (int32_t i = 0; i < out_size; i++)
        {
            if (data[i] < minv)
            {
                new_data[i] = cminv;
            }
            if (data[i] > maxv)
            {
                new_data[i] = cmaxv;
            }
        }
        auto ret = std::make_shared<aipubt::Tensor>(aipubt::TensorShape(out_shape), aipubt::Tensor::FP32);
        ret->set_data(new_data, out_size * aipubt::Tensor::dtype_size(aipubt::Tensor::FP32));
        return aipubt::ComputeTensor::create(ret);
    };

    int coordinate_x_shift = 10;
    int coordinate_y_shift = 10;
    auto inp_t = node->inputs()[0];
    auto out_t = node->outputs()[0];
    NoQuantization::quantize_transparent(out_t, inp_t);

    std::string method = node->params()["method"];
    std::string padding_mode = node->params()["padding_mode"];
    bool align_corners = node->params()["align_corners"];
    bool is_5_dims = node->inputs()[0]->shape().dim() == 5;

    const std::vector<std::string> _supported_method = {"NEAREST", "BILINEAR", "LINEAR"};
    const std::vector<std::string> _supported_padding_mode = {"ZEROS", "BORDER"};

    if (!aipubt::contains(_supported_method, method))
    {
        ERROR() << "unsupported method = " << method << " in gridsample node (" << node << ")";
    }
    if (!aipubt::contains(_supported_padding_mode, padding_mode))
    {
        ERROR() << "unsupported padding_mode = " << padding_mode << " in gridsample node (" << node << ")";
    }

    auto inp0_t = node->inputs()[0];
    auto inp1_t = node->inputs()[1];
    auto grid_bits = inp1_t->quantization().bits();
    float inp1_scale = inp1_t->quantization().scale();
    int inp1_zerop = inp1_t->quantization().offset();
    auto inp1_qmin = dtype_utils::dtype_to_range_min(inp1_t->quantization().aipu_dtype());
    auto inp1_qmax = dtype_utils::dtype_to_range_max(inp1_t->quantization().aipu_dtype());
    auto grid_f_range = aipubt::ComputeTensor::linspace(inp1_qmin, inp1_qmax, inp1_qmax - inp1_qmin + 1);
    grid_f_range = (grid_f_range + inp1_zerop) / inp1_scale;

    auto feature_depth = 1;
    auto feature_height = inp0_t->shape()[1];
    auto feature_width = inp0_t->shape()[2];
    if (is_5_dims)
    {
        feature_depth = inp0_t->shape()[1];
        feature_height = inp0_t->shape()[2];
        feature_width = inp0_t->shape()[3];
    }

    auto grid_x_range = gs_denormalize(grid_f_range, feature_width, align_corners);
    auto grid_y_range = gs_denormalize(grid_f_range, feature_height, align_corners);
    auto grid_z_range = gs_denormalize(grid_f_range, feature_depth, align_corners);

    int32_t grid_x_range_bits = dtype_utils::range_to_bits(grid_x_range.min(), grid_x_range.max(), true);
    int32_t grid_y_range_bits = dtype_utils::range_to_bits(grid_y_range.min(), grid_y_range.max(), true);
    int32_t grid_z_range_bits = dtype_utils::range_to_bits(grid_z_range.min(), grid_z_range.max(), true);

    if (grid_x_range_bits > 16 || grid_y_range_bits > 16 || grid_z_range_bits > 16)
    {
        WARN() << "gridsample op (" << node << ") grid coordinates may exceed the range [-32768, 32767],  which causes accuracy problems";
    }

    if (grid_bits <= 8 && is_5_dims)
    {
        WARN() << node << " is 5 dim input of gridsample, 8bits quantization would lead to big acc drop, so now only support 16bits.";
        grid_bits = 16;
    }

    if (grid_bits <= 8)
    {
        if (method == "NEAREST")
        {
            grid_x_range = grid_x_range.round();
            grid_y_range = grid_y_range.round();
        }

        if (padding_mode == "BORDER")
        {
            float x_min = -0.5;
            float x_max = feature_width - 0.5;
            float y_min = -0.5;
            float y_max = feature_height - 0.5;
            if (align_corners)
            {
                x_min = 0.;
                x_max = feature_width - 1.;
                y_min = 0.;
                y_max = feature_height - 1.;
            }
            grid_x_range = grid_clamp(grid_x_range, x_min, x_max, 0, feature_width - 1);
            grid_y_range = grid_clamp(grid_y_range, y_min, y_max, 0, feature_height - 1);
        }

        aipubt::ComputeTensor lut_x = nullptr;
        aipubt::ComputeTensor lut_y = nullptr;
        if (method == "NEAREST")
        {
            lut_x = grid_x_range.astype(aipubt::Tensor::INT32);
            lut_y = grid_y_range.astype(aipubt::Tensor::INT32);
        }
        else
        {
            lut_x = grid_x_range.floor().astype(aipubt::Tensor::INT32);
            lut_y = grid_y_range.floor().astype(aipubt::Tensor::INT32);

            auto x_terp = ((grid_x_range - lut_x) * std::pow(2, coordinate_x_shift)).astype(aipubt::Tensor::UINT16);
            auto y_terp = ((grid_y_range - lut_y) * std::pow(2, coordinate_y_shift)).astype(aipubt::Tensor::UINT16);
            node->constants()["x_terp"] = x_terp;
            node->constants()["x_terp"]->set_type(aipubt::Tensor::UINT16);
            node->constants()["y_terp"] = y_terp;
            node->constants()["y_terp"]->set_type(aipubt::Tensor::UINT16);
        }

        node->constants()["lutx"] = lut_x.astype(aipubt::Tensor::UINT16);
        node->constants()["lutx"]->set_type(aipubt::Tensor::UINT16);
        node->constants()["luty"] = lut_y.astype(aipubt::Tensor::UINT16);
        node->constants()["luty"]->set_type(aipubt::Tensor::UINT16);
        node->params()["scale_value"] = {1, 1};
        node->params()["scale_type"] = {aipubt::Tensor::UINT8, aipubt::Tensor::UINT8};
        node->params()["shift_value"] = {0, 0};
        node->params()["shift_type"] = {aipubt::Tensor::INT8, aipubt::Tensor::INT8};
        node->params()["coordinate_x_shift"] = coordinate_x_shift;
        node->params()["coordinate_y_shift"] = coordinate_y_shift;
    }
    else
    {
        auto multiplier_grid_x = std::max(16 - grid_x_range_bits, 0);
        auto multiplier_grid_y = std::max(16 - grid_y_range_bits, 0);
        auto multiplier_grid_z = std::max(16 - grid_z_range_bits, 0);

        if (align_corners)
        {
            feature_height -= 1;
            feature_width -= 1;
            feature_depth -= 1;
        }
        auto grid_x_scale = std::pow(2, multiplier_grid_x) * feature_width / inp1_scale / 2;
        auto grid_y_scale = std::pow(2, multiplier_grid_y) * feature_height / inp1_scale / 2;
        auto grid_z_scale = std::pow(2, multiplier_grid_z) * feature_depth / inp1_scale / 2;

        float grid_x_doscale, grid_y_doscale, grid_z_doscale;
        int8_t grid_x_doshift, grid_y_doshift, grid_z_doshift;
        quant_utils::get_scale_approximation_params(grid_x_scale, grid_x_doscale, grid_x_doshift, 15);
        quant_utils::get_scale_approximation_params(grid_y_scale, grid_y_doscale, grid_y_doshift, 15);
        quant_utils::get_scale_approximation_params(grid_z_scale, grid_z_doscale, grid_z_doshift, 15);
        auto scale_dtype = dtype_utils::bits_to_dtype(15, false);
        node->params()["scale_value"] = {int(grid_x_doscale), int(grid_y_doscale)};
        node->params()["scale_type"] = {scale_dtype, scale_dtype};
        node->params()["shift_value"] = {int(grid_x_doshift), int(grid_y_doshift)};
        node->params()["shift_type"] = {aipubt::Tensor::INT8, aipubt::Tensor::INT8};
        node->params()["coordinate_x_shift"] = int(multiplier_grid_x);
        node->params()["coordinate_y_shift"] = int(multiplier_grid_y);

        if (is_5_dims)
        {
            node->params()["scale_value"].push_back(int(grid_z_doscale));
            node->params()["scale_type"].push_back(scale_dtype);
            node->params()["shift_value"].push_back(int(grid_z_doshift));
            node->params()["shift_type"].push_back(aipubt::Tensor::INT8);
            node->params()["coordinate_z_shift"] = int(multiplier_grid_z);
        }
    }
}

REGISTER_QUANTIZATION_OP(GridSample);

}; // namespace qtlibn