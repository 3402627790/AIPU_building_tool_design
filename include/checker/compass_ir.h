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
 * @Date: 2020-01-29 14:28:12
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-07-22 13:59:08
 */
#pragma once

#include <string>

namespace compass_ir
{

enum class version
{
    v1,
    v2,

    latest = v2,
};
enum class head_params
{
    model_name,
    input_tensors,
    output_tensors,
    batch_size,
    layer_number,
};

enum class layer_params
{
    layer_id,
    layer_name,
    layer_type,
    layer_bottom,
    layer_bottom_shape,
    layer_bottom_type,
    layer_top,
    layer_top_shape,
    layer_top_type,
    layer_top_datalayout,
    layer_top_range,
    layer_top_scale,
    layer_top_zp,
    layer_top_ds_shape,
    layer_top_scale_type,
    layer_top_scale_offset,
    layer_top_scale_size,
    layer_top_scale_shape,
    layer_top_zp_type,
    layer_top_zp_offset,
    layer_top_zp_size,
    layer_top_zp_shape,
    layer_top_range_type,
    layer_top_range_offset,
    layer_top_range_size,
    layer_top_range_shape,
    weights_type,
    weights_shape,

};

constexpr const unsigned int required_layer_params[] =
    {
        (unsigned int)layer_params::layer_id,
        (unsigned int)layer_params::layer_name,
        (unsigned int)layer_params::layer_type,
        (unsigned int)layer_params::layer_bottom,
        (unsigned int)layer_params::layer_bottom_shape,
        (unsigned int)layer_params::layer_bottom_type,
        (unsigned int)layer_params::layer_top,
        (unsigned int)layer_params::layer_top_shape,
        (unsigned int)layer_params::layer_top_type,
};

constexpr const unsigned int required_head_params[] =
    {
        (unsigned int)head_params::model_name,
        (unsigned int)head_params::input_tensors,
        (unsigned int)head_params::output_tensors,
};

constexpr const unsigned int head_deprecated[] =
    {
        (unsigned int)head_params::batch_size,
};

constexpr const unsigned int layer_deprecated[] =
    {
        // (unsigned int)head_params::batch_size,
};

namespace v1
{

    constexpr const char *head_params[] =
        {
            "model_name",
            "input_tensors",
            "output_tensors",
            "batch_size",
            "layer_number",
    };
    constexpr const char *layer_params[] = {
        "layer_id",
        "layer_name",
        "layer_type",
        "layer_bottom",
        "layer_bottom_shape",
        "layer_bottom_type",
        "layer_top",
        "layer_top_shape",
        "layer_top_type",
        "layer_top_datalayout",
        "layer_top_range",
        "layer_top_scale",
        "layer_top_zp",
        "layer_top_ds_shape",
        "layer_top_scale_type",
        "layer_top_scale_offset",
        "layer_top_scale_size",
        "layer_top_scale_shape",
        "layer_top_zp_type",
        "layer_top_zp_offset",
        "layer_top_zp_size",
        "layer_top_zp_shape",
        "layer_top_range_type",
        "layer_top_range_offset",
        "layer_top_range_size",
        "layer_top_range_shape",

        "weights_type",
        "weights_shape",

    };
} // namespace v1

namespace v2
{

    constexpr const char *head_params[] =
        {
            "model.name",
            "model.input_tensors",
            "model.output_tensors",
            "batch_size", // invalid
            "layer_number",

    };
    constexpr const char *layer_params[] = {
        "layer.id",
        "layer.name",
        "layer.type",
        "layer.bottom",
        "layer.bottom.shape",
        "layer.bottom.dtype",
        "layer.top",
        "layer.top.shape",
        "layer.top.dtype",
        "layer.top.datalayout",
        "layer.top.range",
        "layer.top.scale",
        "layer.top.zp",
        "layer.top.ds_shape",
        "layer.top.scale.type",
        "layer.top.scale.offset",
        "layer.top.scale.size",
        "layer.top.scale.shape",
        "layer.top.zp.type",
        "layer.top.zp.offset",
        "layer.top.zp.size",
        "layer.top.zp.shape",
        "layer.top.range.type",
        "layer.top.range.offset",
        "layer.top.range.size",
        "layer.top.range.shape",
        "weights.dtype",
        "weights.shape",

    };
} // namespace v2

inline const char *alias_h(head_params p, version v)
{
    if (v == version::v1)
    {
        return v1::head_params[(unsigned)p];
    }
    return v2::head_params[(unsigned)p];
}
inline const char *alias(layer_params p, version v)
{
    if (v == version::v1)
    {
        return v1::layer_params[(unsigned)p];
    }
    return v2::layer_params[(unsigned)p];
}
} // namespace compass_ir