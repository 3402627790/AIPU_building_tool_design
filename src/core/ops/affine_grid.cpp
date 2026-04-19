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
 * @Author: Alex Zheng
 * @Date: 2025-05-06 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2025-05-06 11:35:37
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    template <typename T>
    std::vector<int32_t> get_size_value_(TensorPtr size)
    {
        std::vector<int32_t> size_value;
        auto data = size->data<T>();
        if (size->node()->type() == OpType_Constant)
            data = size->node()->constants()["weights"]->data<T>();
        auto size_len = size->size();
        for (size_t i = 0; i < size_len; i++)
        {
            size_value.push_back(data[i]);
        }
        return size_value;
    }
    static std::vector<int32_t> get_size_value(TensorPtr size)
    {
        switch (size->type())
        {
        case Tensor::UINT8:
            return get_size_value_<uint8_t>(size);
        case Tensor::INT8:
            return get_size_value_<int8_t>(size);
        case Tensor::UINT16:
            return get_size_value_<uint16_t>(size);
        case Tensor::INT16:
            return get_size_value_<int16_t>(size);
        case Tensor::UINT32:
            return get_size_value_<uint32_t>(size);
        case Tensor::INT32:
            return get_size_value_<int32_t>(size);
        default:
            ERROR() << "Unsupport 'size' data type, only for: UINT8,INT8,UINT16,INT16,UINT32,INT32.\n";
            return {};
        }
        return {};
    }

    TensorPtr affine_grid(TensorPtr theta,
                          TensorPtr size,
                          const bool align_corners,
                          const Quantization &quant)
    {
        size_t size_len = size->shape().size();
        if (size->data() == nullptr &&
            size->node()->type() != OpType_Constant)
        {
            ERROR() << "The size tensor only support constants!";
            return nullptr;
        }
        std::vector<int32_t> size_value = get_size_value(size);
        if (size_value.size() != size_len)
        {
            ERROR() << "The size tensor value lenth is not right!" << size_value.size() << ", " << size_len;
            return nullptr;
        }
        TensorShape target_shape(size_value);
        if (size_len == 4)
        {
            if (theta->shape().dim() != 3 || theta->shape()[1] != 2 || theta->shape()[2] != 3)
            {
                ERROR() << "Expected a batch of 2D affine matrices of shape Nx2x3 for size"
                        << target_shape << ". Got " << theta->shape() << ".";
                return nullptr;
            }
        }
        else if (size_len == 5)
        {
            if (theta->shape().dim() != 3 || theta->shape()[1] != 3 || theta->shape()[2] != 4)
            {
                ERROR() << "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
                        << target_shape << ". Got " << theta->shape() << ".";
                return nullptr;
            }
        }
        else
        {
            ERROR() << "affine_grid only supports 4D and 5D sizes, "
                    << "for 2D and 3D affine transforms, respectively. "
                    << "Got size " << target_shape << ".";
            return nullptr;
        }

        auto g = theta->node()->graph();
        if (size_len == 4)
        {
            size_value[3] = 2;
        }
        else if (size_len == 5)
        {
            size_value[4] = 3;
        }

        TensorShape output_shape(size_value);
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("affine_grid"), output_shape);
        add_tensor_info(theta, output, quant);

        auto node = Node::create(g->get_valid_node_name("affine_grid"), OpType_AffineGrid);
        node->add_input(theta);
        node->add_input(size);
        node->add_output(output);
        node->params()["align_corners"] = align_corners ? "True" : "False";
        g->add_node(node);
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt