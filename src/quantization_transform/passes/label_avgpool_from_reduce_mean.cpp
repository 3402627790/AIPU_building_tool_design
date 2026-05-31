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
 * @Date: 2023-02-28 18:55:44
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-03-05 21:55:14
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
static std::vector<int> eliminate_shape(const aipubt::TensorShape &shape)
{
    std::vector<int> new_shape = shape;
    for (auto intr = shape.begin(); intr != shape.end(); ++intr)
    {
        if (*intr != 1)
        {
            new_shape.assign(intr, shape.end());
            break;
        }
    }
    return new_shape;
}

template <typename T>
static int64_t get_data(aipubt::TensorPtr t)
{
    auto data = t->data<T>();
    return static_cast<int64_t>(data[0]);
}

static std::map<aipubt::Tensor::Dtype, std::function<int64_t(aipubt::TensorPtr)>> _get_data_map = {
    {aipubt::Tensor::UINT8, get_data<uint8_t>},
    {aipubt::Tensor::INT8, get_data<int8_t>},
    {aipubt::Tensor::UINT16, get_data<uint16_t>},
    {aipubt::Tensor::INT16, get_data<int16_t>},
    {aipubt::Tensor::UINT32, get_data<uint32_t>},
    {aipubt::Tensor::INT32, get_data<int32_t>},
};

static aipubt::TensorPtr shrink_tensor(aipubt::TensorPtr orig_t)
{
    auto orig_shape = orig_t->shape();
    auto squzeezed_shape = eliminate_shape(orig_shape);
    auto t = orig_t;
    t->shape() = aipubt::TensorShape(squzeezed_shape);
    do
    {
        t = shrink_tensor_data(t);
    } while (t && t->shape().dim() > 0);
    return t;
}
static bool criteria(std::vector<aipubt::NodePtr> &nodes)
{
    auto pool_node = nodes[0];
    auto constant_node = nodes[1];
    auto eltwise_node = nodes[2];
    if (nodes.size() == 4)
    {
        constant_node = nodes[2];
        eltwise_node = nodes[3];
    }
    if (eltwise_node->inputs()[0]->node()->type() == aipubt::OpType_Constant)
    {
        constant_node = nodes[0];
        pool_node = nodes[1];
    }

    if (pool_node->attrs().count("avgpool_from_reducemean") && (bool)pool_node->get_attr("avgpool_from_reducemean"))
    {
        return false;
    }

    std::vector<aipubt::Tensor::Dtype> dtypes = {
        pool_node->outputs()[0]->type(),
        constant_node->outputs()[0]->type(),
        eltwise_node->outputs()[0]->type(),
    };

    // now only support compass quantizedIR
    for (auto d : dtypes)
    {
        if (aipubt::Tensor::is_float(d))
        {
            return false;
        }
    }

    if (constant_node->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    std::string method;
    if (eltwise_node->type() == aipubt::OpType::OpType_Add)
    {
        method = "ADD";
    }
    else if (eltwise_node->type() == aipubt::OpType::OpType_Mul)
    {
        method = "MUL";
    }
    else
    {
        method = (std::string)eltwise_node->params()["method"];
        std::string with_activation = eltwise_node->params()["with_activation"];
        if (with_activation != "NONE")
        {
            return false;
        }
    }

    // now we only support method == add and mul
    if (method != "ADD" && method != "MUL")
    {
        return false;
    }

    // when method==add and mul, limit the constant has only one data, and data == 0
    auto constant_tensor = constant_node->constants()["weights"];
    auto t = shrink_tensor(constant_tensor);
    if (!t || t->shape().dim() > 0)
    {
        return false;
    }
    else
    {
        auto val = _get_data_map[t->type()](t);
        if (val != 0)
        {
            return false;
        }
    }
    // when method == add, the zp of constant should be == 0
    auto constant_zp = constant_tensor->quantization().offsets();
    auto constant_zp_is_zero = constant_zp.all_of([](int32_t z) -> bool { return z == 0 ? true : false; });
    auto constant_zp_is_greater_equal_zero = constant_zp.all_of([](int32_t z) -> bool { return z >= 0 ? true : false; });

    if (method == "ADD" && !constant_zp_is_zero)
    {
        return false;
    }

    if (method == "MUL" && !constant_zp_is_greater_equal_zero)
    {
        return false;
    }

    return true;
}

static void _change_constant_scale(aipubt::NodePtr pool_n, aipubt::NodePtr constant_n)
{
    auto pool_o_scales = pool_n->outputs()[0]->quantization().scales();
    auto const_o_scales = constant_n->outputs()[0]->quantization().scales();
    auto const_o_offsets = constant_n->outputs()[0]->quantization().offsets();
    auto const_w_offsets = constant_n->constants()["weights"]->quantization().offsets();
    if (const_o_offsets[0] == 0 && const_w_offsets[0] == 0)
    {
        constant_n->outputs()[0]->quantization().set_scales(pool_o_scales);
        constant_n->constants()["weights"]->quantization().set_scales(pool_o_scales);
        DEBUG() << "change constant scale = " << const_o_scales[0]
                << " to pool scale = " << pool_o_scales[0] << " in pattern [reducemean->pool+constant+eltwise].";
    }
}

void label_avgpool_from_reducemean(aipubt::GraphPtr g)
{
    std::vector<aipubt::GraphMatcherPattern> patterns = {
        {{{"0", aipubt::OpType_Pooling}, {"1", aipubt::OpType_Constant}, {"2", aipubt::OpType_Eltwise}}, {{"0", "2"}, {"1", "2"}}},
        {{{"1", aipubt::OpType_Pooling}, {"0", aipubt::OpType_Constant}, {"2", aipubt::OpType_Eltwise}}, {{"0", "2"}, {"1", "2"}}},
        {{{"0", aipubt::OpType_Pooling}, {"1", aipubt::OpType_Constant}, {"2", aipubt::OpType_Add}}, {{"0", "2"}, {"1", "2"}}},
        {{{"1", aipubt::OpType_Pooling}, {"0", aipubt::OpType_Constant}, {"2", aipubt::OpType_Add}}, {{"0", "2"}, {"1", "2"}}},
        {{{"0", aipubt::OpType_Pooling}, {"1", aipubt::OpType_Constant}, {"2", aipubt::OpType_Mul}}, {{"0", "2"}, {"1", "2"}}},
        {{{"1", aipubt::OpType_Pooling}, {"0", aipubt::OpType_Constant}, {"2", aipubt::OpType_Mul}}, {{"0", "2"}, {"1", "2"}}},
        {{{"0", aipubt::OpType_Pooling}, {"1", aipubt::OpType_Reshape}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Eltwise}}, {{"0", "1"}, {"1", "3"}, {"2", "3"}}},
        {{{"1", aipubt::OpType_Pooling}, {"0", aipubt::OpType_Reshape}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Eltwise}}, {{"0", "1"}, {"1", "3"}, {"2", "3"}}},
        {{{"0", aipubt::OpType_Pooling}, {"1", aipubt::OpType_Reshape}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Add}}, {{"0", "1"}, {"1", "3"}, {"2", "3"}}},
        {{{"1", aipubt::OpType_Pooling}, {"0", aipubt::OpType_Reshape}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Add}}, {{"0", "1"}, {"1", "3"}, {"2", "3"}}},
        {{{"0", aipubt::OpType_Pooling}, {"1", aipubt::OpType_Reshape}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Mul}}, {{"0", "1"}, {"1", "3"}, {"2", "3"}}},
        {{{"1", aipubt::OpType_Pooling}, {"0", aipubt::OpType_Reshape}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Mul}}, {{"0", "1"}, {"1", "3"}, {"2", "3"}}},
    };
    // bool is_changed = false;
    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                bool ret = criteria(m);
                if (!ret)
                {
                    continue;
                }
                auto eltwise_node = m[2];
                auto pool_node = m[0];
                auto constant_node = m[1];
                if (m.size() == 4)
                {
                    constant_node = m[2];
                    eltwise_node = m[3];
                }
                if (eltwise_node->inputs()[0]->node()->type() == aipubt::OpType_Constant)
                {
                    pool_node = m[1];
                    constant_node = m[0];
                }
                pool_node->set_attr("avgpool_from_reducemean", true);
                _change_constant_scale(pool_node, constant_node);
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    // return is_changed;
}

} // namespace qtlibn
