//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-12-03 15:45:06
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-12-15 09:50:28
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/data_type.h"
#include <functional>

namespace aipubt
{
static std::vector<int> eliminate_shape(const TensorShape &shape)
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
static int64_t get_data(TensorPtr t)
{
    auto data = t->data<T>();
    return static_cast<int64_t>(data[0]);
}

static std::map<Tensor::Dtype, std::function<int64_t(TensorPtr)>> _get_data_map = {
    {Tensor::UINT8, get_data<uint8_t>},
    {Tensor::INT8, get_data<int8_t>},
    {Tensor::UINT16, get_data<uint16_t>},
    {Tensor::INT16, get_data<int16_t>},
    {Tensor::UINT32, get_data<uint32_t>},
    {Tensor::INT32, get_data<int32_t>},
};

static TensorPtr shrink_tensor(TensorPtr orig_t)
{
    auto orig_shape = orig_t->shape();
    auto squzeezed_shape = eliminate_shape(orig_shape);
    auto t = orig_t;
    t->shape() = TensorShape(squzeezed_shape);
    do
    {
        t = shrink_tensor_data(t);
    } while (t && t->shape().dim() > 0);
    return t;
}

static bool criteria(std::vector<NodePtr> &nodes)
{
    auto pool_node = nodes[0];
    auto constant_node = nodes[1];
    auto eltwise_node = nodes[2];
    if (eltwise_node->inputs()[0]->node()->type() == aipubt::OpType_Constant)
    {
        constant_node = nodes[0];
        pool_node = nodes[1];
    }

    std::vector<aipubt::Tensor::Dtype> dtypes = {
        pool_node->outputs()[0]->type(),
        constant_node->outputs()[0]->type(),
        eltwise_node->outputs()[0]->type(),
    };

    // now only support compass quantizedIR
    for (auto d : dtypes)
    {
        if (Tensor::is_float(d))
        {
            return false;
        }
    }

    if (constant_node->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    std::string method;
    if (eltwise_node->type() == OpType::OpType_Add)
    {
        method = "ADD";
    }
    else if (eltwise_node->type() == OpType::OpType_Mul)
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
    auto constant_zp_is_zero = constant_zp.all_of([](int32_t z) -> bool
                                                  { return z == 0 ? true : false; });
    auto constant_zp_is_greater_equal_zero = constant_zp.all_of([](int32_t z) -> bool
                                                                { return z >= 0 ? true : false; });

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

/* fuse_constant_eltwise_to_leakyrelu pass is used to fuse the pattern:
 *  pooling(q/x1+zp1)  --\
 *                        |--> eltwise (q/x3+zp3)
 *  constant(q/x2+zp2) --/
 *  eltwise_add:
 *      which all(constant.data) == 0 and constant.zp == 0. the calculation formula of above pattern is
 *      x3 = ((x1 + zp1) * s1 + (x2 + zp2) * s2) * s3 * 0.5 ** shift - zp3. and if x2==0 and zp2==0,
 *      x3 = ((x1 + zp1) * s1 * s3 * 0.5 ** shift - zp3, now the pattern can be:
 *      pooling(q/x1+zp1) --> leakyrelu(q/x3+zp3), the leakyrelu compass quantization infomation:
 *      negative_slope = 1, negative_shift = 0, so the negative value will not change. doscale = s1 * s3, doshift = shift
 *
 *      the backgroup is when reduce_mean in compass IR, in order to improve performance, usually convert the reduce mean to avgpool
 *      + constant + eltwise, and constant + eltwise is used for the zp compensation.
 *  eltwise_mul:
 *      which all(constant.data) == 0 and constant.zp >= 0. the calculation formula of above pattern is
 *      x3 = (x1 + zp1) * (x2 + zp2) * s * 0.5 ** shift - zp3. and if x2==0 and zp2 >=0,
 *      x3 = ((x1 + zp1) * zp2 * s * 0.5 ** shift - zp3, now the pattern can be:
 *      pooling(q/x1+zp1) --> leakyrelu(q/x3+zp3), the leakyrelu compass quantization infomation:
 *      negative_slope = 1, negative_shift = 0, so the negative value will not change. doscale = zp2 * s, doshift = shift
 *
 *  if doscale > 32767, we need rescale it.
 */

bool fuse_constant_eltwise_to_leakyrelu(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Pooling}, {"1", OpType_Constant}, {"2", OpType_Eltwise}}, {{"0", "2"}, {"1", "2"}}},
        {{{"0", OpType_Pooling}, {"1", OpType_Constant}, {"2", OpType_Add}}, {{"0", "2"}, {"1", "2"}}},
        {{{"0", OpType_Pooling}, {"1", OpType_Constant}, {"2", OpType_Mul}}, {{"0", "2"}, {"1", "2"}}},
    };
    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m))
                {
                    continue;
                }
                auto eltwise_node = m[2];
                // pool + constant --> eltwise
                auto pool_node = m[0];
                auto constant_node = m[1];
                int32_t pool_scale_index_in_eltwise_scales = 1;
                if (eltwise_node->inputs()[0]->node()->type() == aipubt::OpType_Constant)
                {
                    // constant + pool --> eltwise
                    pool_scale_index_in_eltwise_scales = 2;
                    constant_node = m[0];
                    pool_node = m[1];
                }

                std::string method;
                if (eltwise_node->type() == OpType::OpType_Add)
                {
                    method = "ADD";
                }
                else if (eltwise_node->type() == OpType::OpType_Mul)
                {
                    method = "MUL";
                }
                else
                {
                    method = (std::string)eltwise_node->params()["method"];
                }
                auto scales = eltwise_node->params()["scale_value"];
                int32_t shift = eltwise_node->params()["shift_value"];
                float total_scale = 1.0;
                if (method == "ADD")
                {
                    int32_t pool_branch_scale = scales[pool_scale_index_in_eltwise_scales];
                    int32_t eltwise_do_scale = scales[0];
                    total_scale = eltwise_do_scale * pool_branch_scale; // total_scale max bits 30
                }
                else // method == "MUL"
                {
                    // only support per-tensor quantization
                    int32_t eltwise_do_scale = scales[0];
                    int32_t constant_zp = constant_node->outputs()[0]->quantization().offset();
                    total_scale = eltwise_do_scale * constant_zp; // total_scale max bits 23
                }

                while (total_scale > (1 << 15) - 1)
                {
                    total_scale /= 2;
                    shift--;
                }

                NodePtr leakyrelu_node = Node::create(g->get_valid_node_name(m[1]->name() + "/LEAKYRELU"), OpType_Activation);
                leakyrelu_node->params()["method"] = "LEAKYRELU";
                leakyrelu_node->params()["negative_slope_type"] = Tensor::INT8;
                leakyrelu_node->params()["negative_slope_value"] = 1;
                leakyrelu_node->params()["negative_slope_shift"] = 0;
                leakyrelu_node->params()["scale_type"] = dtype_utils::range_to_dtype(0, total_scale, false);
                leakyrelu_node->params()["scale_value"] = static_cast<int32_t>(round(total_scale));
                leakyrelu_node->params()["shift_type"] = Tensor::INT8;
                leakyrelu_node->params()["shift_value"] = shift;
                leakyrelu_node->add_input(pool_node->outputs()[0]);
                leakyrelu_node->add_output(eltwise_node->outputs()[0]);

                ADD_LOCATION_BY_NODE(leakyrelu_node, m[1]);
                constant_node->clear_outputs();
                eltwise_node->clear_outputs();
                eltwise_node->clear_inputs();

                g->remove_node(constant_node);
                g->remove_node(eltwise_node);
                g->add_node(leakyrelu_node);
                leakyrelu_node->infer_shape(true);
                is_changed = true;
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}

} // namespace aipubt
