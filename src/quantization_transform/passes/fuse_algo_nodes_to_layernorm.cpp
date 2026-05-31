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
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-07-05 10:40:45
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/fp16/half.h"
#include "core/ops/ops.h"
#include "quantization_transform/passes/passes.h"

// fuse algorithm to layernorm
namespace qtlibn
{
using namespace aipubt;
namespace
{
    template <typename T>
    int32_t get_data(TensorPtr t)
    {
        int32_t val = t->data<T>()[0];
        return val;
    }

    std::map<Tensor::Dtype, std::function<int32_t(TensorPtr t)>> get_data_table = {
        {Tensor::UINT8, get_data<uint8_t>},
        {Tensor::INT8, get_data<int8_t>},
        {Tensor::UINT16, get_data<uint16_t>},
        {Tensor::INT16, get_data<int16_t>},
        {Tensor::UINT32, get_data<uint32_t>},
        {Tensor::INT32, get_data<int32_t>},
        {Tensor::FP32, get_data<float_t>},
    };

    void fuse_layernorm_axis_continue(GraphPtr g)
    {
        auto check_constant_shape = [](TensorPtr constant, TensorPtr input, std::vector<int32_t> &axis)
        {
            if (constant->shape().dim() > input->shape().dim())
            {
                return false;
            }
            int step = axis.size() - 1;
            for (int i = constant->shape().dim() - 1; i >= 0; --i)
            {
                if (step >= 0)
                {
                    if (constant->shape()[i] != input->shape()[axis[step]])
                    {
                        return false;
                    }
                    step--;
                }
                else
                {
                    if (constant->shape()[i] != 1)
                    {
                        return false;
                    }
                }
            }
            return true;
        };

        auto criteria = [check_constant_shape](std::vector<NodePtr> m)
        {
            NodePtr eltwise_sub = nullptr;
            NodePtr pow = nullptr;
            NodePtrList constants;
            NodePtrList reduces;
            for (auto node : m)
            {
                if (node->type() == OpType_Eltwise)
                {
                    eltwise_sub = node;
                }
                if (node->type() == OpType_Pow)
                {
                    pow = node;
                }
                if (node->type() == OpType_Reduce)
                {
                    reduces.push_back(node);
                }
                if (node->type() == OpType_Constant)
                {
                    constants.push_back(node);
                }
            }

            if ((eltwise_sub == nullptr) || (pow == nullptr))
            {
                return false;
            }
            if (reduces.size() < 2 || constants.size() < 3)
            {
                return false;
            }

            // 1. pow y check
            NodePtr pow_y = pow->inputs()[1]->node();
            auto pow_y_weights = pow_y->constants()["weights"];
            auto pow_y_dtype = pow_y_weights->type();
            if (!get_data_table.count(pow_y_dtype))
            {
                return false;
            }
            int32_t pow_y_value = get_data_table[pow_y_dtype](pow_y_weights);
            if (pow_y_value != 2)
            {
                return false;
            }

            // 2. sub check
            auto sub_params = eltwise_sub->params();
            if (!sub_params.count("method") || sub_params["method"] != "SUB")
            {
                return false;
            }

            // 3. reduce mean check
            auto reduce_0 = reduces[0];
            auto reduce_1 = reduces[1];
            auto reduce_0_params = reduce_0->params();
            auto reduce_1_params = reduce_1->params();
            if (!reduce_0_params.count("method") || reduce_0_params["method"] != "MEAN")
            {
                return false;
            }
            if (!reduce_1_params.count("method") || reduce_1_params["method"] != "MEAN")
            {
                return false;
            }

            std::vector<int32_t> reduce_0_axes = reduce_0_params["axis"];
            std::vector<int32_t> reduce_1_axes = reduce_1_params["axis"];
            for (auto &axis : reduce_0_axes)
            {
                if (axis < 0)
                {
                    axis = axis + reduce_0->inputs()[0]->shape().dim();
                }
            }
            for (auto &axis : reduce_1_axes)
            {
                if (axis < 0)
                {
                    axis = axis + reduce_0->inputs()[1]->shape().dim();
                }
            }
            if (reduce_0_axes.size() != reduce_1_axes.size())
            {
                return false;
            }
            sort(reduce_0_axes.begin(), reduce_0_axes.end());
            reduce_0_axes.erase(unique(reduce_0_axes.begin(), reduce_0_axes.end()), reduce_0_axes.end());

            sort(reduce_1_axes.begin(), reduce_1_axes.end());
            reduce_0_axes.erase(unique(reduce_1_axes.begin(), reduce_1_axes.end()), reduce_1_axes.end());
            for (size_t i = 0; i < reduce_0_axes.size(); ++i)
            {
                if (reduce_0_axes[i] != reduce_1_axes[i])
                {
                    return false;
                }
            }

            // constant shape check
            auto input_node = m[0];
            auto gamma = constants[constants.size() - 2];
            auto beta = constants[constants.size() - 1];
            auto gamma_data = gamma->constants()["weights"];
            auto beta_data = beta->constants()["weights"];
            if (!check_constant_shape(gamma_data, input_node->outputs()[0], reduce_0_axes) ||
                !check_constant_shape(beta_data, input_node->outputs()[0], reduce_0_axes))
            {
                return false;
            }

            return true;
        };

        std::vector<GraphMatcherPattern> patterns = {
            {
                {
                    {"dequantize_0", OpType_DeQuantize},
                    {"reduce_0", OpType_Reduce},
                    {"tile_0", OpType_Tile},
                    {"eltwise_0", OpType_Eltwise},
                    {"pow_y", OpType_Constant},
                    {"pow_0", OpType_Pow},
                    {"reduce_1", OpType_Reduce},
                    {"quantize_0", OpType_Quantize},
                    {"epsilon", OpType_Constant},
                    {"add_0", OpType_Add},
                    {"dequantize_1", OpType_DeQuantize},
                    {"sqrt_0", OpType_Sqrt},
                    {"tile_1", OpType_Tile},
                    {"div_0", OpType_Div},
                    {"quantize_1", OpType_Quantize},
                    {"gamma", OpType_Constant},
                    {"mul_0", OpType_Mul},
                    {"beta", OpType_Constant},
                    {"add_1", OpType_Add},
                },
                {
                    {"dequantize_0", "reduce_0"},
                    {"reduce_0", "tile_0"},
                    {"tile_0", "eltwise_0"},
                    {"dequantize_0", "eltwise_0"},
                    {"eltwise_0", "pow_0"},
                    {"pow_y", "pow_0"},
                    {"pow_0", "reduce_1"},
                    {"reduce_1", "quantize_0"},
                    {"quantize_0", "add_0"},
                    {"epsilon", "add_0"},
                    {"add_0", "dequantize_1"},
                    {"dequantize_1", "sqrt_0"},
                    {"sqrt_0", "tile_1"},
                    {"eltwise_0", "div_0"},
                    {"tile_1", "div_0"},
                    {"div_0", "quantize_1"},
                    {"quantize_1", "mul_0"},
                    {"gamma", "mul_0"},
                    {"mul_0", "add_1"},
                    {"beta", "add_1"},
                },
            },
            {
                {
                    {"dequantize_0", OpType_DeQuantize},
                    {"reduce_0", OpType_Reduce},
                    {"eltwise_0", OpType_Eltwise},
                    {"pow_y", OpType_Constant},
                    {"pow_0", OpType_Pow},
                    {"reduce_1", OpType_Reduce},
                    {"quantize_0", OpType_Quantize},
                    {"epsilon", OpType_Constant},
                    {"add_0", OpType_Add},
                    {"dequantize_1", OpType_DeQuantize},
                    {"sqrt_0", OpType_Sqrt},
                    {"div_0", OpType_Div},
                    {"quantize_1", OpType_Quantize},
                    {"gamma", OpType_Constant},
                    {"mul_0", OpType_Mul},
                    {"beta", OpType_Constant},
                    {"add_1", OpType_Add},
                },
                {
                    {"dequantize_0", "reduce_0"},
                    {"reduce_0", "eltwise_0"},
                    {"dequantize_0", "eltwise_0"},
                    {"eltwise_0", "pow_0"},
                    {"pow_y", "pow_0"},
                    {"pow_0", "reduce_1"},
                    {"reduce_1", "quantize_0"},
                    {"quantize_0", "add_0"},
                    {"epsilon", "add_0"},
                    {"add_0", "dequantize_1"},
                    {"dequantize_1", "sqrt_0"},
                    {"eltwise_0", "div_0"},
                    {"sqrt_0", "div_0"},
                    {"div_0", "quantize_1"},
                    {"quantize_1", "mul_0"},
                    {"gamma", "mul_0"},
                    {"mul_0", "add_1"},
                    {"beta", "add_1"},
                },
            },
            {
                {
                    {"dequantize_0", OpType_DeQuantize},
                    {"reduce_0", OpType_Reduce},
                    {"tile_0", OpType_Tile},
                    {"eltwise_0", OpType_Eltwise},
                    {"pow_y", OpType_Constant},
                    {"pow_0", OpType_Pow},
                    {"reduce_1", OpType_Reduce},
                    {"quantize_0", OpType_Quantize},
                    {"dequantize_1", OpType_DeQuantize},
                    {"sqrt_0", OpType_Sqrt},
                    {"tile_1", OpType_Tile},
                    {"div_0", OpType_Div},
                    {"quantize_1", OpType_Quantize},
                    {"gamma", OpType_Constant},
                    {"mul_0", OpType_Mul},
                    {"beta", OpType_Constant},
                    {"add_1", OpType_Add},
                },
                {
                    {"dequantize_0", "reduce_0"},
                    {"reduce_0", "tile_0"},
                    {"tile_0", "eltwise_0"},
                    {"dequantize_0", "eltwise_0"},
                    {"eltwise_0", "pow_0"},
                    {"pow_y", "pow_0"},
                    {"pow_0", "reduce_1"},
                    {"reduce_1", "quantize_0"},
                    {"quantize_0", "dequantize_1"},
                    {"dequantize_1", "sqrt_0"},
                    {"sqrt_0", "tile_1"},
                    {"eltwise_0", "div_0"},
                    {"tile_1", "div_0"},
                    {"div_0", "quantize_1"},
                    {"quantize_1", "mul_0"},
                    {"gamma", "mul_0"},
                    {"mul_0", "add_1"},
                    {"beta", "add_1"},
                },
            },
        };
        for (auto &pattern : patterns)
        {
            GraphMatcher matcher(pattern);
            matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
            auto matched = matcher.match(g);

            for (auto &&m : matched)
            {

                if (!criteria(m))
                {
                    continue;
                }

                NodePtr mul = nullptr;
                NodePtrList constants;
                NodePtrList reduces;
                for (auto node : m)
                {
                    if (node->type() == OpType_Constant)
                    {
                        constants.push_back(node);
                    }
                    if (node->type() == OpType_Mul)
                    {
                        mul = node;
                    }
                    if (node->type() == OpType_Reduce)
                    {
                        reduces.push_back(node);
                    }
                }
                auto input = m[0]->outputs()[0];
                auto output = m[m.size() - 1]->outputs()[0];
                auto gamma = constants[constants.size() - 2];
                auto beta = constants[constants.size() - 1];

                NodePtr epsilon = nullptr;
                if (constants.size() == 4)
                {
                    epsilon = constants[1];
                }

                // axis
                auto reduce_0 = reduces[0];
                auto reduce_0_params = reduce_0->params();
                std::vector<int32_t> reduce_axes = reduce_0_params["axis"];
                for (auto &axis : reduce_axes)
                {
                    if (axis < 0)
                    {
                        axis = axis + reduce_0->inputs()[0]->shape().dim();
                    }
                }
                sort(reduce_axes.begin(), reduce_axes.end());
                reduce_axes.erase(unique(reduce_axes.begin(), reduce_axes.end()), reduce_axes.end());

                std::vector<int32_t> dims;
                for (auto &axis : reduce_axes)
                {
                    dims.push_back(input->shape()[axis]);
                }

                // weights
                ComputeTensor gamma_ct = gamma->constants()["weights"];
                auto gamma_q = gamma->constants()["weights"]->quantization();
                gamma_ct = gamma_ct.reshape(dims);
                gamma_ct = gamma_ct.astype(Tensor::FP32);
                gamma_ct = (gamma_ct - gamma_q.offset()) * gamma_q.scale();
                gamma_ct = gamma_ct.astype(input->type());
                TensorPtr gamma_t = gamma_ct;
                gamma_t->quantization() = Quantization(1.0, 0);
                gamma_t->set_type(input->type());
                gamma_t->quantization().set_aipu_dtype(input->type());
                gamma_t->quantization().set_data_unchanged(true);
                gamma_t->quantization().set_granularity(Quantization::Per_Tensor);

                // biases
                TensorPtr beta_t_1 = beta->constants()["weights"];
                auto beta_q = beta->constants()["weights"]->quantization();
                ComputeTensor beta_ct = beta->constants()["weights"];
                beta_ct = beta_ct.reshape(dims);
                beta_ct = beta_ct.astype(Tensor::FP32);
                beta_ct = (beta_ct - beta_q.offset()) * beta_q.scale();
                TensorPtr beta_t = beta_ct;
                beta_t->quantization() = Quantization(1.0, 0);
                beta_t->set_type(Tensor::FP32);
                beta_t->quantization().set_aipu_dtype(Tensor::FP32);
                beta_t->quantization().set_data_unchanged(true);
                beta_t->quantization().set_granularity(Quantization::Per_Tensor);

                // epsilon
                float epsilon_value = 1e-5f;
                if (epsilon != nullptr)
                {
                    auto epsilon_weights = epsilon->constants()["weights"];
                    auto epsilon_q = epsilon_weights->quantization();
                    auto epsilon_dtype = epsilon_weights->type();
                    int32_t epsilon_value_i32 = get_data_table[epsilon_dtype](epsilon_weights);
                    epsilon_value = (epsilon_value_i32 - epsilon_q.offset()) * epsilon_q.scale();
                }

                // create new node layernorm
                bool unquantifiable = input->quantization().unquantifiable();
                input->quantization().set_unquantifiable(true);
                auto ln_output = ops::layer_normalization(input, gamma_t, beta_t, reduce_axes, epsilon_value, true, false);
                auto ln = ln_output[0]->node();

                input->quantization().set_unquantifiable(unquantifiable);
                auto qt_output = ops::quantize(ln_output[0], output->type(), ops::QuantizeRoundMode::RoundToEven, output->quantization());
                auto qt = qt_output->node();
                qt->remove_output(0);
                qt->add_output(output);
                for (size_t i = 1; i < m.size(); ++i)
                {
                    g->remove_node(m[i]);
                }
            }
        }
        return;
    }

    void fuse_layernorm_axis_1_transpose(GraphPtr g)
    {
        auto criteria = [](std::vector<NodePtr> m)
        {
            NodePtrList eltwises;
            NodePtr pow = nullptr;
            NodePtrList reduces;
            NodePtrList constants;
            for (auto node : m)
            {
                if (node->type() == OpType_Eltwise)
                {
                    eltwises.push_back(node);
                }
                if (node->type() == OpType_Pow)
                {
                    pow = node;
                }
                if (node->type() == OpType_Reduce)
                {
                    reduces.push_back(node);
                }
                if (node->type() == OpType_Constant)
                {
                    constants.push_back(node);
                }
            }
            // 0. check input shape
            auto input = m[0];
            if (input->inputs()[0]->shape().dim() != 3)
            {
                return false;
            }

            // 1. eltwise check
            if (eltwises.size() < 1)
            {
                return false;
            }
            // sub check
            NodePtr eltwise_sub = eltwises[0];
            auto sub_params = eltwise_sub->params();
            if (!sub_params.count("method") || sub_params["method"] != "SUB")
            {
                return false;
            }
            // mul check
            if (eltwises.size() == 2)
            {
                NodePtr eltwise_mul = eltwises[1];
                auto mul_params = eltwise_mul->params();
                if (!mul_params.count("method") || mul_params["method"] != "MUL")
                {
                    return false;
                }
            }
            // add check
            if (eltwises.size() == 3)
            {
                NodePtr eltwise_add = eltwises[2];
                auto add_params = eltwise_add->params();
                if (!add_params.count("method") || add_params["method"] != "ADD")
                {
                    return false;
                }
            }

            // 2. pow y check
            if (pow == nullptr)
            {
                return false;
            }
            NodePtr pow_y = pow->inputs()[1]->node();
            auto pow_y_weights = pow_y->constants()["weights"];
            auto pow_y_dtype = pow_y_weights->type();
            if (!get_data_table.count(pow_y_dtype))
            {
                return false;
            }
            int32_t pow_y_value = get_data_table[pow_y_dtype](pow_y_weights);
            if (pow_y_value != 2)
            {
                return false;
            }

            // 3. reduce mean check
            if (reduces.size() < 2)
            {
                return false;
            }
            NodePtr reduce_0 = reduces[0];
            NodePtr reduce_1 = reduces[1];
            auto reduce_0_params = reduce_0->params();
            auto reduce_1_params = reduce_1->params();
            if (!reduce_0_params.count("method") || reduce_0_params["method"] != "MEAN")
            {
                return false;
            }
            if (!reduce_1_params.count("method") || reduce_1_params["method"] != "MEAN")
            {
                return false;
            }

            std::vector<int32_t> reduce_0_axes = reduce_0_params["axis"];
            std::vector<int32_t> reduce_1_axes = reduce_1_params["axis"];
            if (reduce_0_axes.size() != 1)
            {
                return false;
            }
            if (reduce_1_axes.size() != 1)
            {
                return false;
            }
            int32_t axis_0 = reduce_0_axes[0];
            int32_t axis_1 = reduce_1_axes[0];
            axis_0 = axis_0 < 0 ? (axis_0 + reduce_0->inputs()[0]->shape().dim()) : axis_0;
            axis_1 = axis_1 < 0 ? (axis_1 + reduce_1->inputs()[0]->shape().dim()) : axis_1;
            if (axis_0 != axis_1 && axis_0 != 1)
            {
                return false;
            }

            // 4. constants shape check
            auto gamma = constants[constants.size() - 2];
            auto beta = constants[constants.size() - 1];
            auto gamma_t = gamma->constants()["weights"];
            auto beta_t = beta->constants()["weights"];
            if (static_cast<int>(gamma_t->size()) != input->outputs()[0]->shape()[axis_0] ||
                static_cast<int>(beta_t->size()) != input->outputs()[0]->shape()[axis_0])
            {
                return false;
            }

            return true;
        };
        std::vector<GraphMatcherPattern> patterns = {
            {
                {
                    {"dequantize_0", OpType_DeQuantize},
                    {"reduce_0", OpType_Reduce},
                    {"tile_0", OpType_Tile},
                    {"eltwise_0", OpType_Eltwise},
                    {"pow_y", OpType_Constant},
                    {"pow_0", OpType_Pow},
                    {"reduce_1", OpType_Reduce},
                    {"quantize_0", OpType_Quantize},
                    {"epsilon", OpType_Constant},
                    {"add_0", OpType_Add},
                    {"dequantize_1", OpType_DeQuantize},
                    {"sqrt_0", OpType_Sqrt},
                    {"tile_1", OpType_Tile},
                    {"div_0", OpType_Div},
                    {"quantize_1", OpType_Quantize},
                    {"gamma", OpType_Constant},
                    {"tile_2", OpType_Tile},
                    {"mul_0", OpType_Eltwise},
                    {"beta", OpType_Constant},
                    {"tile_3", OpType_Tile},
                    {"add_1", OpType_Eltwise},
                },
                {
                    {"dequantize_0", "reduce_0"},
                    {"reduce_0", "tile_0"},
                    {"tile_0", "eltwise_0"},
                    {"dequantize_0", "eltwise_0"},
                    {"eltwise_0", "pow_0"},
                    {"pow_y", "pow_0"},
                    {"pow_0", "reduce_1"},
                    {"reduce_1", "quantize_0"},
                    {"quantize_0", "add_0"},
                    {"epsilon", "add_0"},
                    {"add_0", "dequantize_1"},
                    {"dequantize_1", "sqrt_0"},
                    {"sqrt_0", "tile_1"},
                    {"eltwise_0", "div_0"},
                    {"tile_1", "div_0"},
                    {"div_0", "quantize_1"},
                    {"quantize_1", "mul_0"},
                    {"gamma", "tile_2"},
                    {"tile_2", "mul_0"},
                    {"mul_0", "add_1"},
                    {"beta", "tile_3"},
                    {"tile_3", "add_1"},
                },
            },
            {
                {
                    {"dequantize_0", OpType_DeQuantize},
                    {"reduce_0", OpType_Reduce},
                    {"tile_0", OpType_Tile},
                    {"eltwise_0", OpType_Eltwise},
                    {"pow_y", OpType_Constant},
                    {"pow_0", OpType_Pow},
                    {"reduce_1", OpType_Reduce},
                    {"quantize_0", OpType_Quantize},
                    {"epsilon", OpType_Constant},
                    {"add_0", OpType_Add},
                    {"dequantize_1", OpType_DeQuantize},
                    {"sqrt_0", OpType_Sqrt},
                    {"tile_1", OpType_Tile},
                    {"div_0", OpType_Div},
                    {"quantize_1", OpType_Quantize},
                    {"gamma", OpType_Constant},
                    {"mul_0", OpType_Mul},
                    {"beta", OpType_Constant},
                    {"add_1", OpType_Add},
                },
                {
                    {"dequantize_0", "reduce_0"},
                    {"reduce_0", "tile_0"},
                    {"tile_0", "eltwise_0"},
                    {"dequantize_0", "eltwise_0"},
                    {"eltwise_0", "pow_0"},
                    {"pow_y", "pow_0"},
                    {"pow_0", "reduce_1"},
                    {"reduce_1", "quantize_0"},
                    {"quantize_0", "add_0"},
                    {"epsilon", "add_0"},
                    {"add_0", "dequantize_1"},
                    {"dequantize_1", "sqrt_0"},
                    {"sqrt_0", "tile_1"},
                    {"eltwise_0", "div_0"},
                    {"tile_1", "div_0"},
                    {"div_0", "quantize_1"},
                    {"quantize_1", "mul_0"},
                    {"gamma", "mul_0"},
                    {"mul_0", "add_1"},
                    {"beta", "add_1"},
                },
            },
        };
        for (auto &pattern : patterns)
        {
            GraphMatcher matcher(pattern);
            matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
            auto matched = matcher.match(g);

            for (auto &&m : matched)
            {
                if (!criteria(m))
                {
                    continue;
                }

                NodePtrList constants;
                NodePtrList reduces;
                NodePtrList eltwises;
                for (auto node : m)
                {
                    if (node->type() == OpType_Constant)
                    {
                        constants.push_back(node);
                    }
                    if (node->type() == OpType_Reduce)
                    {
                        reduces.push_back(node);
                    }
                    if (node->type() == OpType_Eltwise)
                    {
                        eltwises.push_back(node);
                    }
                }
                auto input = m[0]->outputs()[0];
                auto output = m[m.size() - 1]->outputs()[0];
                auto gamma = constants[constants.size() - 2];
                auto beta = constants[constants.size() - 1];
                auto add = eltwises[eltwises.size() - 1];
                NodePtr epsilon = nullptr;
                if (constants.size() == 4)
                {
                    epsilon = constants[1];
                }

                // axis
                NodePtr reduce_0 = reduces[0];
                auto reduce_0_params = reduce_0->params();
                std::vector<int32_t> reduce_0_axes = reduce_0_params["axis"];
                int32_t axis_0 = reduce_0_axes[0];
                axis_0 = axis_0 < 0 ? (axis_0 + reduce_0->inputs()[0]->shape().dim()) : axis_0;
                std::vector<int32_t> axis = {-1};
                int32_t dim = output->shape()[axis_0];

                // weights
                ComputeTensor gamma_ct = gamma->constants()["weights"];
                auto gamma_q = gamma->constants()["weights"]->quantization();
                gamma_ct = gamma_ct.reshape(std::vector<int32_t>{dim});
                gamma_ct = gamma_ct.astype(Tensor::FP32);
                gamma_ct = (gamma_ct - gamma_q.offset()) * gamma_q.scale();
                gamma_ct = gamma_ct.astype(input->type());
                TensorPtr gamma_t = gamma_ct;
                gamma_t->quantization() = Quantization(1.0, 0);
                gamma_t->set_type(input->type());
                gamma_t->quantization().set_aipu_dtype(input->type());
                gamma_t->quantization().set_data_unchanged(true);
                gamma_t->quantization().set_granularity(Quantization::Per_Tensor);

                // biases
                TensorPtr beta_t_1 = beta->constants()["weights"];
                auto beta_q = beta->constants()["weights"]->quantization();
                ComputeTensor beta_ct = beta->constants()["weights"];
                beta_ct = beta_ct.reshape(std::vector<int32_t>{dim});
                beta_ct = beta_ct.astype(Tensor::FP32);
                beta_ct = (beta_ct - beta_q.offset()) * beta_q.scale();
                TensorPtr beta_t = beta_ct;
                beta_t->quantization() = Quantization(1.0, 0);
                beta_t->set_type(Tensor::FP32);
                beta_t->quantization().set_aipu_dtype(Tensor::FP32);
                beta_t->quantization().set_data_unchanged(true);
                beta_t->quantization().set_granularity(Quantization::Per_Tensor);

                // epsilon
                float epsilon_value = 1e-5f;
                if (epsilon != nullptr)
                {
                    auto epsilon_weights = epsilon->constants()["weights"];
                    auto epsilon_q = epsilon_weights->quantization();
                    auto epsilon_dtype = epsilon_weights->type();
                    int32_t epsilon_value_i32 = get_data_table[epsilon_dtype](epsilon_weights);
                    epsilon_value = (epsilon_value_i32 - epsilon_q.offset()) * epsilon_q.scale();
                }

                // create new node transpose1 + batchnorm + transpose2 + quantize
                auto transpose1_output = ops::transpose(input, {0, 2, 1});
                bool unquantifiable = transpose1_output->quantization().unquantifiable();

                transpose1_output->quantization().set_unquantifiable(true);
                auto ln_output = ops::layer_normalization(transpose1_output, gamma_t, beta_t, axis, epsilon_value, true, false);
                auto ln = ln_output[0]->node();
                transpose1_output->quantization().set_unquantifiable(unquantifiable);

                auto transpose2_output = ops::transpose(ln_output[0], {0, 2, 1});
                auto transpose2 = transpose2_output->node();

                auto qt_output = ops::quantize(transpose2_output, output->type(), ops::QuantizeRoundMode::RoundToEven, output->quantization());
                auto qt = qt_output->node();
                qt->remove_output(0);
                qt->add_output(output);

                for (size_t i = 1; i < m.size(); ++i)
                {
                    g->remove_node(m[i]);
                }
            }
        }
        return;
    }
} // namespace

void fuse_algo_nodes_to_layernorm(GraphPtr g)
{
    fuse_layernorm_axis_continue(g);
    fuse_layernorm_axis_1_transpose(g);

    return;
}
}; // namespace qtlibn