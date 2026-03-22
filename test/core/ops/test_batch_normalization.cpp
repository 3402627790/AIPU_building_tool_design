//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file
// may only be used by a person authorised under and to the extent permitted
// by a subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2018 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//
//       Release Information:
//
//--------------------------------------------------------------------------------

#include <builder/builder.h>
#include <gcompiler_api.h>

#include <core/graph.h>
#include <core/ops/ops.h>
#include <gcompiler_api.h>

#include "ops_test_base.h"

using namespace aipubt;
namespace test_ops
{
// qtlib will merge gamma/beta/mean/variance into weights/bias
class OpTestBatchNorm : public OpTestBase
{
public:
    OpTestBatchNorm() {};
    virtual ~OpTestBatchNorm() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_batch_norm");
        auto inp = ops::input({1, 1, 40, 4}, Tensor::Dtype::UINT8, {1.0, 128}, g);

        std::vector<int16_t> weight_val = {1, 2, 3, 4};
        int16_t *new_weight_val = new int16_t[weight_val.size()];
        memcpy(new_weight_val, weight_val.data(), weight_val.size() * sizeof(int16_t));
        auto weight = ops::make_weight(g, new_weight_val, TensorShape(weight_val.size()), Tensor::INT16, Quantization(1.0, 20));

        std::vector<int32_t> bias_val = {5, 6, 7, 8};
        int32_t *new_bias_val = new int32_t[bias_val.size()];
        memcpy(new_bias_val, bias_val.data(), bias_val.size() * sizeof(int32_t));
        auto bias = ops::make_bias(g, new_bias_val, bias_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        std::vector<int32_t> mean_val = {5, 6, 7, 8};
        int32_t *new_mean_val = new int32_t[mean_val.size()];
        memcpy(new_mean_val, mean_val.data(), mean_val.size() * sizeof(int32_t));
        auto mean = ops::make_tensor(g, new_mean_val, mean_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        std::vector<int32_t> var_val = {5, 6, 7, 8};
        int32_t *new_var_val = new int32_t[var_val.size()];
        memcpy(new_var_val, var_val.data(), var_val.size() * sizeof(int32_t));
        auto var = ops::make_tensor(g, new_var_val, var_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        // auto a = ops::batch_normalization(inp, weight, bias, mean, var, -1, 1e-5f, Quantization(1.0, 100));
        auto a = ops::batch_normalization(inp, weight, bias, -1, 1e-5f, Quantization(1.0, 100));
        if (!a)
        {
            return false;
        }
        qtlibn::QuantizationTransform asym_sym(g, "Sym", 8);
        asym_sym.quantize();

        g->to_png("batch_norm.png");
        g->serialize("graph_batch_norm.def", "weight_batch_norm.bin");

        return true;
    }
};

class OpTestBatchNormAxis : public OpTestBase
{
public:
    OpTestBatchNormAxis() {};
    virtual ~OpTestBatchNormAxis() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_batch_norm_axis");
        auto inp = ops::input({1, 1, 8, 4}, Tensor::Dtype::UINT8, {1.0, 128}, g);

        std::vector<int16_t> weight_val = {1, 2, 3, 4};
        int16_t *new_weight_val = new int16_t[weight_val.size()];
        memcpy(new_weight_val, weight_val.data(), weight_val.size() * sizeof(int16_t));
        auto weight = ops::make_weight(g, new_weight_val, TensorShape(weight_val.size()), Tensor::INT16, Quantization(1.0, 20));

        std::vector<int32_t> bias_val = {5, 6, 7, 8};
        int32_t *new_bias_val = new int32_t[bias_val.size()];
        memcpy(new_bias_val, bias_val.data(), bias_val.size() * sizeof(int32_t));
        auto bias = ops::make_bias(g, new_bias_val, bias_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        std::vector<int32_t> mean_val = {5, 6, 7, 8};
        int32_t *new_mean_val = new int32_t[mean_val.size()];
        memcpy(new_mean_val, mean_val.data(), mean_val.size() * sizeof(int32_t));
        auto mean = ops::make_tensor(g, new_mean_val, mean_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        std::vector<int32_t> var_val = {5, 6, 7, 8};
        int32_t *new_var_val = new int32_t[var_val.size()];
        memcpy(new_var_val, var_val.data(), var_val.size() * sizeof(int32_t));
        auto var = ops::make_tensor(g, new_var_val, var_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        auto a = ops::batch_normalization(inp, weight, bias, 2, 1e-5f, Quantization(1.0, 100));
        if (!a)
        {

            return false;
        }
        qtlibn::QuantizationTransform asym_sym(g, "Sym", 8);
        asym_sym.quantize();

        g->to_png("batch_norm_axis.png");
        g->serialize("graph_batch_norm_axis.def", "weight_batch_norm_axis.bin");

        return true;
    }
};

class OpTestBatchNormNo4D : public OpTestBase
{
public:
    OpTestBatchNormNo4D() {};
    virtual ~OpTestBatchNormNo4D() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_batch_norm_no_4d");
        auto inp = ops::input({1, 8, 4}, Tensor::Dtype::UINT8, {1.0, 128}, g);

        std::vector<int16_t> weight_val = {1, 2, 3, 4};
        int16_t *new_weight_val = new int16_t[weight_val.size()];
        memcpy(new_weight_val, weight_val.data(), weight_val.size() * sizeof(int16_t));
        auto weight = ops::make_weight(g, new_weight_val, TensorShape(weight_val.size()), Tensor::Dtype::INT16, Quantization(1.0, 20));

        std::vector<int32_t> bias_val = {5, 6, 7, 8};
        int32_t *new_bias_val = new int32_t[bias_val.size()];
        memcpy(new_bias_val, bias_val.data(), bias_val.size() * sizeof(int32_t));
        auto bias = ops::make_bias(g, new_bias_val, bias_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        std::vector<int32_t> mean_val = {0, 0, 0, 0};
        int32_t *new_mean_val = new int32_t[mean_val.size()];
        memcpy(new_mean_val, mean_val.data(), mean_val.size() * sizeof(int32_t));
        auto mean = ops::make_tensor(g, new_mean_val, mean_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        std::vector<int32_t> var_val = {1, 1, 1, 1};
        int32_t *new_var_val = new int32_t[var_val.size()];
        memcpy(new_var_val, var_val.data(), var_val.size() * sizeof(int32_t));
        auto var = ops::make_tensor(g, new_var_val, var_val.size() * sizeof(int32_t), Tensor::INT32, Quantization(1.0, 30));

        auto a = ops::batch_normalization(inp, weight, bias, 2, 1e-5f, Quantization(1.0, 100));
        if (!a)
        {

            return false;
        }
        qtlibn::QuantizationTransform asym_sym(g, "Sym", 8);
        asym_sym.quantize();

        g->to_png("batch_norm_no_4d.png");
        g->serialize("graph_batch_norm_no_4d.def", "weight_batch_norm_no_4d.bin");

        return true;
    }
};

REGISTER_TEST_OP(BatchNorm, "batch_normalization", true);
REGISTER_TEST_OP(BatchNormAxis, "batch_normalization", true);
REGISTER_TEST_OP(BatchNormNo4D, "batch_normalization", true);

} // namespace test_ops