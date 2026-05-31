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
 * @Date: 2022-03-14 19:05:28
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-07 11:20:50
 */

#ifndef __QUANTIZER_H__
#define __QUANTIZER_H__

#include "common.h"
#include "core/node.h"
#include "core/graph.h"
#include "core/subgraph.h"
#include "core/quantization.h"
#include "dataset.h"
#include "quantization/qtransform.h"
#include <cstdint>
#include <string>

namespace qtlibn
{
class Quantizer
{
public:
    // these constructors are for quantizer tool
    Quantizer(aipubt::GraphPtr graph, TensorQInfo_t *tensor_qinfo);
    Quantizer(aipubt::GraphPtr graph, std::string scheme, const int32_t bits);
    Quantizer(aipubt::GraphPtr graph, const int32_t bits);
    Quantizer(aipubt::GraphPtr graph);

    // these constructors are for OpsAPI
    Quantizer(aipubt::NodePtr node, std::string scheme, const int32_t bits);
    Quantizer(aipubt::NodePtr node, const int32_t bits);
    Quantizer(aipubt::NodePtr node);

    // this constructor is for qtransform
    Quantizer() {}

    ~Quantizer();

    void quantize(void);
    void forward(void);
    // void forward(std::vector<float_t *> input_vec);
    void quantize(aipubt::NodePtr node);
    void quantize_subgraph(aipubt::NodePtr node);
    void forward(aipubt::NodePtr node);
    // void statistic(aipubt::NodePtr node);

private:
    // bool _check_node_neccessary_qinfo(aipubt::NodePtr node);
    void _node_quantize(aipubt::NodePtr node);
    aipubt::GraphPtr m_graph = nullptr;
    std::string m_scheme = "Symmetric";
    // Dataset *m_dataset = nullptr;
};

}; // namespace qtlibn

#endif
