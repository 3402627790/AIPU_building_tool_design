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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 14:50:38
 */

#ifndef __QUANTIZATION_TRANSFORM_H__
#define __QUANTIZATION_TRANSFORM_H__

#include "core/graph.h"
#include "core/quantization.h"
#include <cstdint>
#include <string>

namespace qtlibn
{

typedef struct
{
    bool qinvariant;
    bool data_unchanged;
    int32_t bits;
    aipubt::Quantization::Scheme scheme;
    aipubt::Quantization::Granularity granularity;
    aipubt::Quantization::QuantizedRange quantized_range;
} QInfo_t;

typedef struct
{
    QInfo_t activation_qinfo;
    QInfo_t weight_qinfo;
    QInfo_t bias_qinfo;
} TensorQInfo_t;

class QuantizationTransform
{
public:
    QuantizationTransform(aipubt::GraphPtr graph);

    QuantizationTransform(aipubt::GraphPtr graph,
                          std::string run_mode,
                          TensorQInfo_t *tensor_qinfo);

    QuantizationTransform(aipubt::GraphPtr graph,
                          std::string run_mode,
                          int32_t bits);

    QuantizationTransform(std::string run_mode, int32_t bits);

    ~QuantizationTransform();

    void transform(void);
    void quantize(void);
    void forward(void);
    void clear_placeholders(void);
    void run(void);

    void quantize(aipubt::NodePtr node);
    void forward(aipubt::NodePtr node);

    void serialize(const std::string ir_txt, const std::string ir_bin);

private:
    aipubt::GraphPtr m_graph = nullptr;
    std::string m_run_mode = "Normal";
};

}; // namespace qtlibn

#endif