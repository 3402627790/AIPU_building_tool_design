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
 * @Date: 2022-11-22 20:11:32
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-07 11:13:59
 */

#ifndef __QT_INFERENCE_H__
#define __QT_INFERENCE_H__

#include "quantization_transform/common.h"
#include "quantization_transform/dataset.h"

namespace qtlibn
{

// class Dataset;
// class DataLoader;

class Inferencer
{
public:
    Inferencer() {}
    ~Inferencer() {}

    // void inference(void);
    // forward_mode in ['float', 'quantized']
    static void inference(aipubt::NodePtr node, std::string forward_mode = "float");
    static void prepare(aipubt::NodePtr node);
    static void update(aipubt::NodePtr node, aipubt::NodePtr from_node);

    // void set_dataset(DatasetPtr dataset)
    // {
    //     m_dataset = dataset;
    // }

private:
    // aipubt::GraphPtr m_graph = nullptr;
    // aipubt::NodePtr m_node = nullptr;
    // DatasetPtr m_dataset = nullptr;
    // std::string m_forward_mode;
};

} // namespace qtlibn

#endif