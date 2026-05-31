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
 * @Date: 2022-03-11 18:08:52
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-12-28 11:36:33
 */

#include "quantization_transform/adaptor.h"

namespace qtlibn
{

void Adaptor::calculate_max_min(aipubt::NodePtr node)
{
    for (auto &t : node->outputs())
    {
        m_transform->calculate_activation_max_min(t);
    }

    for (auto &c : node->constants())
    {
        m_transform->calculate_constant_max_min(c.second);
    }

    for (auto &p : node->placeholders())
    {
        m_transform->calculate_activation_max_min(p);
    }
}

void Adaptor::dequantize_constant(aipubt::NodePtr node)
{
    for (auto &c : node->constants())
    {
        // INFO() << "constant[" << c.first << "].data_unchanged =  " << (int32_t)(c.second->quantization().data_unchanged());
        if (!c.second->quantization().data_unchanged())
        {
            m_transform->dequantize_constant(c.second);
        }
    }
}

void Adaptor::dequantize_param(aipubt::NodePtr node)
{
    m_transform->dequantize_param(node);
}

void Adaptor::calculate_scale_zp(aipubt::NodePtr node)
{
    for (auto &t : node->outputs())
    {
        if (!t->quantization().quantized())
        {
            m_transform->calculate_activation_scale_zp(t);
        }
    }

    for (auto &c : node->constants())
    {
        if (!c.second->quantization().quantized())
        {
            m_transform->calculate_constant_scale_zp(c.second);
        }
    }

    for (auto &p : node->placeholders())
    {
        if (!p->quantization().quantized())
        {
            m_transform->calculate_placeholder_scale_zp(p);
        }
    }
}

void Adaptor::calculate_activation_max_min(aipubt::TensorPtr tensor)
{
    m_transform->calculate_activation_max_min(tensor);
}

}; // namespace qtlibn