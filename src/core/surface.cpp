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
 * @Author: Kyle Huang
 * @Date: 2026-05-14 12:16:08
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-05-14 17:43:59
 */
#include "core/surface.h"

namespace aipubt
{

Surface::~Surface(){};

Surface::Surface(TensorPtr t) : Tensor(*t)
{
}
void Surface::set_layout(uint64_t layout)
{
    m_surf_desc.layout = layout;
}

void Surface::set_data_size(uint64_t data_type)
{
    m_surf_desc.data_type = data_type;
}

void Surface::set_uma_numa(uint64_t uma_numa)
{
    m_surf_desc.uma_numa = uma_numa;
}

void Surface::set_width(uint64_t width)
{
    m_surf_desc.width = width;
}

void Surface::set_height(uint64_t height)
{
    m_surf_desc.height = height;
}

void Surface::set_depth(uint64_t depth)
{
    m_surf_desc.depth = depth;
}

void Surface::set_surf_desc(const cpsSurfDesc &surf_desc)
{
    m_surf_desc = surf_desc;
}

cpsSurfDesc Surface::get_surf_desc()
{
    return m_surf_desc;
}

} // namespace aipubt
