//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
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

Surface::Surface(TensorPtr t) : Tensor(*t)
{
    this->memory_offset()->set_base_offset(t->memory_offset());
}

void Surface::set_layout(uint32_t layout)
{
    m_surf_desc.layout = layout;
}

void Surface::set_data_size(uint32_t data_size)
{
    m_surf_desc.data_size = data_size;
}

void Surface::set_uma_numa(uint32_t uma_numa)
{
    m_surf_desc.uma_numa = uma_numa;
}

void Surface::set_width(Expr width)
{
    m_surf_desc.width = width;
}

void Surface::set_height(Expr height)
{
    m_surf_desc.height = height;
}

void Surface::set_depth(Expr depth)
{
    m_surf_desc.depth = depth;
}

cpsSurfDesc &Surface::get_surf_desc()
{
    enum SurfLayout : uint32_t
    {
        LAYOUT_LINEAR = 0,
        LAYOUT_NHW1C1W8Cx = 1,
        LAYOUT_M1K1M8Kx = 2,
    };

    uint32_t rank = 0;
    uint32_t layout = m_surf_desc.layout;
    if (layout == LAYOUT_LINEAR)
    {
        if (m_surf_desc.height.is_constant() && (uint32_t)m_surf_desc.height <= 1 &&
            m_surf_desc.depth.is_constant() && (uint32_t)m_surf_desc.depth <= 1)
        {
            rank = 0; // 1D
        }
        else if (m_surf_desc.depth.is_constant() && (uint32_t)m_surf_desc.depth <= 1)
        {
            rank = 1;
        }
        else
        {
            rank = 2;
        }
    }
    else if (layout == LAYOUT_M1K1M8Kx)
    {
        rank = 1;
    }
    else if (layout == LAYOUT_NHW1C1W8Cx)
    {
        rank = 2;
    }
    else
    {
        rank = 0;
    }
    m_surf_desc.rank = rank;

    Expr width = m_surf_desc.width;
    Expr height = m_surf_desc.height;
    Expr depth = m_surf_desc.depth;
    if (rank == 0)
    {
        m_surf_desc.dimension0 = width & 0xFFFFUL;
        m_surf_desc.dimension1 = (width >> 16) & 0xFFFFUL;
    }
    else if (rank == 1)
    {
        m_surf_desc.dimension0 = width & 0xFFFFUL;
        m_surf_desc.dimension1 = ((width >> 16) & 0xFFUL) | ((height & 0xFFFFFFUL) << 8);
    }
    else
    {
        m_surf_desc.dimension0 = width & 0xFFFFUL;
        m_surf_desc.dimension1 = (height & 0xFFFFUL) | ((depth & 0xFFFFUL) << 16);
    }

    return m_surf_desc;
}

} // namespace aipubt
