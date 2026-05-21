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

#ifndef CORE_SURFACE_H
#define CORE_SURFACE_H

#include "core/tensor.h"
#include "logger.h"
#include <functional>

namespace aipubt
{

typedef struct cpsSurfDesc
{
    unsigned long long layout : 3;
    unsigned long long : 5;
    unsigned long long data_type : 7;
    unsigned long long uma_numa : 1;
    unsigned long long width : 16;
    unsigned long long height : 16;
    unsigned long long depth : 16;
} cpsSurfDesc;

/**
 * @brief a class for named Surface descriptor
 *
 */
class Surface : public Tensor
{
public:
    Surface() = delete;
    Surface(TensorPtr t);
    ~Surface();

    void set_layout(uint64_t layout);
    void set_data_size(uint64_t data_size);
    void set_uma_numa(uint64_t uma);
    void set_width(uint64_t width);
    void set_height(uint64_t height);
    void set_depth(uint64_t depth);

    void set_surf_desc(const cpsSurfDesc &surf_desc);
    cpsSurfDesc get_surf_desc();

protected:
    cpsSurfDesc m_surf_desc;
};

using SurfacePtr = std::shared_ptr<Surface>;
using SurfaceRef = std::weak_ptr<Surface>;

} // namespace aipubt
#endif //CORE_SURFACE_H