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

#ifndef CORE_SURFACE_H
#define CORE_SURFACE_H

#include "core/tensor.h"
#include "tongue/symbolic.h"
#include "logger.h"
#include <functional>

namespace aipubt
{
using namespace tongue;

struct cpsSurfDesc
{
    struct Field
    {
        Field &operator=(const Expr &val)
        {
            value = val;
            return *this;
        }

        bool is_constant() const
        {
            if (value)
            {
                return value->is_constant();
            }
            return true;
        }

        operator bool() const
        {
            return value;
        }

        operator Expr() const
        {
            return value;
        }

        operator uint32_t() const
        {
            if (value && value->is_constant())
            {
                return value->value();
            }
            return 0;
        }

        std::string to_string() const
        {
            if (value)
            {
                return value->to_string();
            }
            return std::string();
        }

        Expr value;
        uint32_t shift;
        uint32_t width;
    };

    Expr operator[](int idx) const
    {
        if (idx < 0 || idx > 1)
        {
            return Expr();
        }
        std::vector<Field> fields;
        if (idx == 0)
        {
            fields = {
                layout,
                rank,
                reserved,
                data_size,
                uma_numa,
                dimension0,
            };
        }
        else
        {
            fields = {
                dimension1,
            };
        }
        Expr ret = 0;
        for (auto &it : fields)
        {
            if (!it)
            {
                continue;
            }
            if (it.shift == 0 && it.width == 32)
            {
                ret = it.value;
            }
            else
            {
                auto mask = ((1ul << it.width) - 1u) << it.shift;
                ret = (ret & ~mask) | ((it.value << it.shift) & mask);
            }
        }
        return ret;
    }

    std::map<std::string, std::reference_wrapper<Field>> get_fields()
    {
        std::map<std::string, std::reference_wrapper<Field>> fields = {
            {"layout", layout},
            {"rank", rank},
            {"reserved", reserved},
            {"data_size", data_size},
            {"uma_numa", uma_numa},
            {"dimension0", dimension0},
            {"dimension1", dimension1},
            {"width", width},
            {"height", height},
            {"depth", depth},
        };
        return fields;
    }

    Field layout{nullptr, 0, 3};
    Field rank{nullptr, 3, 2};
    Field reserved{nullptr, 5, 3};
    Field data_size{nullptr, 8, 7};
    Field uma_numa{nullptr, 15, 1};
    Field dimension0{nullptr, 16, 16};
    Field dimension1{nullptr, 0, 32};
    Field width{nullptr, 0, 32};
    Field height{nullptr, 0, 32};
    Field depth{nullptr, 0, 32};
};

/**
 * @brief a class for named Surface descriptor
 *
 */
class Surface : public Tensor
{
public:
    Surface() = delete;
    Surface(TensorPtr t);
    ~Surface() = default;

    void set_layout(uint32_t layout);
    void set_data_size(uint32_t data_size);
    void set_uma_numa(uint32_t uma);
    void set_width(Expr width);
    void set_height(Expr height);
    void set_depth(Expr depth);

    cpsSurfDesc &get_surf_desc();

protected:
    cpsSurfDesc m_surf_desc;
};

using SurfacePtr = std::shared_ptr<Surface>;
using SurfaceRef = std::weak_ptr<Surface>;

} // namespace aipubt
#endif //CORE_SURFACE_H