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
 * @Author: Neo Wang 
 * @Date: 2020-01-29 14:27:25 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-01 17:18:34
 */
#include "builder/readonly_param.h"
#include "core/compute_tensor.h"
#include "core/core.h"
namespace aipubt
{

ReadOnlyParam::ReadOnlyParam(ComputeTensor o) : tensor(o) {}

ReadOnlyParam::ReadOnlyParam(NodeParamValue o)
{
    if (o.t < NodeParamValue::NodeParamType_float)
    {
        value = o.v.i;
    }
    else if (o.t == NodeParamValue::NodeParamType_float)
    {
        float f = o;
        value = *((uint32_t *)&f);
    }
    else
    {
        value = 0;
    }
}

ReadOnlyParam::ReadOnlyParam(int64_t v)
{
    value = *((uint64_t *)&v);
}

ReadOnlyParam::ReadOnlyParam(int32_t v)
{
    value = *((uint32_t *)&v);
}

ReadOnlyParam::ReadOnlyParam(float v)
{
    value = *((uint32_t *)&v);
}

ReadOnlyParam::ReadOnlyParam(half v)
{
    value = *((uint16_t *)&v);
}

ReadOnlyParam::ReadOnlyParam(bfp16 v)
{
    value = *((uint16_t *)&v);
}

ReadOnlyParam::ReadOnlyParam(fp8e5m2 v)
{
    value = *((uint8_t *)&v);
}

ReadOnlyParam::ReadOnlyParam(fp8e4m3fn v)
{
    value = *((uint8_t *)&v);
}

std::ostream &operator<<(std::ostream &os, const ReadOnlyParam &p)
{
    if (p.tensor)
    {
        return os << p.tensor;
    }
    else if (p.expr)
    {
        return os << (p.expr->to_string());
    }
    return os << p.value;
}
}; // namespace aipubt
