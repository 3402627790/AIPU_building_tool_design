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
 * @Date: 2020-01-29 14:24:25 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-10-31 14:18:07
 */
#ifndef BUILDER_READ_ONLY_PARAM_H
#define BUILDER_READ_ONLY_PARAM_H
#include "builder/bldtypes.h"
#include <core/const_tensor.h>
#include <core/fp16.h>
#include <core/placeholder.h>
#include <core/surface.h>
#include <cstdint>
#include <string>
#include <tongue/symbolic.h>
#include <vector>
namespace aipubt
{
using namespace tongue;
struct NodeParamValue;

/**
 * @brief  The ReadOnlyParam is for host-device param, it only accept 32bit number store in uint32_t or tensor, which is the memory ptr
 *          For dynamic shape, a expr may be a RO.
 *          
 *          which value stands for the RO?
 *          1. tensor if tensor is not null and expr is null
 *          2. expr if expr is not null and tensor is null
 *          3. value if tensor and expr are all null
 *          4. define by value, if tensor and expr are not null, if the value!=0, then is tensor, else is expr.
 * 
 */
struct ReadOnlyParam
{
    /**
     * @brief the value parameter in 32bit view
     * 
     */
    uint64_t value = 0;

    /**
     * @brief the memory paramter(reference parameter) you can access in AIPU as a pointer.
     * 
     */
    TensorPtr tensor = 0;

    Expr expr;

    ConstTensorPtr const_tensor = 0;

    SurfacePtr surface = 0;

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a uint64_t to construct the paramter.
     */
    ReadOnlyParam(uint64_t v) : value(v) {}

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a int64_t to construct the paramter, the int64_t will be stored as uint64_t.
     */
    ReadOnlyParam(int64_t v);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a int to construct the paramter, the int will be cast to uint32_t.
     */
    ReadOnlyParam(int32_t v);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a uint32_t to construct the paramter.
     */
    ReadOnlyParam(uint32_t v) : value(v) {}

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a float to construct the paramter.
     */
    ReadOnlyParam(float v);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a half to construct the paramter.
     */
    ReadOnlyParam(half v);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a bfp16 to construct the paramter.
     */
    ReadOnlyParam(bfp16 v);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a fp8e5m2 to construct the paramter.
     */
    ReadOnlyParam(fp8e5m2 v);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param v use a fp8e4m3fn to construct the paramter.
     */
    ReadOnlyParam(fp8e4m3fn v);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param t use a TensorPtr to construct the paramter as a memory paramter.
     */
    ReadOnlyParam(TensorPtr t) : tensor(t)
    {
        if (!t)
        {
            ERROR() << "A null pointer cannot be an argument!";
        }
    }
    ReadOnlyParam(ConstTensorPtr t) : const_tensor(t)
    {
        if (!t)
        {
            ERROR() << "A null pointer cannot be an argument!";
        }
    }
    ReadOnlyParam(SurfacePtr s) : surface(s)
    {
        if (!s)
        {
            ERROR() << "A null pointer cannot be an argument!";
        }
    }

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param t use a PlaceholderPtr to construct the paramter as a memory paramter.
     */
    ReadOnlyParam(PlaceholderPtr t) : tensor(t)
    {
        if (!t)
        {
            ERROR() << "A null pointer cannot be an argument!";
        }
    }

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param t use a TensorPtr to construct the paramter as a memory paramter.
     */
    ReadOnlyParam(ComputeTensor t);

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param o use a Expr to construct the paramter.
     */
    ReadOnlyParam(Expr o) : expr(o) {}

    /**
     * @brief Construct a new Read Only Param object
     * 
     * @param o use a NodeParamValue to construct the paramter, the NodeParamValue will be cast to a uint32_t.
     */
    ReadOnlyParam(NodeParamValue o);

    // /**
    //  * @brief implicitly convertion to uint32_t
    //  *
    //  * @return uint32_t
    //  */
    // operator uint32_t() const
    // {
    //     return value;
    // }

    /**
     * @brief implicitly convertion to uint64_t
     * 
     * @return uint32_t 
     */
    operator uint64_t() const
    {
        return value;
    }

    /**
     * @brief implicitly convertion to TensorPtr
     * 
     * @return TensorPtr 
     */
    operator TensorPtr()
    {
        return tensor;
    }

    operator ConstTensorPtr()
    {
        return const_tensor;
    }

    operator SurfacePtr()
    {
        return surface;
    }
    /**
     * @brief implicitly convertion to PlaceholderPtr
     * 
     * @return PlaceholderPtr 
     */
    operator PlaceholderPtr()
    {
        return tensor;
    }

    /**
     * @brief implicitly convertion to Expr
     * 
     * @return expr 
     */
    operator Expr()
    {
        return expr;
    }
};

inline ReadOnlyParam operator+(ReadOnlyParam p, uint32_t v)
{
    if (p.tensor)
    {
        PlaceholderPtr t = p.tensor;
        return t + v;
    }
    else if (p.expr)
    {
        return p.expr + v;
    }
    return p.value + v;
}
inline ReadOnlyParam operator+(uint32_t v, ReadOnlyParam p)
{
    return p + v;
}

inline ReadOnlyParam operator-(ReadOnlyParam p, uint32_t v)
{
    if (p.tensor)
    {
        PlaceholderPtr t = p.tensor;
        return t - v;
    }
    else if (p.expr)
    {
        return p.expr - v;
    }
    return p.value - v;
}

std::ostream &operator<<(std::ostream &os, const ReadOnlyParam &p);

// this function would complete the filename with different Architecture and Target (Z1_0904,...) and debug mode.
std::string complete_text_code_name(const std::string &base_name);
BuilderText get_text_content_by_filename(const std::string &filename);
void cache_text_content(const std::string &filename, BuilderText text);
} // namespace aipubt
#endif //!BUILDER_READ_ONLY_PARAM_H
