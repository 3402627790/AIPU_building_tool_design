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
 * @Date: 2026-04-09 14:23:41
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-09 14:23:41
 */
#ifndef BUILDER_IBUILDER_H
#define BUILDER_IBUILDER_H

#include "builder/cps_elf.h"
#include "core/graph.h"
#include <memory>

namespace aipubt
{

using GraphPtr = std::shared_ptr<Graph>;

class IBuilder
{
public:
    virtual ~IBuilder() = default;

    virtual bool build() = 0;
    virtual GraphPtr get_graph() = 0;
    virtual HybridELFPtr get_elf() = 0;
    virtual const std::string &get_header() const = 0;
};

} // namespace aipubt
#endif //!BUILDER_IBUILDER_H