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
#ifndef BUILDER_MQ_IR_CACHE_H
#define BUILDER_MQ_IR_CACHE_H

#include "core/graph.h"

#include <string>
#include <vector>

namespace aipubt
{

GraphPtr load_mq_ir_cache(const std::string &dir);
bool load_mq_ir_cache_attributes(GraphPtr graph, const std::string &dir);
void save_mq_ir_cache(GraphPtr graph, const std::string &dir);
void save_mq_ir_cache(GraphPtr graph, const std::string &dir, const std::string &stage);
std::string load_mq_ir_cache_stage(const std::string &dir);
bool save_mq_ir_cache_schedule(const std::vector<std::vector<NodePtr>> &levels, const std::string &dir);
bool load_mq_ir_cache_schedule(GraphPtr graph, const std::string &dir, std::vector<std::vector<NodePtr>> &levels);

} // namespace aipubt

#endif //!BUILDER_MQ_IR_CACHE_H
