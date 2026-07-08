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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:23:23
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-11-15 10:06:27
 */
#pragma once
#include <builder/tools.h>
namespace aipubt
{
GraphPtr aipurun(GraphPtr g,
                 std::vector<TensorPtr> &input_tensors,
                 std::map<std::string, aipubt::Any> &arg_map);
GraphPtr aipugb(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map);

bool run_simulator(std::string dump_file, std::map<std::string, aipubt::Any> &arg_map);
bool aipudumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map);
bool cpsdumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map);
bool elfdumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map, GraphELFBuilder *builder = nullptr);
GraphBuilder *run_gb(GraphPtr g, std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map);
GraphPtr aipuestimator(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map);

} // namespace aipubt