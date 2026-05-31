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
 * @Date: 2022-04-25 13:38:38
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-12 19:55:58
 */

#ifndef __QT_QINFO_H__
#define __QT_QINFO_H__

#include "quantization/qtransform.h"
#include "quantization_transform/common.h"
#include "quantization_transform/run_mode.h"

namespace qtlibn
{

// void update_qinfo_to_tensor(aipubt::TensorPtr tensor, QInfo_t &qinfo);
void _update(aipubt::NodePtr n, TensorQInfo_t *tqinfo);
void _update_subgraph(aipubt::NodePtr n, TensorQInfo_t *tqinfo);
void update_qinfo_to_tensor_in_graph(aipubt::GraphPtr graph, TensorQInfo_t *tensor_qinfo);
void update_qinfo_to_tensor_in_node(aipubt::NodePtr node, TensorQInfo_t *tensor_qinfo);
TensorQInfo_t get_tensor_qinfo(const std::string scheme, const int32_t bits, bool is_run_mode);
void update_graph_qinfo(aipubt::GraphPtr graph,
                        const std::string run_mode_or_scheme,
                        const int32_t bits,
                        bool is_run_mode);
void update_node_qinfo(aipubt::NodePtr node,
                       const std::string run_mode_or_scheme,
                       const int32_t bits,
                       bool is_run_mode);
void update_aipu_dtype(aipubt::NodePtr node);
bool check_node_neccessary_qinfo(aipubt::NodePtr node);
void set_default_qinfo_to_tensor(aipubt::TensorPtr t, const int32_t bits = 8);

} // namespace qtlibn

#endif
