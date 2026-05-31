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
 * @Date: 2020-05-17 12:09:56
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-03-03 14:34:47
 */

#ifndef __QTLIB_RNN_UTILS_H__
#define __QTLIB_RNN_UTILS_H__
#include "quantization_transform/common.h"
#include "quantization_transform/ops/base_linear.h"
#include "quantization_transform/utils/data_type.h"
#include <algorithm>
#include <cmath>
#include <core/core.h>
#include <core/node.h>

namespace qtlibn
{
static std::map<int8_t, std::map<aipubt::OpType, std::vector<float_t>>> g_rnn_sigmoid_clamp = {
    {8, {{aipubt::OpType_RNN, {-4.564205, 4.564205}}, {aipubt::OpType_GRUv1, {-4.564202, 4.564202}}, {aipubt::OpType_GRUv3, {-4.8, 4.8}}, {aipubt::OpType_BasicLSTM, {-4.564205, 4.564205}}}},
    {16, {{aipubt::OpType_RNN, {-10.0, 10.0}}, {aipubt::OpType_GRUv1, {-10.0, 10.0}}, {aipubt::OpType_GRUv3, {-10.0, 10.0}}, {aipubt::OpType_BasicLSTM, {-10.0, 10.0}}}}};

static std::map<int, std::map<aipubt::OpType, std::vector<float>>> g_rnn_tanh_clamp = {
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
 * @Date: 2020-05-17 12:09:56
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-03-03 14:34:47
 */

#ifndef __QTLIB_RNN_UTILS_H__
#define __QTLIB_RNN_UTILS_H__
#include "quantization_transform/common.h"
#include "quantization_transform/ops/base_linear.h"
#include "quantization_transform/utils/data_type.h"
#include <algorithm>
#include <cmath>
#include <core/core.h>
#include <core/node.h>

namespace qtlibn
{
static std::map<int8_t, std::map<aipubt::OpType, std::vector<float_t>>> g_rnn_sigmoid_clamp = {
    {8, {{aipubt::OpType_RNN, {-4.564205, 4.564205}}, {aipubt::OpType_GRUv1, {-4.564202, 4.564202}}, {aipubt::OpType_GRUv3, {-4.8, 4.8}}, {aipubt::OpType_BasicLSTM, {-4.564205, 4.564205}}}},
    {16, {{aipubt::OpType_RNN, {-10.0, 10.0}}, {aipubt::OpType_GRUv1, {-10.0, 10.0}}, {aipubt::OpType_GRUv3, {-10.0, 10.0}}, {aipubt::OpType_BasicLSTM, {-10.0, 10.0}}}}};

static std::map<int, std::map<aipubt::OpType, std::vector<float>>> g_rnn_tanh_clamp = {
