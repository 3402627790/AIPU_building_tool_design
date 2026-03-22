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
 * @Date: 2020-01-29 14:24:30
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-01 10:38:07
 */
#ifndef GCOMPILER_API_H
#define GCOMPILER_API_H

#include "builder/builder.h"
#include "builder/linker/elfreader.h"
#include "builder/profiler.h"
#include "builder/readonly_param.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/graph.h"
#include "core/graph_matcher.h"
#include "core/logger.h"
#include "core/memory_object.h"
#include "core/node.h"
#include "core/ops/ops.h"
#include "core/optype.h"
#include "core/quantization.h"
#include "core/tensor.h"
#include "core/tensor_shape.h"
#include "quantization/qtransform.h"
#include "quantization/quantize_asym_to_sym.h"
#endif //!GCOMPILER_API_H