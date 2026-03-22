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
 * @Date: 2020-05-27 10:15:36 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-10-15 15:25:48
 */
#ifndef BUILDER_BLDTYPES_H
#define BUILDER_BLDTYPES_H
#include <cstddef>
#include <memory>
#include <vector>
namespace aipubt
{
struct ReadOnlyParam;
class BaseNode;
class Tensor;
class ComputeTensor;
using BaseNodePtr = std::shared_ptr<BaseNode>;
typedef std::shared_ptr<Tensor> TensorPtr;
typedef std::vector<TensorPtr> TensorList;

typedef std::vector<ReadOnlyParam> BuilderParams;
typedef std::pair<const char *, size_t> BuilderText;
typedef std::vector<BuilderText> BuilderTextList;

} //namespace aipubt
#endif //!BUILDER_BLDTYPES_H