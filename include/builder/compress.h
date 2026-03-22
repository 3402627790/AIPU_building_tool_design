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
 * @Author: Alex Zheng
 * @Date: 2023-05-29 14:23:41
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-05-29 14:23:41
 */
#include "builder/readonly_param.h"
#include "core/node.h"

#ifndef BUILDER_COMPRESS_H
#define BUILDER_COMPRESS_H
namespace aipubt
{
class Compress
{
public:
    Compress() {}
    ~Compress() {}
    bool compress_desc(std::vector<NodePtr> &refer_node,
                       NodePtr &node,
                       std::vector<ReadOnlyParam> &descriptor,
                       std::vector<ReadOnlyParam> &compress_data);
    void do_compress(std::vector<ReadOnlyParam> &desc,
                     std::vector<ReadOnlyParam> &ref_desc,
                     std::vector<ReadOnlyParam> &compress_data);

private:
    const float m_ratio = 0.4;
    const uint32_t m_min_range = 128;
};

} // namespace aipubt
#endif //!BUILDER_BUILDER_H