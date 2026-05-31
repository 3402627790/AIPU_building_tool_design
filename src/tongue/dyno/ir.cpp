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
 * @Date: 2023-04-03 16:46:56 
 * @Last Modified by:   Neo Wang 
 * @Last Modified time: 2023-04-03 16:46:56 
 */
#include "ir.h"

namespace tongue
{

namespace dyno
{
    const std::string &to_string(Operation op)
    {
        return ExprImpl::op_to_string(op);
    }
    std::ostream &operator<<(std::ostream &oss, LoopNode &ln)
    {
        return oss << ln.dump();
    }
    std::ostream &operator<<(std::ostream &oss, const Loop &ln)
    {
        return oss << "for " << ln.var.name() << " in " << ln.size << "|tail " << ln.tail;
    }
    std::ostream &operator<<(std::ostream &oss, Node &ln)
    {
        return oss << ln.dump();
    }

    static std::stringstream &print_indent(std::stringstream &oss, int indent)
    {
        while (indent > 0)
        {
            oss << " ";
            indent--;
        }
        return oss;
    }
    const std::string LoopNode::dump()
    {
        if (m_ptr)
        {
            std::stringstream oss;
            print_indent(oss, m_ptr->depth);
            if (m_ptr->is_loop)
            {
                oss << m_ptr->loop << "\n";
                for (auto c : m_ptr->children)
                {
                    oss << c;
                }
            }
            else
            {
                oss << m_ptr->node << "\n";
            }
            return oss.str();
        }
        return "";
    }

} // namespace dyno
} // namespace tongue