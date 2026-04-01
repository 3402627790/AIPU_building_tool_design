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
 * @Date: 2022-05-13 14:36:30 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-05-13 15:52:29
 */

#include <core/basepass.h>
#include <iostream>

namespace aipubt
{
class AnonymityPass : public Pass
{
    std::string m_name;
    PassFunc m_func;

public:
    AnonymityPass(const std::string &name, PassFunc func) : m_name(name), m_func(func) {}
    virtual ~AnonymityPass() = default;
    const std::string &name() const override { return m_name; }
    PassType type() const override { return PassType_Common; }
    bool operator()(GraphPtr g) const override { return m_func(g); }
};

std::ostream &operator<<(std::ostream &os, PassPtr ptr)
{
    return os << ptr->name();
}

PassPtr Pass::create(const std::string &name, PassFunc func)
{
    return PassPtr(new AnonymityPass(name, func));
}
PassPtr Pass::create(PassFunc func)
{
    static uint32_t pass_id = 0;
    pass_id++;
    return PassPtr(new AnonymityPass(std::string("_AnonPass_") + std::to_string(pass_id), func));
}
// bool test_pass(GraphPtr  g)
// {
//     std::cout<<"testing pass\n";
//     return false;
// }

// class TestPass:public FuncPass<TestPass,test_pass>{};

} //namespace aipubt