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
 * @Date: 2024-12-02 13:55:52 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-02 14:14:47
 */
#pragma once
#include <cinttypes>
#include <string>
namespace tongue
{
namespace HAL
{
    struct HWConfig
    {
        uint32_t num_mtp = 2;
        uint32_t num_itp = 4;
        uint32_t mtp_mac_num = 2048;
        uint32_t wbuf_size = 50 * 1024;
        uint32_t actbuf_size = 64 * 1024;
        uint32_t unb_size = 256 * 1024;

        static const HWConfig get_conf(const std::string &arch)
        {
            return HWConfig();
        }
    };
} //namespace HAL
} //namespace tongue