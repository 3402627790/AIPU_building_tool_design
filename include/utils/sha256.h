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
#ifndef UTILS_SHA256_H
#define UTILS_SHA256_H
#include <stdint.h>
#include <string>
#include <vector>
namespace utils
{
const std::string sha256_hex(const void *src, size_t n_bytes);
const std::vector<uint8_t> sha256_bytes(const void *src, size_t n_bytes);
void sha256_hex(const void *src, size_t n_bytes, char *dst_hex65);
void sha256_bytes(const void *src, size_t n_bytes, void *dst_bytes32);
} // namespace utils

#endif //!UTILS_SHA256_H