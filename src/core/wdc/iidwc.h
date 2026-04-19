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
 * @Author: Weibin Li
 * @Date: 2024-05-17 10:15:08
 * @Email: weibin.li@armchina.com
 */

#ifndef __SUPER_WDC_H__
#define __SUPER_WDC_H__

#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#define __likely(x) __builtin_expect(!!(x), 1)
#define __unlikely(x) __builtin_expect(!!(x), 0)

#define _WDC_NAMESPACE_NAME zhouyi
#define _WDC_BEGIN_NAMESPACE_ZHOUYI \
    namespace _WDC_NAMESPACE_NAME   \
    {
#define _WDC_END_NAMESPACE_ZHOUYI }
#define _WDC_USING_NAMESPACE_ZHOUYI using namespace _WDC_NAMESPACE_NAME;

_WDC_BEGIN_NAMESPACE_ZHOUYI

class IIWdc
{
public:
    union frame_t
    {
        uint8_t u8[32];
        uint16_t u16[16];
        uint32_t u32[8];
        uint64_t u64[4];
    };

    static constexpr int k_frame_bits = sizeof(frame_t) * 8;
    virtual ~IIWdc() {}

    virtual int encode(const void *raw, size_t rlen, void *cdata,
                       size_t clen, void *cxt) const = 0;

    virtual int decode(const void *cdata, size_t clen, const void *cxt,
                       void *raw, size_t rlen) const
    {
        const auto total = rlen;
        const auto raw_end = (int8_t *)raw + rlen;
        const auto *frame = (const frame_t *)cdata;
        const auto *frame_end = (const frame_t *)((int8_t *)cdata + clen);

        uint8_t head = 0;
        uint32_t tail_bits = 0;

        while (raw < raw_end && frame < frame_end)
        {
            int n = decode_frame(*frame, cxt, raw, rlen, head, tail_bits);
            rlen -= n;
            raw = (int8_t *)raw + n;
            ++frame;
        }

        return total - rlen;
    }

protected:
    virtual int decode_frame(const frame_t &frame, const void *cxt, void *raw,
                             size_t len, uint8_t &, uint32_t &) const = 0;
};

template <class _T, std::enable_if_t<std::is_integral<_T>::value, bool> = true>
static _T sign_extend(_T value, uint8_t sign_pos)
{
    if ((value >> sign_pos) & 0x1)
    {
        const auto ext_bits = (sizeof(_T) << 3) - sign_pos;
        const auto sign_ext = ((1 << ext_bits) - 1) << sign_pos;
        return sign_ext | value;
    }
    return value;
}
_WDC_END_NAMESPACE_ZHOUYI

#endif //__SUPER_WDC_H__
