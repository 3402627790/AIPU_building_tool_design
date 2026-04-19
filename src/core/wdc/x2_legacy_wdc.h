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
 * @Date: 2024-06-03 15:48:21
 * @Email: weibin.li@armchina.com
 */

#ifndef __X2_LEGACY_WDC_H__
#define __X2_LEGACY_WDC_H__

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <type_traits>
#include <utility>

#include "iiwdc.h"

_WDC_BEGIN_NAMESPACE_ZHOUYI

//WdcLegacy<int8_t> for int8 mode
//WdcLegacy<int16_t> for int16 mode

template <class _Tp, std::enable_if_t<std::is_same<_Tp, int8_t>::value ||
                                          std::is_same<_Tp, int16_t>::value,
                                      bool> = true>
class WdcLegacy : public IIWdc
{
    using elem_type = _Tp;

    struct head_t
    {
        uint32_t wgs;      // bytes of non-zero
        uint32_t wgt_size; // bytes of original weight
        uint32_t next_addr;
        uint32_t next_len;
        uint8_t wmb[]; // weight bit map; align 16B
        // int16_t compressed_wgt[]; //align 16B
    };

    static constexpr uint32_t k_head_size = sizeof(head_t);
    static constexpr uint32_t elem_shift = sizeof(elem_type) - 1;
    static constexpr uint32_t elem_count = 1024 >> elem_shift >> elem_shift;

protected:
    static std::pair<size_t, size_t>
    on_encode(const elem_type *rdata, uint32_t rlen, void *cdata,
              uint32_t clen, uint32_t offset, uint32_t **next_len, uint32_t actual_count)
    {
        actual_count = actual_count < rlen ? actual_count : rlen;

        auto head = (head_t *)cdata;
        head->wgs = 0;
        head->wgt_size = actual_count << elem_shift;

        uint8_t *wmb = head->wmb;
        const uint32_t wmb_size = ((actual_count >> 3) + 15) & (~15);

        elem_type *cw = (elem_type *)(wmb + wmb_size);
        for (auto i = 0u; i < actual_count; ++i)
        {
            if (rdata[i])
            {
                div_t rc = {(int)i >> 3, (int)i & 0x7};
                wmb[rc.quot] |= uint8_t(1 << rc.rem);
                cw[head->wgs++] = rdata[i];
            }
        }

        head->wgs <<= elem_shift;
        const uint32_t cw_size = (head->wgs + 15) & ~15;
        const uint32_t packet_size = k_head_size + wmb_size + cw_size;
        head->next_addr = packet_size + offset;

        **next_len = packet_size;
        *next_len = &(head->next_len);

        return {actual_count, packet_size};
    }

    static std::pair<size_t, size_t>
    on_decode(const void *cdata, uint32_t clen, elem_type *rdata,
              uint32_t rlen, uint32_t &next_len)
    {
        const auto head = (const head_t *)cdata;

        const uint8_t *wmb = head->wmb;
        const uint32_t wmb_size = ((head->wgt_size >> (elem_shift + 3)) + 15) & (~15);

        const elem_type *cw = (const elem_type *)(wmb + wmb_size);
        const auto cw_size = head->wgs >> elem_shift;

        for (auto wmb_index = 0u, cw_index = 0u; cw_index < cw_size; ++wmb_index)
        {
            if (!wmb[wmb_index])
                continue;

            const auto rdata_index = wmb_index << 3;
            for (auto i = __builtin_ctz(wmb[wmb_index]); i < 8; ++i)
            {
                if ((wmb[wmb_index] >> i) & 0b1)
                {
                    rdata[rdata_index + i] = cw[cw_index++];
                }
            }
        }
        next_len = head->next_len;
        return {head->next_addr, head->wgt_size >> elem_shift};
    }

    int decode_frame(const frame_t &, const void *, void *,
                     size_t, uint8_t &, uint32_t &) const override
    {
        return -ENOTSUP;
    }

public:
    int encode(const void *raw, size_t rlen, void *cdata,
               size_t clen, void *cxt) const override
    {
        memset(cdata, 0, clen);

        rlen >>= elem_shift;
        const elem_type *rdata = (const elem_type *)raw;

        const auto encode_byte = clen;
        uint32_t *next_len = (uint32_t *)cxt;
        uint64_t frame_info = *(uint64_t *)cxt;
        const uint32_t frame_l = frame_info >> 40;
        uint32_t frame_n = (frame_info >> 16) & 0xffffff;
        const uint32_t frame_m = (frame_info >> 2) & 0x3fff;
        const uint32_t tail_count = ((frame_info & 0b11) * 256) >> elem_shift;
        if (frame_n == 0 && frame_m == 0)
        {
            frame_n = 1;
        }
        uint32_t offset = 0;
        for (auto i = 0u; (i < frame_l) && (rlen > 0); ++i)
        {
            const auto rc = on_encode(rdata, rlen, cdata, clen, offset, &next_len, elem_count);
            rdata += rc.first;
            rlen -= rc.first;

            cdata = (uint8_t *)cdata + rc.second;
            clen -= rc.second;
            offset += rc.second;
        }
        while (rlen > 0 && clen > 0)
        {
            for (auto i = 0u; i < frame_n; ++i)
            {
                const auto rc = on_encode(rdata, rlen, cdata, clen, offset, &next_len, elem_count);
                rdata += rc.first;
                rlen -= rc.first;

                cdata = (uint8_t *)cdata + rc.second;
                clen -= rc.second;
                offset += rc.second;
            }
            if (tail_count)
            {
                for (auto i = 0u; i < frame_m; ++i)
                {
                    const auto rc = on_encode(rdata, rlen, cdata, clen, offset, &next_len, tail_count);
                    rdata += rc.first;
                    rlen -= rc.first;

                    cdata = (uint8_t *)cdata + rc.second;
                    clen -= rc.second;
                    offset += rc.second;
                }
            }
        }
        return encode_byte - clen;
    }

    int decode(const void *cdata, size_t clen, const void *cxt,
               void *raw, size_t rlen) const override
    {
        memset(raw, 0, rlen);

        elem_type *rdata = (elem_type *)raw;
        rlen >>= elem_shift;

        uint32_t offset = 0;
        uint32_t packet_size = *((uint32_t *)cxt);
        while (packet_size > 0)
        {
            const auto rc = on_decode((const uint8_t *)cdata + offset, clen,
                                      rdata, rlen, packet_size);
            offset = rc.first;
            rdata += rc.second;
            rlen -= rc.second;
        }
        assert(offset <= clen && rlen >= 0 && "overflow");
        return (uint8_t *)rdata - (uint8_t *)raw;
    }
};

_WDC_END_NAMESPACE_ZHOUYI

#endif //__X2_LEGACY_WDC_H__
