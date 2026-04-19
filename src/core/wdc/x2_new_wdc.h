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
 * @Date: 2024-05-17 17:05:29
 * @Email: weibin.li@armchina.com
 */

#ifndef __X2_NEW_WDC_H__
#define __X2_NEW_WDC_H__

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "iiwdc.h"

_WDC_BEGIN_NAMESPACE_ZHOUYI

class WdcI8ToNBit;  // int8 for 5bit or 6bit mode
class WdcI16ToNBit; // int16 for 10bit or 12bit mode

template <class _Tp, std::enable_if_t<std::is_same<_Tp, int8_t>::value |
                                          std::is_same<_Tp, int16_t>::value,
                                      bool> = true>
class WdcNBitImpl
{
    uint8_t compress_bits_ = 0;
    uint8_t compress_sign_bit_ = 0;
    _Tp compress_mask_ = 0;
    _Tp pos_switch_value_ = 0;
    _Tp neg_switch_value_ = 0;

    uint8_t uncompress_bits_ = 0;
    uint8_t uncompress_sign_bit_ = 0;

    uint8_t uncompress_vbits_ = 0;
    uint64_t uncompress_vmask_ = 0;

    uint64_t uncompress_mask0_ = 0;
    uint64_t uncompress_mask1_ = 0;

    int encode_value(_Tp value, uint64_t &result) const
    {
        if (value <= neg_switch_value_ || value >= pos_switch_value_)
        {
            result = value;
            result &= uncompress_mask0_; // or uncompress_mask1_
            result <<= compress_bits_;
            result |= pos_switch_value_;

            return compress_bits_ + uncompress_bits_;
        }
        result = (value & compress_mask_);
        return compress_bits_;
    }

public:
    using frame_t = IIWdc::frame_t;
    static constexpr int k_frame_bits = IIWdc::k_frame_bits;

    void calc_context(uint8_t range)
    {
        if (range == compress_bits_)
            return;

        compress_bits_ = range;
        compress_sign_bit_ = range - 1;
        compress_mask_ = (1 << range) - 1;
        pos_switch_value_ = 1 << (range - 1);
        neg_switch_value_ = -pos_switch_value_;

        uncompress_bits_ = range << 1;
        uncompress_vbits_ = uncompress_bits_ - 2;
        uncompress_sign_bit_ = uncompress_vbits_ - 1;
        uncompress_vmask_ = (1 << uncompress_vbits_) - 1;

        uncompress_mask0_ = (1 << (uncompress_bits_ - 2)) - 1;
        uncompress_mask1_ = (1 << uncompress_bits_) - 1;
    }

    int decode_frame(const frame_t &frame, _Tp *raw, int len,
                     uint8_t &head, uint32_t &tail_bits) const
    {
        assert(compress_bits_ && "pls invoke calc_context to initializing!");

        const auto *raw_end = raw + len;
        const auto *raw_start = raw;

        int decoded_bits = compress_bits_ - head;
        uint64_t decode_word = (uint64_t)frame.u32[0] << head | tail_bits;

        head = 0, tail_bits = 0;

        _Tp value = sign_extend<_Tp>(decode_word & compress_mask_, compress_sign_bit_);
        if (value == neg_switch_value_)
        {
            value = sign_extend<_Tp>(
                (decode_word >> compress_bits_) & uncompress_vmask_,
                uncompress_sign_bit_);
            decoded_bits += uncompress_bits_;
        }
        *raw++ = value;

        while (raw < raw_end && decoded_bits < k_frame_bits)
        {
            div_t rc = {decoded_bits >> 6, decoded_bits & 0x3f};
            decode_word = frame.u64[rc.quot] >> rc.rem;

            auto lane_bits = 64 - rc.rem;
            if (lane_bits < compress_bits_)
            {
                if (rc.quot == 3)
                {
                    head = lane_bits;
                    tail_bits = decode_word;
                    return raw - raw_start;
                }
                decode_word = (uint64_t)frame.u64[rc.quot + 1] << lane_bits | decode_word;
            }

            value = sign_extend<_Tp>(decode_word & compress_mask_, compress_sign_bit_);
            decoded_bits += compress_bits_;

            if (value == neg_switch_value_)
            {
                if (decoded_bits + uncompress_bits_ > k_frame_bits)
                {
                    head = k_frame_bits - (decoded_bits - compress_bits_);
                    tail_bits = decode_word;
                    return raw - raw_start;
                }

                rc = {decoded_bits >> 6, decoded_bits & 0x3f};
                decode_word = frame.u64[rc.quot] >> rc.rem;

                lane_bits = 64 - rc.rem;
                if (lane_bits < uncompress_bits_)
                {
                    decode_word = (uint64_t)frame.u64[rc.quot + 1] << lane_bits | decode_word;
                }

                value = sign_extend<_Tp>(decode_word & uncompress_vmask_, uncompress_sign_bit_);
                decoded_bits += uncompress_bits_;
            }

            *raw++ = value;
        }
        return raw - raw_start;
    }

    int on_encode(const _Tp *raw, size_t rlen, uint8_t *cdata, size_t clen) const
    {
        int encode_bytes = 0;
        int use_bits = 0;
        frame_t frame = {{0}};

        auto save_frame = [&](int bytes = 32) {
            memcpy(cdata + encode_bytes, frame.u8, bytes);
            memset(frame.u8, 0, bytes);
            encode_bytes += bytes;
            assert(encode_bytes <= (int)clen);
        };

        for (auto i = 0u; i < rlen; ++i)
        {
            uint64_t result;
            const int write_bits = encode_value(raw[i], result);
            const div_t bit = {use_bits >> 6, use_bits & 0x3f};
            const auto idle_bits = 64 - bit.rem;
            if (__likely(idle_bits >= write_bits))
            {
                frame.u64[bit.quot] |= result << bit.rem;
                use_bits += write_bits;
            }
            else
            {
                uint8_t next_write_bits = write_bits - idle_bits;
                frame.u64[bit.quot] |= ((result & ((1ul << idle_bits) - 1ul)) << bit.rem);

                const auto remain_result = result >> idle_bits;
                if (__likely(bit.quot < 3))
                {
                    frame.u64[bit.quot + 1] |= remain_result;
                    use_bits += write_bits;
                }
                else
                {
                    save_frame();

                    frame.u64[0] |= remain_result;
                    use_bits = next_write_bits;
                }
            }

            if (__unlikely(use_bits == k_frame_bits))
            {
                save_frame();
                use_bits = 0;
            }
        }

        if (use_bits != 0)
            save_frame();

        return encode_bytes;
    }
};

class WdcI8ToNBit : public IIWdc
{
private:
    using elem_type = int8_t;
    using impl_type = WdcNBitImpl<elem_type>;
    impl_type impl;

protected:
    int decode_frame(const frame_t &frame, const void *, void *raw,
                     size_t len, uint8_t &head, uint32_t &tail_bits) const override
    {
        return impl.decode_frame(frame, (elem_type *)raw, len, head, tail_bits);
    }

public:
    int encode(const void *raw, size_t raw_len, uint8_t range, void *cdata, size_t clen) const
    {
        const_cast<impl_type &>(impl).calc_context(range);
        return impl.on_encode((const elem_type *)raw, raw_len, (uint8_t *)cdata, clen);
    }

    int encode(const void *vraw, size_t raw_len, void *cdata,
               size_t clen, void *cxt) const override
    {
        auto &range = *((uint8_t *)cxt);
        const elem_type *raw = (const elem_type *)vraw;
        enum
        {
            five_bits,
            six_bits,
            k_max_bits
        };
        int count[k_max_bits + 1] = {0};
        for (auto i = 0u; i < raw_len; ++i)
        {
            if (raw[i] > -16 && raw[i] < 16)
                count[five_bits] += 1;
            else if (raw[i] > -32 && raw[i] < 32)
                count[six_bits] += 1;
            else
                count[k_max_bits] += 1;
        }
        const auto five_bits_compress = raw_len * 5 + (raw_len - count[five_bits]) * 10;
        const auto six_bits_compress = raw_len * 6 + count[k_max_bits] * 12;
        range = five_bits_compress < six_bits_compress ? 5 : 6;
        return encode(raw, raw_len, range, cdata, clen);
    }

    int decode(const void *cdata, size_t clen, const void *cxt,
               void *raw, size_t raw_len) const override
    {
        const auto &range = *((const uint8_t *)cxt);
        const_cast<impl_type &>(impl).calc_context(range);
        return IIWdc::decode(cdata, clen, cxt, raw, raw_len);
    }
};

class WdcI16ToNBit : public IIWdc
{
private:
    using elem_type = int16_t;
    using impl_type = WdcNBitImpl<elem_type>;
    impl_type impl;

protected:
    int decode_frame(const frame_t &frame, const void *, void *raw,
                     size_t len, uint8_t &head, uint32_t &tail_bits) const override
    {
        return impl.decode_frame(frame, (elem_type *)raw, len / 2, head, tail_bits) << 1;
    }

public:
    int encode(const void *raw, size_t raw_len, uint8_t range,
               void *cdata, size_t clen) const
    {
        const_cast<impl_type &>(impl).calc_context(range);
        return impl.on_encode((const elem_type *)raw, raw_len / 2, (uint8_t *)cdata, clen);
    }

    int encode(const void *raw, size_t raw_len, void *cdata,
               size_t clen, void *cxt) const override
    {
        auto &range = *((uint8_t *)cxt);
        enum
        {
            ten_bits,
            twelve_bits,
            k_max_bits
        };
        const elem_type *p = (const elem_type *)raw;
        const int len = raw_len / sizeof(elem_type);
        int count[k_max_bits + 1] = {0};
        for (int i = 0; i < len; ++i)
        {
            if (p[i] > -512 && p[i] < 512)
                count[ten_bits] += 1;
            else if (p[i] > -2048 && p[i] < 2048)
                count[twelve_bits] += 1;
            else
                count[k_max_bits] += 1;
        }
        const auto ten_bits_compress = len * 10 + (len - count[ten_bits]) * 20;
        const auto twelve_bits_compress = len * 12 + count[k_max_bits] * 24;
        range = ten_bits_compress < twelve_bits_compress ? 10 : 12;

        return encode(raw, raw_len, range, cdata, clen);
    }

    int decode(const void *cdata, size_t clen, const void *cxt,
               void *raw, size_t rlen) const override
    {
        const auto &range = *((const uint8_t *)cxt);
        const_cast<impl_type &>(impl).calc_context(range);
        return IIWdc::decode(cdata, clen, cxt, raw, rlen);
    }
};

_WDC_END_NAMESPACE_ZHOUYI

#endif //!__X2_NEW_WDC_H__
