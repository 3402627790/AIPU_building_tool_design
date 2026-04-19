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
 * @Date: 2024-04-09 10:15:03
 * @Email: weibin.li@armchina.com
 */

#ifndef __X3_WDC_H__
#define __X3_WDC_H__

#include <assert.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <type_traits>

#include "iiwdc.h"

_WDC_BEGIN_NAMESPACE_ZHOUYI

class WdcInt4M1; // for allone mode
class WdcInt4M2; // for 333331 mode
class WdcInt4M3; // for 772 mdoe
class WdcInt8M1; // for allone mode
class WdcInt8M2; // for 333331 mode

class IWdc : public IIWdc
{
protected:
    using value_t = uint8_t;
    using count_t = uint32_t;
    using rank_t = uint8_t;

    static constexpr int k_head_bits = 5;
    static constexpr int k_head_mask = (1 << k_head_bits) - 1;

    template <class _Tp, std::enable_if_t<std::is_integral<_Tp>::value, bool> = true>
    static constexpr int __countr_one(_Tp __t) noexcept
    {
        return __t != std::numeric_limits<_Tp>::max() ? __builtin_ctzl(static_cast<_Tp>(~__t)) : std::numeric_limits<_Tp>::digits;
    }

    class Lut
    {
    public:
        Lut(const value_t *raw, int len)
        {
            build_tab(raw, len);
        }

        rank_t rank(value_t value) const
        {
            return value_rank_tab_[do_cast(value)];
        };

        void get_rankings(value_t lut[16]) const
        {
            memcpy(lut, value_sorted_tab_.data(), 16);
        }

    protected:
        Lut() = default;
        void build_tab(const value_t *raw, int len)
        {
            std::array<count_t, 16> count_tab{0};
            for (int i = 0; i < len; ++i)
                count_tab[do_cast(raw[i])] += 1;

            std::multimap<count_t, value_t, std::greater<count_t>> cv_map;
            for (auto i = 0u; i < 16; ++i)
                cv_map.emplace(count_tab[i], i);

            auto it = cv_map.cbegin();
            for (auto i = 0u; i < 16; ++i, ++it)
            {
                value_sorted_tab_[i] = it->second;
                value_rank_tab_[it->second] = i;
            }
        }

        virtual value_t do_cast(const value_t &val) const { return val & 0xf; }

    private:
        std::array<value_t, 16> value_sorted_tab_{0};
        std::array<value_t, 16> value_rank_tab_{0};
    };
};

class WdcInt4M1 : public IWdc
{
public:
    int encode(const void *vraw, size_t rlen, void *cdata,
               size_t clen, void *cxt) const override
    {
        const value_t *raw = (const value_t *)vraw;
        value_t *lut = (value_t *)cxt;

        Lut tab(raw, rlen);
        tab.get_rankings(lut);

        frame_t frame = {{0}};
        int use_bits = k_head_bits;
        int encode_bytes = 0;
        auto save_frame = [&](int bytes = 32) {
            memcpy((uint8_t *)cdata + encode_bytes, frame.u8, bytes);
            memset(frame.u8, 0, bytes);
            encode_bytes += bytes;
            assert(encode_bytes <= (int)clen);
        };

        for (auto i = 0u; i < rlen; i += 2)
        {
            uint16_t result0;
            const int write_bits0 = encode_rank(tab.rank(raw[i]), result0);
            uint16_t result1;
            const int write_bits1 = encode_rank(tab.rank(raw[i + 1]), result1);

            const uint32_t result = ((uint32_t)result1 << write_bits0) | (uint32_t)result0;
            const int write_bits = write_bits0 + write_bits1;

            const div_t bit = {use_bits >> 5, use_bits & 0x1f};
            const auto idle_bits = 32 - bit.rem;
            if (__likely(idle_bits >= write_bits))
            {
                frame.u32[bit.quot] |= result << bit.rem;
                use_bits += write_bits;
            }
            else
            {
                uint8_t next_write_bits = write_bits - idle_bits;
                frame.u32[bit.quot] |= (result & ((1 << idle_bits) - 1)) << bit.rem;
                const uint32_t remain_result = result >> idle_bits;
                if (__likely(bit.quot < 7))
                {
                    frame.u32[bit.quot + 1] |= remain_result;
                    use_bits += write_bits;
                }
                else
                {
                    //update header
                    frame.u8[0] |= idle_bits; //(next_write_bits ? idle_bits : 0);
                    //save frame
                    save_frame();
                    //write tail
                    frame.u64[0] |= ((uint64_t)remain_result << k_head_bits);
                    use_bits = k_head_bits + next_write_bits;
                }
            }

            if (__unlikely(use_bits == k_frame_bits))
            {
                save_frame();
                use_bits = k_head_bits;
            }
        }

        if (use_bits > k_head_bits)
            save_frame();

        return encode_bytes;
    }

protected:
    int decode_frame(const frame_t &frame, const void *cxt, void *vraw, size_t len,
                     uint8_t &head, uint32_t &tail_bits) const override
    {
        value_t *raw = (value_t *)vraw;
        const value_t *lut = (const value_t *)cxt;

        const auto raw_start = raw;
        const auto raw_end = raw + len;

        uint64_t decode_word = (frame.u64[0] >> k_head_bits) << head | tail_bits;
        auto count_one = __countr_one(decode_word);
        *raw++ = sign_extend(lut[count_one], 3);
        decode_word >>= count_one + 1;
        int decode_bits = k_head_bits + count_one + 1;

        count_one = __countr_one(decode_word);
        *raw++ = sign_extend(lut[count_one], 3);
        decode_bits = (decode_bits + count_one + 1) - head;

        head = frame.u8[0] & k_head_mask;
        tail_bits = (uint64_t)frame.u32[7] >> (32 - head);

        const int frame_bits = k_frame_bits - head;
        while (decode_bits < frame_bits && raw < raw_end)
        {
            const div_t bit = {decode_bits >> 6, decode_bits & 0x3f};
            decode_word = frame.u64[bit.quot] >> bit.rem;
            count_one = __countr_one(decode_word);

            if (__unlikely(bit.rem + count_one == 64))
                count_one += __countr_one(frame.u64[bit.quot + 1]);

            *raw++ = sign_extend(lut[count_one], 3);
            decode_bits += count_one + 1;
        }

        return raw - raw_start;
    }

    virtual int encode_rank(rank_t rank, uint16_t &result) const
    {
        result = 0xffff >> (16 - rank);
        return rank + 1;
    }
};

class WdcInt4M2 : public WdcInt4M1
{
protected:
    int decode_frame(const frame_t &frame, const void *cxt, void *vraw, size_t len,
                     uint8_t &head, uint32_t &tail_bits) const override
    {
        value_t *raw = (value_t *)vraw;
        const value_t *lut = (const value_t *)cxt;

        const auto raw_start = raw;
        const auto raw_end = raw + len;

        auto decode_word = (frame.u64[0] >> k_head_bits) << head | tail_bits;
        auto count_one = __countr_one(decode_word);
        div_t rc = {count_one >> 1, count_one & 1};
        count_one -= rc.rem;
        decode_word >>= count_one;
        int rank = rc.quot * 3 + (decode_word & 0b11);
        decode_word >>= 2;
        *raw++ = sign_extend(lut[rank], 3);
        int decode_bits = k_head_bits + count_one + 2;

        count_one = __countr_one(decode_word);
        rc = {count_one >> 1, count_one & 1};
        count_one -= rc.rem;
        rank = rc.quot * 3 + ((decode_word >> count_one) & 0b11);
        *raw++ = sign_extend(lut[rank], 3);
        decode_bits = (decode_bits + count_one + 2) - head;

        head = frame.u8[0] & k_head_mask;
        tail_bits = (uint64_t)frame.u32[7] >> (32 - head);

        const int frame_bits = k_frame_bits - head;
        while (decode_bits < frame_bits && raw < raw_end)
        {
            const div_t bit = {decode_bits >> 6, decode_bits & 0x3f};
            decode_word = frame.u64[bit.quot] >> bit.rem;
            auto rshift = __countr_one(decode_word);
            count_one = rshift;

            const auto has_enc_bit = count_one & 1;
            auto lane_bits = bit.rem + count_one - has_enc_bit;
            if (__unlikely(lane_bits + 2 > 64))
            {
                const auto remain_bits = 64 - lane_bits;
                decode_word >>= rshift - has_enc_bit;
                decode_word |= frame.u64[bit.quot + 1] << remain_bits;
                rshift = __countr_one(decode_word);
                count_one += rshift - has_enc_bit;
            }

            rc = {count_one >> 1, count_one & 1};
            rank = rc.quot * 3 + ((decode_word >> (rshift - rc.rem)) & 0b11);
            *raw++ = sign_extend(lut[rank], 3);
            decode_bits += count_one - rc.rem + 2;
        }

        return raw - raw_start;
    }
    int encode_rank(rank_t rank, uint16_t &result) const override
    {
        const auto rc = std::div(rank, 3);
        int bits = 0;
        switch (rc.quot)
        {
        case 0:
            result = rc.rem;
            bits = 2;
            break;
        case 1:
            result = 0b0011 | rc.rem << 2;
            bits = 4;
            break;
        case 2:
            result = 0b001111 | rc.rem << 4;
            bits = 6;
            break;
        case 3:
            result = 0b00111111 | rc.rem << 6;
            bits = 8;
            break;
        case 4:
            result = 0b0011111111 | rc.rem << 8;
            bits = 10;
            break;
        case 5:
            result = 0b001111111111 | rc.rem << 10;
            bits = 12;
            break;
        default:
            assert(false);
            break;
        }
        return bits;
    }
};

class WdcInt4M3 : public WdcInt4M1
{
protected:
    int decode_frame(const frame_t &frame, const void *cxt, void *vraw, size_t len,
                     uint8_t &head, uint32_t &tail_bits) const override
    {
        value_t *raw = (value_t *)vraw;
        const value_t *lut = (const value_t *)cxt;

        const auto raw_start = raw;
        const auto raw_end = raw + len;

        auto decode_word = (frame.u64[0] >> k_head_bits) << head | tail_bits;
        auto count_one = __countr_one(decode_word);
        auto level = count_one * 0x56 >> 8;     //std::div(count_one, 3);
        auto level_bits = level + (level << 1); // <==> level * 3;
        decode_word >>= level_bits;
        int rank = level * 7 + (decode_word & 0b111);
        decode_word >>= 3;
        *raw++ = sign_extend(lut[rank], 3);
        int decode_bits = k_head_bits + (level + 1) * 3;

        count_one = __countr_one(decode_word);
        level = count_one * 0x56 >> 8;
        level_bits = level + (level << 1);
        decode_word >>= level_bits;
        rank = level * 7 + (decode_word & 0b111);
        *raw++ = sign_extend(lut[rank], 3);
        decode_bits = (decode_bits + (level + 1) * 3) - head;

        head = frame.u8[0] & k_head_mask;
        tail_bits = (uint64_t)frame.u32[7] >> (32 - head);

        const int frame_bits = k_frame_bits - head;
        while (decode_bits < frame_bits && raw < raw_end)
        {
            const div_t bit = {decode_bits >> 6, decode_bits & 0x3f};
            decode_word = frame.u64[bit.quot] >> bit.rem;
            count_one = __countr_one(decode_word);

            level = count_one * 0x56 >> 8; //std::div(count_one, 3);
            level_bits = level + (level << 1);
            decode_word >>= level_bits;

            const auto lane_bits = bit.rem + level_bits;
            if (__unlikely(lane_bits + 3 > 64))
            {
                const auto rem = count_one % 3;
                const auto remain_bits = 64 - lane_bits;
                decode_word |= frame.u64[bit.quot + 1] << remain_bits;
                const auto rshift = __countr_one(decode_word);
                decode_word >>= (rshift * 0x56 >> 8) * 3; //<=> rshift / 3 * 3;
                count_one += rshift - rem;
                level = count_one * 0x56 >> 8;
            }
            //!(level<<3)-level <==> level * 7
            rank = (level << 3) - level + (decode_word & 0b111);
            *raw++ = sign_extend(lut[rank], 3);
            decode_bits += (level + 1) * 3;
        }

        return raw - raw_start;
    }

    int encode_rank(rank_t rank, uint16_t &result) const override
    {
        int bits = 0;
        const auto rc = std::div(rank, 7);
        switch (rc.quot)
        {
        case 0:
            result = rc.rem;
            bits = 3;
            break;
        case 1:
            result = 0b000111 | rc.rem << 3;
            bits = 6;
            break;
        case 2:
            result = 0b000111111 | rc.rem << 6;
            bits = 9;
            break;
        default:
            assert(false);
            break;
        }
        return bits;
    }
};

class WdcInt8M1 : public IWdc
{
public:
    int encode(const void *vraw, size_t rlen, void *cdata,
               size_t clen, void *cxt) const override
    {
        const value_t *raw = (const value_t *)vraw;
        value_t *lut = (value_t *)cxt;

        LutI8 tab(raw, rlen);
        tab.get_rankings(lut);

        frame_t frame = {{0}};
        int use_bits = k_head_bits;
        int encode_bytes = 0;
        auto save_frame = [&](int bytes = 32) {
            memcpy((uint8_t *)cdata + encode_bytes, frame.u8, bytes);
            memset(frame.u8, 0, bytes);
            encode_bytes += bytes;
            assert(encode_bytes <= (int)clen);
        };

        for (auto i = 0u; i < rlen; ++i)
        {
            uint32_t result;
            const int write_bits = encode_rank(tab.rank(raw[i]), raw[i], result);
            const div_t bit = {use_bits >> 5, use_bits & 0x1f};
            const auto idle_bits = 32 - bit.rem;
            if (__likely(idle_bits >= write_bits))
            {
                frame.u32[bit.quot] |= result << bit.rem;
                use_bits += write_bits;
            }
            else
            {
                uint8_t next_write_bits = write_bits - idle_bits;
                frame.u32[bit.quot] |= (result & ((1 << idle_bits) - 1)) << bit.rem;
                const uint32_t remain_result = result >> idle_bits;
                if (__unlikely(bit.quot == 7))
                {
                    //update header
                    frame.u8[0] |= idle_bits;
                    //save frame
                    save_frame();
                    //write tail
                    frame.u64[0] |= remain_result << k_head_bits;
                    use_bits = k_head_bits + next_write_bits;
                }
                else
                {
                    frame.u32[bit.quot + 1] |= remain_result;
                    use_bits += write_bits;
                }
            }

            if (__unlikely(use_bits == k_frame_bits))
            {
                save_frame();
                use_bits = k_head_bits;
            }
        }

        if (use_bits > k_head_bits)
            save_frame();

        return encode_bytes;
    }

protected:
    int decode_frame(const frame_t &frame, const void *cxt, void *vraw, size_t len,
                     uint8_t &head, uint32_t &tail_bits) const override
    {
        value_t *raw = (value_t *)vraw;
        const value_t *lut = (const value_t *)cxt;

        const auto raw_start = raw;
        const auto raw_end = raw + len;

        auto decode_word = (frame.u64[0] >> k_head_bits) << head | tail_bits;
        uint8_t low = decode_word & 0xf;
        auto count_one = __countr_one(decode_word >> 4);
        *raw++ = lut[count_one] << 4 | low;
        int decode_bits = k_head_bits + count_one + 1 + 4 - head;

        head = frame.u8[0] & k_head_mask;
        tail_bits = (uint64_t)frame.u32[7] >> (32 - head);

        const int frame_bits = k_frame_bits - head;
        while (decode_bits < frame_bits && raw < raw_end)
        {
            const div_t bit = {decode_bits >> 6, decode_bits & 0x3f};
            decode_word = frame.u64[bit.quot] >> bit.rem;

            const auto need_bits = bit.rem + 4;
            if (__unlikely(need_bits >= 64))
                decode_word |= frame.u64[bit.quot + 1] << (64 - bit.rem);

            low = decode_word & 0xf;
            decode_word >>= 4;

            auto count_one = __countr_one(decode_word);
            if (__unlikely(need_bits + count_one == 64))
                count_one += __countr_one(frame.u64[bit.quot + 1]);

            *raw++ = lut[count_one] << 4 | low;
            decode_bits += 4 + count_one + 1;
        }

        return raw - raw_start;
    }

    struct LutI8 : IWdc::Lut
    {
        LutI8(const value_t *raw, int len) : Lut()
        {
            build_tab(raw, len);
        }

    protected:
        value_t do_cast(const value_t &val) const override { return val >> 4; }
    };

    virtual int encode_rank(rank_t rank, value_t value, uint32_t &result) const
    {
        result = (0xffff >> (16 - rank)) << 4 | (value & 0xf);
        return rank + 1 + 4;
    }
};

class WdcInt8M2 : public WdcInt8M1
{
protected:
    int decode_frame(const frame_t &frame, const void *cxt, void *vraw, size_t len,
                     uint8_t &head, uint32_t &tail_bits) const override
    {
        value_t *raw = (value_t *)vraw;
        const value_t *lut = (const value_t *)cxt;

        const auto raw_start = raw;
        const auto raw_end = raw + len;

        auto decode_word = (frame.u64[0] >> k_head_bits) << head | tail_bits;
        uint8_t low = decode_word & 0xf;
        decode_word >>= 4;

        auto count_one = __countr_one(decode_word);
        auto quot = count_one >> 1;
        decode_word >>= (count_one & (~1));
        int rank = quot + (quot << 1) + (decode_word & 0b11);
        *raw++ = lut[rank] << 4 | low;

        int decode_bits = k_head_bits + ((quot + 3) << 1) - head;

        head = frame.u8[0] & k_head_mask;
        tail_bits = (uint64_t)frame.u32[7] >> (32 - head);

        const int frame_bits = k_frame_bits - head;
        while (decode_bits < frame_bits && raw < raw_end)
        {
            const div_t bit = {decode_bits >> 6, decode_bits & 0x3f};
            decode_word = frame.u64[bit.quot] >> bit.rem;

            const auto need_bits = bit.rem + 4;
            if (__unlikely(need_bits >= 64))
                decode_word |= frame.u64[bit.quot + 1] << (64 - bit.rem);

            low = decode_word & 0xf;
            decode_word >>= 4;
            count_one = __countr_one(decode_word);

            if (__unlikely(need_bits + count_one >= 63))
            {
                decode_word |= frame.u64[bit.quot + 1] << (64 - need_bits);
                count_one = __countr_one(decode_word);
            }

            quot = count_one >> 1;
            decode_word >>= (count_one & (~1));
            rank = quot + (quot << 1) + (decode_word & 0b11);

            *raw++ = lut[rank] << 4 | low;
            decode_bits += (quot + 3) << 1; //<=>leve_bits(rc.quot*2)+enc_bits(2)+low_bits(4)
        }

        return raw - raw_start;
    }

    int encode_rank(rank_t rank, value_t value, uint32_t &result) const override
    {
        const div_t rc = std::div(rank, 3);
        int bits = 0;
        switch (rc.quot)
        {
        case 0:
            result = rc.rem << 4;
            bits = 6;
            break;
        case 1:
            result = 0b00110000 | rc.rem << 6;
            bits = 8;
            break;
        case 2:
            result = 0b0011110000 | rc.rem << 8;
            bits = 10;
            break;
        case 3:
            result = 0b001111110000 | rc.rem << 10;
            bits = 12;
            break;
        case 4:
            result = 0b00111111110000 | rc.rem << 12;
            bits = 14;
            break;
        case 5:
            result = 0b0011111111110000 | rc.rem << 14;
            bits = 16;
            break;
        default:
            assert(false);
            break;
        }
        result |= (value & 0xf);
        return bits;
    }
};

_WDC_END_NAMESPACE_ZHOUYI

#endif // !__X3_WDC_H__
