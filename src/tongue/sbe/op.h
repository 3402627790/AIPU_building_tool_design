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
 * @Date: 2023-04-03 17:08:15 
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-08 11:45:20
 */
#pragma once
#include <set>

namespace tongue
{
namespace sbe
{
    enum class Dtype : int
    {
        none = 0,
        i8 = 1,
        u8 = 2,
        i16 = 3,
        u16 = 4,
        i32 = 5,
        u32 = 6,
        f32 = 7,
    };

    static std::set<Dtype> INT_DTYPE = {
        Dtype::i8,
        Dtype::i16,
        Dtype::i32,
    };

    static std::set<Dtype> UINT_DTYPE = {
        Dtype::u8,
        Dtype::u16,
        Dtype::u32,
    };

    inline bool is_scalar(Dtype d)
    {
        return d == Dtype::f32 || UINT_DTYPE.count(d) || INT_DTYPE.count(d);
    }
    inline bool is_integral(Dtype d)
    {
        return UINT_DTYPE.count(d) || INT_DTYPE.count(d);
    }
    inline bool is_none(Dtype d)
    {
        return d == Dtype::none;
    }

    inline std::ostream &operator<<(std::ostream &oss, Dtype d)
    {
        switch (d)
        {
        case Dtype::none:
            return oss << "none";
        case Dtype::i8:
            return oss << "i8";
        case Dtype::u8:
            return oss << "u8";
        case Dtype::i16:
            return oss << "i16";
        case Dtype::u16:
            return oss << "u16";
        case Dtype::i32:
            return oss << "i32";
        case Dtype::u32:
            return oss << "u32";
        case Dtype::f32:
            return oss << "f32";
        default:
            return oss << "[?]";
        }
    }

    enum class Op : int
    {
        constant = 0,
        add = 1,
        sub = 2,
        mul = 3,
        mulh = 4,
        div = 5,
        mod = 6,
        ret = 7,
        alloc = 8,
        // phi node syntax: %a = phi [(v0, %block0), (v1, %block1), ...]
        // phi must be the begin of a code block
        phi = 9,
        label = 10,
        call = 11,
        fun = 12,
        pop = 13,
        push = 14,
        cbz = 15,
        cbnz = 16,
        load = 17,
        store = 18,
        lshit = 19,
        rshit = 20,
        band = 21,
        bor = 22,
        bxor = 23,
        slt = 24,
        sle = 25,
        sgt = 26,
        sge = 27,
        seq = 28,
        sne = 29,
        jmp = 30, // relative jump
        mov = 31,
        itof = 32,
        ftoi = 33,
        max = 34,
        min = 35,
        rint = 36,
        log2 = 37,
        sqrt = 38,
        rsqrt = 39,
        keep_live = 40, // make a variable live

        aDMA = 41,  //asynchronous DMA
        sDMA = 42,  //synchronous DMA
        aAIFF = 43, //asynchronous AIFF
        sAIFF = 44, //synchronous AIFF
        sync = 45,  //synchronize, wait for flag
    };

    static std::set<Op> NO_DEPS_OP = {
        Op::ret,
        Op::push,
        Op::label,
        Op::fun,
        Op::store,
        Op::jmp,
        Op::cbz,
        Op::cbnz,
        Op::keep_live,
    };

    static std::set<Op> BRANCH_OP = {
        Op::ret,
        Op::call,
        Op::cbnz,
        Op::cbz,
        Op::jmp,
    };

    inline std::ostream &operator<<(std::ostream &oss, Op op)
    {
        switch (op)
        {
        case Op::constant:
            return oss << "constant";
        case Op::add:
            return oss << "add";
        case Op::sub:
            return oss << "sub";
        case Op::mul:
            return oss << "mul";
        case Op::mulh:
            return oss << "mulh";
        case Op::div:
            return oss << "div";
        case Op::mod:
            return oss << "mod";
        case Op::ret:
            return oss << "ret";
        case Op::alloc:
            return oss << "alloc";
        case Op::phi:
            return oss << "phi";
        case Op::label:
            return oss << "label";
        case Op::call:
            return oss << "call";
        case Op::fun:
            return oss << "fun";
        case Op::pop:
            return oss << "pop";
        case Op::push:
            return oss << "push";
        case Op::cbz:
            return oss << "cbz";
        case Op::cbnz:
            return oss << "cbnz";
        case Op::load:
            return oss << "load";
        case Op::store:
            return oss << "store";
        case Op::lshit:
            return oss << "lshit";
        case Op::rshit:
            return oss << "rshit";
        case Op::band:
            return oss << "band";
        case Op::bor:
            return oss << "bor";
        case Op::bxor:
            return oss << "bxor";
        case Op::slt:
            return oss << "slt";
        case Op::sle:
            return oss << "sle";
        case Op::sgt:
            return oss << "sgt";
        case Op::sge:
            return oss << "sge";
        case Op::seq:
            return oss << "seq";
        case Op::sne:
            return oss << "sne";
        case Op::jmp:
            return oss << "jmp";
        case Op::mov:
            return oss << "mov";
        case Op::itof:
            return oss << "itof";
        case Op::ftoi:
            return oss << "ftoi";
        case Op::max:
            return oss << "max";
        case Op::min:
            return oss << "min";
        case Op::rint:
            return oss << "rint";
        case Op::log2:
            return oss << "log2";
        case Op::sqrt:
            return oss << "sqrt";
        case Op::rsqrt:
            return oss << "rsqrt";
        case Op::keep_live:
            return oss << "keep_live";
        case Op::aDMA:
            return oss << "aDMA";
        case Op::sDMA:
            return oss << "sDMA";
        case Op::aAIFF:
            return oss << "aAIFF";
        case Op::sAIFF:
            return oss << "sAIFF";
        case Op::sync:
            return oss << "sync";
        default:
            return oss << "unkown";
        }
    }
} // namespace sbe
} // namespace tongue
