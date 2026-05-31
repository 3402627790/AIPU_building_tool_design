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
 * @Date: 2024-06-21 16:38:09 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-17 15:58:55
 */
#include "fp_ins.h"

namespace aipubt
{

static inline Xbyak::Xmm ymm_to_xmm(Xbyak::Ymm ymm)
{
    return Xbyak::Xmm(ymm.getIdx());
}

void pg_to_ymm(Xbyak::CodeGenerator &cg, Xbyak::Reg32 pg, Xbyak::Ymm ymm, Xbyak::Ymm ytmp, Xbyak::Reg64 tmp)
{
    using namespace Xbyak::util;

    static const YMMValue mask1 = {0x00, 0x00, 0x00, 0x00,
                                   0x00, 0x00, 0x00, 0x00,
                                   0x01, 0x01, 0x01, 0x01,
                                   0x01, 0x01, 0x01, 0x01,
                                   0x02, 0x02, 0x02, 0x02,
                                   0x02, 0x02, 0x02, 0x02,
                                   0x03, 0x03, 0x03, 0x03,
                                   0x03, 0x03, 0x03, 0x03};
    static const YMMValue mask2 = {0x01, 0x02, 0x04, 0x08,
                                   0x10, 0x20, 0x40, 0x80,
                                   0x01, 0x02, 0x04, 0x08,
                                   0x10, 0x20, 0x40, 0x80,
                                   0x01, 0x02, 0x04, 0x08,
                                   0x10, 0x20, 0x40, 0x80,
                                   0x01, 0x02, 0x04, 0x08,
                                   0x10, 0x20, 0x40, 0x80};
    auto xmm = Xbyak::Xmm(ymm.getIdx());
    cg.mov(tmp, (uint64_t)&mask1);
    cg.vmovd(xmm, pg.cvt32());
    cg.vpxor(ytmp, ytmp, ytmp);
    cg.vpbroadcastd(ymm, xmm);
    cg.vpshufb(ymm, ymm, ptr[tmp]);
    cg.mov(tmp, (uint64_t)&mask2);
    cg.vpandn(ymm, ymm, ptr[tmp]);
    cg.vpcmpeqb(ymm, ymm, ytmp);
}

void nan_propagation(Xbyak::CodeGenerator &cg, RegMap &reg_map, Xbyak::Ymm a, Xbyak::Ymm b, Xbyak::Ymm c)
{
    using namespace Xbyak::util;
    // the propagation strategy is to propagate the NaN value from a or b to c
    // and it will conver the x86 default nan into zhouyi nan
    // if a is nan then c=a|0x200
    // if b is nan then c=b|0x200
    static const YMMValue MASK[] = {
        {0, 0x7c, 0, 0x7c,
         0, 0x7c, 0, 0x7c,
         0, 0x7c, 0, 0x7c,
         0, 0x7c, 0, 0x7c,
         0, 0x7c, 0, 0x7c,
         0, 0x7c, 0, 0x7c,
         0, 0x7c, 0, 0x7c,
         0, 0x7c, 0, 0x7c},
        {0xff, 0x3, 0xff, 0x3,
         0xff, 0x3, 0xff, 0x3,
         0xff, 0x3, 0xff, 0x3,
         0xff, 0x3, 0xff, 0x3,
         0xff, 0x3, 0xff, 0x3,
         0xff, 0x3, 0xff, 0x3,
         0xff, 0x3, 0xff, 0x3,
         0xff, 0x3, 0xff, 0x3},
        {0, 0x2, 0, 0x2,
         0, 0x2, 0, 0x2,
         0, 0x2, 0, 0x2,
         0, 0x2, 0, 0x2,
         0, 0x2, 0, 0x2,
         0, 0x2, 0, 0x2,
         0, 0x2, 0, 0x2,
         0, 0x2, 0, 0x2},
        {0, 0xfe, 0, 0xfe, // x86_default nan
         0, 0xfe, 0, 0xfe,
         0, 0xfe, 0, 0xfe,
         0, 0xfe, 0, 0xfe,
         0, 0xfe, 0, 0xfe,
         0, 0xfe, 0, 0xfe,
         0, 0xfe, 0, 0xfe,
         0, 0xfe, 0, 0xfe},
        {0, 0x7e, 0, 0x7e, // ZY_default nan
         0, 0x7e, 0, 0x7e,
         0, 0x7e, 0, 0x7e,
         0, 0x7e, 0, 0x7e,
         0, 0x7e, 0, 0x7e,
         0, 0x7e, 0, 0x7e,
         0, 0x7e, 0, 0x7e,
         0, 0x7e, 0, 0x7e},

    };
    auto tmp = reg_map.temp_treg(cg);
    auto tmp2 = reg_map.temp_treg(cg);
    auto msk = reg_map.temp_reg32(cg).cvt64();
    cg.mov(msk, (uint64_t)&MASK);
    // convert x86 default nan into ZY default nan
    cg.vpcmpeqw(tmp, c, ptr[msk + 96]);
    cg.vpandn(c, tmp, c);
    cg.vpand(tmp, tmp, ptr[msk + 128]);
    cg.vpor(c, c, tmp);

    // if a is nan
    cg.vpand(tmp, a, ptr[msk]);       //msk= 0x7c00
    cg.vpcmpeqw(tmp, tmp, ptr[msk]);  // if inf or nan, tmp =0xffff
    cg.vpand(tmp2, a, ptr[msk + 32]); // msk = 0x03FF
    cg.vpand(tmp, tmp, tmp2);         //  tmp = frac bits
    cg.vpxor(tmp2, tmp2, tmp2);       // tmp2 = 0x0000
    cg.vpcmpgtw(tmp, tmp, tmp2);      // tmp =0xffff
    cg.vpor(tmp2, a, ptr[msk + 64]);  // msk = 0x200, tmp2=a|0x200
    cg.vpand(tmp2, tmp2, tmp);        // tmp2 is the nan value
    cg.vpandn(c, tmp, c);
    cg.vpor(c, c, tmp2);

    cg.vpandn(tmp, tmp, b); // tmp is the rest of b

    cg.vpand(tmp, tmp, ptr[msk]);     //msk= 0x7c00
    cg.vpcmpeqw(tmp, tmp, ptr[msk]);  // if inf or nan, tmp =0xffff
    cg.vpand(tmp2, b, ptr[msk + 32]); // msk = 0x01FF
    cg.vpand(tmp, tmp, tmp2);         //  tmp = frac bits
    cg.vpxor(tmp2, tmp2, tmp2);       // tmp2 = 0x0000
    cg.vpcmpgtw(tmp, tmp, tmp2);      // tmp =0xffff
    cg.vpor(tmp2, b, ptr[msk + 64]);  // msk = 0x200, tmp3=b|0x200
    cg.vpand(tmp2, tmp2, tmp);        // tmp2 is the nan value
    cg.vpandn(c, tmp, c);
    cg.vpor(c, c, tmp2);
    reg_map.release_reg(tmp);
    reg_map.release_reg(tmp2);
    reg_map.release_reg(msk);
}
void fp_vadd(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto td = reg_map.get_treg(ins.operands[0], cg);
    auto tn = reg_map.get_treg(ins.operands[1], cg);
    auto tm = reg_map.get_treg(ins.operands[2], cg);
    auto pg = ins.operands[3].value;
    if (pg == 7)
    {
        cg.vaddps(td, tn, tm);
    }
    else
    {
        auto temp = reg_map.temp_treg(cg);
        cg.vaddps(temp, tn, tm);
        apply_pg_mask(cg, reg_map, td, temp, pg);
        reg_map.release_reg(temp);
    }
}
void fp16_vadd(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx)
{
    using namespace Xbyak::util;
    auto td = reg_map.get_treg(ins.operands[0], cg);
    auto tn = reg_map.get_treg(ins.operands[1], cg);
    auto tm = reg_map.get_treg(ins.operands[2], cg);
    auto pg = ins.operands[3].value;

    auto temp = reg_map.temp_treg(cg);
    auto t0 = reg_map.temp_treg(cg);
    auto t1 = reg_map.temp_treg(cg);
    cg.vcvtph2ps(t0, ymm_to_xmm(tm));
    cg.vcvtph2ps(t1, ymm_to_xmm(tn));
    cg.vaddps(temp, t0, t1);
    cg.vcvtps2ph(ymm_to_xmm(temp), temp, 0);
    cg.vextracti128(ymm_to_xmm(t0), tm, 0x1);
    cg.vextracti128(ymm_to_xmm(t1), tn, 0x1);
    cg.vcvtph2ps(t0, ymm_to_xmm(t0));
    cg.vcvtph2ps(t1, ymm_to_xmm(t1));
    cg.vaddps(t0, t0, t1);
    cg.vcvtps2ph(ymm_to_xmm(t0), t0, 0);
    cg.vinserti128(temp, temp, ymm_to_xmm(t0), 1);

    reg_map.release_reg(t0);
    reg_map.release_reg(t1);
    nan_propagation(cg, reg_map, tn, tm, temp);
    if (pg == 7)
    {
        cg.vmovaps(td, temp);
    }
    else
    {
        apply_pg_mask(cg, reg_map, td, temp, pg, 2);
    }
    reg_map.release_reg(temp);
}
}; //namespace aipubt