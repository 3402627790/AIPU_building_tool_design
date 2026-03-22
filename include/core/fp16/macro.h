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
 * @Date: 2023-03-09 12:16:08
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2025-10-17 17:40:54
 */
#ifndef DECLARE_BUILTIN_FUNCS
#define DECLARE_BUILTIN_FUNCS(FPFUNC1A, FPFUNC1b, FPFUNC2A, FPFUNC2b) \
    FPFUNC1A(abs);                                                    \
    FPFUNC1A(fabs);                                                   \
    FPFUNC1A(sin);                                                    \
    FPFUNC1A(cos);                                                    \
    FPFUNC1A(tan);                                                    \
    FPFUNC1A(asin);                                                   \
    FPFUNC1A(acos);                                                   \
    FPFUNC1A(atan);                                                   \
    FPFUNC1A(sinh);                                                   \
    FPFUNC1A(cosh);                                                   \
    FPFUNC1A(tanh);                                                   \
    FPFUNC1A(asinh);                                                  \
    FPFUNC1A(acosh);                                                  \
    FPFUNC1A(atanh);                                                  \
    FPFUNC1A(erf);                                                    \
    FPFUNC1A(erfc);                                                   \
    FPFUNC1A(exp);                                                    \
    FPFUNC1A(expm1);                                                  \
    FPFUNC1A(exp2);                                                   \
    FPFUNC1A(log);                                                    \
    FPFUNC1A(log10);                                                  \
    FPFUNC1A(log1p);                                                  \
    FPFUNC1A(log2);                                                   \
    FPFUNC1A(sqrt);                                                   \
    FPFUNC1A(cbrt);                                                   \
    FPFUNC2A(fmod);                                                   \
    FPFUNC2A(remainder);                                              \
    FPFUNC2A(fmax);                                                   \
    FPFUNC2A(fmin);                                                   \
    FPFUNC2A(fdim);                                                   \
    FPFUNC2A(hypot);                                                  \
    FPFUNC2A(pow);                                                    \
    FPFUNC1A(lgamma);                                                 \
    FPFUNC1A(tgamma);                                                 \
    FPFUNC1A(ceil);                                                   \
    FPFUNC1A(floor);                                                  \
    FPFUNC1A(trunc);                                                  \
    FPFUNC1A(round);                                                  \
    FPFUNC1A(lround);                                                 \
    FPFUNC1A(nearbyint);                                              \
    FPFUNC1A(rint);                                                   \
    FPFUNC1A(lrint);                                                  \
    FPFUNC1A(llround);                                                \
    FPFUNC1A(llrint);                                                 \
    FPFUNC1A(ilogb);                                                  \
    FPFUNC1A(logb);                                                   \
    FPFUNC1b(isfinite);                                               \
    FPFUNC1b(isinf);                                                  \
    FPFUNC1b(isnan);                                                  \
    FPFUNC1b(isnormal);                                               \
    FPFUNC1b(signbit);                                                \
    FPFUNC2A(atan2);                                                  \
    FPFUNC2b(isgreater);                                              \
    FPFUNC2b(isgreaterequal);                                         \
    FPFUNC2b(isless);                                                 \
    FPFUNC2b(islessequal);                                            \
    FPFUNC2b(islessgreater);                                          \
    FPFUNC2b(isunordered);
#endif // DECLARE_BUILTIN_FUNCS