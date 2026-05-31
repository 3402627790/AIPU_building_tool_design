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
 * @Date: 2024-11-13 11:58:41 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-02 14:49:15
 */
#pragma once

namespace ffc
{

/**
 * @brief Design of uOp
 * uop is micro op in term of NN, which is a small operation compared with NN operation
 * but it is a Large operation comparing to instruction
 * 
 * uOp is designed for NPU with fix function core(FF).
 * we suppose FF has NO register base operation.
 * So all uOps operate on memory, therefore there is no Load/Store operation.
 * 
 * 
 */
enum class uOp
{
    None = 0,
    Argument = 1,
    MemCpy,
    MemSet,
    MatMul,
    FC,
    Conv,
    DwConv,
    DeConv,
    Add,
    Sub,
    Mul,
    RShift,
    Lut,
    Pool,
    Resize,
    ReduceMax,
    ReduceSum,

    //internal op
    Slice,
    StrideSlice,
    LayoutCvt,
    Cast,

    //external func call

};

} //namespace ffc