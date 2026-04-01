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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
#ifndef INTERNAL_OPTYPE_H
#define INTERNAL_OPTYPE_H

namespace internal_ops
{
    enum OpType : int
    {
        OpType_Variance = 5120,
        OpType_WordRsqrt = 5121,
        OpType_NGammaQuant = 5122,
        OpType_Norm = 5123,
        OpType_NormGammaBeta = 5124,
        OpType_SquareSum = 5125,
        OpType_SurfaceSum = 5126,
        OpType_SurfaceSubSum = 5127,
        OpType_LayerSum = 5128,
        OpType_LayerMean = 5129,
        OpType_LayerSubMean = 5130,
        OpType_RMSRsqrt = 5131,
    };
};

#endif //! INTERNAL_OPTYPE_H
