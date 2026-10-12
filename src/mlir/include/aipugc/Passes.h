//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------

//===- Passes.h - AIPU Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for AIPU.
//
//===----------------------------------------------------------------------===//

#ifndef AIPU_PASSES_H
#define AIPU_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir
{
class Pass;

namespace aipugc
{
    std::unique_ptr<Pass> createShapeInferencePass();
    std::unique_ptr<Pass> createDCEPass();
    void registerLowerToTOSAPass();
    std::unique_ptr<Pass> createLowerToTOSAPass();
    std::unique_ptr<Pass> createConvertTosaToLinalg();
    std::unique_ptr<Pass> createConvertAffineToLLVM(bool useBarePtrCallConv = false);
    std::unique_ptr<Pass> createKernelABIPass();
    std::unique_ptr<Pass> createSetSPIRVABIAttrsPass();
    void registerSetSPIRVABIAttrsPass();
    std::unique_ptr<Pass> createSimplifyGPULaunchPass();
    void registerSimplifyGPULaunchPass();
    void buildTestSPIRVCPURunnerPipeline(mlir::OpPassManager &passManager);
    void buildTestVulkanRunnerPipeline(mlir::OpPassManager &passManager);
} // namespace aipugc
} // namespace mlir

#endif // AIPU_PASSES_H
