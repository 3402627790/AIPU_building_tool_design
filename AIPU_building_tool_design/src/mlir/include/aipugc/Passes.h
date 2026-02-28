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
    std::unique_ptr<Pass> createConvertAffineToLLVM();
    std::unique_ptr<Pass> createSetSPIRVABIAttrsPass();
    void registerSetSPIRVABIAttrsPass();
    std::unique_ptr<Pass> createSimplifyGPULaunchPass();
    void registerSimplifyGPULaunchPass();
    void buildTestSPIRVCPURunnerPipeline(mlir::OpPassManager &passManager);
    void buildTestVulkanRunnerPipeline(mlir::OpPassManager &passManager);
} // namespace aipugc
} // namespace mlir

#endif // AIPU_PASSES_H
