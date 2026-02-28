#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace mlir {
namespace aipugc {

void buildTestSPIRVCPURunnerPipeline(OpPassManager &passManager) {
  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(createConvertGPUToSPIRVPass(/*mapMemorySpace=*/true));

  OpPassManager &nestedPM = passManager.nest<spirv::ModuleOp>();
  nestedPM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  nestedPM.addPass(spirv::createSPIRVUpdateVCEPass());
  passManager.addPass(createLowerHostCodeToLLVMPass());
  passManager.addPass(createConvertSPIRVToLLVMPass());
}

} // namespace aipugc
} // namespace mlir