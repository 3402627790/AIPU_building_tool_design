#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

using namespace mlir;

// Defined in the test directory, no public header.
namespace mlir::aipugc {
std::unique_ptr<Pass> createTestConvertToSPIRVPass(bool convertGPUModules,
                                                   bool nestInGPUModule);
}

namespace mlir {
namespace aipugc {

void buildTestVulkanRunnerPipeline(OpPassManager &passManager) {
  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());

  GpuSPIRVAttachTargetOptions attachTargetOptions{};
  attachTargetOptions.spirvVersion = "v1.0";
  attachTargetOptions.spirvCapabilities.push_back("Shader");
  attachTargetOptions.spirvExtensions.push_back(
      "SPV_KHR_storage_buffer_storage_class");
  passManager.addPass(createGpuSPIRVAttachTarget(attachTargetOptions));

  passManager.addPass(aipugc::createTestConvertToSPIRVPass(
    /*convertGPUModules=*/true, /*nestInGPUModule=*/true));

  OpPassManager &spirvModulePM =
      passManager.nest<gpu::GPUModuleOp>().nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
  spirvModulePM.addPass(spirv::createSPIRVUpdateVCEPass());

  passManager.addPass(createGpuModuleToBinaryPass());

  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.nest<func::FuncOp>().addPass(LLVM::createLLVMRequestCWrappersPass());
  GpuToLLVMConversionPassOptions opt;
  opt.hostBarePtrCallConv = false;
  opt.kernelBarePtrCallConv = true;
  opt.kernelIntersperseSizeCallConv = true;
  passManager.addPass(createGpuToLLVMConversionPass(opt));
}

} // namespace aipugc
} // namespace mlir