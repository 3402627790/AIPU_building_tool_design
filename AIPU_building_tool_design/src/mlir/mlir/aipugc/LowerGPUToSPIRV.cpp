#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::aipugc;

namespace {

struct SetSPIRVABIAttrsPass : public PassWrapper<SetSPIRVABIAttrsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetSPIRVABIAttrsPass)

  SetSPIRVABIAttrsPass() = default;
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    std::array<int32_t, 3> wg = {16, 1, 1};
    auto wgDenseAttr = DenseI32ArrayAttr::get(&getContext(), 
        llvm::ArrayRef<int32_t>(wg.data(), wg.size()));

    std::optional<int> subgroupSize = std::nullopt;
    std::optional<int> targetWidth = std::nullopt;

    auto entryPointAbiAttr = spirv::EntryPointABIAttr::get(&getContext(), 
        wgDenseAttr, subgroupSize, targetWidth);

    module.walk([&](gpu::GPUFuncOp gpuFunc) {
      gpuFunc->setAttr("spirv.entry_point_abi", entryPointAbiAttr);
    //   gpuFunc->setAttr("workgroup_size", wgDenseAttr);
    });
  }

  StringRef getArgument() const final { return "set-spirv-abi-attrs"; }
  StringRef getDescription() const final { 
    return "Set SPIR-V entry point ABI attributes on GPU functions"; 
  }
};

} // namespace

namespace mlir {
namespace aipugc {

std::unique_ptr<Pass> createSetSPIRVABIAttrsPass() {
  return std::make_unique<SetSPIRVABIAttrsPass>();
}

void registerSetSPIRVABIAttrsPass() {
  PassPipelineRegistration<>(
      "set-spirv-abi-attrs", 
      "Set SPIR-V entry point ABI attributes on GPU functions",
      [](OpPassManager &pm) { 
        pm.addPass(createSetSPIRVABIAttrsPass()); 
      });
}

} // namespace aipugc
} // namespace mlir