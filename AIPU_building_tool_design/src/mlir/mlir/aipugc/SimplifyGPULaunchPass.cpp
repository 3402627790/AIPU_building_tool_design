
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {

struct SimplifyGPULaunchPass : public PassWrapper<SimplifyGPULaunchPass, 
                                                 OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyGPULaunchPass)

    SimplifyGPULaunchPass() = default;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<gpu::GPUDialect>();
        registry.insert<arith::ArithDialect>();
    }

    StringRef getArgument() const final { return "simplify-gpu-launch"; }
    StringRef getDescription() const final { 
        return "Simplify GPU launch by removing unnecessary index parameters"; 
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        llvm::StringMap<gpu::GPUFuncOp> gpuFuncMap;
        module.walk([&](gpu::GPUFuncOp gpuFunc) {
            gpuFuncMap[gpuFunc.getName()] = gpuFunc;
        });

        module.walk([&](gpu::LaunchFuncOp launchOp) {
            auto kernelNameAttr = launchOp.getKernelName();
            if (!kernelNameAttr) return;

            StringRef kernelName = kernelNameAttr.getValue();
            auto it = gpuFuncMap.find(kernelName);
            if (it == gpuFuncMap.end()) return;

            gpu::GPUFuncOp gpuFunc = it->second;

            auto operands = launchOp.getKernelOperands();
            if (operands.size() < 2) return;

            Value firstOp = operands[0];
            Value secondOp = operands[1];

            APInt firstValue, secondValue;
            if (matchPattern(firstOp, m_ConstantInt(&firstValue)) &&
                matchPattern(secondOp, m_ConstantInt(&secondValue))) {
                if (firstValue == 0 && secondValue == 0) {
                    SmallVector<Value> newOperands;
                    for (unsigned i = 2; i < operands.size(); ++i) {
                        newOperands.push_back(operands[i]);
                    }

                    launchOp.getKernelOperandsMutable().assign(newOperands);
                }
            }
        });

        module.walk([&](gpu::GPUFuncOp gpuFunc) {
            if (gpuFunc.getNumArguments() >= 2) {
                if (gpuFunc.getArgument(0).getType().isIndex() && 
                    gpuFunc.getArgument(1).getType().isIndex()) {

                    OpBuilder builder(gpuFunc.getBody());
                    Value zero = builder.create<arith::ConstantIndexOp>(
                        gpuFunc.getLoc(), 0);

                    gpuFunc.getArgument(0).replaceAllUsesWith(zero);
                    gpuFunc.getArgument(1).replaceAllUsesWith(zero);

                    BitVector indicesToErase(gpuFunc.getNumArguments());
                    indicesToErase.set(0);
                    indicesToErase.set(1);

                    LogicalResult result = gpuFunc.eraseArguments(indicesToErase);
                    (void)result;
                }
            }
        });
    }
};

} // namespace

namespace mlir {
namespace aipugc {

std::unique_ptr<Pass> createSimplifyGPULaunchPass() {
    return std::make_unique<SimplifyGPULaunchPass>();
}

void registerSimplifyGPULaunchPass() {
    PassPipelineRegistration<>(
        "simplify-gpu-launch", 
        "Simplify GPU launch by removing unnecessary index parameters",
        [](OpPassManager &pm) { 
            pm.addPass(createSimplifyGPULaunchPass()); 
        });
}

} // namespace aipugc
} // namespace mlir