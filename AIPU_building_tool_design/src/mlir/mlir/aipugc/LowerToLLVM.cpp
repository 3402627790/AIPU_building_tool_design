#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "aipugc/QuantUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-to-llvm"

using namespace mlir;
using namespace mlir::aipugc;

namespace {

class AbsOpLowering : public OpConversionPattern<aipugc::AbsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(aipugc::AbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Type resultType = input.getType();

    auto tosaAbs = rewriter.create<tosa::AbsOp>(loc, resultType, input);

    rewriter.replaceOp(op, tosaAbs.getResult());
    return success();
  }
};

struct LowerToLLVMPass
    : public PassWrapper<LowerToLLVMPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace mlir {
namespace aipugc {

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMPass>();
}
void registerLowerToLLVMPass() {
  PassPipelineRegistration<>(
      "tosa-lower-llvm", "Lower TOSA dialect to LLVM dialect",
      [](OpPassManager &pm) { pm.addPass(createLowerToLLVMPass()); });
}

} // namespace aipugc
} // namespace mlir
