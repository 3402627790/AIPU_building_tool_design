#include "aipugc/Passes.h"
#include "aipugc/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"





#define DEBUG_TYPE "lowering-tosa-to-linalg"

using namespace mlir;
using namespace mlir::aipugc;

namespace {
struct TosaToLinalgConverter : public PassWrapper<TosaToLinalgConverter, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TosaToLinalgConverter)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect,
                    tensor::TensorDialect,
                    arith::ArithDialect,
                    math::MathDialect,
                    func::FuncDialect
                  >();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    
    // Set up conversion target
    ConversionTarget target(*ctx);
    target.addLegalDialect<linalg::LinalgDialect,
                          tensor::TensorDialect,
                          arith::ArithDialect,
                          math::MathDialect,
                          func::FuncDialect
                        >();
    target.addIllegalDialect<tosa::TosaDialect>();
    
    // Add necessary legal operations
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<func::FuncOp, func::ReturnOp>();

    // Create type converter
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    // Create pattern set
    RewritePatternSet patterns(ctx);
    
    // legacy MLIR requires TypeConverter parameter
    tosa::populateTosaToLinalgConversionPatterns(typeConverter, &patterns);

    // Apply full module conversion
    ModuleOp module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

// Register Pass
std::unique_ptr<Pass> mlir::aipugc::createConvertTosaToLinalg() {
  return std::make_unique<TosaToLinalgConverter>();
}