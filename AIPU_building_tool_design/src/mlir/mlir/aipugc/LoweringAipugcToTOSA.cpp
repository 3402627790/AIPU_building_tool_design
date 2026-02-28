#include "aipugc/Passes.h"
#include "aipugc/Dialect.h"
#include "aipugc/QuantUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-aipugc-to-tosa"

using namespace mlir;
using namespace mlir::aipugc;

namespace {

Value convertToF32(ConversionPatternRewriter &rewriter, Location loc, Value value) {
  Type valueType = value.getType();
  if (auto tensorType = dyn_cast<TensorType>(valueType)) {
    Type elementType = tensorType.getElementType();
    if (auto quantType = dyn_cast<QuantizedType>(elementType)) {
      Type newType = RankedTensorType::get(
        tensorType.getShape(),
        quantType.getExpressedType()
      );
      return rewriter.create<UnrealizedConversionCastOp>(loc, newType, value).getResult(0);
    }
  }
  return value;
}

Value convertToQuant(ConversionPatternRewriter &rewriter, Location loc, Value value, Type origType) {
  if (auto origTensorType = dyn_cast<TensorType>(origType)) {
    Type origElementType = origTensorType.getElementType();
    if (isa<QuantizedType>(origElementType)) {
      return rewriter.create<UnrealizedConversionCastOp>(loc, origType, value).getResult(0);
    }
  }
  return value;
}

class QuantizeOpLowering : public OpConversionPattern<aipugc::QuantizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::QuantizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value output = op.getOutput();
    rewriter.replaceOp(op, input);
    return success();
  }
};

class FuncOpLowering : public OpConversionPattern<aipugc::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto funcType = FunctionType::get(
      getContext(),
      op.getArgumentTypes(),
      op.getResultTypes()
    );
    llvm::SmallVector<NamedAttribute> attrs;

    auto func = rewriter.create<func::FuncOp>(
      op.getLoc(),
      op.getName(),
      funcType,
      attrs
    );

    rewriter.inlineRegionBefore(op.getBody(), func.getBody(), func.end());
    rewriter.eraseOp(op);

    return success();
  }
};

class AddOpLowering : public OpConversionPattern<aipugc::AddOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Type origResultType = op.getType();

    lhs = convertToF32(rewriter, loc, lhs);
    rhs = convertToF32(rewriter, loc, rhs);

    Type f32ResultType = origResultType;

    if (auto origTensorType = dyn_cast<TensorType>(origResultType)) {
      if (auto quantType = dyn_cast<QuantizedType>(origTensorType.getElementType())) {
        f32ResultType = RankedTensorType::get(
          origTensorType.getShape(),
          quantType.getExpressedType()
        );
      }
    }

    auto tosaAdd = rewriter.create<tosa::AddOp>(
      loc,
      f32ResultType,
      lhs,
      rhs
    );

    Value result = tosaAdd.getResult();
    result = convertToQuant(rewriter, loc, result, origResultType);
    rewriter.replaceOp(op, result);

    return success();
  }
};

class ReturnOpLowering : public OpConversionPattern<aipugc::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    rewriter.create<func::ReturnOp>(op.getLoc(), adaptor.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

class FunctionArgLowering : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type> newArgTypes;
    for (Type argType : op.getFunctionType().getInputs()) {
      if (auto tensorType = dyn_cast<TensorType>(argType)) {
        if (auto quantType = dyn_cast<QuantizedType>(tensorType.getElementType())) {
          newArgTypes.push_back(RankedTensorType::get(
            tensorType.getShape(),
            quantType.getExpressedType()));
          continue;
        }
      }
      newArgTypes.push_back(argType);
    }

    SmallVector<Type> newResultTypes;
    for (Type resultType : op.getFunctionType().getResults()) {
      if (auto tensorType = dyn_cast<TensorType>(resultType)) {
        if (auto quantType = dyn_cast<QuantizedType>(tensorType.getElementType())) {
          newResultTypes.push_back(RankedTensorType::get(
            tensorType.getShape(),
            quantType.getExpressedType())
          );
          continue;
        }
      }
      newResultTypes.push_back(resultType);
    }

    auto newFuncType = FunctionType::get(getContext(), newArgTypes, newResultTypes);

    auto newFunc = rewriter.create<func::FuncOp>(
        op.getLoc(),
        op.getName(),
        newFuncType,
        op->getAttrs()
    );

    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());

    Block &entryBlock = newFunc.getBody().front();
    for (unsigned i = 0; i < newArgTypes.size(); ++i) {
      if (i < entryBlock.getNumArguments()) {
        entryBlock.getArgument(i).setType(newArgTypes[i]);
      }
    }

    rewriter.eraseOp(op);
    return success();}
};

class ConstantOpLowering : public OpConversionPattern<aipugc::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    aipugc::ConstantOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    ElementsAttr valueAttr = op.getValue();
    Type resultType = op.getType();

    Value newConst;
    Type elementType = getElementTypeOrSelf(resultType);

    if (auto uniformQuantType = dyn_cast<UniformQuantizedType>(elementType)) {
      return handleUniformQuantization(rewriter, loc, op, uniformQuantType, resultType);
    }
    else if (auto perAxisQuantType = dyn_cast<UniformQuantizedPerAxisType>(elementType)) {
      return handlePerAxisQuantization(rewriter, loc, op, perAxisQuantType, resultType);
    }
    else {
      newConst = rewriter.create<tosa::ConstOp>(
        loc, resultType, valueAttr
      );
      rewriter.replaceOp(op, newConst);
      return success();
    }
  }

private:
  ArrayRef<int64_t> getTensorShape(Type type) const {
    if (auto tensorType = dyn_cast<TensorType>(type)) {
      return tensorType.getShape();
    }

    return {};
}

  LogicalResult handleUniformQuantization(ConversionPatternRewriter &rewriter,
                                        Location loc,
                                        ConstantOp op,
                                        UniformQuantizedType uniformQuantType,
                                        Type resultType) const {
    ElementsAttr valueAttr = op.getValue();
    double scale = uniformQuantType.getScale();
    int64_t zeroPoint = uniformQuantType.getZeroPoint();
    Type expressedType = uniformQuantType.getExpressedType();

    ArrayRef<int64_t> shape = getTensorShape(resultType);
    Type f32TensorType = RankedTensorType::get(shape, expressedType);

    if (auto denseIntAttr = dyn_cast<DenseIntElementsAttr>(valueAttr)) {
      SmallVector<APFloat> floatValues;
      for (APInt v : denseIntAttr.getValues<APInt>()) {
        double floatValue = (v.getSExtValue() + zeroPoint) / scale;
        floatValues.push_back(APFloat(floatValue));
      }

      auto newAttr = DenseFPElementsAttr::get(cast<RankedTensorType>(f32TensorType), floatValues);

      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, f32TensorType, newAttr);
      return success();
    }
    else {
      return op.emitError("Unsupported attribute type for uniform quantization");
    }
  }

  LogicalResult handlePerAxisQuantization(ConversionPatternRewriter &rewriter,
                                        Location loc, 
                                        ConstantOp op,
                                        UniformQuantizedPerAxisType perAxisQuantType,
                                        Type resultType) const {
    ElementsAttr valueAttr = op.getValue();
    ArrayRef<double> scales = perAxisQuantType.getScales();
    ArrayRef<int64_t> zeroPoints = perAxisQuantType.getZeroPoints();
    Type expressedType = perAxisQuantType.getExpressedType();
    int32_t quantizedDimension = perAxisQuantType.getQuantizedDimension();

    ArrayRef<int64_t> shape = getTensorShape(resultType);
    if (quantizedDimension < 0 || quantizedDimension >= static_cast<int32_t>(shape.size())) {
      return op.emitError("Invalid quantized dimension");
    }

    Type f32TensorType = RankedTensorType::get(shape, expressedType);
    auto type = cast<RankedTensorType>(f32TensorType);

    if (auto denseIntAttr = dyn_cast<DenseIntElementsAttr>(valueAttr)) {
      auto values = denseIntAttr.getValues<APInt>();
      int64_t step = 1;
      for (int i = shape.size() - 1; i > quantizedDimension; --i) {
        step *= shape[i];
      }

      int64_t numChannels = shape[quantizedDimension];
      SmallVector<APFloat> dequantizedValues;
      dequantizedValues.reserve(values.size());

      for (auto it = values.begin(), end = values.end(); it != end; ++it) {
        int64_t linearIndex = std::distance(values.begin(), it);
        int64_t channelIndex = (linearIndex / step) % numChannels;
        if (channelIndex >= numChannels) channelIndex = numChannels - 1;

        int64_t quantizedValue = (*it).getSExtValue();
        double scale = scales[channelIndex];
        int64_t zeroPoint = zeroPoints[channelIndex];

        double floatValue = (quantizedValue + zeroPoint) / scale;
        dequantizedValues.push_back(APFloat(floatValue));
      }

      auto newAttr = DenseFPElementsAttr::get(type, dequantizedValues);
      rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, f32TensorType, newAttr);

      return success();
    }
    else {
      return op.emitError("Unsupported attribute type for per-axis quantization");
    }
  }
};


// class AbsOpLowering : public OpConversionPattern<aipugc::ActivationOp> {
//   using OpConversionPattern<aipugc::ActivationOp>::OpConversionPattern;

//   LogicalResult
//   matchAndRewrite(aipugc::ActivationOp op, OpAdaptor adaptor,
//                   ConversionPatternRewriter &rewriter) const final {
//     rewriter.modifyOpInPlace(op, [&] { op->setOperands(adaptor.getOperands()); });
//     return success();
//   }
// };

class AbsOpLowering : public OpConversionPattern<aipugc::AbsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::AbsOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Type resultType = input.getType();

    auto tosaAbs = rewriter.create<tosa::AbsOp>(
      loc,
      resultType,
      input
    );

    rewriter.replaceOp(op, tosaAbs.getResult());
    return success();
  }
};

struct LowerToTOSAPass
    : public PassWrapper<LowerToTOSAPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToTOSAPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<tosa::TosaDialect>();
  }

  void runOnOperation() override {

    ModuleOp module = getOperation();
    if (!module) {
      signalPassFailure();
      return;
    }

    // Phase 1: Convert func and return ops
    {
      ConversionTarget phase1Target(getContext());
      RewritePatternSet phase1Patterns(&getContext());

      phase1Target.addLegalDialect<func::FuncDialect>();
      phase1Target.addLegalDialect<tosa::TosaDialect>();

      phase1Target.addLegalOp<ModuleOp>();
      phase1Target.addLegalOp<func::FuncOp>();
      phase1Target.addIllegalOp<aipugc::FuncOp, aipugc::ReturnOp>();
      phase1Target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

      phase1Patterns.add<FuncOpLowering>(&getContext());
      phase1Patterns.add<ReturnOpLowering>(&getContext());

      if (failed(applyPartialConversion(module, phase1Target, std::move(phase1Patterns)))) {
        LLVM_DEBUG({
          llvm::dbgs() << "\n=== After phase1 failed conversion ===\n";
          module.dump();
        });
        signalPassFailure();
        return;
      }
    }

    // Phase 2: Full conversion
    {
      ConversionTarget target(getContext());
      RewritePatternSet patterns(&getContext());

      target.addLegalDialect<tosa::TosaDialect>();
      target.addLegalDialect<func::FuncDialect>();
      target.addLegalOp<ModuleOp>();
      target.addLegalOp<func::FuncOp>();
      target.addLegalOp<func::ReturnOp>();
      target.addLegalOp<UnrealizedConversionCastOp>();
      target.addDynamicallyLegalOp<aipugc::AbsOp>([](aipugc::AbsOp op){return true;});

      target.addIllegalDialect<aipugc::AIPUDialect>();
    //   target.addDynamicallyLegalOp<aipugc::AbsOp>([](aipugc::AbsOp op) {
    //     return true;
    //   });

      target.addIllegalOp<aipugc::ConstantOp, aipugc::AddOp, aipugc::QuantizeOp, aipugc::AbsOp>();
      patterns.add<FunctionArgLowering>(&getContext());
      patterns.add<AddOpLowering>(&getContext());
      patterns.add<ConstantOpLowering>(&getContext());
      patterns.add<QuantizeOpLowering>(&getContext());
      patterns.add<AbsOpLowering>(&getContext());

      if (failed(applyFullConversion(module, target, std::move(patterns)))) {
        LLVM_DEBUG({
          llvm::dbgs() << "\n=== Failed to legalize operations ===\n";
          module.walk([&](Operation* op) {
            if (!target.isLegal(op)) {
              llvm::dbgs() << "  - Illegal op: " << op->getName() << " at " << op->getLoc() << "\n";
            }
          });
          llvm::dbgs() << "Remaining operations:\n";
          module.dump();
        });
        signalPassFailure();
        return;
      }
    }

    if (failed(mlir::verify(module))) {
      module.emitError("Module verification failed after conversion");
      signalPassFailure();
      return;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "\n=== After successful conversion ===\n";
      module.dump();
    });
  }
};

} // namespace

void mlir::aipugc::registerLowerToTOSAPass() {
  PassPipelineRegistration<>(
    "aipu-lower-tosa",
    "Lower AIPU dialect to TOSA dialect",
    [](OpPassManager &pm) {
      pm.addPass(createLowerToTOSAPass());
    }
  );
}

std::unique_ptr<Pass> mlir::aipugc::createLowerToTOSAPass() {
  return std::make_unique<LowerToTOSAPass>();
}