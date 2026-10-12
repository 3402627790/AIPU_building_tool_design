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

#include "aipugc/Passes.h"
#include "aipugc/Dialect.h"
#include "aipugc/QuantUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lowering-aipugc-to-tosa"

using namespace mlir;
using namespace mlir::aipugc;

namespace {

class QuantizeOpLowering : public OpConversionPattern<aipugc::QuantizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::QuantizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
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

    Type resultType = getTypeConverter()->convertType(op.getType());

    auto tosaAdd = rewriter.create<tosa::AddOp>(
      loc,
      resultType,
      lhs,
      rhs
    );

    Value result = tosaAdd.getResult();
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

class EltwiseOpLowering : public OpConversionPattern<aipugc::EltwiseOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::EltwiseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    if (op.getMethod() != "ADD")
      return failure();

    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Type resultType = getTypeConverter()->convertType(op.getType());

    auto tosaAdd = rewriter.create<tosa::AddOp>(
      loc,
      resultType,
      lhs,
      rhs
    );

    rewriter.replaceOp(op, tosaAdd.getResult());
    return success();
  }
};

class ActivationOpLowering : public OpConversionPattern<aipugc::ActivationOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      aipugc::ActivationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    if (op.getMethod() != "RELU")
      return failure();

    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    Type resultType = getTypeConverter()->convertType(op.getType());
    Type elemType = getElementTypeOrSelf(resultType);

    auto f32Type = cast<FloatType>(elemType);
    auto minVal = rewriter.getFloatAttr(f32Type, 0.0);
    auto maxVal = rewriter.getFloatAttr(
        f32Type, APFloat::getLargest(f32Type.getFloatSemantics()));

    rewriter.replaceOpWithNewOp<tosa::ClampOp>(
        op, resultType, input, minVal, maxVal);

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

    // Phase 2: Full conversion with a TypeConverter that maps quantized tensor
    // types to their expressed (f32) tensor types.
    {
      TypeConverter typeConverter;
      typeConverter.addConversion([](Type type) {
        if (auto tt = dyn_cast<TensorType>(type)) {
          if (auto quantType = dyn_cast<QuantizedType>(tt.getElementType())) {
            return std::optional<Type>(
                RankedTensorType::get(tt.getShape(), quantType.getExpressedType()));
          }
        }
        return std::optional<Type>(type);
      });
      typeConverter.addSourceMaterialization(
          [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
                .getResult(0);
          });
      typeConverter.addTargetMaterialization(
          [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
                .getResult(0);
          });

      ConversionTarget target(getContext());
      RewritePatternSet patterns(&getContext());

      target.addLegalDialect<tosa::TosaDialect>();
      target.addLegalDialect<func::FuncDialect>();
      target.addLegalOp<ModuleOp>();
      target.addLegalOp<UnrealizedConversionCastOp>();

      // func.func / func.return are legal once their signatures (and return
      // operand types) no longer contain quantized types.
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType());
      });
      target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
        return typeConverter.isLegal(op.getOperandTypes());
      });

      target.addIllegalDialect<aipugc::AIPUDialect>();
      target.addIllegalOp<aipugc::ConstantOp, aipugc::AddOp, aipugc::QuantizeOp,
                           aipugc::AbsOp, aipugc::EltwiseOp,
                           aipugc::ActivationOp>();

      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
          patterns, typeConverter);
      populateReturnOpTypeConversionPattern(patterns, typeConverter);
      patterns.add<AddOpLowering>(typeConverter, &getContext());
      patterns.add<ConstantOpLowering>(typeConverter, &getContext());
      patterns.add<QuantizeOpLowering>(typeConverter, &getContext());
      patterns.add<AbsOpLowering>(typeConverter, &getContext());
      patterns.add<EltwiseOpLowering>(typeConverter, &getContext());
      patterns.add<ActivationOpLowering>(typeConverter, &getContext());

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
