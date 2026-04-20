#include "Plugin/Plugin.h"
#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "FuseMulAddToBatchNormPass"

using namespace mlir;

namespace {

struct FuseMulAddToBatchNorm : public OpRewritePattern<aipugc::AddOp> {
  FuseMulAddToBatchNorm(MLIRContext *context)
      : OpRewritePattern<aipugc::AddOp>(context, /*benefit=*/5) {}

  // Match and rewrite the following pattern:
  //   %w = aipugc.constant   (weight)
  //   %b = aipugc.constant   (bias)
  //   %0 = "aipugc.mul"(%input, %w)            -> MulOp defines %0
  //   %1 = "aipugc.add"(%0, %b)                -> AddOp uses %0, defines %1
  // Into:
  //   %1 = "aipugc.BatchNorm"(%input, %w, %b)
  LogicalResult matchAndRewrite(aipugc::AddOp addOp,
                                PatternRewriter &rewriter) const override {
    Value addLhs = addOp.getLhs();
    Value addRhs = addOp.getRhs();

    // Pattern: add(mul(...), bias) or add(bias, mul(...))
    aipugc::MulOp mulOp = nullptr;
    Value bias = nullptr;

    if (auto mul = addLhs.getDefiningOp<aipugc::MulOp>()) {
      mulOp = mul;
      bias = addRhs;
    } else if (auto mul = addRhs.getDefiningOp<aipugc::MulOp>()) {
      mulOp = mul;
      bias = addLhs;
    }
    if (!mulOp)
      return failure();

    // Require both weight and bias to be constant for quantization fusion.
    if (!bias.getDefiningOp<aipugc::ConstantOp>())
      return failure();

    // Determine input vs weight inside mul by comparing tensor ranks:
    Value mulLhs = mulOp.getLhs();
    Value mulRhs = mulOp.getRhs();
    auto lhsType = llvm::dyn_cast<RankedTensorType>(mulLhs.getType());
    auto rhsType = llvm::dyn_cast<RankedTensorType>(mulRhs.getType());
    if (!lhsType || !rhsType)
      return failure();

    Value input, weight;
    if (lhsType.getRank() >= rhsType.getRank()) {
      input = mulLhs;
      weight = mulRhs;
    } else {
      input = mulRhs;
      weight = mulLhs;
    }

    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto weightType = llvm::dyn_cast<RankedTensorType>(weight.getType());
    if (!inputType || !weightType || weightType.getRank() != 1)
      return failure();

    // Require weight to be constant.
    if (!weight.getDefiningOp<aipugc::ConstantOp>())
      return failure();

    SmallVector<int64_t> broadcastShape;
    if (!OpTrait::util::getBroadcastedShape(inputType.getShape(),
                                            weightType.getShape(),
                                            broadcastShape))
      return failure();

    int32_t axis = static_cast<int32_t>(inputType.getRank() - 1);

    LLVM_DEBUG(llvm::dbgs() << "FuseMulAddToBatchNorm: matched pattern "
                               "add(mul(input, weight), bias), axis="
                            << axis << "\n");

    auto axisAttr =
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), axis);
    auto epsilonAttr =
        rewriter.getFloatAttr(rewriter.getF32Type(), /*epsilon=*/0.0);

    auto bnOp = rewriter.create<aipugc::BatchNormOp>(
        addOp.getLoc(), addOp.getType(), input, weight, bias, axisAttr,
        epsilonAttr, /*dictAttr=*/nullptr);

    rewriter.replaceOp(addOp, bnOp.getResult());
    return success();
  }
};

struct FuseMulAddToBatchNormPass
    : public PassWrapper<FuseMulAddToBatchNormPass,
                         OperationPass<aipugc::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMulAddToBatchNormPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMulAddToBatchNorm>(&getContext());

    GreedyRewriteConfig config;
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

REGISTER_PASS_PLUGIN(FuseMulAddToBatchNormPass)
