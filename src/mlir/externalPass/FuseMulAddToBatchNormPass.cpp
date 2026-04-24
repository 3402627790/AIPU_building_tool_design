#include "Plugin/Plugin.h"
#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "FuseMulAddToBatchNormPass"

using namespace mlir;

namespace {

struct FuseMulAddToBatchNorm : public OpRewritePattern<aipugc::AddOp> {
  FuseMulAddToBatchNorm(MLIRContext *context)
      : OpRewritePattern<aipugc::AddOp>(context, 5) {}

  // Match and rewrite the following pattern:
  //   %w = aipugc.constant   (weight)
  //   %b = aipugc.constant   (bias)
  //   %0 = "aipugc.mul"(%input, %w)            -> MulOp defines %0
  //   %1 = "aipugc.add"(%0, %b)                -> AddOp uses %0, defines %1
  // Into:
  //   %1 = "aipugc.BatchNorm"(%input, %w, %b)
  llvm::LogicalResult matchAndRewrite(aipugc::AddOp addOp,
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

    // If mul result is directly returned, keep mul as-is and skip fusion.
    // This guard is added so that in non-return cases, mul can become dead
    // after fusion and then be removed by the downstream DCE pass.
    for (Operation *user : mulOp->getUsers()) {
      if (isa<aipugc::ReturnOp>(user))
        return failure();
    }

    // check MUL bias to be constant
    if (!bias.getDefiningOp<aipugc::ConstantOp>())
      return failure();

    // Determine input vs weight inside mul:
    // ConstantOp is treated as weight, the other side as input.
    Value mulLhs = mulOp.getLhs();
    Value mulRhs = mulOp.getRhs();
    auto mulLhsType = llvm::dyn_cast<RankedTensorType>(mulLhs.getType());
    auto mulRhsType = llvm::dyn_cast<RankedTensorType>(mulRhs.getType());
    if (!mulLhsType || !mulRhsType)
      return llvm::failure();

    Value input, weight;
    bool mulLhsIsConst = static_cast<bool>(mulLhs.getDefiningOp<aipugc::ConstantOp>());
    bool mulRhsIsConst = static_cast<bool>(mulRhs.getDefiningOp<aipugc::ConstantOp>());
    if (mulLhsIsConst == mulRhsIsConst) {
      // Expect exactly one constant operand in mul.
      return mlir::failure();
    } else if (mulLhsIsConst){
      input = mulRhs;
      weight = mulLhs;
    } else {
      input = mulLhs;
      weight = mulRhs;
    }

    auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
    auto weightType = llvm::dyn_cast<RankedTensorType>(weight.getType());
    auto biasType = llvm::dyn_cast<RankedTensorType>(bias.getType());
    if (!inputType || !weightType || !biasType || weightType.getRank() != 1 || biasType.getRank() != 1)
      return failure();

    SmallVector<int64_t> broadcastShape;
    if (!OpTrait::util::getBroadcastedShape(inputType.getShape(),
                                            weightType.getShape(),
                                            broadcastShape))
      return failure();

    int32_t axis = static_cast<int32_t>(inputType.getRank() - 1);
    int64_t lastDimSize = inputType.getDimSize(axis);
    if (lastDimSize == ShapedType::kDynamic)
      return failure();
    if (lastDimSize != weightType.getDimSize(0) || lastDimSize != biasType.getDimSize(0))
      return failure();

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
  MLIR_DEFINE_EXPLICIT_FALLBACK_TYPE_ID(FuseMulAddToBatchNormPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMulAddToBatchNorm>(&getContext());

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

REGISTER_PASS_PLUGIN(FuseMulAddToBatchNormPass)
