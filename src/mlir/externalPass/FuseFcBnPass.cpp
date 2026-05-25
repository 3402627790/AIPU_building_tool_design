#include "Plugin/Plugin.h"
#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "FuseFcBnPass"

using namespace mlir;

/// Fold FC + BN into FC (BatchNormOp $weight / $bias are the per-channel scale
/// and shift; same math as gamma/beta on the affine after normalization).
///   W'[i,:] = W[i,:] * bnWeight[i]
///   b'[i]   = b[i] * bnWeight[i] + bnBias[i]
static DenseElementsAttr getDenseConstant(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return {};
  auto cst = dyn_cast<aipugc::ConstantOp>(def);
  if (!cst)
    return {};
  return dyn_cast<DenseElementsAttr>(cst.getValue());
}

static llvm::LogicalResult fuseFullyConnectedBatchNormConstants(
    RankedTensorType fcWeightType, DenseElementsAttr fcWeight,
    RankedTensorType fcBiasType, DenseElementsAttr fcBias,
    RankedTensorType bnWeightType, DenseElementsAttr bnWeight,
    RankedTensorType bnBiasType, DenseElementsAttr bnBias,
    DenseElementsAttr &fusedWeight, DenseElementsAttr &fusedBias
) {
  if (!isa<FloatType, BFloat16Type>(fcWeightType.getElementType())) {
    return llvm::failure();
  }

  auto fcWeightShape = fcWeightType.getShape();
  if (fcWeightShape.size() != 2) {
    return llvm::failure();
  }
  int64_t outc = fcWeightShape[0];
  int64_t inc = fcWeightShape[1];

  auto fcBiasShape = fcBiasType.getShape();
  if (fcBiasShape.size() != 1 || fcBiasShape[0] != outc) {
    return llvm::failure();
  }
  if (bnWeightType.getShape() != fcBiasShape || bnBiasType.getShape() != fcBiasShape) {
    return llvm::failure();
  }

  auto fcWeightVals = fcWeight.getValues<APFloat>();
  auto fcBiasVals = fcBias.getValues<APFloat>();
  auto bnWeightVals = bnWeight.getValues<APFloat>();
  auto bnBiasVals = bnBias.getValues<APFloat>();

  if (fcWeightVals.size() != outc * inc ||
      fcBiasVals.size() != outc || 
      bnWeightVals.size() != outc ||
      bnBiasVals.size() != outc) {
    return llvm::failure();
  }

  constexpr auto round_mode = APFloat::rmNearestTiesToEven;
  SmallVector<APFloat> newWeight;
  for (int64_t oc = 0; oc < outc; oc++) {
    APFloat scale = bnWeightVals[oc];
    for (int64_t ic = 0; ic < inc; ic++) {
      APFloat wv = fcWeightVals[oc * inc + ic];
      wv.multiply(scale, round_mode);
      newWeight.push_back(wv);
    }
  }

  SmallVector<APFloat> newBias;
  for (int64_t oc = 0; oc < outc; oc++) {
    APFloat bv = fcBiasVals[oc];
    bv.multiply(bnWeightVals[oc], round_mode);
    bv.add(bnBiasVals[oc], round_mode);
    newBias.push_back(bv);
  }

  fusedWeight = DenseElementsAttr::get(fcWeightType, newWeight);
  fusedBias = DenseElementsAttr::get(fcBiasType, newBias);
  return llvm::success();
}


struct FuseFullyConnectedBatchNorm : public OpRewritePattern<aipugc::BatchNormOp> {
  
  FuseFullyConnectedBatchNorm(MLIRContext* context)
    : OpRewritePattern<aipugc::BatchNormOp>(context, 5) {}

  // Match: %y = FullyConnected(%x, %w_fc, %b_fc)
  //        %out = BatchNorm(%y, %w_bn, %b_bn)
  // Into:  %out = FullyConnected(%x, %w_new, %b_new)
  llvm::LogicalResult matchAndRewrite(aipugc::BatchNormOp bnOp,
                                      PatternRewriter &rewriter) const override {
    Value bnInput = bnOp.getInput();
    auto fcOp = bnInput.getDefiningOp<aipugc::FullyConnectedOp>();
    
    auto fcOutType = llvm::dyn_cast<RankedTensorType>(fcOp.getResult().getType());
    auto bnInpType = llvm::dyn_cast<RankedTensorType>(bnInput.getType());
    if (!fcOutType || !bnInpType || fcOutType != bnInpType) {
        return llvm::failure();
    }

    auto rank = bnInpType.getRank();
    auto axis = bnOp.getAxis();
    if (axis < 0)
      axis += rank;
    // NOTE: BN after FC on channel/feature dim (axis 1 for 2D).
    if (rank != 2 || axis != 1)
      return llvm::failure();
    
    DenseElementsAttr fcWeightAttr = getDenseConstant(fcOp.getWeight());
    DenseElementsAttr fcBiasAttr = getDenseConstant(fcOp.getBias());
    DenseElementsAttr bnWeightAttr = getDenseConstant(bnOp.getWeight());
    DenseElementsAttr bnBiasAttr = getDenseConstant(bnOp.getBias());
    if (!fcWeightAttr || !fcBiasAttr || !bnWeightAttr || !bnBiasAttr) {
      return llvm::failure();
    }

    auto fcWeightType = llvm::dyn_cast<RankedTensorType>(fcOp.getWeight().getType());
    auto fcBiasType = llvm::dyn_cast<RankedTensorType>(fcOp.getBias().getType());
    auto bnWeightType = llvm::dyn_cast<RankedTensorType>(bnOp.getWeight().getType());
    auto bnBiasType = llvm::dyn_cast<RankedTensorType>(bnOp.getBias().getType());
    if (!fcWeightType || !fcBiasType || !bnWeightType || !bnBiasType) {
      return llvm::failure();
    }

    DenseElementsAttr fusedWeightAttr, fusedBiasAttr;
    if (failed(fuseFullyConnectedBatchNormConstants(
        fcWeightType, fcWeightAttr, fcBiasType, fcBiasAttr,
        bnWeightType, bnWeightAttr, bnBiasType, bnBiasAttr,
        fusedWeightAttr, fusedBiasAttr
    ))) {
      return llvm::failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "FuseFcBn: folding BatchNorm into FullyConnected.\n");

    Location loc = bnOp.getLoc();
    auto fusedWeight = rewriter.create<aipugc::ConstantOp>(loc, fusedWeightAttr);
    auto fusedBias = rewriter.create<aipugc::ConstantOp>(loc, fusedBiasAttr);
    DictionaryAttr dictAttr = rewriter.getDictionaryAttr({});
    if (auto attr = fcOp->getAttr("dictAttr")) {
      dictAttr = cast<DictionaryAttr>(attr);
    }

    auto newFc = rewriter.create<aipugc::FullyConnectedOp>(
      loc, bnOp.getType(), fcOp.getInput(),
      fusedWeight, fusedBias, dictAttr
    );
    
    rewriter.replaceOp(bnOp, newFc.getResult());
    rewriter.eraseOp(fcOp);

    return llvm::success();
  }
};

struct FuseFcBnPass
    : public PassWrapper<FuseFcBnPass, OperationPass<aipugc::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseFcBnPass);

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseFullyConnectedBatchNorm>(&getContext());

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

REGISTER_PASS_PLUGIN(FuseFcBnPass)
