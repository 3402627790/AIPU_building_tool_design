#include "Plugin/Plugin.h"
#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "EliminateRedundantTransposePass"

using namespace mlir;

namespace {
struct EliminateRedundantTransposePass
    : public PassWrapper<EliminateRedundantTransposePass,
                         OperationPass<aipugc::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateRedundantTransposePass)

  StringRef getArgument() const final { return "eliminate-redundant-transpose"; }
  StringRef getDescription() const final { return "Eliminate redundant transpose operations"; }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyRedundantTranspose>(&getContext());
    
    GreedyRewriteConfig config;
    // config.enableRegionSimplification = GreedySimplifyRegionLevel::Disabled;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
        signalPassFailure();
    }
  }

private:
  struct SimplifyRedundantTranspose
      : public OpRewritePattern<aipugc::TransposeOp> {
    SimplifyRedundantTranspose(MLIRContext *context)
        : OpRewritePattern<aipugc::TransposeOp>(context, /*benefit=*/3) {}

    LogicalResult matchAndRewrite(aipugc::TransposeOp op,
                                  PatternRewriter &rewriter) const override {
      // Check input operation chain
      Value input = op.getInput();
      llvm::dbgs() << "matchAndRewrite";
      auto inputTranspose = input.getDefiningOp<aipugc::TransposeOp>();
      if (!inputTranspose)
        return failure();

      // Access permutation attributes
      auto currPerm = op.getPermAttr();
      auto inputPerm = inputTranspose.getPermAttr();

      // Validate inverse permutation
      if (currPerm.size() != inputPerm.size())
        return failure();
      for (size_t i = 0; i < currPerm.size(); ++i) {
        int64_t inputVal = llvm::cast<IntegerAttr>(inputPerm[i]).getInt();
        int64_t currVal = llvm::cast<IntegerAttr>(currPerm[inputVal]).getInt();
        if (currVal != static_cast<int64_t>(i))
          return failure();
      }

      // Replace operation
      rewriter.replaceOp(op, inputTranspose.getInput());
      return success();
    }
  };
};
} // namespace

static mlir::PassRegistration<EliminateRedundantTransposePass> passRegElimTranspose;

REGISTER_PASS_PLUGIN(EliminateRedundantTransposePass)