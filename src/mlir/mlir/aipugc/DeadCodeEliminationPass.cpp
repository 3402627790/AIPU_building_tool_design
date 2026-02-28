#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

#define DEBUG_TYPE "dce"

using namespace mlir;
using namespace aipugc;

namespace {
struct DCEPass : public PassWrapper<DCEPass,
                                    OperationPass<FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DCEPass)

  void runOnOperation() override {
    FuncOp func = getOperation();
    bool changed;

    do {
      changed = false;
      SmallVector<Operation*> deadOps;

      // Collect operations to delete
      func.walk([&](Operation *op) {
        // Skip terminators (e.g., return, branch)
        if (op->hasTrait<OpTrait::IsTerminator>()) return;

        // Check if the operation is trivially dead using MLIR's built-in method
        if (!mlir::isOpTriviallyDead(op)) return;

        deadOps.push_back(op);
      });

      // Erase collected dead operations
      if (!deadOps.empty()) {
        changed = true;
        for (Operation *op : deadOps) {
          op->erase();
        }
      }
    } while (changed);  // Loop until no more dead operations are found
  }
};
} // namespace

std::unique_ptr<Pass> aipugc::createDCEPass() {
  return std::make_unique<DCEPass>();
}