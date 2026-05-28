/// CacheHintAnalysisPass.cpp - Analyze tensor usage to generate cache hints
///
/// This pass walks the graph in topological order and for each operation's
/// output tensor computes:
///   - num_consumers:      how many distinct ops consume this tensor
///   - last_use_distance:  topological distance from producer to last consumer
///   - is_last_use:        whether there are no further consumers after this point
///   - cache_priority:     a composite score (higher => more valuable to cache)
///
/// The priority heuristic (usage density):
///   priority = num_consumers * 100 / distance
///
/// Tensors with high usage density (many consumers over short distance)
/// are most valuable to cache; tensors consumed only by return get priority 0.
///
/// Results are written as integer attributes on each producing operation:
///   "cache_hint.num_consumers"     : i32
///   "cache_hint.last_use_distance" : i32
///   "cache_hint.priority"          : i32

#include "Plugin/Plugin.h"
#include "aipugc/Dialect.h"
#include "aipugc/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <vector>

#define DEBUG_TYPE "CacheHintAnalysisPass"

using namespace mlir;

namespace {

struct CacheHintAnalysisPass
    : public PassWrapper<CacheHintAnalysisPass,
                         OperationPass<aipugc::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_FALLBACK_TYPE_ID(CacheHintAnalysisPass)

  StringRef getArgument() const final { return "cache-hint-analysis"; }
  StringRef getDescription() const final {
    return "Analyze output tensor usage frequency and assign cache priority "
           "attributes to each operation";
  }

  void runOnOperation() override {
    aipugc::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();

    llvm::errs() << "[CacheHintAnalysis] >>> Pass started on function: "
                 << func.getName() << "\n";

    // Helper: extract the op name from its Location (NameLoc stores layer_name)
    auto getOpName = [](Operation *op) {
      if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc()))
        return nameLoc.getName().str();
      return op->getName().getStringRef().str();
    };

    // ---- Step 1: assign a topological index to every operation ----------
    llvm::DenseMap<Operation *, int32_t> topoIndex;
    int32_t idx = 0;
    func.walk([&](Operation *op) { topoIndex[op] = idx++; });

    int32_t maxIndex = idx; // for normalization
    llvm::errs() << "[CacheHintAnalysis] Total ops indexed: " << maxIndex
                 << "\n";

    // ---- Step 2: for each op, compute per-result cache statistics -------
    func.walk([&](Operation *op) {
      // Skip terminators and constants – they don't produce cacheable outputs
      if (op->hasTrait<OpTrait::IsTerminator>())
        return;

      llvm::errs() << "[CacheHintAnalysis] Visiting op: "
                   << op->getName().getStringRef()
                   << " [" << getOpName(op) << "]"
                   << " (topo#" << topoIndex[op] << ")\n";

      // Iterate over every result value (output tensor) of this op
      for (OpResult result : op->getResults()) {
        // Collect consumer info
        int32_t numConsumers = 0;
        int32_t lastUseIdx = topoIndex[op]; // start with producer position
        bool consumedByReturn = false;

        for (Operation *user : result.getUsers()) {
          ++numConsumers;
          int32_t userIdx = topoIndex[user];
          llvm::errs() << "[CacheHintAnalysis]   consumer: "
                       << user->getName().getStringRef()
                       << " [" << getOpName(user) << "]"
                       << " (topo#" << userIdx << ")\n";
          if (userIdx > lastUseIdx)
            lastUseIdx = userIdx;
          if (isa<aipugc::ReturnOp>(user))
            consumedByReturn = true;
        }

        if (numConsumers == 0) {
          llvm::errs() << "[CacheHintAnalysis]   result#"
                       << result.getResultNumber()
                       << " has no consumers, skipping\n";
          continue;
        }

        int32_t distance = lastUseIdx - topoIndex[op];

        // Priority = usage density: consumers / distance
        //   - More consumers  => higher priority (reused many times)
        //   - Larger distance => lower priority (occupies cache without use)
        //   - Network outputs (consumed only by ReturnOp) => priority 0
        int32_t priority;
        if (consumedByReturn && numConsumers == 1) {
          priority = 0; // final output, evict immediately
        } else {
          // Scale by 100 to preserve precision in integer arithmetic
          priority = numConsumers * 100 / std::max(distance, (int32_t)1);
        }

        llvm::errs() << "[CacheHintAnalysis]   result#"
                     << result.getResultNumber()
                     << " => consumers=" << numConsumers
                     << ", distance=" << distance
                     << ", priority=" << priority
                     << (consumedByReturn ? " (consumed by return)" : "")
                     << "\n";

        // ---- Step 3: attach attributes to the producing op --------------
        // If the op has multiple results, suffix the attribute name with the
        // result number; for the common single-result case, use plain names.
        auto suffix = [&](StringRef base) -> std::string {
          if (op->getNumResults() == 1)
            return base.str();
          return (base + "." + llvm::Twine(result.getResultNumber())).str();
        };

        auto i32Ty = IntegerType::get(ctx, 32);
        op->setAttr(suffix("cache_hint.num_consumers"),
                    IntegerAttr::get(i32Ty, numConsumers));
        op->setAttr(suffix("cache_hint.last_use_distance"),
                    IntegerAttr::get(i32Ty, distance));
        op->setAttr(suffix("cache_hint.priority"),
                    IntegerAttr::get(i32Ty, priority));
      }
    });

    llvm::errs() << "[CacheHintAnalysis] <<< Pass finished\n";
  }
};

} // namespace

static mlir::PassRegistration<CacheHintAnalysisPass> passReg;

REGISTER_PASS_PLUGIN(CacheHintAnalysisPass)
