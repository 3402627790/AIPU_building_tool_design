#ifndef AIPUTILE_PASSES_H
#define AIPUTILE_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <cstdint>
#include <memory>

namespace mlir::aiputile {
std::unique_ptr<Pass> createLowerAipugcToAIPUTilePass();
std::unique_ptr<Pass> createTileTilingPass(int64_t tileSize = 256);
std::unique_ptr<Pass> createTileFusionPass();
// Address space is a device ABI choice, never a claim of SRAM residency.
std::unique_ptr<Pass> createTileLoweringPass(unsigned addressSpace = 0);
std::unique_ptr<Pass> createTileToLLVMPass(bool barePointers = true,
                                           unsigned indexBitwidth = 64);
void registerAIPUTilePasses();
} // namespace mlir::aiputile
#endif
