---
name: mlir-passes
description: "Use when: working with MLIR passes, aipugc dialect, transformation passes, or files in src/mlir/"
applyTo:
  - "src/mlir/**/*.cpp"
  - "src/mlir/**/*.h"
  - "**/*Pass*.cpp"
  - "**/*Pass*.h"
---

# MLIR Pass Development Guidelines

## Pass Structure

MLIR passes in AIPUBuilder follow this pattern:

```cpp
#include "mlir/Pass/Pass.h"
#include "aipugc/Passes.h"

namespace {
struct YourPass : public PassWrapper<YourPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    // Pass implementation
  }
  
  StringRef getArgument() const override { return "your-pass"; }
  StringRef getDescription() const override { return "Description"; }
};
} // namespace

std::unique_ptr<Pass> createYourPass() {
  return std::make_unique<YourPass>();
}
```

## Registration

1. **Declare in `Passes.h`**:
```cpp
std::unique_ptr<Pass> createYourPass();
```

2. **Register in `Passes.cpp`**:
```cpp
void registerPasses() {
  PassRegistration<YourPass>();
}
```

3. **Add to pipeline** in appropriate builder file

## Common Patterns

### Walking Operations
```cpp
getOperation().walk([&](SomeOp op) {
  // Process each operation of type SomeOp
});
```

### Pattern Matching & Rewriting
```cpp
#include "mlir/IR/PatternMatch.h"

struct YourPattern : public OpRewritePattern<YourOp> {
  using OpRewritePattern<YourOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(YourOp op, PatternRewriter &rewriter) const override {
    // Match conditions
    if (!shouldRewrite(op))
      return failure();
    
    // Rewrite
    rewriter.replaceOpWithNewOp<NewOp>(op, ...);
    return success();
  }
};
```

### Analysis Passes
```cpp
// Store analysis results without modifying IR
void runOnOperation() override {
  auto module = getOperation();
  
  // Perform analysis
  auto &analysis = getAnalysis<SomeAnalysis>();
  
  // Don't mark anything as modified if pure analysis
  markAllAnalysesPreserved();
}
```

## aipugc Dialect Specifics

- Operations defined in `Dialect.cpp`
- Shape inference: Implement `ShapeInferenceInterface`
- Quantization utilities in `QuantUtils.h`
- Type utilities in `utils.h`

## Testing Passes

```bash
# Dump IR before pass
aipugb model.def --serialize-before-pass your-pass

# Dump IR after pass
aipugb model.def --serialize-after-pass your-pass

# Visualize before/after
aipugb model.def --svg-before-pass your-pass --svg-after-pass your-pass

# Disable specific pass
aipugb model.def --disable-pass your-pass

# Enable only specific pass
aipugb model.def --enable-pass your-pass
```

## Pass Dependencies

- Declare analysis dependencies with `getAnalysis<T>()`
- Preserve analyses with `markAnalysesPreserved<T>()`
- Specify required dialects in pass constructor

## Common Issues

1. **Memory leaks**: Use MLIR's allocation APIs, not `new`/`delete`
2. **Invalid IR**: Always verify with `verify()` after modifications
3. **Type mismatches**: Use `getType()` and type casting carefully
4. **Dangling uses**: Replace operations properly with `replaceAllUsesWith()`

## Debugging

```cpp
// Enable pass debug output
#define DEBUG_TYPE "your-pass"
#include "llvm/Support/Debug.h"

LLVM_DEBUG(llvm::dbgs() << "Debug message\n");
```

## References

- `doc/Builder/How_to_write_a_pass.md`
- `doc/parser/how_to_write_a_pass.md`
- MLIR documentation: https://mlir.llvm.org/docs/
