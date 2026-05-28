---
name: write-pass
description: Write a new optimization pass for AIPUBuilder (MLIR or builder-level)
parameters:
  - name: pass_name
    description: "Name of the pass (e.g., 'FuseBatchNorm', 'EliminateDeadCode')"
    required: true
  - name: pass_type
    description: "Type: mlir (MLIR-level) or builder (builder-level)"
    required: true
  - name: pass_purpose
    description: "What the pass optimizes (e.g., 'Fuse BatchNorm into Conv', 'Remove unused tensors')"
    required: true
  - name: analysis_only
    description: "Is this an analysis-only pass? (true/false)"
    required: false
---

# Write Optimization Pass: {{pass_name}}

**Purpose**: {{pass_purpose}}
**Type**: {{pass_type}} pass
**Analysis only**: {{analysis_only}}

## Pass Implementation Template

{% if pass_type == "mlir" %}
### MLIR Pass Implementation

**Location**: `src/mlir/lib/aipugc/`

Create `{{pass_name}}.cpp`:

```cpp
#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "aipugc/Passes.h"
#include "aipugc/Dialect.h"

namespace {

{% if analysis_only == "true" %}
// Analysis-only pass - doesn't modify IR
struct {{pass_name}} : public PassWrapper<{{pass_name}}, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    
    // Perform analysis
    module.walk([&](Operation *op) {
      // Analyze operations
      // Collect statistics
      // Build analysis data structures
    });
    
    // Don't modify IR
    markAllAnalysesPreserved();
  }
  
  StringRef getArgument() const override { return "{{pass_name | lower | replace(' ', '-')}}"; }
  StringRef getDescription() const override { return "{{pass_purpose}}"; }
};
{% else %}
// Transformation pass - modifies IR
struct {{pass_name}} : public PassWrapper<{{pass_name}}, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    
    // Collect operations to transform
    SmallVector<Operation*> opsToTransform;
    module.walk([&](Operation *op) {
      if (shouldTransform(op)) {
        opsToTransform.push_back(op);
      }
    });
    
    // Apply transformations
    for (auto *op : opsToTransform) {
      transform(op);
    }
  }
  
private:
  bool shouldTransform(Operation *op) {
    // Return true if op should be transformed
    // Check op type, attributes, operands, etc.
    return false;
  }
  
  void transform(Operation *op) {
    OpBuilder builder(op);
    
    // Create replacement operations
    // builder.create<NewOp>(op->getLoc(), ...);
    
    // Replace old operation
    // op->replaceAllUsesWith(newOp);
    // op->erase();
  }
  
public:
  StringRef getArgument() const override { return "{{pass_name | lower | replace(' ', '-')}}"; }
  StringRef getDescription() const override { return "{{pass_purpose}}"; }
};
{% endif %}

} // namespace

// Factory function
std::unique_ptr<Pass> create{{pass_name}}() {
  return std::make_unique<{{pass_name}}>();
}

``` 

### Pattern-Based MLIR Pass (Alternative)

For pattern-matching transformations:

```cpp
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

// Define rewrite pattern
struct YourPattern : public OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const override {
    // Match conditions
    if (!shouldRewrite(op))
      return failure();
    
    // Rewrite
    auto newOp = rewriter.create<TargetOp>(op.getLoc(), ...);
    rewriter.replaceOp(op, newOp.getResults());
    
    return success();
  }
};

struct {{pass_name}} : public PassWrapper<{{pass_name}}, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<YourPattern>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
  
  StringRef getArgument() const override { return "{{pass_name | lower | replace(' ', '-')}}"; }
  StringRef getDescription() const override { return "{{pass_purpose}}"; }
};

} // namespace
```

{% elif pass_type == "builder" %}
### Builder-Level Pass Implementation

**Location**: `src/builder/` or `src/builder/lowering/`

Create `{{pass_name}}.h` and `{{pass_name}}.cpp`:

```cpp
// {{pass_name}}.h
#pragma once
#include "core.h"
#include "builder/optimizer.h"

class {{pass_name}} : public PassBase {
public:
  {{pass_name}}() : PassBase("{{pass_name}}") {}
  
  bool run(Graph *graph) override;
  
private:
  bool shouldTransform(Node *node);
  void transform(Node *node, Graph *graph);
};
```

```cpp
// {{pass_name}}.cpp
#include "{{pass_name}}.h"

bool {{pass_name}}::run(Graph *graph) {
  bool changed = false;
  
  {% if analysis_only == "true" %}
  // Analysis pass - don't modify graph
  for (auto *node : graph->nodes()) {
    // Analyze node
    // Collect statistics
    // Build analysis data
  }
  
  return false;  // Indicate no modifications
  {% else %}
  // Transformation pass
  std::vector<Node*> nodesToTransform;
  
  // Collect nodes to transform
  for (auto *node : graph->nodes()) {
    if (shouldTransform(node)) {
      nodesToTransform.push_back(node);
    }
  }
  
  // Apply transformations
  for (auto *node : nodesToTransform) {
    transform(node, graph);
    changed = true;
  }
  
  return changed;
  {% endif %}
}

bool {{pass_name}}::shouldTransform(Node *node) {
  // Check if node should be transformed
  // Consider:
  // - Node type
  // - Node attributes
  // - Input/output shapes
  // - Data types
  // - Predecessors/successors
  
  return false;
}

void {{pass_name}}::transform(Node *node, Graph *graph) {
  // Transform the node
  // Options:
  // 1. Modify node in-place
  // 2. Replace with new node(s)
  // 3. Remove node
  // 4. Fuse with other nodes
  
  // Example: Replace node
  // auto *newNode = graph->createNode(NewOpType, node->inputs());
  // graph->replaceNode(node, newNode);
}
```
{% endif %}

## Step 2: Register the Pass

{% if pass_type == "mlir" %}
**In `src/mlir/include/aipugc/Passes.h`**:
```cpp
std::unique_ptr<Pass> create{{pass_name}}();
```

**In `src/mlir/lib/aipugc/Passes.cpp`**:
```cpp
void registerPasses() {
  PassRegistration<{{pass_name}}>();
}
```
{% elif pass_type == "builder" %}
**In appropriate builder pipeline file**:
```cpp
#include "{{pass_name}}.h"

void runOptimizationPipeline(Graph *graph) {
  // Add to pipeline
  {{pass_name}}().run(graph);
}
```
{% endif %}

## Step 3: Add Tests

**Location**: `test/` or `src/mlir/test/`

{% if pass_type == "mlir" %}
Create `{{pass_name | lower | replace(' ', '-')}}.mlir`:

```mlir
// RUN: aipugc-opt %s -{{pass_name | lower | replace(' ', '-')}} | FileCheck %s

// CHECK-LABEL: func @test_{{pass_name | lower}}
func @test_{{pass_name | lower}}(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32> {
  // Test case for {{pass_purpose}}
  // CHECK: Expected output after transformation
  %0 = "aipugc.some_op"(%arg0) : (tensor<1x3x224x224xf32>) -> tensor<1x64x112x112xf32>
  return %0 : tensor<1x64x112x112xf32>
}
```
{% elif pass_type == "builder" %}
Create `test_{{pass_name}}.cpp`:

```cpp
#include "test_core.h"
#include "{{pass_name}}.h"

TEST({{pass_name}}Test, BasicTransformation) {
  // Create test graph
  Graph graph;
  
  // Add nodes to test
  auto *node = graph.createNode(OpType::TEST_OP);
  
  // Run pass
  {{pass_name}} pass;
  bool changed = pass.run(&graph);
  
  // Verify results
  EXPECT_TRUE(changed);
  // Check graph structure
  // Verify transformations
}

TEST({{pass_name}}Test, NoChange) {
  // Test case where pass doesn't modify graph
  Graph graph;
  // ... setup graph
  
  {{pass_name}} pass;
  bool changed = pass.run(&graph);
  
  EXPECT_FALSE(changed);
}
```
{% endif %}

## Step 4: Test the Pass

```bash
{% if pass_type == "mlir" %}
# Build
cmake --build build

# Run MLIR tests
cd build && ninja check-aipugc

# Test on real model
aipugb model.def -w model.bin2 \
  --serialize-before-pass {{pass_name | lower | replace(' ', '-')}} \
  --serialize-after-pass {{pass_name | lower | replace(' ', '-')}}

# Visualize changes
aipugb model.def -w model.bin2 \
  --svg-before-pass {{pass_name | lower | replace(' ', '-')}} \
  --svg-after-pass {{pass_name | lower | replace(' ', '-')}}
{% elif pass_type == "builder" %}
# Build and run tests
mkdir -p build && cd build
cmake .. && make
./test_{{pass_name}}

# Test on real model
aipugb model.def -w model.bin2 -o output.bin

# Verify output correctness
aipurun model.def -w model.bin2 -i input.bin -o output.bin
{% endif %}
```

## Common Pass Patterns

### Fusion Pass
Combine multiple operations into one:
- Conv + BatchNorm → ConvBN
- Add + ReLU → AddReLU
- Reduces memory traffic and improves performance

### Elimination Pass
Remove unnecessary operations:
- Dead code elimination
- Identity operation removal
- Constant folding

### Canonicalization Pass
Transform to canonical form:
- Normalize operation order
- Simplify expression trees
- Constant propagation

### Layout Pass
Optimize data layout:
- NCHW ↔ NHWC conversion
- Tensor format optimization
- Memory layout optimization

## Checklist

- [ ] Pass implemented with proper structure
- [ ] Pass registered in appropriate file
- [ ] Tests added and passing
- [ ] Pass doesn't break existing tests
- [ ] Pass tested on real models
- [ ] Documentation added
- [ ] Performance impact measured
- [ ] Edge cases handled

## Debugging Tips

```cpp
// Enable debug output
#define DEBUG_TYPE "{{pass_name | lower}}"
#include "llvm/Support/Debug.h"

LLVM_DEBUG(llvm::dbgs() << "Processing node: " << node->name() << "\n");

// Dump IR for inspection
module.dump();
node->dump();

// Verify IR is valid
assert(module.verify().succeeded() && "IR verification failed");
```

## References

- `doc/Builder/How_to_write_a_pass.md`
- `doc/parser/how_to_write_a_pass.md`
- Existing passes in `src/mlir/lib/aipugc/` or `src/builder/`
