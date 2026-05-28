---
name: builder-ops
description: "Use when: working on builder operations, lowering, scheduling, memory allocation, or code generation in src/builder/"
applyTo:
  - "src/builder/**/*.cpp"
  - "src/builder/**/*.h"
  - "src/core/**/*.cpp"
  - "src/core/**/*.h"
---

# Builder Operations Development Guidelines

## Builder Pipeline Overview

The builder transforms quantized IR into AIPU executable code:

1. **Lowering**: MLIR/IR → AIPU-specific IR (`src/builder/lowering/`)
2. **Optimization**: Graph optimization passes (`src/builder/optimizer.h`)
3. **Scheduling**: Instruction scheduling (`src/builder/mc_scheduler.h`, `soms_scheduler.h`)
4. **Memory Allocation**: SRAM and external memory (`src/builder/sram_allocator.h`, `mmu_allocator.h`)
5. **Code Generation**: Assembly and binary generation (`src/builder/emitter/`)

## Adding New Operations

### 1. Define Op in Core IR

In `src/core/ops/`:

```cpp
#include "core.h"

class YourOp : public OpBase {
public:
  YourOp(const std::string &name, const std::vector<Tensor*> &inputs);
  
  // Shape inference
  std::vector<Shape> inferShape() const override;
  
  // Type inference
  DataType inferType() const override;
  
  // Execution
  void compute(Context &ctx) override;
  
  // Serialization
  void serialize(Serializer &s) const override;
  void deserialize(Deserializer &d) override;
};
```

### 2. Register Op

In `src/core/internal_optype.h`:
```cpp
enum class OpType {
  // ... existing ops
  YOUR_OP,
};
```

In op factory:
```cpp
OpBase* createOp(OpType type, const std::string &name) {
  switch (type) {
    // ... existing cases
    case OpType::YOUR_OP:
      return new YourOp(name);
  }
}
```

### 3. Implement Lowering

In `src/builder/lowering/`:

```cpp
class YourOpLowering : public OpLoweringBase {
public:
  void lower(YourOp *op, BuilderContext &ctx) override {
    // Transform high-level op to low-level instructions
    
    // Allocate tensors
    auto input = ctx.getTensor(op->input(0));
    auto output = ctx.allocTensor(op->output(0));
    
    // Generate instructions
    ctx.emit(InstructionType::YOUR_INST, input, output, params);
  }
};
```

## Memory Management

### SRAM Allocation

SRAM is precious - optimize aggressively:

```cpp
// In src/builder/sram_allocator.h
class SRAMAllocator {
  // Allocate tensor in SRAM
  bool allocate(Tensor *tensor, size_t size);
  
  // Free SRAM after last use
  void free(Tensor *tensor);
  
  // Reuse SRAM for non-overlapping tensors
  void reuseMemory();
};
```

### Memory Remapping

Optimize memory layout in `src/builder/mem_remapping.h`:

```cpp
// Reorder memory to improve cache locality
void remapMemory(std::vector<Tensor*> &tensors) {
  // Group tensors by access pattern
  // Minimize memory fragmentation
  // Maximize SRAM utilization
}
```

## Scheduling

### Multi-Core Scheduling

In `src/builder/mc_scheduler.h`:

```cpp
class MCScheduler {
  // Schedule ops across multiple AIPU cores
  void schedule(Graph *graph, int num_cores);
  
  // Handle data dependencies
  void resolveDependencies();
  
  // Balance load across cores
  void balanceLoad();
};
```

### SOMS Scheduling

In `src/builder/soms_scheduler.h`:

```cpp
// Schedule with SOMS (Spilling and Overlapping Memory Scheduler)
class SOMSScheduler {
  // Minimize memory usage while respecting dependencies
  void schedule(Graph *graph);
  
  // Decide when to spill to external memory
  bool shouldSpill(Tensor *tensor);
};
```

## Code Generation

### Assembly Generation

In `src/asm/`:

```cpp
#include "asm/asm.h"

// Generate assembly instruction
void emitInstruction(OpCode opcode, 
                    const Operand &dst,
                    const Operand &src1,
                    const Operand &src2) {
  AsmInstruction inst;
  inst.opcode = opcode;
  inst.operands = {dst, src1, src2};
  emitAsm(inst);
}
```

### Binary Generation

In `src/builder/emitter/code_emitter.h`:

```cpp
class CodeEmitter {
  // Emit binary code for AIPU
  void emit(const Instruction &inst, std::vector<uint8_t> &binary);
  
  // Handle relocations
  void addRelocation(uint32_t offset, const Symbol &symbol);
};
```

## Optimization Passes

### Reorder Optimizer

In `src/builder/reorder_optimizer.h`:

```cpp
// Reorder operations for better performance
class ReorderOptimizer {
  // Move independent ops to fill pipeline bubbles
  void reorderOps(Graph *graph);
  
  // Respect data dependencies
  bool canReorder(Op *op1, Op *op2);
};
```

### SRAM Optimization

In `src/builder/sram_optimize.h`:

```cpp
// Optimize SRAM usage patterns
class SRAMOptimizer {
  // Determine optimal tensor placement
  void optimizePlacement(Graph *graph);
  
  // Minimize SRAM footprint
  void minimizeFootprint();
};
```

## Target-Specific Code

Different AIPU versions have different capabilities:

```cpp
#include "builder/target.h"

void generateCode(Target target) {
  switch (target) {
    case Target::ZY_SC_000:
      // Use X2 features
      generateX2Code();
      break;
    case Target::ZY_SC_003:
      // Use X3 features
      generateX3Code();
      break;
  }
}
```

## TCB (Task Control Block)

In `src/builder/dumper/tcb/`:

```cpp
#include "tcb/tcb.h"

// Generate TCB for AIPU runtime
class TCBGenerator {
  // Initialize grid/group
  void generateGridInit(int grid_id);
  void generateGroupInit(int group_id);
  
  // Generate task
  void generateTask(Task *task);
  
  // Synchronization
  void generateSync(SyncType type);
};
```

## Testing Builder Code

```bash
# Build model
aipugb model.def -w model.bin2 -o output.bin --target ZY_SC_000

# Dump intermediate files
aipugb model.def -w model.bin2 --dump

# This generates:
# - .ro file (read-only data)
# - .text file (code)
# - .data file (data segment)

# Run on simulator with profiling
aipurun model.def -w model.bin2 -i input.bin -p --profile_event 0x01

# Debug with dumped IR
aipugb model.def --dumpIR debug_ir
# Generates debug_ir.def and debug_ir.bin
```

## Common Patterns

### Tiling Large Tensors

```cpp
// Break large tensor into tiles that fit in SRAM
void tileConvolution(ConvOp *op) {
  auto [oh_tile, ow_tile] = computeTileSize(op, sram_size);
  
  for (int oh = 0; oh < output_height; oh += oh_tile) {
    for (int ow = 0; ow < output_width; ow += ow_tile) {
      // Generate code for tile
      emitTileCode(oh, ow, oh_tile, ow_tile);
    }
  }
}
```

### Handling Data Types

```cpp
#include "dtypes/op_data_types.h"

// Use AIPU data types
DataType getDType(Tensor *tensor) {
  // INT8, UINT8, INT16, UINT16, INT32, FP16, etc.
  return tensor->dtype();
}
```

## Debugging

```bash
# Enable debug mode
aipugb model.def -D

# Show all passes
aipugb --show-all-passes

# Disable specific pass
aipugb model.def --disable-pass problematic-pass

# Serialize before/after pass
aipugb model.def --serialize-before-pass my-pass --serialize-after-pass my-pass
```

## References

- `doc/AIPUBuilder_CPP_design.md` - Builder architecture
- `doc/Builder/How_to_write_a_pass.md` - Pass development
- `src/builder/README.md` - Builder component docs
