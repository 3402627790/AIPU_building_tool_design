---
name: aipu-builder-context
description: Core context and patterns for AIPUBuilder compiler development
---

# AIPUBuilder Project Context

## What is AIPUBuilder

AIPUBuilder is an end-to-end graph compiler for ARM China AIPU (AI Processing Unit) that compiles deep learning models to executable binaries. It supports TensorFlow, TFLite, Caffe, and ONNX.

**Three-stage pipeline:**
1. **Parser** (Front-end): Parse 3rd party models into AIPU IR
2. **Quantization** (Middle-end): Quantize float32 models to int8
3. **Builder** (Back-end): Optimize and compile to AIPU executable

## Project Architecture

```
src/
  asm/         - Assembly generation
  builder/     - Backend compiler (optimizer, scheduler, allocator)
  core/        - Core IR and operations
  mlir/        - MLIR-based transformation passes
python/
  AIPUBuilder/ - Python frontend (parser, quantization tool)
modules/
  runtime/     - AIPU runtime
  btrt/        - Binary runtime
doc/           - Architecture and development guides
```

## Key Components

### MLIR Infrastructure
- Dialect: `aipugc` - AIPU-specific MLIR dialect
- Passes are in `src/mlir/include/aipugc/Passes.h`
- Lowering: MLIR → AIPU IR in `src/builder/lowering/`

### Builder Pipeline
- **Scheduler**: `mc_scheduler.h`, `soms_scheduler.h` - Instruction scheduling
- **Allocator**: `mmu_allocator.h`, `sram_allocator.h` - Memory allocation
- **Optimizer**: `optimizer.h`, `reorder_optimizer.h` - Graph optimization
- **Emitter**: `code_emitter.h` - Code generation

### Quantization
- Located in `python/src/AIPUBuilder/Optimizer/`
- Supports SYMMETRIC and ASYMMETRIC quantization
- Per-channel quantization for specific ops

## Coding Conventions

### C++ Code
- Follow `.clang-format` in project root
- Header guards: `#ifndef`, not `#pragma once`
- Namespace: Use project-specific namespaces
- MLIR passes: Implement pass interfaces, register in `Passes.cpp`

### Python Code
- Follow AI SW coding style guidelines
- Use type hints where applicable
- Config files use INI format

### Pass Development
- See `doc/Builder/How_to_write_a_pass.md`
- See `doc/parser/how_to_write_a_pass.md`
- Passes must be registered in dialect
- Test with `--serialize-before-pass` and `--serialize-after-pass` flags

## Common Targets

Hardware targets (AIPU versions):
- `ZY_SC_000` (default)
- `ZY_SC_001`
- `ZY_SC_002`
- `ZY_SC_003`

Specify with `--target` flag when building.

## Build System

- **CMake-based**: `CMakeLists.txt` at root
- **Environment setup**: `env_setup.sh` or `env_setup.csh`
- **Python package**: Distributed as wheel (`.whl`)

## Testing

- C++ tests: `test/` directory
- Run tests after pass modifications
- Validate with simulator before committing

## Common Commands

```bash
# Parser
aipuparser -c config.cfg

# Quantization
aipuaqt --cfg config.cfg --graph model.def --bin model.bin2

# Builder (Python)
aipugb model.def -w model.bin2 -o output.bin --target ZY_SC_000

# Builder (C++)
aipugb model.def --weight model.bin2 --output output.bin

# Run on simulator
aipurun model.def -w model.bin2 -i input.bin --simulator /path/to/sim
```

## When Working On...

### Adding new operations
1. Define op in IR (`src/core/`)
2. Add quantization support (`python/src/AIPUBuilder/Optimizer/ops/`)
3. Implement lowering pass if needed
4. Add builder support (`src/builder/ops/`)
5. Update tests

### Writing optimization passes
1. Create pass in `src/mlir/` or `src/builder/`
2. Register in appropriate `Passes.cpp`
3. Add to pass pipeline
4. Test with `--serialize-before-pass` / `--serialize-after-pass`

### Debugging quantization issues
1. Enable logging: `--log` flag
2. Dump intermediate results: `--dump --dump_dir output/`
3. Check statistics: `ts_max_file` and `ts_min_file`
4. Visualize with `--png graph.png`

## Important Files

- `doc/AIPU_building_tool_dev_guide.md` - Developer guide
- `doc/AIPUBuilder_CPP_design.md` - C++ builder design
- `doc/AIPUBuilder_user_guide.md` - User guide
- `doc/introduction.md` - Project introduction

## Memory Patterns

- **IR**: Use internal representation, not framework-specific format
- **Quantization**: Statistics-based, calibration dataset required
- **Allocation**: Two-level (SRAM + external memory)
- **Scheduling**: Consider data dependencies and hardware constraints
