#include <cuda.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstdint>

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllDialects.h"

using namespace mlir;


static std::string readFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Failed to open " << path << "\n";
    std::exit(1);
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

static void writeFile(const std::string &path, const std::string &data) {
  std::ofstream out(path);
  if (!out) {
    std::cerr << "Failed to write " << path << "\n";
    std::exit(1);
  }
  out << data;
}

static std::string extractAndWrapKernel(StringRef inputMlir, MLIRContext &ctx) {
  DialectRegistry registry;
  registerAllDialects(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(inputMlir, &ctx);
  if (!module) {
    std::cerr << "Error: could not parse MLIR input\n";
    std::exit(1);
  }

  gpu::GPUModuleOp gpuMod;
  for (auto op : module->getOps<gpu::GPUModuleOp>()) {
    gpuMod = op;
    break;
  }
  if (!gpuMod) {
    std::cerr << "Error: no gpu.module found\n";
    std::exit(1);
  }

  LLVM::LLVMFuncOp kernelFunc;
  Block *bodyBlock = gpuMod.getBody();
  if (!bodyBlock) {
    std::cerr << "gpu.module has no body\n";
    std::exit(1);
  }
  for (Operation &op : bodyBlock->getOperations()) {
    if (auto f = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      if (f.getName() == "Add_kernel") {
        kernelFunc = f;
        break;
      }
    }
  }
  if (!kernelFunc) {
    std::cerr << "Error: llvm.func @Add_kernel not found inside gpu.module\n";
    std::exit(1);
  }

  // Create new module and clone only the function
  OpBuilder builder(&ctx);
  ModuleOp newMod = ModuleOp::create(module->getLoc());
  builder.setInsertionPointToStart(newMod.getBody());
  Operation *cloned = builder.clone(*kernelFunc.getOperation());

  // Mark as kernel so we get .visible .entry
  if (auto f = dyn_cast<LLVM::LLVMFuncOp>(cloned)) {
    f->setAttr("nvvm.kernel", UnitAttr::get(&ctx));
  }

  std::string out;
  {
    llvm::raw_string_ostream os(out);
    newMod.print(os);
  }
  return out;
}

static void runCmd(const std::string &cmd) {
  std::cerr << "[running cmd] " << cmd << "\n";
  int code = std::system(cmd.c_str());
  if (code != 0) {
    std::cerr << "Command failed with exit code " << code << "\n";
    std::exit(1);
  }
}


int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " temp.mlir\n";
    return 1;
  }

  // Read the lowered MLIR that contains gpu.module + host code
  std::string fullMlir = readFile(argv[1]);

  // Extract only the device kernel function and wrap in module { … }
  MLIRContext context;
  std::string gpuModuleStr = extractAndWrapKernel(fullMlir, context);
  writeFile("gpu.mlir", gpuModuleStr);

  // NVVM MLIR → LLVM IR
  runCmd(
    "mlir-translate "
    "--mlir-to-llvmir gpu.mlir "
    "-o gpu.ll"
  );

  // LLVM IR → PTX
  runCmd(
    "llc "
    "-march=nvptx64 "
    "-mcpu=sm_70 "
    "gpu.ll "
    "-o gpu.ptx"
  );

  // Load PTX and run via CUDA Driver API
  std::string ptx = readFile("gpu.ptx");

  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) { std::cerr << "cuInit failed\n"; return 1; }

  CUdevice device;
  err = cuDeviceGet(&device, 0);
  if (err != CUDA_SUCCESS) { std::cerr << "cuDeviceGet failed\n"; return 1; }

  CUcontext ctx;
  err = cuCtxCreate(&ctx, 0, device);
  if (err != CUDA_SUCCESS) { std::cerr << "cuCtxCreate failed\n"; return 1; }

  CUmodule module;
  err = cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr);
  if (err != CUDA_SUCCESS) { std::cerr << "cuModuleLoadDataEx failed\n"; return 1; }

  CUfunction kernel;
  err = cuModuleGetFunction(&kernel, module, "Add_kernel");
  if (err != CUDA_SUCCESS) { std::cerr << "cuModuleGetFunction failed\n"; return 1; }

  const int gridDim[3] = {2, 1, 1};
  const int blockDim[3] = {2, 1, 1};
  size_t N = gridDim[0] * blockDim[0];
  std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N, 0.0f);
  CUdeviceptr dA, dB, dC;
  cuMemAlloc(&dA, N * sizeof(float));
  cuMemAlloc(&dB, N * sizeof(float));
  cuMemAlloc(&dC, N * sizeof(float));
  cuMemcpyHtoD(dA, A.data(), N * sizeof(float));
  cuMemcpyHtoD(dB, B.data(), N * sizeof(float));

  int64_t grid_offset = 0, block_offset = 0;
  // grid dim offset, block dim offset, inp1 ptr, inp2 ptr, out ptr
  void *args[] = { &grid_offset, &block_offset, &dA, &dB, &dC }; 

  err = cuLaunchKernel(
    kernel,
    gridDim[0], gridDim[1], gridDim[2],
    blockDim[0], blockDim[1], blockDim[2],
    0, nullptr,
    args, nullptr
  );
  if (err != CUDA_SUCCESS) { std::cerr << "cuLaunchKernel failed\n"; return 1; }

  cuCtxSynchronize();
  cuMemcpyDtoH(C.data(), dC, N * sizeof(float));

  std::cout << "Result C:\n";
  for (size_t i = 0; i < N; ++i)
    std::cout << "C[" << i << "] = " << C[i] << "\n";

  cuMemFree(dA);
  cuMemFree(dB);
  cuMemFree(dC);
  cuModuleUnload(module);
  cuCtxDestroy(ctx);
  return 0;
}
