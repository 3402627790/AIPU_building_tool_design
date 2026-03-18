#include "Plugin/PluginLoader.h"
#include "Target/Compass/CompassEmitter.h"
#include "aipugc/Dialect.h"
#include "aipugc/MLIRGen.h"
#include "aipugc/Passes.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <tools/argparser.h>
#include <tools/common_options.h>
#include <utils/utils.h>
#include "llvm/Support/FileUtilities.h"

// MLIR core infrastructure
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

// MLIR dialect definitions
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"

using namespace aipubt;
namespace cl = llvm::cl;

enum Action {
  None,
  DumpAIPUGC,
  DumpTosa,
  DumpLinalg,
  DumpAffine,
  DumpCPU,
  DumpCUDA,
  DumpVulkan,
  DumpLL,
  RunJIT
};

static cl::opt<enum Action> emitAction(
  "emit",
  cl::desc("Specify the compilation stage to output."),
  cl::values(clEnumValN(DumpAIPUGC, "aipugc", "output the optimized AIPUGC-level MLIR")),
  cl::values(clEnumValN(DumpTosa, "tosa", "output the TOSA-level MLIR")),
  cl::values(clEnumValN(DumpLinalg, "linalg", "output the Linalg-level MLIR")),
  cl::values(clEnumValN(DumpAffine, "affine", "output the Affine-Level MLIR")),
  cl::values(clEnumValN(DumpCPU, "cpu", "backend to cpu")),
  cl::values(clEnumValN(DumpCUDA, "cuda", "backend to cuda")),
  cl::values(clEnumValN(DumpVulkan, "vulkan", "backend to vulkan")),
  cl::values(clEnumValN(DumpLL, "ll", "output the LLVMIR")),
  cl::values(clEnumValN(RunJIT, "jit", "JIT and run the code"))
);

static cl::opt<std::string> graphFilename(
  cl::Positional,
  cl::desc("<input graph.def file>"),
  cl::init("-"),
  cl::value_desc("graph.def filename")
);

static cl::opt<std::string> weightFilename(
  "w",
  cl::desc("<input weight.bin file>"),
  cl::value_desc("weight.bin filename")
);

static cl::opt<bool> enableTransToCompass(
  "mlir-to-compass",
  cl::desc("Enable Translate to Compass")
);

static cl::opt<std::string> dump(
  "dump",
  cl::desc("<save mlir file name>"),
  cl::value_desc("save mlir file name")
);

int main(int argc, const char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Compass IR compiler\n");

  auto graphFile = llvm::StringRef(graphFilename);
  auto weightFile = llvm::StringRef(weightFilename);
  if (!utils::file_exists(graphFile.str())) {
    ERROR() << "Cannot Open file:" << graphFile.str();
    return -1;
  }
  GraphPtr g = Graph::parse(graphFile.str());
  if (!g) {
    ERROR() << "The file:" << graphFile.str() << " parse failed!";
    return -1;
  }
  if (utils::file_exists(weightFile.str())) {
    g->load_weight(weightFile.str());
  }

  // Register all upstream dialects and extensions.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();

  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<mlir::aipugc::AIPUDialect>();
  context.appendDialectRegistry(registry);

  // Get mlir module from context
  mlir::OwningOpRef<mlir::ModuleOp> module = aipugc::mlirGen(context, g);
  if (!module) {
    llvm::errs() << "Error: MLIR generation or module verification failed.\n";
    return 1;
  }

  if (emitAction != Action::None) {
    mlir::PassManager pm(module.get()->getName());
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
      return 4;
    }

    bool isLoweringToTOSA = emitAction >= Action::DumpTosa;
    bool isLoweringToLinalg = emitAction >= Action::DumpLinalg;
    bool isLoweringToAffine = emitAction >= Action::DumpAffine;
    // Backend Targets
    bool isCpuPath = emitAction == Action::DumpCPU;         // CPU Path: Affine -> LLVM
    bool isCudaPath = emitAction == Action::DumpCUDA;       // CUDA Path: Affine -> NVVM -> PTX -> LLVM
    bool isVulkanPath = emitAction == Action::DumpVulkan;   // Vulkan Path: Affine -> GPU -> SPIRV
    bool isRunJit = emitAction >= Action::RunJIT;

    mlir::OpPassManager &optPM = pm.nest<mlir::aipugc::FuncOp>();
    mlir::aipugc::AipuApplyPassPlugin(optPM);
    optPM.addPass(mlir::aipugc::createShapeInferencePass());
    optPM.addPass(mlir::aipugc::createDCEPass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    if (isLoweringToTOSA) {
      pm.addPass(mlir::aipugc::createLowerToTOSAPass());
      pm.addPass(mlir::createCanonicalizerPass());
    }

    if (isLoweringToLinalg) {
      pm.addPass(mlir::aipugc::createConvertTosaToLinalg());
      pm.addPass(mlir::createCanonicalizerPass());
    }

    if (isLoweringToAffine) {
      mlir::bufferization::OneShotBufferizePassOptions options;
      options.bufferizeFunctionBoundaries = true;
      options.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::IdentityLayoutMap;     
      pm.addPass(mlir::bufferization::createOneShotBufferizePass(options)); // eliminate unknown stride and offset
      // pm.addPass(mlir::bufferization::createOneShotBufferizePass()); // with dynamic input layout

      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    }

    if (isCpuPath) {
      /* isLoweringToLLVM */ {
        pm.addPass(mlir::aipugc::createConvertAffineToLLVM());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      }
    }

    if (isCudaPath) {
      /* isLoweringToNVVM */ {
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        pm.addPass(mlir::memref::createNormalizeMemRefsPass());

        // pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps());
        mlir::ConvertGpuOpsToNVVMOpsOptions options;
        options.indexBitwidth = 0;
        options.useBarePtrCallConv = true; // pack multi pointer
        pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps(options));

        mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
        nvvmTargetOptions.triple = "nvptx64-nvidia-cuda";
        nvvmTargetOptions.chip = "sm_70";
        nvvmTargetOptions.features = "+sm_70";
        pm.addPass(mlir::createGpuNVVMAttachTarget(nvvmTargetOptions));
        pm.addPass(mlir::createCanonicalizerPass());
      }

      /* isLoweringToPTX */ {
        pm.addPass(mlir::createConvertNVVMToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        mlir::GpuToLLVMConversionPassOptions options;
        options.hostBarePtrCallConv = true; // pack multi pointer
        options.kernelBarePtrCallConv = true; // pack multi pointer
        pm.addPass(mlir::createGpuToLLVMConversionPass(options));
        pm.addPass(mlir::createCanonicalizerPass());
      }

      /* isLoweringToLLVM */ {
        pm.addPass(mlir::aipugc::createConvertAffineToLLVM());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      }
    }

    if (isVulkanPath) {
      /* isLoweringToGPU */ {
        pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createAffineLoopInvariantCodeMotionPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertAffineForToGPUPass());
        pm.addPass(mlir::createGpuKernelOutliningPass());
        pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createGpuDecomposeMemrefsPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::aipugc::createSimplifyGPULaunchPass());
      }

      /* isLoweringToSPIRV */ {
        context.loadDialect<mlir::gpu::GPUDialect>();
        context.loadDialect<mlir::spirv::SPIRVDialect>();

        mlir::spirv::Capability caps[] = {
          mlir::spirv::Capability::Shader,
        };
        mlir::spirv::Extension exts[] = {
          mlir::spirv::Extension::SPV_KHR_storage_buffer_storage_class,
        };

        auto triple = mlir::spirv::VerCapExtAttr::get(
          mlir::spirv::Version::V_1_0,
          caps,
          exts,
          &context
        );

        auto maxWorkgroupSize = mlir::ArrayAttr::get(&context, {
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), 1024),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), 1024),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32), 64)
        });

        auto limits = mlir::spirv::ResourceLimitsAttr::get(
          &context,
          32768,  // maxComputeSharedMemorySize
          1024,   // maxComputeWorkgroupInvocations
          maxWorkgroupSize, // maxComputeWorkgroupSize
          32,     // subgroupSize
          1,      // minSubgroupSize
          32,     // maxSubgroupSize
          nullptr, // cooperativeMatrixPropertiesKHR
          nullptr  // cooperativeMatrixPropertiesNV
        );

        auto targetEnv = mlir::spirv::TargetEnvAttr::get(
          triple,
          limits,
          mlir::spirv::ClientAPI::Vulkan,
          mlir::spirv::Vendor::Unknown,
          mlir::spirv::DeviceType::Unknown,
          0       // deviceID
        );

        module->getOperation()->setAttr("spirv.target_env", targetEnv);
        pm.addPass(mlir::aipugc::createSetSPIRVABIAttrsPass());
        // mlir::aipugc::buildTestSPIRVCPURunnerPipeline(pm);
        mlir::aipugc::buildTestVulkanRunnerPipeline(pm);
      }
    }


    if (mlir::failed(pm.run(*module))) {
      llvm::errs() << "Error: Pass manager failed to execute.\n";
      return 4;
    }
  }

  module->dump();

  if (emitAction == Action::DumpLL) {
    dumpLLVMIR(*module, dump);
  }

  if (emitAction == Action::RunJIT) {
    runJit(*module);
  }

  if (!dump.empty() && emitAction != Action::DumpLL) {
    llvm::StringRef fileName(dump);
    if (llvm::sys::fs::exists(fileName)) {
      llvm::errs() << "Info: Overwriting existing file '" << fileName << "'\n";
    }
    std::error_code ec;
    llvm::raw_fd_ostream os(fileName, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Error: Could not open '" << fileName << "': " << ec.message() << "\n";
      return false;
    }
    module->print(os);
  }

  if (enableTransToCompass) {
    if (mlir::failed(mlir::compass::translateToCompass(module)))
      return 4;
  }

  return 0;
}