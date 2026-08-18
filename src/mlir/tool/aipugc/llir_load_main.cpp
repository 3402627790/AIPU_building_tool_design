#include "aipugc/Dialect.h"
#include "aipugc/LLIRLoader.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace cl = llvm::cl;
using namespace aipubt;

static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input .ll file>"),
    cl::init("src/mlir/isNanOp/kernel/isnan_x3u_kernel_obj.ll"),
    cl::value_desc("llir filename"));

static cl::opt<bool> dumpMLIR(
    "dump-mlir",
    cl::desc("Print the imported MLIR LLVM dialect module"),
    cl::init(false));

int main(int argc, const char **argv)
{
    cl::ParseCommandLineOptions(argc, argv, "AIPUGC LLIR loader\n");

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::aipugc::AIPUDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    mlir::OwningOpRef<mlir::ModuleOp> module = aipugc::loadLLIRFile(context, inputFilename);
    if (!module)
    {
        llvm::errs() << "Failed to load LLIR file: " << inputFilename << "\n";
        return 1;
    }

    aipugc::printLLIRLoadSummary(*module, llvm::outs());

    if (dumpMLIR)
    {
        llvm::outs() << "\nImported MLIR module:\n";
        module->print(llvm::outs(), mlir::OpPrintingFlags().elideLargeElementsAttrs(32));
        llvm::outs() << "\n";
    }

    return 0;
}
