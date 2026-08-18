#include "aipugc/LLIRLoader.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>
#include <utility>

namespace aipubt
{
namespace aipugc
{

static std::unique_ptr<llvm::MemoryBuffer> readLLIR(llvm::StringRef filename)
{
    auto bufferOrError = llvm::MemoryBuffer::getFile(filename);
    if (!bufferOrError)
    {
        llvm::errs() << "Failed to read LLIR file: " << filename << ": " << bufferOrError.getError().message()
                     << "\n";
        return nullptr;
    }

    return std::move(*bufferOrError);
}

mlir::OwningOpRef<mlir::ModuleOp> loadLLIRFile(mlir::MLIRContext &context, llvm::StringRef filename)
{
    std::unique_ptr<llvm::MemoryBuffer> buffer = readLLIR(filename);
    if (!buffer)
    {
        return {};
    }

    llvm::LLVMContext llvmContext;
    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> llvmModule = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, llvmContext);
    if (!llvmModule)
    {
        diagnostic.print("aipugc-llir-load", llvm::errs());
        return {};
    }

#if LLVM_VERSION_MAJOR >= 19
    return mlir::translateLLVMIRToModule(std::move(llvmModule), &context,
                                         /*emitExpensiveWarnings=*/false,
                                         /*dropDICompositeTypeElements=*/false,
                                         /*loadAllDialects=*/true,
                                         /*preferUnregisteredIntrinsics=*/true);
#else
    return mlir::translateLLVMIRToModule(std::move(llvmModule), &context);
#endif
}

void printLLIRLoadSummary(mlir::ModuleOp module, llvm::raw_ostream &os)
{
    unsigned funcCount = 0;
    os << "Loaded LLIR into MLIR module successfully.\n";
    os << "Functions:\n";
    module.walk([&](mlir::LLVM::LLVMFuncOp func) {
        ++funcCount;
        os << "  [" << funcCount << "] @" << func.getName() << " : ";
        mlir::Type(func.getFunctionType()).print(os);
        os << "\n";
    });
    os << "Total LLVM functions: " << funcCount << "\n";
}

} // namespace aipugc
} // namespace aipubt
