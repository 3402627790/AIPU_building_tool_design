//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------

#include "aipugc/LLIRTool.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>

namespace aipubt
{
namespace aipugc
{

static constexpr const char *kPriorityAttrName = "cpsPriority";
static constexpr const char *kPriorityMLIRAttrName = "cps.priority";

mlir::OwningOpRef<mlir::ModuleOp> loadLLIRFile(mlir::MLIRContext &context,
                                               llvm::StringRef filename,
                                               const KernelPriorityMap &priorities,
                                               int defaultPriority)
{
    llvm::LLVMContext llvmContext;
    llvm::SMDiagnostic diagnostic;
    // parseIRFile accepts both textual IR (.ll) and bitcode (.bc);
    // it detects the format from the file contents
    std::unique_ptr<llvm::Module> llvmModule = llvm::parseIRFile(filename, diagnostic, llvmContext);
    if (!llvmModule)
    {
        diagnostic.print("aipugc-llir", llvm::errs());
        return {};
    }

    // Explicitly named functions must exist in the module and be kernel functions.
    bool error = false;
    for (const auto &Entry : priorities)
    {
        llvm::Function *F = llvmModule->getFunction(Entry.getKey());
        if (!F)
        {
            llvm::errs() << "error: kernel function '" << Entry.getKey()
                         << "' not found in module\n";
            error = true;
            continue;
        }
        if (F->getCallingConv() != llvm::CallingConv::Zhouyi_Kernel)
        {
            llvm::errs() << "error: function '" << Entry.getKey()
                         << "' is not a zhouyi_kernel function\n";
            error = true;
        }
    }
    if (error)
        return {};

    // Attach cpsPriority as a string function attribute. Only zhouyi_kernel
    // functions carry a priority: explicit entries win, the rest get the
    // default priority.
    for (llvm::Function &F : *llvmModule)
    {
        if (F.getCallingConv() != llvm::CallingConv::Zhouyi_Kernel)
            continue;

        auto It = priorities.find(F.getName());
        int value = (It != priorities.end()) ? It->second : defaultPriority;
        F.addFnAttr(kPriorityAttrName, std::to_string(value));
    }

    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::translateLLVMIRToModule(
        std::move(llvmModule), &context, false, false, true, true);
    if (!module)
        return {};

    // Expose the priority as a first-class i32 attribute on each LLVMFuncOp.
    module->walk([&](mlir::LLVM::LLVMFuncOp funcOp) {
        std::optional<mlir::ArrayAttr> passthrough = funcOp.getPassthrough();
        if (!passthrough)
            return;
        for (mlir::Attribute Attr : *passthrough)
        {
            auto Pair = mlir::dyn_cast<mlir::ArrayAttr>(Attr);
            if (!Pair || Pair.size() != 2)
                continue;
            auto Key = mlir::dyn_cast<mlir::StringAttr>(Pair[0]);
            if (!Key || Key.getValue() != kPriorityAttrName)
                continue;
            auto Val = mlir::dyn_cast<mlir::StringAttr>(Pair[1]);
            if (!Val)
                continue;
            int Priority = 0;
            if (Val.getValue().getAsInteger(10, Priority))
                continue;
            funcOp->setAttr(kPriorityMLIRAttrName,
                            mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32),
                                                   Priority));
            break;
        }
    });

    return module;
}

// Lower the LLVM IR module to assembly text for the given target triple.
static std::unique_ptr<llvm::Module> compileLLIRToAssembly(llvm::StringRef inputFilename,
                                                           llvm::LLVMContext &llvmContext,
                                                           llvm::SmallVectorImpl<char> &asmBuffer,
                                                           const LLIRCompileOptions &options)
{
    llvm::SMDiagnostic diagnostic;
    // Accept both textual IR (.ll) and bitcode (.bc) inputs.
    std::unique_ptr<llvm::Module> module = llvm::parseIRFile(inputFilename, diagnostic, llvmContext);
    if (!module)
    {
        diagnostic.print("aipugc-llir", llvm::errs());
        return nullptr;
    }

    llvm::Triple triple(options.triple);
    std::string error;
    const llvm::Target *theTarget =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (!theTarget)
    {
        llvm::errs() << "error: " << error << "\n";
        return nullptr;
    }

    llvm::CodeGenOptLevel optLevel;
    switch (options.optLevel)
    {
    case 0:
        optLevel = llvm::CodeGenOptLevel::None;
        break;
    case 1:
        optLevel = llvm::CodeGenOptLevel::Less;
        break;
    case 3:
        optLevel = llvm::CodeGenOptLevel::Aggressive;
        break;
    default:
        optLevel = llvm::CodeGenOptLevel::Default;
        break;
    }

    llvm::TargetOptions targetOptions;
    targetOptions.FunctionSections = options.functionSections;
    std::unique_ptr<llvm::TargetMachine> targetMachine(theTarget->createTargetMachine(
        triple, options.cpu, options.features, targetOptions, std::nullopt,
        std::nullopt, optLevel));
    if (!targetMachine)
    {
        llvm::errs() << "error: failed to create target machine for '" << options.triple << "'\n";
        return nullptr;
    }

    llvm::raw_svector_ostream os(asmBuffer);
    llvm::legacy::PassManager passManager;
    llvm::MachineModuleInfoWrapperPass *mmiwp =
        new llvm::MachineModuleInfoWrapperPass(targetMachine.get());
    if (targetMachine->addPassesToEmitFile(
            passManager, os, /*DwoOut=*/nullptr,
            llvm::CodeGenFileType::AssemblyFile, /*DisableVerify=*/true, mmiwp))
    {
        llvm::errs() << "error: target '" << options.triple
                     << "' does not support assembly file emission\n";
        return nullptr;
    }

    passManager.run(*module);
    return module;
}

// Assemble assembly text into an ELF relocatable object file.
static bool assembleToObject(llvm::StringRef assembly, llvm::StringRef outputFilename,
                             const LLIRCompileOptions &options)
{
    llvm::Triple triple(options.triple);
    std::string error;
    const llvm::Target *theTarget =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (!theTarget)
    {
        llvm::errs() << "error: " << error << "\n";
        return false;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(assembly, "llir_assembly", /*RequiresNullTerminator=*/false),
        llvm::SMLoc());

    llvm::MCTargetOptions mcOptions;
    std::unique_ptr<llvm::MCRegisterInfo> mri(theTarget->createMCRegInfo(options.triple));
    if (!mri)
    {
        llvm::errs() << "error: unable to create target register info\n";
        return false;
    }
    std::unique_ptr<llvm::MCAsmInfo> mai(
        theTarget->createMCAsmInfo(*mri, options.triple, mcOptions));
    if (!mai)
    {
        llvm::errs() << "error: unable to create target asm info\n";
        return false;
    }
    std::unique_ptr<llvm::MCSubtargetInfo> sti(
        theTarget->createMCSubtargetInfo(options.triple, options.cpu, options.features));
    if (!sti)
    {
        llvm::errs() << "error: unable to create subtarget info\n";
        return false;
    }

    llvm::MCContext mcContext(triple, mai.get(), mri.get(), sti.get(), &sourceMgr,
                              &mcOptions);
    std::unique_ptr<llvm::MCObjectFileInfo> mofi(
        theTarget->createMCObjectFileInfo(mcContext, /*PIC=*/false));
    mcContext.setObjectFileInfo(mofi.get());

    std::unique_ptr<llvm::MCInstrInfo> mcii(theTarget->createMCInstrInfo());
    if (!mcii)
    {
        llvm::errs() << "error: unable to create instruction info\n";
        return false;
    }

    std::error_code ec;
    llvm::raw_fd_ostream out(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec)
    {
        llvm::errs() << "error: could not open '" << outputFilename << "': "
                     << ec.message() << "\n";
        return false;
    }

    llvm::MCCodeEmitter *codeEmitter = theTarget->createMCCodeEmitter(*mcii, mcContext);
    llvm::MCAsmBackend  *asmBackend  = theTarget->createMCAsmBackend(*sti, *mri, mcOptions);
    std::unique_ptr<llvm::MCStreamer> streamer(theTarget->createMCObjectStreamer(
        triple, mcContext, std::unique_ptr<llvm::MCAsmBackend>(asmBackend),
        asmBackend->createObjectWriter(out),
        std::unique_ptr<llvm::MCCodeEmitter>(codeEmitter), *sti));
    if (!streamer)
    {
        llvm::errs() << "error: unable to create object streamer\n";
        return false;
    }

    std::unique_ptr<llvm::MCAsmParser> parser(
        llvm::createMCAsmParser(sourceMgr, mcContext, *streamer, *mai));
    std::unique_ptr<llvm::MCTargetAsmParser> targetParser(
        theTarget->createMCAsmParser(*sti, *parser, *mcii, mcOptions));
    if (!targetParser)
    {
        llvm::errs() << "error: target '" << options.triple
                     << "' does not support assembly parsing\n";
        return false;
    }

    parser->setTargetParser(*targetParser);
    if (parser->Run(/*NoInitialTextSection=*/false))
    {
        llvm::errs() << "error: assembly of '" << options.triple << "' failed\n";
        return false;
    }

    // streamer->finish();  // to avoid rewrite twice.
    out.flush();
    return true;
}

bool compileLLIRToObject(llvm::StringRef inputFilename, llvm::StringRef outputFilename,
                         const LLIRCompileOptions &options)
{
    // Register the Zhouyi target once so the backend and MC components are
    // available. The declarations are generated by TargetSelect.h via the
    // Targets.def / AsmParsers.def / AsmPrinters.def files.
    static bool initialized = false;
    if (!initialized)
    {
        LLVMInitializeZhouyiTargetInfo();
        LLVMInitializeZhouyiTarget();
        LLVMInitializeZhouyiTargetMC();
        LLVMInitializeZhouyiAsmPrinter();
        LLVMInitializeZhouyiAsmParser();
        initialized = true;
    }

    llvm::LLVMContext llvmContext;
    llvm::SmallVector<char, 0> asmBuffer;
    if (!compileLLIRToAssembly(inputFilename, llvmContext, asmBuffer, options))
        return false;

    return assembleToObject(llvm::StringRef(asmBuffer.data(), asmBuffer.size()), outputFilename, options);
}

void printLLIRLoadSummary(mlir::ModuleOp module, llvm::raw_ostream &os)
{
    unsigned funcCount = 0;
    os << "Loaded LLIR into MLIR module successfully.\n";
    os << "Functions:\n";
    module.walk([&](mlir::LLVM::LLVMFuncOp func) {
        ++funcCount;
        os << "  [" << funcCount << "] @" << func.getName() << " : ";
        mlir::Type type = func.getFunctionType();
        type.print(os);
        mlir::IntegerAttr priority =
            func->getAttrOfType<mlir::IntegerAttr>(kPriorityMLIRAttrName);
        if (priority)
            os << " priority=" << priority.getInt();
        else
            os << " priority=N/A";
        os << "\n";
    });
    os << "Total LLVM functions: " << funcCount << "\n";
}

} // namespace aipugc
} // namespace aipubt
