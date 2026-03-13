//===- MLIRGen.cpp - MLIR Generation from a Compass IR----------------------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Compass IR.
//
//===----------------------------------------------------------------------===//

#include "aipugc/MLIRGen.h"
#include "aipugc/Dialect.h"
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;
namespace {
using namespace mlir::aipugc;
using namespace mlir::aipugc::utils;
/// A variable type with shape information.
struct VarType {
  std::vector<int64_t> shape;
};

/// Implementation of a simple MLIR emission from the Compass IR.
///
/// This will emit operations that are specific to the AIPU, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  /// Public API: convert the Compass IR (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(aipubt::GraphPtr g) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    // TODO: graph as main function, subgraph as a function.

    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(theModule.getBody());
    auto location = mlir::NameLoc::get(builder.getStringAttr(""));

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type> inputTypes;
    for (auto input : g->get_input_tensors()) {
      inputTypes.push_back(getType(builder, input));
    }
    llvm::SmallVector<mlir::Type> outputTypes;
    for (auto output : g->get_output_tensors()) {
      outputTypes.push_back(getType(builder, output));
    }
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);

    mlir::aipugc::FuncOp function =
        builder.create<mlir::aipugc::FuncOp>(location, g->name(), funcType);
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();

    // Declare all the function arguments in the symbol table.
    for (size_t i = 0; i < g->get_input_tensors().size(); ++i) {
      if (failed(declare(g->get_input_tensors()[i]->name(),
                         entryBlock.getArguments()[i])))
        return nullptr;
    }

    builder.setInsertionPointToStart(&entryBlock);
    for (auto &n : g->nodes()) {
      if (failed(mlirGen(n))) {
        theModule.emitError("node Gen error");
        return nullptr;
      }
    }

    std::vector<mlir::Value> outputValues;
    for (auto output : g->get_output_tensors()) {
      auto outputValue = MLIRGenBase::getsymbol(symbolTable, output->name());
      if (outputValue != nullptr) {
        outputValues.push_back(outputValue);
      }
    }
    if (outputValues.empty()) {
      builder.create<ReturnOp>(location, ArrayRef<mlir::Value>());
    } else {
      builder.create<ReturnOp>(location, ArrayRef<mlir::Value>(outputValues));
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the AIPU operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  //  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;
  std::map<std::string, mlir::Value> symbolTable;

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  llvm::LogicalResult declare(std::string var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(std::make_pair(var, value));
    return mlir::success();
  }

  mlir::Location loc(const aipubt::NodePtr &node) {
    auto nodeLoc = node->loc()->to_string();
    if (nodeLoc.length() > 2)
      nodeLoc = nodeLoc.substr(1, nodeLoc.length() - 2);
    return mlir::NameLoc::get(builder.getStringAttr(nodeLoc));
  }

  /// Emit a new function and add it to the MLIR module.
  llvm::LogicalResult mlirGen(const aipubt::NodePtr &node) {
    // Create a scope in the symbol table to hold variable declarations.
    // ScopedHashTableScope<std::string, mlir::Value> varScope(symbolTable);
    auto location = loc(node);
    std::vector<mlir::Value> outputs;
    if (node->type() == aipubt::OpType_Input) {
      return mlir::success();
    }
    if (!MLIRGenRegistry::get(node->type())) {
      return mlir::failure();
    }
    MLIRGenBase *mlirGenClass = MLIRGenRegistry::get(node->type());
    outputs = mlirGenClass->mlirGen(builder, node, symbolTable);

    if (outputs.size() != node->outputs().size()) {
      llvm::errs() << "MLIRGen output count mismatch for node " << node->name()
                   << ": expected " << node->outputs().size()
                   << ", got " << outputs.size() << "\n";
      return mlir::failure();
    }

    // Declare all the function arguments in the symbol table.
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      if (failed(declare(node->outputs()[i]->name(), outputs[i])))
        return mlir::failure();
    }
    return mlir::success();
  }
};

} // namespace

namespace aipugc {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          aipubt::GraphPtr g) {
  return MLIRGenImpl(context).mlirGen(g);
}

} // namespace aipugc
