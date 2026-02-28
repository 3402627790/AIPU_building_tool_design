//===- MLIRGenRegister.h - MLIR Generation function 'mlirGen'from a Compass Node
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface for registering the MLIR generation
// function 'mlirGen' for each AIPU OP.
//
//===----------------------------------------------------------------------===//
#include "aipugc/Dialect.h"
#include "aipugc/MLIRGen.h"
#include "aipugc/utils.h"
#include "llvm/ADT/ScopedHashTable.h"

#ifndef AIPUGC_MLIRGEN_REGISTER_H
#define AIPUGC_MLIRGEN_REGISTER_H
namespace mlir {
namespace aipugc {

class MLIRGenBaseTest {
public:
  MLIRGenBaseTest() {}
  virtual ~MLIRGenBaseTest() = default;
};

class MLIRGenBase {
public:
  MLIRGenBase() {}
  virtual ~MLIRGenBase() = default;
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) = 0;

  /// Helper conversion for a Compass IR location to an MLIR location.
  static mlir::Location loc(mlir::OpBuilder &builder,
                            const aipubt::NodePtr &node) {
    // vitual location
    auto nodeLoc = node->loc()->to_string();
    if (nodeLoc.length() > 2)
      nodeLoc = nodeLoc.substr(1, nodeLoc.length() - 2);
    return mlir::NameLoc::get(builder.getStringAttr(nodeLoc));
  }

  static mlir::Value getsymbol(std::map<std::string, mlir::Value> &symbolTable,
                               std::string symbolName) {
    if (symbolTable.count((symbolName)))
      return symbolTable[symbolName];
    return nullptr;
  }

protected:
  Attribute ParamsToAttribute(mlir::OpBuilder &builder,
                              const aipubt::NodeParamValue &paramValue);
  DictionaryAttr ParamsToMlirAttr(mlir::OpBuilder &builder,
                                  aipubt::NodePtr node,
                                  std::set<std::string> uniqueKeys = {});
  mlir::Value
  ConstantTensorMlirGen(mlir::OpBuilder &builder, mlir::Location loc,
                        aipubt::TensorPtr tensor,
                        std::map<std::string, mlir::Value> &symbolTable);
};

class MLIRGenRegistry {
public:
  typedef std::map<aipubt::OpType, MLIRGenBase *> MLIRGenMap;

  static MLIRGenMap &getMLIRGenMapEntry() {
    static MLIRGenMap registry;
    return registry;
  }

  static MLIRGenBase *get(const aipubt::OpType type) {
    MLIRGenMap &registered = getMLIRGenMapEntry();

    if ((registered).count(type) == 0) {
      llvm::errs() << "The aipu op type: " << aipubt::optype_to_string(type)
                   << " don't supported.\n";
      return nullptr;
    }

    return registered[type];
  }

  static void registerType(const aipubt::OpType &aipuOpType,
                           MLIRGenBase *registerClass) {
    MLIRGenMap &registered = getMLIRGenMapEntry();
    if (registered.count(aipuOpType) != 0) {
      llvm::errs() << "The aipu op type: "
                   << aipubt::optype_to_string(aipuOpType)
                   << " has registered to MLIRGen map.\n";
      return;
    }
    registered[aipuOpType] = registerClass;
  }

private:
  MLIRGenRegistry() {}
};

class MLIRGenRegisterer {
public:
  MLIRGenRegisterer(const aipubt::OpType &aipuOpType,
                    MLIRGenBase *registerClass) {
    MLIRGenRegistry::registerType(aipuOpType, registerClass);
  }
};

#define REGISTER_MLIRGEN(classType)                                            \
  static MLIRGen##classType classType##Register;                               \
  static MLIRGenRegisterer creator##classType(aipubt::OpType_##classType,      \
                                              &classType##Register)

#define REGISTER_OP_MLIRGEN(aipuOpType, classType)                             \
  static MLIRGen##classType classType##Register;                               \
  static MLIRGenRegisterer __attribute__((used)) creator##classType(           \
      aipuOpType, &classType##Register)

} // namespace aipugc
} // namespace mlir
#endif // AIPUGC_MLIRGEN_REGISTER_H