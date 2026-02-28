#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenSquaredDifference : public MLIRGenBase {
public:
  MLIRGenSquaredDifference() = default;
  ~MLIRGenSquaredDifference() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value xValue;
    mlir::Value yValue;
    
    if (!node->inputs().empty()) {
      xValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!xValue) {
      llvm::errs() << "SquaredDifference missing x input: " << node->name() << "\n";
      return {};
    }

    if (node->inputs().size() > 1) {
      yValue = getsymbol(symbolTable, node->inputs()[1]->name());
    }
    if (!yValue) {
      llvm::errs() << "SquaredDifference missing y input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<SquaredDifferenceOp>(
        location,
        outType,
        xValue,
        yValue,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_SquaredDifference, SquaredDifference);

} // namespace aipugc
} // namespace mlir