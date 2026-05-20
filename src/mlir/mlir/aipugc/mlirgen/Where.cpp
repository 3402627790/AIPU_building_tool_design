#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenWhere : public MLIRGenBase {
public:
  MLIRGenWhere() = default;
  ~MLIRGenWhere() = default;

  std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_condition = node->inputs()[0];
    auto ts_x = node->inputs()[1];
    auto ts_y = node->inputs()[2];
    auto ts_output = node->outputs()[0];

    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);

    mlir::Value condition = getsymbol(symbolTable, ts_condition->name());
    if (!condition) {
      llvm::errs() << "The condition input " << ts_condition->name()
                   << " is nullptr\n";
      return {};
    }

    mlir::Value x = getsymbol(symbolTable, ts_x->name());
    if (!x) {
      llvm::errs() << "The x input " << ts_x->name() << " is nullptr\n";
      return {};
    }

    mlir::Value y = getsymbol(symbolTable, ts_y->name());
    if (!y) {
      llvm::errs() << "The y input " << ts_y->name() << " is nullptr\n";
      return {};
    }

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<WhereOp>(location, outType, condition, x, y,
                                          opAttrs);
    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Where, Where);

} // namespace aipugc
} // namespace mlir
