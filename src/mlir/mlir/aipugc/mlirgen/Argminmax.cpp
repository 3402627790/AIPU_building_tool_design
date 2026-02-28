#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenArgMinMax : public MLIRGenBase {
public:
  MLIRGenArgMinMax() = default;
  ~MLIRGenArgMinMax() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "ArgMinMaxOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto &params = node->params();
    int32_t axis = params.at("axis");
    auto axisAttr = builder.getIntegerAttr(builder.getIntegerType(32), axis);

    std::string method = params.at("method");
    auto methodAttr = builder.getStringAttr(method);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<ArgMinMaxOp>(location, outType, inputValue,
                                              axisAttr, methodAttr, opAttrs);

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_ArgMinMax, ArgMinMax);

} // namespace aipugc
} // namespace mlir

