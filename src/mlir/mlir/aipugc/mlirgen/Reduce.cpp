#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenReduce : public MLIRGenBase {
public:
  MLIRGenReduce() = default;
  ~MLIRGenReduce() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override
  {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "ReduceOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();
    std::vector<int64_t> axes = params.at("axis");
    auto axisAttr = builder.getDenseI64ArrayAttr(axes);

    std::string method = params.at("method");
    auto methodAttr = builder.getStringAttr(method);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<ReduceOp>(
        location,
        outType,
        inputValue,
        axisAttr,
        methodAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Reduce, Reduce);

} // namespace aipugc
} // namespace mlir