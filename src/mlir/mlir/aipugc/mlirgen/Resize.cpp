#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenResize : public MLIRGenBase {
public:
  MLIRGenResize() = default;
  ~MLIRGenResize() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "ResizeOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();

    float ratio_x = params.at("ratio_x");
    float ratio_y = params.at("ratio_y");
    auto ratioXAttr = builder.getF32FloatAttr(ratio_x);
    auto ratioYAttr = builder.getF32FloatAttr(ratio_y);

    std::string method = params.at("method");
    auto methodAttr = builder.getStringAttr(method);

    std::string mode = params.at("mode");
    auto modeAttr = builder.getStringAttr(mode);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<ResizeOp>(
        location,
        outType,
        inputValue,
        ratioXAttr,
        ratioYAttr,
        methodAttr,
        modeAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Resize, Resize);

} // namespace aipugc
} // namespace mlir