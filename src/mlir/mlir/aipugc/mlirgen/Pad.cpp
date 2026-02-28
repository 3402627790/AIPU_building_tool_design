#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenPad : public MLIRGenBase {
public:
  MLIRGenPad() = default;
  ~MLIRGenPad() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "PadOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    std::vector<std::vector<int64_t>> padsVec = node->params()["pads"];

    // convert to MLIR attr
    llvm::SmallVector<mlir::Attribute> padAttrs;
    for (auto& padPair : padsVec) {
      auto densePad = builder.getDenseI64ArrayAttr(padPair);
      padAttrs.push_back(densePad);
    }
    auto padsAttr = builder.getArrayAttr(padAttrs);

    std::string modeStr = node->params().at("mode");
    auto modeAttr = builder.getStringAttr(modeStr);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<PadOp>(
        location,
        outType,
        inputValue,
        padsAttr,
        modeAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Pad, Pad);

} // namespace aipugc
} // namespace mlir