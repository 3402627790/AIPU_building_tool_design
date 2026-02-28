#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenSpaceToBatch : public MLIRGenBase {
public:
  MLIRGenSpaceToBatch() = default;
  ~MLIRGenSpaceToBatch() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "SpaceToBatchOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto &params = node->params();

    int32_t block_y = params.at("block_size_y");
    int32_t block_x = params.at("block_size_x");
    auto blockYAttr = builder.getIntegerAttr(builder.getI32Type(), block_y);
    auto blockXAttr = builder.getIntegerAttr(builder.getI32Type(), block_x);

    int32_t pad_top = params.at("pad_top");
    int32_t pad_bottom = params.at("pad_bottom");
    int32_t pad_left = params.at("pad_left");
    int32_t pad_right = params.at("pad_right");

    auto padTopAttr = builder.getIntegerAttr(builder.getI32Type(), pad_top);
    auto padBottomAttr =
        builder.getIntegerAttr(builder.getI32Type(), pad_bottom);
    auto padLeftAttr = builder.getIntegerAttr(builder.getI32Type(), pad_left);
    auto padRightAttr =
        builder.getIntegerAttr(builder.getI32Type(), pad_right);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<SpaceToBatchOp>(
        location, outType, inputValue, blockYAttr, blockXAttr, padTopAttr,
        padBottomAttr, padLeftAttr, padRightAttr, opAttrs);

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_SpaceToBatch, SpaceToBatch);

} // namespace aipugc
} // namespace mlir
