#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenBatchToSpace : public MLIRGenBase {
public:
  MLIRGenBatchToSpace() = default;
  ~MLIRGenBatchToSpace() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "BatchToSpaceOp missing input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto &params = node->params();

    int32_t block_y = params.at("block_size_y");
    int32_t block_x = params.at("block_size_x");
    auto blockYAttr = builder.getIntegerAttr(builder.getI32Type(), block_y);
    auto blockXAttr = builder.getIntegerAttr(builder.getI32Type(), block_x);

    int32_t crop_top = params.at("crop_top");
    int32_t crop_bottom = params.at("crop_bottom");
    int32_t crop_left = params.at("crop_left");
    int32_t crop_right = params.at("crop_right");

    auto cropTopAttr = builder.getIntegerAttr(builder.getI32Type(), crop_top);
    auto cropBottomAttr =
        builder.getIntegerAttr(builder.getI32Type(), crop_bottom);
    auto cropLeftAttr = builder.getIntegerAttr(builder.getI32Type(), crop_left);
    auto cropRightAttr =
        builder.getIntegerAttr(builder.getI32Type(), crop_right);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<BatchToSpaceOp>(
        location, outType, inputValue, blockYAttr, blockXAttr, cropTopAttr,
        cropBottomAttr, cropLeftAttr, cropRightAttr, opAttrs);

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_BatchToSpace, BatchToSpace);

} // namespace aipugc
} // namespace mlir

