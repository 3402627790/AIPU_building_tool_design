#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenGridSample : public MLIRGenBase {
public:
  MLIRGenGridSample() = default;
  ~MLIRGenGridSample() = default;

  std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    // Fetch input tensor value.
    mlir::Value inputValue =
        getsymbol(symbolTable, node->inputs()[0]->name());
    if (!inputValue) {
      llvm::errs() << "GridSampleOp missing input: " << node->name() << "\n";
      return {};
    }

    // Fetch grid tensor value.
    mlir::Value gridValue =
        getsymbol(symbolTable, node->inputs()[1]->name());
    if (!gridValue) {
      llvm::errs() << "GridSampleOp missing grid: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto &params = node->params();

    // Align corners.
    bool alignCorners = params.at("align_corners");
    auto alignCornersAttr = builder.getBoolAttr(alignCorners);

    // Interpolation method, e.g. "BILINEAR", "NEAREST", "LINEAR", "BICUBIC".
    std::string method = params.at("method");
    auto methodAttr = builder.getStringAttr(method);

    // Padding mode, e.g. "ZEROS", "BORDER", "REFLECTION".
    std::string paddingMode = params.at("padding_mode");
    auto paddingModeAttr = builder.getStringAttr(paddingMode);

    // Collect remaining attributes into a dictionary for quantization etc.
    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<GridSampleOp>(
        location,
        outType,
        inputValue,
        gridValue,
        methodAttr,
        paddingModeAttr,
        alignCornersAttr,
        opAttrs);

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_GridSample, GridSample);

} // namespace aipugc
} // namespace mlir

