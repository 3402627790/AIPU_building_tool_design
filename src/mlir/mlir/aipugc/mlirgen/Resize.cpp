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

    float ratio_x = 0.0f;
    float ratio_y = 0.0f;
    bool hasRatio = params.count("ratio_x") || params.count("ratio_y");
    bool hasSize = params.count("size");

    if (hasRatio && hasSize) {
      llvm::errs() << "ResizeOp expects either ratio_x/ratio_y or size, not both\n";
      return {};
    }

    if (hasRatio) {
      if (!params.count("ratio_x") || !params.count("ratio_y")) {
        llvm::errs() << "ResizeOp requires both ratio_x and ratio_y\n";
        return {};
      }
      ratio_x = params.at("ratio_x");
      ratio_y = params.at("ratio_y");
    } else if (hasSize) {
      auto inputTensor = node->inputs()[0];
      auto inputShape = inputTensor->shape();
      std::vector<int64_t> outputSize = params.at("size");

      if (inputShape.size() != 4 || outputSize.size() != 4) {
        llvm::errs() << "ResizeOp expects 4D NHWC input and size, but got input rank "
                     << inputShape.size() << " and output rank "
                     << outputSize.size() << "\n";
        return {};
      }

      if (inputShape[1] == 0 || inputShape[2] == 0) {
        llvm::errs() << "ResizeOp input H/W cannot be zero\n";
        return {};
      }

      ratio_y = static_cast<float>(outputSize[1]) /
                static_cast<float>(inputShape[1]);
      ratio_x = static_cast<float>(outputSize[2]) /
                static_cast<float>(inputShape[2]);
    } else {
      llvm::errs() << "ResizeOp missing required params: either ratio_x/ratio_y "
                   << "or size must be provided\n";
      return {};
    }

    if (ratio_x <= 0.0f || ratio_y <= 0.0f) {
      llvm::errs() << "ResizeOp ratio_x and ratio_y must be positive\n";
      return {};
    }

    auto ratioXAttr = builder.getF32FloatAttr(ratio_x);
    auto ratioYAttr = builder.getF32FloatAttr(ratio_y);

    if (!params.count("method")) {
      llvm::errs() << "ResizeOp missing required param: method\n";
      return {};
    }

    std::string method = params.at("method");
    auto methodAttr = builder.getStringAttr(method);

    std::string mode = params.count("mode") ? std::string(params.at("mode"))
                                            : "HALF_PIXEL";
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