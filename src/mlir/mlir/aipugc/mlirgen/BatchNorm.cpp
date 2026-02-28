#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenBatchNorm : public MLIRGenBase {
public:
  MLIRGenBatchNorm() = default;
  ~MLIRGenBatchNorm() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_input = node->inputs()[0];
    auto ts_weights = node->constants()["weights"];
    auto ts_biases = node->constants()["biases"];
    auto ts_output = node->outputs()[0];

    auto outType = utils::getType(builder, ts_output);
    auto location = loc(builder, node);

    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "The input " << ts_input->name() << "is nullptr \n";
      return {};
    }

    mlir::Value mlir_weights = getsymbol(symbolTable, ts_weights->name());
    if (!mlir_weights) {
      mlir_weights =
          ConstantTensorMlirGen(builder, location, ts_weights, symbolTable);
    }

    mlir::Value mlir_biases = getsymbol(symbolTable, ts_biases->name());
    if (!mlir_biases) {
      mlir_biases =
          ConstantTensorMlirGen(builder, location, ts_biases, symbolTable);
    }

    int32_t axis = node->params()["axis"];
    float epsilon = node->params()["epsilon"];

    auto axisAttr = builder.getIntegerAttr(builder.getIntegerType(32), axis);
    auto epsilonAttr = builder.getFloatAttr(builder.getF32Type(), epsilon);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<BatchNormOp>(location, outType, mlir_input,
                                              mlir_weights, mlir_biases,
                                              axisAttr, epsilonAttr, opAttrs);
    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_BatchNorm, BatchNorm);

} // namespace aipugc
} // namespace mlir

