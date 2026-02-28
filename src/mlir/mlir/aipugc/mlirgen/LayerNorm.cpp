#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenLayerNorm : public MLIRGenBase {
public:
  MLIRGenLayerNorm() = default;
  ~MLIRGenLayerNorm() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_input = node->inputs()[0];
    auto ts_output = node->outputs()[0];

    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "LayerNorm missing input: " << ts_input->name() << "\n";
      return {};
    }

    auto location = loc(builder, node);
    auto outType = utils::getType(builder, ts_output);

    const auto &params = node->params();

    // gamma, beta: 0 or 2 operands (weights, biases)
    llvm::SmallVector<mlir::Value, 2> gammaBetaValues;
    auto it_weights = node->constants().find("weights");
    auto it_biases = node->constants().find("biases");
    if (it_weights != node->constants().end() &&
        it_biases != node->constants().end()) {
      auto ts_gamma = it_weights->second;
      auto ts_beta = it_biases->second;
      mlir::Value mlir_gamma = getsymbol(symbolTable, ts_gamma->name());
      if (!mlir_gamma)
        mlir_gamma =
            ConstantTensorMlirGen(builder, location, ts_gamma, symbolTable);
      mlir::Value mlir_beta = getsymbol(symbolTable, ts_beta->name());
      if (!mlir_beta)
        mlir_beta =
            ConstantTensorMlirGen(builder, location, ts_beta, symbolTable);
      gammaBetaValues.push_back(mlir_gamma);
      gammaBetaValues.push_back(mlir_beta);
    }

    // axis: normalized axis list (same convention as Reduce)
    std::vector<int64_t> axisVec = params.at("axis");
    auto axisAttr = builder.getDenseI64ArrayAttr(axisVec);

    float epsilon = params.at("epsilon");
    auto epsilonAttr = builder.getFloatAttr(builder.getF32Type(), epsilon);

    // Optional params (stash_type, out_mean_invstddev, quants, etc.) are
    // carried inside the generic dictAttr.
    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    bool out_mean_invstddev = (node->outputs().size() >= 3);

    llvm::SmallVector<mlir::Type, 2> meanInvstddevTypes;
    if (out_mean_invstddev) {
      meanInvstddevTypes.push_back(
          utils::getType(builder, node->outputs()[1]));
      meanInvstddevTypes.push_back(
          utils::getType(builder, node->outputs()[2]));
    }

    auto op = builder.create<LayerNormOp>(
        location, outType, mlir_input, gammaBetaValues, axisAttr, epsilonAttr,
        opAttrs, meanInvstddevTypes);

    std::vector<mlir::Value> results;
    results.push_back(op.getOutput());
    if (out_mean_invstddev) {
      results.push_back(op.getMeanInvstddev()[0]);
      results.push_back(op.getMeanInvstddev()[1]);
    }
    return results;
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_LayerNorm, LayerNorm);

} // namespace aipugc
} // namespace mlir
