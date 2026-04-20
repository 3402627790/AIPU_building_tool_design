#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenRMSNorm : public MLIRGenBase {
public:
  MLIRGenRMSNorm() = default;
  ~MLIRGenRMSNorm() = default;

  std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_input = node->inputs()[0];
    auto ts_output = node->outputs()[0];

    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "RMSNorm missing input: " << ts_input->name() << "\n";
      return {};
    }

    auto location = loc(builder, node);
    auto outType = utils::getType(builder, ts_output);

    const auto &params = node->params();

    auto it_weights = node->constants().find("weights");
    if (it_weights == node->constants().end()) {
      llvm::errs() << "RMSNorm missing weights constant\n";
      return {};
    }
    auto ts_weight = it_weights->second;
    mlir::Value mlir_weight = getsymbol(symbolTable, ts_weight->name());
    if (!mlir_weight)
      mlir_weight =
          ConstantTensorMlirGen(builder, location, ts_weight, symbolTable);

    std::vector<int64_t> axisVec = params.at("axis");
    auto axisAttr = builder.getDenseI64ArrayAttr(axisVec);

    float epsilon = params.at("epsilon");
    auto epsilonAttr = builder.getFloatAttr(builder.getF32Type(), epsilon);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto op = builder.create<RMSNormOp>(location, outType, mlir_input,
                                        mlir_weight, axisAttr, epsilonAttr,
                                        opAttrs);

    std::vector<mlir::Value> results;
    results.push_back(op.getOutput());
    return results;
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_RMSNorm, RMSNorm);

} // namespace aipugc
} // namespace mlir

