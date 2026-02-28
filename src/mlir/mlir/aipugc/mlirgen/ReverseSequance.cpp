#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenReverseSequence : public MLIRGenBase {
public:
  MLIRGenReverseSequence() = default;
  ~MLIRGenReverseSequence() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    mlir::Value seqLensValue;
    
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "ReverseSequence missing feature input: " << node->name() << "\n";
      return {};
    }

    if (node->inputs().size() > 1) {
      seqLensValue = getsymbol(symbolTable, node->inputs()[1]->name());
    }
    if (!seqLensValue) {
      llvm::errs() << "ReverseSequence missing seq_lens input: " << node->name() << "\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();
    
    int64_t batch_axis = params.at("batch_axis");
    int64_t time_axis = params.at("time_axis");
    if (batch_axis < 0 || time_axis < 0) {
      llvm::errs() << "Invalid axis value <0 in " << node->name();
      return {};
    }

    auto batchAxisAttr = builder.getI64IntegerAttr(batch_axis);
    auto timeAxisAttr = builder.getI64IntegerAttr(time_axis);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<ReverseSequenceOp>(
        location,
        outType,
        inputValue,
        seqLensValue,
        batchAxisAttr,
        timeAxisAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_ReverseSequence, ReverseSequence);

} // namespace aipugc
} // namespace mlir