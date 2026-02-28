#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenBasicLSTM : public MLIRGenBase {
public:
  MLIRGenBasicLSTM() = default;
  ~MLIRGenBasicLSTM() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    mlir::Value hiddenStateValue;
    mlir::Value ceilStateValue;

    if (node->inputs().size() != 3) {
        llvm::errs() << "BasicLSTM node has 3 inputs, but provide " 
        << node->inputs().size() << " inputs." << "\n";
    }
    
    inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    hiddenStateValue = getsymbol(symbolTable, node->inputs()[1]->name());
    ceilStateValue = getsymbol(symbolTable, node->inputs()[2]->name());

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();
    std::string outSeq = params.at("out_sequence");
    std::string activations = params.at("activations");
    std::string direction = params.at("direction");
    auto outSeqAttr = builder.getStringAttr(outSeq);
    auto activationsAttr = builder.getStringAttr(activations);
    auto directionAttr = builder.getStringAttr(direction);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<BasicLSTMOp>(
        location,
        outType,
        inputValue,
        hiddenStateValue,
        ceilStateValue,
        outSeqAttr,
        activationsAttr,
        directionAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_BasicLSTM, BasicLSTM);

} // namespace aipugc
} // namespace mlir