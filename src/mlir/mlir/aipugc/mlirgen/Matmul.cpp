#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenMatMul : public MLIRGenBase {
public:
  MLIRGenMatMul() = default;
  ~MLIRGenMatMul() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    mlir::Value weightValue;

    if (node->inputs().size() != 2) {
        llvm::errs() << "Matmul node has inputs and weights, but provide " 
        << node->inputs().size() << " inputs." << "\n";
    }
    
    inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    weightValue = getsymbol(symbolTable, node->inputs()[1]->name());

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();
    bool trans_a = params.at("trans_a");
    bool trans_b = params.at("trans_b");
    BoolAttr trans_a_attr = builder.getBoolAttr(trans_a);
    BoolAttr trans_b_attr = builder.getBoolAttr(trans_b);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<MatMulOp>(
        location,
        outType,
        inputValue,
        weightValue,
        trans_a_attr,
        trans_b_attr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_MatMul, MatMul);

} // namespace aipugc
} // namespace mlir