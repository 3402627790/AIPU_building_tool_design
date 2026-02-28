#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {
class MLIRGenDiv : public MLIRGenBase {
public:
  MLIRGenDiv() = default;
  ~MLIRGenDiv() = default;

  virtual std::vector<mlir::Value> mlirGen(OpBuilder &builder, aipubt::NodePtr node, std::map<std::string, Value>&symbolTable) override {
    auto lhs = node->inputs()[0];
    auto rhs = node->inputs()[1];
    auto out = node->outputs()[0];

    auto outType = utils::getType(builder, out);
    auto location = loc(builder, node);
    
    Value lhs_mlir = getsymbol(symbolTable, lhs->name());
    Value rhs_mlir = getsymbol(symbolTable, rhs->name());
    if (!lhs_mlir || !rhs_mlir) {
      llvm::errs() << "One of the inputs is nullptr: "
                   << (lhs_mlir ? "" : lhs->name()) << ""
                   << (rhs_mlir ? "" : rhs->name()) << "\n";
    }

    DictionaryAttr opAttr = ParamsToMlirAttr(builder, node);

    auto output = builder.create<DivOp>(
        location,
        outType,
        lhs_mlir,
        rhs_mlir,
        opAttr
    );

    return {output};
  }
};

// Registration macro for MLIR generator
// Associates OpType_Div with our MLIRGenDiv implementation
// Enables framework to dispatch generation for Div operations
REGISTER_OP_MLIRGEN(aipubt::OpType_Div, Div);

}
}