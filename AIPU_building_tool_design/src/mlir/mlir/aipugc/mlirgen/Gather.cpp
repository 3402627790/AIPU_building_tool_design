#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenGather : public MLIRGenBase {
public:
  MLIRGenGather() = default;
  ~MLIRGenGather() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override 
  {
    mlir::Value inputValue;
    mlir::Value indicesValue;
    
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (node->inputs().size() > 1) {
      indicesValue = getsymbol(symbolTable, node->inputs()[1]->name());
    }
    
    if (!inputValue || !indicesValue) {
      llvm::errs() << "GatherOp missing inputs: " << node->name() 
                  << " requires 2 inputs\n";
      return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();
    
    if (!params.count("axis") || !params.count("batch_dims")) {
      llvm::errs() << "GatherOp missing required parameters\n";
      return {};
    }

    int64_t axis = params.at("axis");
    auto axisAttr = builder.getI64IntegerAttr(axis);
    
    int64_t batch_dims = params.at("batch_dims");
    auto batchDimsAttr = builder.getI64IntegerAttr(batch_dims);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<GatherOp>(
        location,
        outType,
        inputValue,    // params
        indicesValue,  // indices
        axisAttr,
        batchDimsAttr,
        opAttrs
    );

    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Gather, Gather);

} // namespace aipugc
} // namespace mlir