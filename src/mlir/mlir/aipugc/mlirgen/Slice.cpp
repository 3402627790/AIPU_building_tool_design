#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenSlice : public MLIRGenBase {
public:
  MLIRGenSlice() = default;
  ~MLIRGenSlice() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override
  {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
        inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
        llvm::errs() << "SliceOp missing inputs:: " << node->name() << "\n";
        return {};
    }

    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);

    const auto& params = node->params();

    if (!params.count("begin") || !params.count("end") 
        || !params.count("strides")) {
        llvm::errs() << "SliceOp missing required params\n";
        return {};
    }

    std::vector<int64_t> begin = params.at("begin");
    std::vector<int64_t> end = params.at("end");
    std::vector<int64_t> strides = params.at("strides");
    auto beginAttr = builder.getDenseI64ArrayAttr(begin);
    auto endAttr = builder.getDenseI64ArrayAttr(end);
    auto stridesAttr = builder.getDenseI64ArrayAttr(strides);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto output = builder.create<SliceOp>(
        location,
        outType,
        inputValue,
        beginAttr,
        endAttr,
        stridesAttr,
        opAttrs
    );
    
    return {output};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Slice, Slice);

} // namespace aipugc
} // namespace mlir