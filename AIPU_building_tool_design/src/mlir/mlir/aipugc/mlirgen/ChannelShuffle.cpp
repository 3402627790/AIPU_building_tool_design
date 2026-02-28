#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenChannelShuffle : public MLIRGenBase {
public:
  MLIRGenChannelShuffle() = default;
  ~MLIRGenChannelShuffle() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value inputValue;
    if (!node->inputs().empty()) {
      inputValue = getsymbol(symbolTable, node->inputs()[0]->name());
    }
    if (!inputValue) {
      llvm::errs() << "ChannelShuffleOp missing input: " << node->name()
                   << "\n";
      return {};
    }

    llvm::SmallVector<mlir::Type> outputTypes;
    for (auto &output : node->outputs()) {
      outputTypes.push_back(utils::getType(builder, output));
    }

    auto location = loc(builder, node);
    const auto &params = node->params();

    // ChannelShuffle Op must have axis = -1
    int32_t group = params.at("group");
    int32_t splits = params.at("splits");

    auto groupAttr = builder.getIntegerAttr(builder.getI32Type(), group);
    auto splitsAttr = builder.getIntegerAttr(builder.getI32Type(), splits);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto channelShuffleOp = builder.create<ChannelShuffleOp>(
        location, outputTypes, inputValue, groupAttr, splitsAttr, opAttrs);

    std::vector<mlir::Value> results;
    for (auto result : channelShuffleOp->getResults()) {
      results.push_back(result);
    }
    return results;
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_ChannelShuffle, ChannelShuffle);

} // namespace aipugc
} // namespace mlir
