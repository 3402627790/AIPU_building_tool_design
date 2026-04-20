#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace aipugc {

namespace {

llvm::SmallVector<int64_t, 4>
layerNormWeightBiasShape(RankedTensorType inType,
                         llvm::ArrayRef<int64_t> axisVals) {
  llvm::SmallVector<int64_t, 4> out;
  const int64_t rank = inType.getRank();
  for (int64_t ax : axisVals) {
    const int64_t n = ax < 0 ? ax + rank : ax;
    if (n < 0 || n >= rank)
      continue;
    int64_t d = inType.getDimSize(n);
    if (ShapedType::isDynamic(d))
      d = 1;
    out.push_back(d);
  }
  if (out.empty())
    out.push_back(1);
  return out;
}

DenseElementsAttr splatElements(RankedTensorType rty, double fpVal, int64_t intVal) {
  Type et = rty.getElementType();
  if (auto ft = dyn_cast<FloatType>(et)) {
    // FloatAttr avoids LLVM 21+ deleted APFloat(semantics, double) constructor.
    return DenseElementsAttr::get(rty, FloatAttr::get(ft, fpVal));
  }
  if (auto it = dyn_cast<IntegerType>(et)) {
    const unsigned w = it.getWidth();
    llvm::APInt v;
    if (it.isUnsigned())
      v = llvm::APInt(w, static_cast<uint64_t>(intVal != 0 ? 1 : 0), false);
    else
      v = llvm::APInt(w, static_cast<uint64_t>(intVal), true);
    return DenseElementsAttr::get(rty, v);
  }
  return {};
}

std::pair<Value, Value>
makePlaceholderWeightBias(OpBuilder &builder, Location loc, Value mlirInput,
                          aipubt::TensorPtr tsInput,
                          const std::vector<int64_t> &axisVec) {
  Type elemTy = nullptr;
  if (auto inTy = dyn_cast<RankedTensorType>(mlirInput.getType()))
    elemTy = inTy.getElementType();
  if (!elemTy)
    elemTy = utils::mapTensorDtypeToMLIRType(builder, tsInput->type());
  if (!elemTy)
    elemTy = builder.getF32Type();

  llvm::SmallVector<int64_t, 4> shape;
  if (auto inTy = dyn_cast<RankedTensorType>(mlirInput.getType()))
    shape = layerNormWeightBiasShape(inTy, axisVec);
  else
    shape.push_back(1);

  RankedTensorType rty = RankedTensorType::get(shape, elemTy);
  DenseElementsAttr wAttr = splatElements(rty, 1.0, 1);
  DenseElementsAttr bAttr = splatElements(rty, 0.0, 0);
  if (!wAttr || !bAttr) {
    rty = RankedTensorType::get(shape, builder.getF32Type());
    wAttr = DenseElementsAttr::get(rty, 1.0f);
    bAttr = DenseElementsAttr::get(rty, 0.0f);
  }
  auto w = builder.create<ConstantOp>(loc, rty, wAttr);
  auto b = builder.create<ConstantOp>(loc, rty, bAttr);
  return {w, b};
}

} // namespace

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

    std::vector<int64_t> axisVec = params.at("axis");

    mlir::Value mlir_weight;
    mlir::Value mlir_bias;
    auto it_weights = node->constants().find("weights");
    auto it_biases = node->constants().find("biases");
    if (it_weights != node->constants().end() &&
        it_biases != node->constants().end()) {
      auto ts_weight = it_weights->second;
      auto ts_bias = it_biases->second;
      mlir_weight = getsymbol(symbolTable, ts_weight->name());
      if (!mlir_weight)
        mlir_weight =
            ConstantTensorMlirGen(builder, location, ts_weight, symbolTable);
      mlir_bias = getsymbol(symbolTable, ts_bias->name());
      if (!mlir_bias)
        mlir_bias =
            ConstantTensorMlirGen(builder, location, ts_bias, symbolTable);
    } else {
      std::tie(mlir_weight, mlir_bias) =
          makePlaceholderWeightBias(builder, location, mlir_input, ts_input,
                                    axisVec);
    }

    // axis: normalized axis list (same convention as Reduce)
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
        location, outType, mlir_input, mlir_weight, mlir_bias, axisAttr,
        epsilonAttr, opAttrs, meanInvstddevTypes);

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
