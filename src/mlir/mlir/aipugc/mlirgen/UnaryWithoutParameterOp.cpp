#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {
template <typename MLIROPType> class MLIRGenUnary : public MLIRGenBase {
public:
  MLIRGenUnary() {}
  ~MLIRGenUnary() {}
  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    mlir::Value input = getsymbol(symbolTable, node->inputs()[0]->name());
    if (!input) {
      llvm::errs() << "The input is nullptr \n ";
      return {};
    }
    auto outType = utils::getType(builder, node->outputs()[0]);
    auto location = loc(builder, node);
    DictionaryAttr op_attrs = ParamsToMlirAttr(builder, node);
    auto output =
        builder.create<MLIROPType>(location, outType, input, op_attrs);
    return {output};
  }
};

// class MLIRGenReshape : public MLIRGenUnary<ReshapeOp> {
// public:
//   using MLIRGenUnary::MLIRGenUnary;
//   ~MLIRGenReshape() {}
// };

class MLIRGenAbs : public MLIRGenUnary<AbsOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenAbs() {}
};

class MLIRGenNegative : public MLIRGenUnary<NegativeOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenNegative() {}
};

class MLIRGenSqrt : public MLIRGenUnary<SqrtOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenSqrt() {}
};

class MLIRGenRsqrt : public MLIRGenUnary<RsqrtOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenRsqrt() {}
};

class MLIRGenCosine : public MLIRGenUnary<CosineOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenCosine() {}
};

class MLIRGenExp : public MLIRGenUnary<ExpOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenExp() {}
};

class MLIRGenLog : public MLIRGenUnary<LogOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenLog() {}
};

class MLIRGenSquare : public MLIRGenUnary<SquareOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenSquare() {}
};

class MLIRGenQuantize : public MLIRGenUnary<QuantizeOp> {
public:
  using MLIRGenUnary::MLIRGenUnary;
  ~MLIRGenQuantize() {}
};

// REGISTER_MLIRGEN(Reshape);
REGISTER_MLIRGEN(Abs);
REGISTER_MLIRGEN(Negative);
REGISTER_MLIRGEN(Sqrt);
REGISTER_MLIRGEN(Rsqrt);
REGISTER_MLIRGEN(Cosine);
REGISTER_MLIRGEN(Exp);
REGISTER_MLIRGEN(Log);
REGISTER_MLIRGEN(Square);
REGISTER_MLIRGEN(Quantize);

} // namespace aipugc
} // namespace mlir