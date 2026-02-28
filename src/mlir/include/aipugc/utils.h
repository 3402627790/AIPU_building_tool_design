#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include <gcompiler_api.h>
namespace mlir {
namespace aipugc {
namespace utils {
mlir::Type mapTensorDtypeToMLIRType(mlir::OpBuilder &builder,
                                    aipubt::Tensor::Dtype dtype);
aipubt::Tensor::Dtype GetDtype(mlir::Type type);

mlir::Type getType(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> shape,
                   aipubt::Tensor::Dtype dtype);
mlir::Type getType(mlir::OpBuilder &builder, aipubt::TensorShape shape,
                   aipubt::Tensor::Dtype dtype);
mlir::Type getType(llvm::ArrayRef<int64_t> shape, mlir::Type elementType);
mlir::Type getType(aipubt::TensorShape shape, mlir::Type elementType);
mlir::Type getType(mlir::OpBuilder &builder, aipubt::TensorPtr tensor);
mlir::DenseElementsAttr DenseElementsAttrGet(mlir::OpBuilder &builder,
                                             aipubt::TensorPtr tensor);
mlir::DenseI64ArrayAttr DenseI64ArrayAttrGet(mlir::OpBuilder &builder,
                                             std::vector<int32_t> val);
} // namespace utils
} // namespace aipugc
} // namespace mlir