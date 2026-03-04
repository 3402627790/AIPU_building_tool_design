#include "aipugc/utils.h"
#include "aipugc/QuantUtils.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "utils"

#include <numeric>
namespace mlir {
namespace aipugc {
namespace utils {
/**
 * @brief Convert aipubt::Tensor data type to corresponding MLIR element type.
 *
 * This function maps various tensor data types (enums) to their equivalent MLIR
 * element types. It handles integer types of different bit widths (including
 * special cases like 4-bit and 12-bit), floating-point types, and boolean type.
 * For unsupported or unknown types, returns nullptr.
 *
 * @param dtype The tensor data type enum to convert
 * @param context MLIR context used for type creation
 * @return mlir::Type Corresponding MLIR element type, or nullptr if:
 *         - dtype is VOID
 *         - dtype is invalid/unsupported
 *         - context is nullptr (implicitly, through MLIR type creation)
 *
 * @note Special type mappings:
 * - Boolean: 1-bit integer
 * - UINT8/INT8: 8-bit integer
 * - 4-bit types (ALIGNED/COMPACT): 4-bit integer
 * - 12-bit types (ALIGNED/COMPACT): 12-bit integer
 * - FP16/BFP16: 16-bit float (different representations)
 * - FP32/FP64: Standard 32/64-bit floats
 */
mlir::Type mapTensorDtypeToMLIRType(mlir::OpBuilder &builder,
                                    aipubt::Tensor::Dtype dtype) {
  switch (dtype) {
  case aipubt::Tensor::Dtype::VOID:
    return nullptr;
  case aipubt::Tensor::Dtype::BOOL:
    return builder.getIntegerType(1, false);
  case aipubt::Tensor::Dtype::UINT8:
    return builder.getIntegerType(8, false);
  case aipubt::Tensor::Dtype::INT8:
    return builder.getIntegerType(8);
  case aipubt::Tensor::Dtype::UINT16:
    return builder.getIntegerType(16, false);
  case aipubt::Tensor::Dtype::INT16:
    return builder.getIntegerType(16);
  case aipubt::Tensor::Dtype::UINT32:
    return builder.getIntegerType(32, false);
  case aipubt::Tensor::Dtype::INT32:
    return builder.getIntegerType(32);
  case aipubt::Tensor::Dtype::UINT64:
    return builder.getIntegerType(64, false);
  case aipubt::Tensor::Dtype::INT64:
    return builder.getIntegerType(64);
  case aipubt::Tensor::Dtype::FP16:
    return builder.getF16Type();
  case aipubt::Tensor::Dtype::FP32:
    return builder.getF32Type();
  case aipubt::Tensor::Dtype::FP64:
    return builder.getF64Type();
  case aipubt::Tensor::Dtype::BFP16:
    return builder.getBF16Type();
  case aipubt::Tensor::Dtype::ALIGNED_UINT4:
  case aipubt::Tensor::Dtype::COMPACT_UINT4:
    return builder.getIntegerType(4, false);
  case aipubt::Tensor::Dtype::ALIGNED_INT4:
  case aipubt::Tensor::Dtype::COMPACT_INT4:
    return builder.getIntegerType(4);
  case aipubt::Tensor::Dtype::ALIGNED_UINT12:
  case aipubt::Tensor::Dtype::COMPACT_UINT12:
    return builder.getIntegerType(12, false);
  case aipubt::Tensor::Dtype::ALIGNED_INT12:
  case aipubt::Tensor::Dtype::COMPACT_INT12:
    return builder.getIntegerType(12);
  default:
    return nullptr;
  }
}

aipubt::Tensor::Dtype GetDtype(mlir::Type type) {
  if (type.isBF16()) {
    return aipubt::Tensor::Dtype::BFP16;
  } else if (type.isF16()) {
    return aipubt::Tensor::Dtype::FP16;
  } else if (type.isF32()) {
    return aipubt::Tensor::Dtype::FP32;
  } else if (type.isF64()) {
    return aipubt::Tensor::Dtype::FP64;
  } else if (type.isUnsignedInteger(1)) {
    return aipubt::Tensor::Dtype::BOOL;
  }
  /* Signed Integer */
  else if (type.isSignedInteger(4)) {
    return aipubt::Tensor::Dtype::ALIGNED_INT4;
  } else if (type.isSignedInteger(8)) {
    return aipubt::Tensor::Dtype::INT8;
  } else if (type.isSignedInteger(12)) {
    return aipubt::Tensor::Dtype::ALIGNED_INT12;
  } else if (type.isSignedInteger(16)) {
    return aipubt::Tensor::Dtype::INT16;
  } else if (type.isSignedInteger(32)) {
    return aipubt::Tensor::Dtype::INT32;
  } else if (type.isSignedInteger(64)) {
    return aipubt::Tensor::Dtype::INT64;
  }
  /* Unsigned Integer */
  else if (type.isUnsignedInteger(4)) {
    return aipubt::Tensor::Dtype::ALIGNED_UINT4;
  } else if (type.isUnsignedInteger(8)) {
    return aipubt::Tensor::Dtype::UINT8;
  } else if (type.isUnsignedInteger(12)) {
    return aipubt::Tensor::Dtype::ALIGNED_UINT12;
  } else if (type.isUnsignedInteger(16)) {
    return aipubt::Tensor::Dtype::UINT16;
  } else if (type.isUnsignedInteger(32)) {
    return aipubt::Tensor::Dtype::UINT32;
  } else if (type.isUnsignedInteger(64)) {
    return aipubt::Tensor::Dtype::UINT64;
  }
  /* Integer */
  else if (type.isInteger(4)) {
    return aipubt::Tensor::Dtype::ALIGNED_INT4;
  } else if (type.isInteger(8)) {
    return aipubt::Tensor::Dtype::INT8;
  } else if (type.isInteger(12)) {
    return aipubt::Tensor::Dtype::ALIGNED_INT12;
  } else if (type.isInteger(16)) {
    return aipubt::Tensor::Dtype::INT16;
  } else if (type.isInteger(32)) {
    return aipubt::Tensor::Dtype::INT32;
  } else if (type.isInteger(64)) {
    return aipubt::Tensor::Dtype::INT64;
  }
  /* Default */
  else {
    return aipubt::Tensor::Dtype::VOID;
  }
}


mlir::Type getType(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> shape,
                   aipubt::Tensor::Dtype dtype) {
  mlir::Type elementType = mapTensorDtypeToMLIRType(builder, dtype);
  if (!elementType)
    return nullptr;

  return shape.empty()
             ? mlir::Type(mlir::UnrankedTensorType::get(elementType))
             : mlir::Type(mlir::RankedTensorType::get(shape, elementType));
}

mlir::Type getType(mlir::OpBuilder &builder, aipubt::TensorShape shape,
                   aipubt::Tensor::Dtype dtype) {
  std::vector<int64_t> dims(shape.begin(), shape.end());
  return getType(builder, dims, dtype);
}

mlir::Type getType(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  return shape.empty()
             ? mlir::Type(mlir::UnrankedTensorType::get(elementType))
             : mlir::Type(mlir::RankedTensorType::get(shape, elementType));
}

mlir::Type getType(aipubt::TensorShape shape, mlir::Type elementType) {
  std::vector<int64_t> dims(shape.begin(), shape.end());
  return getType(dims, elementType);
}

mlir::Type getType(mlir::OpBuilder &builder, aipubt::TensorPtr tensor) {
  mlir::Type elementType;
  if (!tensor->quantization().empty() &&
      !aipubt::Tensor::is_float(tensor->type())) {
    if (tensor->quantization().scales().size() > 1)
      elementType = QuantizationToPerAxisType(builder, tensor);
    else
      elementType = QuantizationToType(builder, tensor);
  } else {
    elementType = mapTensorDtypeToMLIRType(builder, tensor->type());
  }
  if (!elementType)
    return nullptr;
  std::vector<int64_t> dims(tensor->shape().begin(), tensor->shape().end());
  llvm::ArrayRef<int64_t> shape(dims);
  return shape.empty()
             ? mlir::Type(mlir::UnrankedTensorType::get(elementType))
             : mlir::Type(mlir::RankedTensorType::get(shape, elementType));
}

// Collect tensor data into a DenseElementsAttr with typed element access.
// Copies `size` elements of type T from the tensor's buffer into a local
// vector, then wraps it as an MLIR dense attribute.
template <typename T>
mlir::DenseElementsAttr collectData(mlir::RankedTensorType &type,
                                    aipubt::TensorPtr tensor, size_t size) {
  std::vector<T> dst;
  dst.resize(size);
  if (tensor->data<T>() != nullptr) {
    memcpy(dst.data(), tensor->data<T>(), size * sizeof(T));
  }
  return mlir::DenseElementsAttr::get(type, llvm::ArrayRef(dst));
}

// Collect tensor data into a DenseElementsAttr using raw byte-level copy,
// bypassing element type interpretation. Unlike collectData<T>, this operates
// on untyped char buffers, suitable for types without native C++ equivalents.
template <size_t ElementByteSize>
mlir::DenseElementsAttr collectDataRaw(mlir::RankedTensorType &type,
                                       aipubt::TensorPtr tensor, size_t size) {
  size_t byteSize = size * ElementByteSize;
  std::vector<char> dst(byteSize, 0);
  if (tensor->data<char>() != nullptr) {
    memcpy(dst.data(), tensor->data<char>(), byteSize);
  }
  return mlir::DenseElementsAttr::getFromRawBuffer(type,
                                                   llvm::ArrayRef<char>(dst));
}

using collectDataFunc = std::function<mlir::DenseElementsAttr(
    mlir::RankedTensorType &, aipubt::TensorPtr, size_t size)>;

static std::map<aipubt::Tensor::Dtype, collectDataFunc> collectDataMaps = {
    {aipubt::Tensor::INT8, collectData<int8_t>},
    {aipubt::Tensor::UINT8, collectData<uint8_t>},
    {aipubt::Tensor::UINT16, collectData<uint16_t>},
    {aipubt::Tensor::INT16, collectData<int16_t>},
    {aipubt::Tensor::UINT32, collectData<uint32_t>},
    {aipubt::Tensor::INT32, collectData<int32_t>},
    {aipubt::Tensor::UINT64, collectData<uint64_t>},
    {aipubt::Tensor::INT64, collectData<int64_t>},
    {aipubt::Tensor::FP16, collectDataRaw<2>},
    {aipubt::Tensor::FP32, collectData<float_t>},
    {aipubt::Tensor::FP64, collectData<double>},
    {aipubt::Tensor::BFP16, collectDataRaw<2>},
};

mlir::DenseElementsAttr DenseElementsAttrGet(mlir::OpBuilder &builder,
                                             aipubt::TensorPtr tensor) {
  std::vector<int64_t> dims(tensor->shape().begin(), tensor->shape().end());
  if (dims.size() == 0)
    dims.push_back(tensor->size());
  llvm::ArrayRef<int64_t> dimsArray(dims);
  auto elementType = mapTensorDtypeToMLIRType(builder, tensor->type());
  auto dataType = mlir::RankedTensorType::get(dimsArray, elementType);
  size_t dataSize = tensor->size();
  if (collectDataMaps.count(tensor->type())) {
    return collectDataMaps[tensor->type()](dataType, tensor, dataSize);
  } else {
    llvm::errs() << "Unsupport AIPU tensor type: "
                 << aipubt::Tensor::dtype_to_string(tensor->type()) << "\n";
    return {};
  }
}

mlir::DenseI64ArrayAttr DenseI64ArrayAttrGet(mlir::OpBuilder &builder,
                                             std::vector<int32_t> val) {
  std::vector<int64_t> val_i64(val.begin(), val.end());
  llvm::ArrayRef<int64_t> arr(val_i64);
  return builder.getDenseI64ArrayAttr(arr);
}

} // namespace utils
} // namespace aipugc
} // namespace mlir