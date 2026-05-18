# QuantUtils.cpp 代码文档

## 概述

本文件实现了 AIPU 量化工具函数，包含量化类型的定义、验证、类型转换以及打印功能。属于 MLIR AIPU Graph Compiler (aipugc) 方言的一部分。

## 命名空间

```
mlir::aipugc
```

## 核心类型

### QuantizedType

量化类型的基类，提供量化类型的通用接口。

| 方法 | 说明 |
|------|------|
| `getFlags()` | 获取量化标志位（如是否有符号） |
| `classof(Type)` | 判断类型是否属于 AIPU 方言 |
| `verify(...)` | 验证量化参数合法性（存储类型、位宽、min/max 范围） |
| `getStorageType()` | 获取底层存储类型（如 i8, u8, i16 等） |
| `getStorageTypeMin()` | 获取存储类型最小值 |
| `getStorageTypeMax()` | 获取存储类型最大值 |
| `getStorageTypeIntegralWidth()` | 获取存储类型位宽 |
| `getExpressedType()` | 获取表达类型（通常为浮点类型） |
| `isCompatibleExpressedType(Type)` | 检查候选表达类型是否兼容 |
| `getQuantizedElementType(Type)` | 从容器类型中提取量化元素类型 |
| `castFromStorageType(Type)` | 从存储类型转换为量化类型 |
| `castToStorageType(Type)` | 从量化类型转换为存储类型 |
| `castFromExpressedType(Type)` | 从表达类型转换为量化类型 |
| `castToExpressedType(Type)` | 从量化类型转换为表达类型 |
| `castExpressedToStorageType(Type)` | 从表达类型直接转换为存储类型 |

### UniformQuantizedType

均匀量化类型（per-tensor），使用单一 scale 和 zeroPoint。

| 方法 | 说明 |
|------|------|
| `get(flags, storageType, expressedType, scale, zeroPoint, min, max)` | 创建实例 |
| `getChecked(...)` | 带错误检查的创建方法 |
| `verify(...)` | 验证参数（scale > 0, expressedType 为浮点类型） |
| `getScale()` | 获取量化缩放因子 |
| `getZeroPoint()` | 获取零点偏移 |

### UniformQuantizedPerAxisType

按轴/通道量化类型，每个通道有独立的 scale 和 zeroPoint。

| 方法 | 说明 |
|------|------|
| `get(flags, storageType, expressedType, scales, zeroPoints, dim, min, max)` | 创建实例 |
| `getChecked(...)` | 带错误检查的创建方法 |
| `verify(...)` | 验证参数（scales 和 zeroPoints 数量一致，scale > 0） |
| `getScales()` | 获取各通道缩放因子数组 |
| `getZeroPoints()` | 获取各通道零点数组 |
| `getQuantizedDimension()` | 获取量化维度（axis） |

## 内部辅助函数

### getDefaultStorageParams

```cpp
bool getDefaultStorageParams(unsigned numBits, bool isSigned, MLIRContext *ctx,
                             Type &storageType, int64_t &qmin, int64_t &qmax)
```

根据位宽和符号性获取默认存储参数：

| numBits | isSigned | 范围 |
|---------|----------|------|
| ≤ 8 | true | [-128, 127] |
| ≤ 8 | false | [0, 255] |
| ≤ 16 | true | [-32768, 32767] |
| ≤ 16 | false | [0, 65535] |
| ≤ 32 | true | [INT32_MIN, INT32_MAX] |
| ≤ 32 | false | [0, UINT32_MAX] |
| ≤ 64 | true | [INT64_MIN, INT64_MAX] |
| ≤ 64 | false | [0, UINT64_MAX] |

## 打印函数

| 函数 | 输出格式 |
|------|----------|
| `printUniformQuantizedType` | `quant_per_axis<i8:f32, 0.003906:0>` |
| `printUniformQuantizedPerAxisType` | `quant<i8:f32:0, {0.001:0,0.002:1}>` |

## 转换函数（aipubt ↔ MLIR）

### aipubt → MLIR

| 函数 | 说明 |
|------|------|
| `QuantizationToType(builder, tensor)` | 将 aipubt Tensor 的量化信息转为 `UniformQuantizedType` |
| `QuantizationToPerAxisType(builder, tensor)` | 将 aipubt Tensor 的量化信息转为 `UniformQuantizedPerAxisType` |

### MLIR → aipubt

| 函数 | 说明 |
|------|------|
| `UniformTypeToQuantization(type)` | 将 `UniformQuantizedType` 转回 `aipubt::Quantization` |
| `UniformPerAxisTypeToQuantization(type)` | 将 `UniformQuantizedPerAxisType` 转回 `aipubt::Quantization` |
| `GetQuantization(type)` | 统一入口，根据类型自动选择转换方法 |

## 依赖

- `aipugc/QuantUtils.h` — 头文件声明
- `aipugc/Dialect.h` — AIPU 方言定义
- `aipugc/utils.h` — 工具函数（如 `GetDtype`）
- `mlir/IR/DialectImplementation.h` — MLIR 方言基础设施
- `llvm/Support/Debug.h` — 调试宏

## 数据流

```
aipubt::Tensor (量化参数)
        │
        ▼
QuantizationToType / QuantizationToPerAxisType
        │
        ▼
UniformQuantizedType / UniformQuantizedPerAxisType (MLIR IR 中使用)
        │
        ▼
GetQuantization
        │
        ▼
aipubt::Quantization (输出/序列化)
```
