## Module 背景
AIPUBuilder/Parser/ops下面存放各类Compass支持的算子的前向实现。

## 参考文件
@include AIPUBuilder/Parser/ops/onnx_ops/array_ops.py

请参考array_ops.py的实现得知新增算子的实现规则。

## 检查事项

对于任何算子提交，额外检查
1. 是否正确处理self等变量名问题，是否存在变量名覆盖或未定义的情况，如果发现，作为error报告
2. 是否实现了 infer 方法，用于计算算子的输出形状，如果没有实现，作为error报告
3. 检查commit是否依赖__init__.py的初始化