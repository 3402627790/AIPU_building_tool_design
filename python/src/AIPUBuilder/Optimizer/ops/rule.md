## Module 背景
AIPUBuilder/Optimizer/ops下面存放各类Compass支持的算子的前向以及量化function。

## 参考文件
@include AIPUBuilder/Optimizer/ops/abs.py
@include AIPUBuilder/Optimizer/ops/__init__.py

请参考abs算子的实现得知算子实现规则。

## 检查事项

对于任何算子提交，额外检查
1. 是否正确处理self等变量名问题
2. 是否额外使用了.clone()等可能会造成张量复制的操作，如果发现，作为warning报告
3. 检查commit是否依赖__init__.py的初始化
