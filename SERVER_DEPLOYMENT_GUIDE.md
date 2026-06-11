# MQ IR Cache 功能 - 服务器部署指南

## 问题说明

你的公司服务器代码版本较旧，缺少 `Constants::mq_save_ir` 和 `Constants::mq_load_ir` 定义，导致编译错误：

```
error: 'mq_save_ir' is not a member of 'aipubt::Constants'
error: 'mq_load_ir' is not a member of 'aipubt::Constants'
```

## 解决方案

需要在服务器代码中添加这两个常量定义。

### 方法 1: 手动修改（推荐）

在服务器上编辑文件 `include/core/constants.h`，在适当位置（例如第 188 行后）添加：

```cpp
constexpr const char *mq_save_ir = "mq_save_ir";
constexpr const char *mq_load_ir = "mq_load_ir";
```

**具体步骤**：

1. SSH 登录到服务器
2. 编辑 constants.h：
   ```bash
   cd /project/ai/scratch01/qidson01/code/AIPU_building_tool_design
   vim include/core/constants.h
   ```

3. 找到类似这些行的位置（通常在 180-190 行之间）：
   ```cpp
   constexpr const char *remove_origin_constant = "remove_origin_constant";
   constexpr const char *specify_wt_gm = "specify_wt_gm";
   ```

4. 在这些行**之后**添加：
   ```cpp
   constexpr const char *mq_save_ir = "mq_save_ir";
   constexpr const char *mq_load_ir = "mq_load_ir";
   ```

5. 保存并退出（`:wq`）

6. 重新编译：
   ```bash
   make -j$(nproc)
   ```

### 方法 2: 使用 patch 文件

我已经创建了 patch 文件：`mq_cache_constants.patch`

在服务器上应用：

```bash
cd /project/ai/scratch01/qidson01/code/AIPU_building_tool_design

# 将 patch 文件上传到服务器后
patch -p1 < mq_cache_constants.patch

# 或者直接手动添加
cat >> include/core/constants.h << 'EOF'
    constexpr const char *mq_save_ir = "mq_save_ir";
    constexpr const char *mq_load_ir = "mq_load_ir";
EOF
```

### 方法 3: 直接使用字符串（临时方案）

如果不想修改 constants.h，可以在 `src/builder/tools.cpp` 中直接使用字符串：

**修改前**（第 760-767 行）：
```cpp
if (arg_map.count("mq_save_ir"))
{
    aipubt::BuilderConfig::set(Constants::mq_save_ir, cast_any<std::string>(arg_map["mq_save_ir"]));
}
if (arg_map.count("mq_load_ir"))
{
    aipubt::BuilderConfig::set(Constants::mq_load_ir, cast_any<std::string>(arg_map["mq_load_ir"]));
}
```

**修改后**：
```cpp
if (arg_map.count("mq_save_ir"))
{
    aipubt::BuilderConfig::set("mq_save_ir", cast_any<std::string>(arg_map["mq_save_ir"]));
}
if (arg_map.count("mq_load_ir"))
{
    aipubt::BuilderConfig::set("mq_load_ir", cast_any<std::string>(arg_map["mq_load_ir"]));
}
```

同时修改 `src/builder/mqbuilder.cpp`（第 144 和 181 行）：

**修改前**：
```cpp
auto mq_load_ir = BuilderConfig::get<std::string>(Constants::mq_load_ir);
auto mq_save_ir = BuilderConfig::get<std::string>(Constants::mq_save_ir);
```

**修改后**：
```cpp
auto mq_load_ir = BuilderConfig::get<std::string>("mq_load_ir");
auto mq_save_ir = BuilderConfig::get<std::string>("mq_save_ir");
```

## 验证修改

修改后重新编译：

```bash
cd /project/ai/scratch01/qidson01/code/AIPU_building_tool_design
make clean
make -j$(nproc)
```

应该能成功编译。然后测试：

```bash
# 检查帮助信息中是否有新参数
./tools/aipugb --help-hidden | grep mq

# 应该看到：
# --mq_save_ir    save the post-lowering MQ graph IR/weight...
# --mq_load_ir    load a previously saved MQ post-lowering graph...
```

## 完整的修改文件清单

如果服务器代码完全没有这个功能，需要修改以下文件：

### 1. include/core/constants.h
添加常量定义（2 行）

### 2. include/tools/common_options.h  
添加命令行参数定义（已存在则跳过）

### 3. src/builder/tools.cpp
添加参数处理逻辑（已存在则跳过）

### 4. src/builder/mqbuilder.cpp
添加序列化/反序列化逻辑（已存在则跳过）

## 检查服务器代码版本

可以通过以下命令检查哪些文件需要修改：

```bash
# 检查 constants.h 是否有定义
grep -n "mq_save_ir\|mq_load_ir" include/core/constants.h

# 检查 common_options.h 是否有参数
grep -n "mq_save_ir\|mq_load_ir" include/tools/common_options.h

# 检查 tools.cpp 是否有处理
grep -n "mq_save_ir\|mq_load_ir" src/builder/tools.cpp

# 检查 mqbuilder.cpp 是否有功能
grep -n "save_mq_ir_cache\|load_mq_ir_cache" src/builder/mqbuilder.cpp
```

如果以上命令没有输出，说明服务器代码完全没有这个功能，需要完整实现。

## 快速修复脚本

创建并运行此脚本自动修复：

```bash
#!/bin/bash
# 自动添加 mq_save_ir 和 mq_load_ir 常量定义

CONSTANTS_FILE="include/core/constants.h"

# 检查是否已经存在
if grep -q "mq_save_ir" "$CONSTANTS_FILE"; then
    echo "常量已存在，无需修改"
    exit 0
fi

# 找到插入位置（specify_wt_gm 之后）
LINE_NUM=$(grep -n "specify_wt_gm" "$CONSTANTS_FILE" | cut -d: -f1)

if [ -z "$LINE_NUM" ]; then
    echo "错误: 找不到插入位置"
    exit 1
fi

# 备份原文件
cp "$CONSTANTS_FILE" "${CONSTANTS_FILE}.bak"

# 插入新行
sed -i "${LINE_NUM}a\\    constexpr const char *mq_save_ir = \"mq_save_ir\";\n    constexpr const char *mq_load_ir = \"mq_load_ir\";" "$CONSTANTS_FILE"

echo "✓ 已添加常量定义到 $CONSTANTS_FILE"
echo "✓ 备份文件: ${CONSTANTS_FILE}.bak"
echo ""
echo "现在可以重新编译了："
echo "  make -j\$(nproc)"
```

保存为 `fix_constants.sh`，然后运行：
```bash
chmod +x fix_constants.sh
./fix_constants.sh
```

## 遇到问题？

如果修改后仍有问题，请提供：
1. 服务器上 `grep -n "specify_wt_gm" include/core/constants.h` 的输出
2. 完整的编译错误日志
3. 服务器代码的大致版本或提交日期
