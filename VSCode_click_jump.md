这是一场非常硬核且经典的底层编译环境搭建实战！作为 AI 编译器工程师，能够在离线/本地 Linux 环境中，把庞大的 LLVM 源码树和自定义的 MLIR 子项目完美融合，绝对是一项“基建”大工程。

为了方便你日后查阅（或者在新服务器上重新部署），我把你这次遇到的所有“坑”以及最终的完美解决方案，系统地整理成了这份**《MLIR/LLVM 跨项目开发环境搭建避坑指南》**：

------

### 🚀 第一阶段：突破离线编译 `clangd` 的环境冲突

**面临问题：** 服务器无法联网下载 `clangd` 语言服务器，利用 LLVM 源码自行编译时，CMake 被环境变量中的 AIPU 交叉编译工具链（`aipuld`）拦截，导致测试程序链接失败；随后又遇到系统原生 GCC 缺少 `lld` 链接器的问题。

**解决心法：** 强制 CMake 绕过交叉编译工具链，降级使用系统原生的编译器，并关闭对高级链接器的依赖。

- 清理 `CMakeCache.txt`。
- 使用关键参数：`-DCMAKE_C_COMPILER=gcc` 和 `-DCMAKE_CXX_COMPILER=g++`。
- 关闭 LLD：`-DLLVM_ENABLE_LLD=OFF`。
- 开启编译命令导出：`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`。
- 最终通过 `ninja clangd` 成功编译出专属的语言服务器二进制文件。

------

### ⚙️ 第二阶段：VS Code 核心配置与插件防冲突

**面临问题：** VS Code 一直弹窗提示 `clangd` 未安装，且图形界面设置不生效。代码无法进行任何跳转。

**解决心法：** 因为是直接在 Linux 桌面运行 VS Code（非 SSH Remote），必须将配置写入**全局用户设置（User `settings.json`）**，并禁用微软自带的 C++ 插件智能感知，防止两者抢夺资源。

- **核心配置片段：**

  JSON

  ```
  "C_Cpp.intelliSenseEngine": "disabled",
  "clangd.path": "/project/ai/scratch01/qidson01/code/llvm-project-llvmorg-21.1.0/build/bin/clangd",
  "clangd.arguments": [
      "--background-index",
      "-j=8"
  ]
  ```

------

### 🗺️ 第三阶段：攻克跨项目与嵌套模块的跳转迷局

**面临问题：**

1. `AIPU_building_tool_design` 里的代码无法跳转到外部的 LLVM/MLIR 源码（显示为黑盒）。
2. `src/mlir` 作为独立的子模块，`clangd` 无法解析其中的自定义 Dialect 头文件（如 `aipugc/Passes.h`），导致大面积红线报错。

**解决心法：** 采用**“多根工作区 (Multi-root Workspace) + 根目录软链接 + 统一 `.clangd` 强行挂载”**的三重保险策略。

1. 在 AIPU 项目和 LLVM 项目的根目录下，分别建立指向各自 `build/compile_commands.json` 的**软链接**。

2. 在 AIPU 项目**最外层根目录**建立**全局唯一的 `.clangd`** 配置文件，强行注入所有的绝对 Include 路径，彻底解决 CMake 相对路径的迷失问题：

   YAML

   ```
   CompileFlags:
     Add: 
       # LLVM/原生 MLIR 头文件
       - "-I/.../llvm-project-llvmorg-21.1.0/llvm/include"
       - "-I/.../llvm-project-llvmorg-21.1.0/mlir/include"
       - "-I/.../llvm-project-llvmorg-21.1.0/build/tools/mlir/include"
       - "-I/.../llvm-project-llvmorg-21.1.0/build/include"
       # 自定义 AIPU MLIR 项目头文件
       - "-I/.../AIPU_building_tool_design/src/mlir/include"
       # TableGen 自动生成的 .inc 头文件
       - "-I/.../AIPU_building_tool_design/src/mlir/build/include"
       - "-I/.../AIPU_building_tool_design/src/mlir/build/tools/aipugc/include"
   ```

------

### 🧩 第四阶段：TableGen (`.td`) 与生成文件 (`.inc`) 丝滑解析

**面临问题：** VS Code 默认不认识 `.inc` 结尾的代码片段，`clangd` 拒绝解析，导致宏定义和生成的算子接口无法跳转。

**解决心法：** 在 `settings.json` 中强制进行文件类型认亲，并配合前一步的 `.clangd` 配置。

- **配置认亲：**

  JSON

  ```
  "files.associations": {
      "*.inc": "cpp",
      "*.td": "mlir"
  }
  ```

- **最佳实践：** 确保每次修改 `.td` 后先在终端跑一下编译，让真实的 `.inc` 文件落地在 build 目录中，`clangd` 就能瞬间顺藤摸瓜跳进去。

------

### 🐞 第五阶段：图形化 GDB 调试精准入轨

**面临问题：** 配置 `launch.json` 时，终端别名（alias）导致 GDB 找不到可执行文件（`does not exist`）；同时多根工作区导致 `${workspaceFolder}` 变量指代不明。

**解决心法：** 明确指定具体的项目名，并顺藤摸瓜找出可执行文件的**绝对真实路径**。

- **防坑写法：** 使用 `${workspaceFolder:AIPU_building_tool_design}` 限定项目。
- **精准打击：** 把 `"program"` 指向 `.../src/mlir/build/tool/aipugc/aipugc`。
- **原味环境：** 把 `"cwd"` 工作目录设定在 `.../src/mlir/build`，确保运行时对相对路径（如 `graph.def`）的读取与终端手敲命令完全一致。

------

### 🤝 第六阶段：团队协作保护机制

**面临问题：** 本地舒爽的专属绝对路径配置文件（`.clangd`）和软链接（`compile_commands.json`），极易被提交到 Git，导致其他同事环境崩溃。

**解决心法：** 第一时间将这些本地 IDE 配置文件加入 `.gitignore`。

Plaintext

```
# clangd configuration
.clangd
compile_commands.json
```

真正做到了“本地开发爽如飞，远端代码净如水”。

------

**最终状态：** 你现在拥有了一个固定保存的 `.code-workspace` 工作区。以后只需点开它，LLVM 源码的深邃海洋和你自己设计的 NPU 算子逻辑就能完美交织，F5 一键 Debug 随时待命。

这套环境已经非常成熟了！如果未来在编写算子时，你需要统一整个团队的代码风格，随时告诉我，我们可以继续把标准 LLVM 规范的 `.clang-format` 配置文件也安排上。