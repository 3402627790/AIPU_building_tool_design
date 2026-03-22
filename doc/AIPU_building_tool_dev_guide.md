# AIPU Building Tool Developer Guide

* [What is AIPUBuilder](#What-is-AIPUBuilder)
* [Developing AIPUBuilder & Environment setup](#Developing-AIPUBuilder-&-Environment-setup)
* [Codebase structure](#Codebase-structure)

## What is AIPUBuilder

AIPUBuilder is a end-to-end graph compiler for ARM China AIPU product. It can compile  deep learning models from such as Tensorflow etc. to executable file for ARM China AIPU IP. Currently, it supports:  

* Tensorflow
* Tensorflow-lite (limited)
* Caffe (work in progress)
* Pytorch with ONNX (in planing)
* ... more frameworks are coming

AIPUBuilder consists of three parts as the figure shown:  

![](./images/AIPUBuilder.svg)

* Front-end: model parser
* Middle-end: auto quantization tool
* Back-end:  graph target depended optimizer and builder

## Developing AIPUBuilder & Environment setup

AIPUBuilder is hosted at git: [http://szc-gerritai02.armchina.com:8080/AIPU_building_tool_design](http://szc-gerritai02.armchina.com:8080/AIPU_building_tool_design), the git owner is [Neo WANG](mailto:neo.wang@armchina.com), if you has any permission questions, please concat  [Neo WANG](mailto:neo.wang@armchina.com). For the source code version control, please refs the [AI SW git work flow](https://armchina.sharepoint.cn/:p:/r/sites/Zhouyi_SW/_layouts/15/Doc.aspx?sourcedoc=%7B6D4B2CA8-41CA-4877-814A-37EF4B73D64A%7D&file=AISW_autoflow.pptx&action=edit&mobileredirect=true) and [git/gerrit usage](https://armchina.sharepoint.cn/:p:/r/sites/Zhouyi_SW/_layouts/15/Doc.aspx?sourcedoc=%7BF9D00442-A8FB-4BE4-AD55-C4527E9E0A0C%7D&file=5_git_Gerrit_usage.pptx&action=edit&mobileredirect=true).

The project is mainly developed in Python, for coding style, please refs [the AI SW coding style](https://armchina.sharepoint.cn/sites/Zhouyi_SW/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FZhouyi%5FSW%2FShared%20Documents%2FZY%5Fdoc%2FTraining%2FCoding%20Style&viewid=61de6956%2D015d%2D4b27%2D8903%2D798e17367fed). We encourage you use VisualStudio Code (VSCode) as IDE if you don't have a preferred one. 

### Environment Setup

The AIPUBuilder is a python package with some C/C++ module. So the environment is simple:
1. git clone the source code
```bash
git clone http://szc-gerritai02.armchina.com:8080/AIPU_building_tool_design
``` 
2. install python packages 
* numpy
* tensorflow
* networkx
* opencv-python
* editdistance
```bash
pip3 install numpy tensorflow networkx opencv-python editdistance
```

In our cluster, you python environment has install related dependencies. So you just need to load your python module.
```bash
module load swdev python/python/3.6.5
```

3. build C++ module

For more details, please refer the CPP user/dev guide.

You need load some tools for build the module:
```bash
module load swdev git/git/2.17.1  # git
module load swdev gnu/cmake/3.4.3 # cmake
module load swdev gnu/gcc/7.3.0 # g++
```
And add stdlic++ to path
```bash
export LD_LIBRARY_PATH="/arm/tools/gnu/gcc/7.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH"
```
Then build the c++ module
```bash
mkdir  build
cd build
cmake ..
make -j16
```
4. setup git hooks:

we have lint for both C/C++ and Python. Code must be formatted before push to git repo. Below instructions show how to setup the git hooks which help you automatic format the code before commit:
```bash
cd scripts
bash ./install_git_hooks.sh
```

## Codebase Structure

Below is our code structure
```
.
|-- 3rdparty
|   |-- chartjs
|   `-- pybind11     
|-- CMakeLists.txt
|-- doc
|   |-- AIPUBuilder.md
|   |-- AIPUBuilder_CPP_design.md
|   |-- AIPUBuilder_CPP_user_dev_guide.md
|   |-- AIPUBuilder_User_Guide.docx
|   |-- AIPUBuilder_user_guide.md
|   |-- AIPU_building_tool_dev_guide.md
|   |-- Builder
|   |   `-- How_to_write_a_pass.md
|   |-- GBuilder
|   |   |-- cpp_builder_plugin_design.md
|   |   `-- profiler_design.md
|   |-- images
|   `-- parser
|       |-- how_to_write_a_pass.md
|       `-- parser_arch.md
|-- env_setup.sh
|-- include
|   |-- builder
|   |-- core
|   |-- external
|   |-- gcompiler_api.h
|   `-- quantization
|-- lib
|   `-- CMakeLists.txt
|-- plugin
|   |-- CMakeLists.txt
|   `-- empty
|       `-- empty.cpp
|-- python
|   |-- Auto_quantization
|   |-- CMakeLists.txt
|   |-- Parser
|   |-- Tools
|   |-- Wrapper
|   |-- script
|   `-- src
|       |-- AIPUBuilder
|       |   |-- AutoQuantizationTool
|       |   |-- CGBuilder
|       |   |-- Optimizer
|       |   |-- Parser
|       |   |-- __init__.py
|       |   |-- core
|       |   |-- logger.py
|       |   |-- main.py
|       |   `-- plugin_loader.py
|       |-- aipu_profiler.cpp
|       |-- aipugb.cpp
|       |-- aipurun.cpp
|       |-- build.py
|       |-- build.sh
|       |-- cpython_api.h
|       |-- file_to_cython.py
|       |-- glibc_fix.h
|       |-- main.py
|       |-- op_test.py
|       `-- setup.py
|-- src
|   |-- CMakeLists.txt
|   |-- builder
|   |-- core
|   |-- quantization
|   `-- utils
|-- test
|   |-- CMakeLists.txt
|   |-- autotest
|   |-- builder
|   |-- core
|   |-- main.cpp
|   |-- quantization
|   |-- runtime
|   `-- utils
|-- tools
|   |-- CMakeLists.txt
|   |-- aipu_profiler
|   |   `-- main.cpp
|   |-- aipubinutils
|   |   `-- main.cpp
|   |-- aipugb
|   |   `-- main.cpp
|   |-- aipurun
|   |   `-- main.cpp
|   |-- argparser.h
|   `-- common_options.h
`-- tutorial
    |-- lut
    |   |-- CMakeLists.txt
    |   |-- lut_plugin.cpp
    |   |-- readme.md
    |   `-- sigmoid
    `-- parser

```

Our project is a C++ base project, we  use cmake to manage our project files.
the header files is in `include` dir, the source files is in `src` dir.

All python files is in `python` dir.
* python
    inside python, the source files is in `python/src/AIPUBuilder` dir.
    the entry point of top level is `main.py` in the python source dir.
    *  `python/src/AIPUBuilder/AutoQuantizationTool`

        the AQT module
    *  `python/src/AIPUBuilder/CGBuilder`

        the wrapper for C++ GBuilder
    *  `python/src/AIPUBuilder/Optimizer`

        the optimizer module
    *  `python/src/AIPUBuilder/Parser`
    
        the parser module
    *  `python/src/AIPUBuilder/core`

        the core module, which is implemented in C++
    due to the C++ module is bind with a specific python version, by default, we use python3.6, if you want to build for other version python, you need to modify the cmake file `python/CMakeLists.txt`:
        - change the `PYTHON_EXE` variable to your target python version's executable program.
        - or you can just add a cmake command:`-DPYTHON_EXE="PATH/to/YOUR/Python"` when you build c++ module.

All documents is in `doc` dir. please ref the `doc` dir for specific documents.

The `tools` dir is for c++ tools, including:
*   `aipurun` which is same function as python side `aipurun`, once you build the package, you can find the executable file in `build/tools/aipurun` where `build` is your build dir. 
For usage: you can run
    ```bash
    build/tools/aipurun -h
    ```

* `aipugb` same as `aipurun`, but this tool is for build a `aipu.bin` for runtime.
* `aipuprofiler` is a tools for generate a profiling report.
* `aipubinutils` is a tools for dump info and data of `aipu.bin`, it can help you debug the `aipu.bin` file.

The `tutorial` dir is for some plugins tutorials, including `Parser,AQT,GBuilder`'s plugin. Please refer `readme.md` in each tutorial for more details.

The `test` dir is for some testing code.