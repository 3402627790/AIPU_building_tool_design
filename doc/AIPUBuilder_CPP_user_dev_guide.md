# AIPUBuilder C++ User Guide

## Installation
 prereuqests
 * git  `module load swdev git/git/2.17.1`
 * cmake   `module load swdev gnu/cmake/3.4.3`
 * gcc/g++ `module load swdev gnu/gcc/7.3.0`

Because we use gcc 7.3.0, the stdlib should use the corresponding version, so we need add the lib path to env
```bash
export LD_LIBRARY_PATH="/arm/tools/gnu/gcc/7.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH"
```

Run the scripts for build the gbuilder
```bash
git clone "ssh://xxxxxx@szc-gerritai02.armchina.com:29418/AIPU_building_tool_design"
cd AIPU_building_tool_design
mkdir  build
cd build
cmake ..
make -j16
```

## For plugin Dev
For plugin Dev, please do use the prebuilt lib for dev.

The latest prebuilt lib is on `/project/ai/scratch01/AIPUBuilder/`. So for dev, you need:

* add include path `/project/ai/scratch01/AIPUBuilder/include`
* add library path for LD `/project/ai/scratch01/AIPUBuilder/lib/`
* add the lib `aipubuildtool` in your compile command.

here is a simple example:
```bash
g++ test.cpp -fPIC -shared -o libTest.so -I /project/ai/scratch01/AIPUBuilder/include -L /project/ai/scratch01/AIPUBuilder/lib/ -laipubuildtool 
```

or you can ref the `tutorial`, there is a cmake project for the plugin dev.

## For user and tester

There is also a prebuilt app in `/project/ai/scratch01/AIPUBuilder/bin` that you can use it.
But you need to set up the LD_LIBRARY_PATH before: add `/project/ai/scratch01/AIPUBuilder/lib` to LD_LIBRARY_PATH.
So we recommend that you add these to your env source `.bashsrc` or `.cshrc`
```bash
export PATH="/project/ai/scratch01/AIPUBuilder/bin:$PATH"
export LD_LIBRARY_PATH="/project/ai/scratch01/AIPUBuilder/lib:/arm/tools/gnu/gcc/7.3.0/rhe7-x86_64/lib64:$LD_LIBRARY_PATH"
```
Then you can just run `aipurun` in any terminal.

For running a network, you need hwa lib, so you need make a directory `plugin` in your working directory and add the 'libhwa.so' to this folder.
```bash
mkdir -p plugin
cp /path/to/libhwa.so plugin
aipurun -help
```

You can see these log if you have load the plugin successfully:
```bash
[D] [plugin_loader.cpp:  68] loading plugin: 
name: hwa_conv
version: 0x10000000
[D] [plugin_loader.cpp:  68] loading plugin: 
name: hwa_conv_eltwise
version: 0x10000000
```