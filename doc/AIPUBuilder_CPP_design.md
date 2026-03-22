# AIPUBuilder C++ desgin and roadmap

## Introduction 

### Purpose: Why a new C++ version
The AIPUBuilder python version has developed over one year and has a rich feature set and relative stable, so why we develope a new C++ version?

There are 5 points:   

1. the urgent request is that the Android NNAPI can not use over python builder, hence Android runtime need to implement a simplified builder for NNAPI. If we can provide a C++ version, the Android runtime will get benefit from it.    

2. The python version is designed on module, there is no a global design. So there are many redundant code it python version, and each module should maintain these code. The C++ version will regard the Builder as one single tool, we will solve this problem.

3. The python version is designed for fast implementing and fast feature enabling but the performance does not count in design. The C++ version should be faster and lower memory consumption than python.

4. Enabling the plugin feature. The python version is difficult to add a plugin, but in C++ version, we can allow use to develop themselves' plugin for special requirement. 

5. The python version is difficult to protect our source code due to the python is a interpreted language, although we use cython to convert the source to binary file, but it relays cython's implementation. The C++ version will not have this problem.

### C++ version design 

In global the C++ version will following the python version's design, but we will explicitly introduce a new module: core function, the core will be the holder for other module: builder, quantization tool and parser. 

Below is a rough file structure of the project. 

```
|-- CMakeLists.txt
|-- build
|-- include
|   |-- builder
|   |-- core
|   |-- parser
|   |-- quantization
|   `-- aipubuilder_api.h
|-- plugin
|   |-- CMakeLists.txt
|   `-- plugin_examples
|-- src
|   |-- CMakeLists.txt
|   |-- builder
|   |-- core
|   |-- parser
|   |-- quantization
|   `-- utils
|-- test
`-- tools
    |-- CMakeLists.txt
    |-- aipugb
    |-- aipurun
    |-- aqt
    `-- parser
```

We use CMake for project management.  
* `build` directory is for build the project. 
* `include` is for all external API's header files, the internal headers should be in `src`
* `plugin` is just some example plugins for guide customer how to implement a plugin.
* `src` is all the src file for the project. `core` is all the core function, `builder` is the backend part, `quantization` is the  for AQT, `parser` is the frontend. The `core` is necessary for other part. the aqt, parser may be implemented later.
* ` test` is for some unit test functions.
* `tools` is the cmd tool for customer, such as `aipurun`,`aipubuilder`, it just an entry point.


### Core design

We follow the Python version design concept.
`Graph` is a container of `Node`, `Node` will be the host of data and operation.

Also, we will introduce the `SubGraph` for graph hierarchy, as well as `SubGraphNode` will be introduced as a `Node` of a subgraph in a parent graph.

`Tensor` will be as the data container, `TensorShape` also will be introduced for shape operation.



### Builder Plugin design

All HWA node should be a plugin in C++ design, so that the lib development will be independent the Builder development.

For the HWA interface, here is draft design:

1. register all available plugin
2. get plugin score and sort in decrease order
3. get graph pattern for plugin
4. give matched nodes to plugin, plugin will check the params for checking whether the plugin eats it.
5. if yes, the pattern will convert to a SubgraphNode, and feed to the plugin
6. the plugin will setup the node: setup attributes such as `use_mmu`
7. Builder will allocate memory for all node, including plugin's subgraph node
8. if plugin supports mmu, the allocator will allocate mmu if possible, and set to the Node attributes
9. builder call plugin to generator params and text


