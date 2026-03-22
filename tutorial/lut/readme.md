# Tutorial: How to write a CPP Builder Plugin for Layerlib

We are using LUT plugin as an example, which is a Activation type in IR, you can found this tutorial and code at: /project/ai/scratch01/AIPUBuilder/tutorial

we need 4 step to add a plugin:

1. writing a pattern for telling Builder which layer we goto support, setup score
2. check the parameter is ok and setup.
3. generate the ro, the ro should match the assembly code's param.
4. generate the code name.

## 1. Pattern & score
the pattern is combined by two list: node list and edge list:

* a node is defined by `std::pair<std::string,aipubt::OpType>` {name,type}, for example `{"n1",OpType_Activation}`

* a edge is defined by `std::pair<std::string,std::string>` {node1's name, node2's name}, the two nodes' names of the edge, for example `{"n1","n2"}`

Our case is simple, we only have one node, and 0 edge, so our code is as follow:
```cpp
const GraphMatcherPattern get_graph_pattern()
{
        return {{{"n1", OpType_Activation}}, {}};
}
```
The score is a unsigned int number. Builder will choose the highest score plugins, if several plugin has same pattern. So in our case, we return a 10. For all TPC layers, we suggest you can return 10;
```cpp
uint32_t get_score()
{
    return 10;
};
```

## 2. Parameter check and setup

In this section, you need check the parameter such as the height/width of feature is supported by your asm code.
In our example we just return true, no checking.

The set_target is for check if your plugin support this target, the target is the target code such as "Z1_0904". Here we return true for all target.

The setup is for other attributes: such workspace_size,  suppport_mmu and layouts' info. The layouts info is must.

#### required attributes

* __valid_input_layouts__: for telling builder what kinds of input layout supported by the node
    This is a `std::vector<std::vector<aipubt:: DataLayout>>` type.
* __valid_output_layouts__: for telling builder what kinds of input layout supported by the node
    This is a `std::vector<std::vector<aipubt:: DataLayout>>` type.
* __keeping_layout__: Bool type, if you node need keep same layout for input/output tensors, set this attribute as true

#### optional attributes

* __workspace_size__ : `int` type, how large temp buffer need in this node.
* __support_mmu__:  `bool` type, if the node support seg mmu, if support, builder will allocate a sram or mmu buffer for it.
* others: you can set any other attributes for later use in RO/Code generation stage.

```cpp
bool check_params(NodePtrList nodes)
{
    return true;
}
bool set_target(const char *target)
{
    m_target_name = target;
    return true;
};
void setup(SubGraphNodePtr node, NodePtrList match_list)
{
    node->set_attr("keeping_layout", true);
    LayoutCombinations valid_in = {
        {DataLayout_NCHWC32},
        {DataLayout_NCHWC16},
        {DataLayout_NHWC},
        {DataLayout_NCHW},
    };
    LayoutCombinations valid_out = {
        {DataLayout_NCHWC32},
        {DataLayout_NCHWC16},
        {DataLayout_NHWC},
        {DataLayout_NCHW},
    };
    node->set_attr("valid_input_layouts", valid_in);
    node->set_attr("valid_output_layouts", valid_out);
}
```

## 3. RO generation

In this phase, you will return a RO list. This is the core of the plugin, to generate asm's parameters according to the node's info.

The first args of the function is `SubGraphNode*` which is a node the represent you plugin. The second arg is a Nodelist `std::vector<NodePtr >` which is original nodes. The origin nodes has all info for your generate ro. !! Nodes is readonly, we should not modify them.

In our case, the nodes only has one node, because our pattern is a single node pattern. And the sgnode is also only represent one node. For TPC layer, generally we do not to care sgnode.

The ReadOnlyPram is a wrapper of `uint32_t` and `Tensor*`, so you can push back a `uint32_t` number ot a `Tensor*` tensor(feature map)

#### The IR v.s. Node

Below table show how to get a IR value via the Node object. For more details, please refer the node.h, tensor.h, tensor_shape.h

In summary

* a layer in IR is a node 
* a layer top/bottom is a Tensor, which has shape, type attribute
* a weights/biases/LUT/scale(per channel)/shift(per channel) is a Tensor in node->constants(), you can access with key: `weights/biases/lut/scale/shift`
* other parameters are in node->params(), and accessible by their key in IR.

| IR                                        | Node Object                                         |
|-------------------------------------------|-----------------------------------------------------|
| layer_id=3                                | not care                                            |
| layer_name=featuresPre/Tanh               | node->name()                                        |
| layer_type=Activation                     | node->type()                                        |
| layer_bottom=featuresPre/conv2d/BiasAdd_0 | node->inputs()                                      |
| layer_bottom_shape=[1,1,390,128]          | node->inputs()[i]->shape()                          |
| layer_bottom_type=int8                    | node->inputs()[i]->type()                           |
| layer_top=featuresPre/Tanh_0              | node->outputs()                                     |
| layer_top_shape=[1,1,390,128]             | node->outputs()[i]->shape()                         |
| layer_top_type=int8                       | node->outputs()[i]->type()                           |
| method=TANH                               | node->params()["method"]                            |
| scale_type=Q0.8                           | node->params()["scale_type"]                        |
| scale_value=0.98828125                    | node->params()["scale_value"]                       |
| shift_type=int8                           | node->params()["shift_type"]                        |
| shift_value=7                             | node->params()["shift_value"]                       |
| lut_type=int8                             | node->constants()["lut"]->type()                    |
| lut_offset=3456                           | node->constants()["lut"]->memory_object()->offset() |
| lut_size=256                              | node->constants()["lut"]->mem_size()                |
| lut_shape=256,1                           | node->constants()["lut"]->shape()                   |
```cpp
BuilderParams generate_params(SubGraphNodePtr sgnode, std::vector<NodePtr > &nodes)
{
    auto node = nodes[0];
    BuilderParams ro;
    int input_shape_dim = node->inputs()[0]->shape().dim();
    int start_idx = input_shape_dim < 4 ? 0 : 1;
    for (int i = 0; i < 3; i++)
    {
        if (i < input_shape_dim)
        {
            ro.push_back(node->inputs()[0]->shape()[start_idx + i]);
        }
        else
        {
            ro.push_back(1);
        }
    }
    // push the Tensor* lut, which is a lut table
    ro.push_back(sgnode->constants()[node->name()+"/lut"]);
    // push the input feature map
    ro.push_back(sgnode->inputs()[0]);
    // push the output feature map
    ro.push_back(sgnode->outputs()[0]);

    return ro;
}
```

#### Important Notice

Different between sgnode and nodes

The sgnode is a subgraph node, which contains all nodes, but it acts as a node. And buildtool would treat it as a node. 

All nodes in the subgraph is invisible in buildtool. __all tensors inputs outputs and cosntants should use the sgnode one.__

The key of constants (weights/biases/scale/shift/lut) will change, because there may be many nodes has weights. The key would be added a scope of node name, for example: 

* nodes[0]'s weights would has a key: nodes[0]->name()+"/weights",


## 4 Code generation and register

The section is just return which bin file you return. For different target, we use the function `complete_text_code_name` to complete the name.

the register is a macro func, you need  provide the plugin class name and a version. the plugin type is fixed for builder.

```cpp
const std::string generate_code_name(SubGraphNodePtr sgnode, NodePtrList &nodes) override
{
    auto node = nodes[0];
    std::string activation_method = node->params()["method"];
    std::string name;
    if (activation_method == "sigmoid")
    {
        name = "sigmoid";
    }
    else if (activation_method == "tanh")
    {
        name = "tanh";
    }
    name += ".bin";
    //if your bin file name is follow the suffix-target rule:
    // for zy_v1 target: would add a suffix: _zy_sc002/_zy_sc003, for qian add _z1_1104
    // you can use the function complete_text_code_name to complete your code name

    return complete_text_code_name(name);
    // else you need deal with the bin filename(function name) by your self.
    if(m_target_name=="Z1_1104")
    {
        name="qian_1104_"+name;
    }
    else
    {
        //other cases
    }
    return name;
}

// classname, version(here is 1.1.0), pluginType
REGISTER_PLUGIN(LUTPlugin, 0x01010000, BUILDERPLUGIN_TYPE);

```


## How to build


```bash
mkdir build
cd build
cmake ..
make -j4
```

Now the build folder will have a <pluginname>.so, which is your plugin.

## How to run

you plugin can not run alone. It need cpp builder.
The cpp builder will search all .so files  a folder named `plugin` in your current working directory(cwd).

so we make a dir named plugin and copy our .so file into it.
```bash
mkdir plugin
cp *.so plugin
```

Now we can a see run `/project/ai/scratch01/AIPUBuilder/bin/aipurun -v` to check if our plugin loaded, and prepare a graph IR and data to test our plugin.
```log
/project/ai/scratch01/AIPUBuilder/bin/aipurun -v

[D] [plugin_loader.cpp:  71] loaded plugins: EmptyPlugin EmptyPlugin2 HwaActivation HwaBn HwaConv HwaConvEltwise HwaConvEltwisePooling HwaConvLut HwaConvPooling HwaFc HwaFcLut HwaGru HwaLstm HwaRnn LUTPlugin 
/project/ai/scratch01/AIPUBuilder/bin/aipurun   version: 2.0.681
plugin versions
HwaActivation                   version:1.0.102
HwaBn                           version:1.0.102
HwaConv                         version:1.0.102
HwaConvEltwise                  version:1.0.102
HwaConvEltwisePooling           version:1.0.102
HwaConvLut                      version:1.0.102
HwaConvPooling                  version:1.0.102
HwaFc                           version:1.0.102
HwaFcLut                        version:1.0.102
HwaGru                          version:1.0.102
HwaLstm                         version:1.0.102
HwaRnn                          version:1.0.102
LUTPlugin                       version:16.0.0
[D] [plugin_loader.cpp:  81] unload plugin: /project/ai/scratch01/AIPUBuilder/bin//plugin/libempty.so
[D] [plugin_loader.cpp:  81] unload plugin: /project/ai/scratch01/AIPUBuilder/bin//plugin/libhwa.so
[D] [plugin_loader.cpp:  81] unload plugin: plugin/liblutplugin.so
Total errors: 0,  warnings: 0
```
Now you plugin is completed and can be run by aipurun.