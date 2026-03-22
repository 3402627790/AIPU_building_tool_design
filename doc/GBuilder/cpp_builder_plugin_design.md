# Builder CPP version Plugin design

The new CPP version Builder uses the plugin mechanisms for operator integration. We have two types of plugin in Builder:
* external plugin
* internal plugin

The external plugin is for customer's self design operator the internal plugin is the built-in operator of build tool, such as HWA lib and others. Both these two types share same code, the only different is the internal will built-in the build tool.

Below is a plugin interface design:
```cpp
class BuilderOpPlugin
{
public:
    virtual ~BuilderOpPlugin() = default;
    virtual const GraphMatcherPattern get_graph_pattern() = 0;
    virtual uint32_t get_score() = 0;
    virtual bool set_target(const char *target) = 0;
    virtual bool check_params(NodePtrList nodes) { return true; }
    virtual void setup(SubGraphNode *node, NodePtrList match_list){};
    virtual BuilderParams generate_params(SubGraphNode *node, NodePtrList &nodes) = 0;
    virtual const std::string generate_code_name(SubGraphNode *node, NodePtrList &nodes) = 0;
};
```
We have 5 interface for plugin:
* `get_score` is to return the performance score of this operator implementation, higher is mean faster.
* `get_graph_pattern` return the graph pattern for subgraph that the plugin can handle.
* `set_target` is telling the plugin what the `target` we are building now.
* `check_params` is for checking the matched pattern's parameters, whether the parameters is in the plugin's range.
* `setup` is setup the node, such as `support_mmu`.
* `generate_params` is for generate params.
* `generate_code_name` is for generate code.

The following figure shows how the plugin commutes with the build tool.
![](images/cpp/plugin_interface.svg)

1. the build tool would register all available plugins, including external/internal plugins
2. set target for all plugin
3. get score from all plugin and we sort then in descending order
4. for each plugin we try to match the pattern in our graph
5. if there are matched subgraph, we check the parameters
6. if the plugin can deal with the subgraph, builder will merge the subgraph as node, and feed to plugin, let the plugin setup the node.

### setup Node

The setup node is at very end stage, 

There is a `attr` member in Node, which is for general purpose attribute, we will use this member to store all info for communication between Core and plugin. 

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
* __next_descriptor_length__ : `Tensor*` type, this tensor is used as a pointer for dynamic change the next descriptor length, if you are using AIFF descriptor, please set up this attribute and make sure its offset is 0, if you don't use descriptor or you don't know what is descriptor **DO NOT set this attribute**.
* __next_descriptor_offset__ : `Tensor*` type, this tensor is used as a pointer for dynamic change the next descriptor offset, if you are using AIFF descriptor, please set up this attribute and make sure its offset is 0, if you don't use descriptor or you don't know what is descriptor **DO NOT set this attribute**.
* __next_descriptor_eof__ : `Tensor*` type, this tensor is used as a pointer for dynamic change the next descriptor eof tag, if you are using AIFF descriptor, please set up this attribute and make sure its offset is 0x81, if you don't use descriptor or you don't know what is descriptor **DO NOT set this attribute**.
* __deny_layouts_comb__: a data layout combination map, is a type of `std::map<std::vector<aipubt::DataLayout>,std::vector<std::vector<aipubt::DataLayout>>>`, where the map's key `std::vector<aipubt::DataLayout>` is the inputs' layout, and the value of the map `std::vector<std::vector<aipubt::DataLayout>>` is a list of forbidden outputs' layout. By default the map is empty.

Below is an example for setup node:
```cpp
typedef std::vector<ReadOnlyParam> BuilderParams;
typedef std::pair<const char *, size_t> BuilderText;
typedef std::vector<NodePtr> NodePtrList;

BuilderText generate_text(NodePtrList &nodes);
BuilderParams generate_params(NodePtrList &nodes);

void setup(SubGraphNode *node, NodePtrList match_list)
{
    node->set_attr("support_mmu", true);
    node->set_attr("workspace_size", 0x80000);
}
```

### Code emit

At the very end stage, the builder will call the `generate_params` and `generate_code_name`, in this stage, the memory has be allocated.

In this stage, you need to generate ro and the code name(your bin file name) or function name(if using elf file). 

You can get all inputs/outputs tensors layout by tensor->shape().layout(). you need generate the RIGHT ro/code for different valid layout.

#### Some attributes you can get at this stage

* __workspace__:  `Tensor*` type, you can get the workspace tensor for some buffer usage by `get_attr`, for example: `node->get_attr<Tensor*>("workspace")`.
* __parambase__:  `Tensor*` type,  if the value is not `nullptr`, this tensor is pointed to this layer's ro base.
* __first_descriptor_length__:  `Tensor*` type,  if the value is not `nullptr`, this tensor is pointed to last layer descriptor stub of `next_descriptor_length`, you can set it's offset as your first descriptor length to telling last descriptor your descriptor length.
* __first_descriptor_offset__: `Tensor*` type,  if the value is not `nullptr`, this tensor is pointed to last layer descriptor stub of `next_descriptor_offset`, you can set it's offset as your first descriptor offset to telling last descriptor your descriptor offset.
* __previous_descriptor_eof__:  `Tensor*` type,  if the value is not `nullptr`, this tensor is pointed to last layer descriptor stub of `descriptor_eof`, you can set it's offset as **0x18** if this layer using descriptor too, if you don't use descriptor or you don't know what is descriptor **DO NOT modify this tensor** .

### Memory Object

The memory object is defined as below.
```cpp
struct MemoryObject
{
    uint32_t relative_offset = 0;
    uint32_t m_base = 0;
    uint32_t align = 128;
    uint32_t mask = 0;
    uint32_t size = 0;
    MemoryObject *base_offset = nullptr;
    // if the memory object is a virtual memory buffer, contains with several physical memory,
    // the we use a linked list for the physical memory
    MemoryObject *next_block = nullptr;

    MemoryObject(uint32_t base, int off = 0);
    MemoryObject(int base, int off = 0);
    MemoryObject(MemoryObject *base, int off = 0);
    uint32_t offset();
    void set_base_offset(uint32_t base);
    void set_base_offset(MemoryObject *base);
    MemoryObject *get_base_object();
    MemoryObject *get_mmu_object();
    uint32_t get_base_memory();
};
```

For `MMU`, if is the segmmu in ZYv1, the memory object is made by two physical memory which are stored at `base_offset` and `next_block`. If the object is one physical memory, not mmu, the `next_block` would be null.

## Relative classes of builder

### Node/SubGraphNode

Node is a core object for representing a layer/operation in the neural network graph.

below is the Node definition, we have simplified the definition. We only focus on the main functions need for plugin development.

```cpp

class Node
{
public:
    Node(std::string name = "unamed", OpType optype = OpType_NoOp);
    virtual ~Node();
    // return the optype
    const OpType type() const;
    const std::string &name() const;
    // the parents' node
    const std::vector<Node *> parents();
    // the children's node
    const std::vector<Node *> children();
    std::vector<Tensor *> &outputs();
    // input tensors
    std::vector<Tensor *> &inputs();
    // placeholders, if you allocate a tensors, but not use by other node(not inputs/outputs/constants), and the tensor is shared other's memory, you should put it in placeholders
    std::vector<Tensor *> &placeholders();
    // access constants Tensors: weights/biases/lut/scale/shift
    std::map<std::string, Tensor *> &constants();
    // access parameters
    NodeParamRef operator[](const std::string &key);
    // get parameters map
    std::map<std::string, NodeParamValue> &params();
    // set an attribute
    void set_attr(const std::string &key, Any v);
    // get an attribute
    template <typename T>
    inline T get_attr(const std::string &key)
    {
        if (m_attrs.count(key))
        {
            return cast_any<T>(m_attrs[key]);
        }
        else
        {
            return T();
        }
    }
};
```


### NodeParamValue

NodeParamValue is a class for represent all parameters in IR. 
It support :
```cpp
enum NodeParamType
{
    NodeParamType_bool = 0, // true/false in IR
    NodeParamType_dtype, // data type: int8/uint8/int16/uint16/int32/uint32/Q0.8 etc.
    NodeParamType_int, // int number
    NodeParamType_uint,// unsigned int number
    NodeParamType_float,// float number
    NodeParamType_string, // string
    NodeParamType_shape, // shape: [1,1,1]
};
```
The NodeParamValue has many conversion functions, so you can just use it like:

```cpp
Tensorshape shape=node->params()["m_shape"];
int integer=node->params()["scale"];
std::string method=node->params()["method"];
```

### Tensor

Tensor is a class represent sorts of data in graph.

```cpp
class Tensor
{
public:
    enum Dtype
    {
        VOID = 0x0,
        BOOL = 0x1,
        UINT8 = 0x2,
        INT8 = 0x3,
        UINT16 = 0x4,
        INT16 = 0x5,
        UINT32 = 0x6,
        INT32 = 0x7,
        UINT64 = 0x8,
        INT64 = 0x9,
        FP16 = 0xa,
        FP32 = 0xb,
        FP64 = 0xc,
        BFP16 = 0xd,

        Q0_8 = 0x20,
        Q8_8 = 0x21,
        Q0_16 = 0x22,
        Q8_16 = 0x23,
        Q16_16 = 0x24,
        Q0_32 = 0x25,
    };


    Tensor(const string &name, const TensorShape &shape, Dtype type = UINT8);
    Tensor(const string &name, void *data = 0, size_t size = 0);
    Tensor() = delete;
    //copy
    Tensor(const Tensor &t);
    Tensor &operator=(const Tensor &t);
    ~Tensor();
    // return a tensor by name if the tensor has be created, else return null
    // static Tensor *get_tensor(const string &tensor_name);
    // get the tensor data if have
    template <typename T>
    T *data()
    {
        return reinterpret_cast<T *>(m_data);
    }
    TensorShape &shape();
    // the size of the tensor, which is product of shape
    size_t size();
    // the memory size of the tensor
    size_t mem_size();
    // get the memory object
    MemoryObject *memory_offset();
    // set data
    void set_data(void *d, uint32_t size);
    void set_type(Dtype t);
    Dtype type();
    const std::string &type_s();
    const std::string &name() const;
    bool set_name(const std::string &name);
};
```

### TensorShape

Tensorshape is for representing a shape.
```cpp
class TensorShape
{
public:
    TensorShape(int d0 = 1);
    TensorShape(int d0, int d1);
    TensorShape(int d0, int d1, int d2);
    TensorShape(int d0, int d1, int d2, int d3);
    TensorShape(std::vector<int> &shape);
    TensorShape(const TensorShape &ts);
    TensorShape(const std::string &str);
    //access by index
    int &operator[](int index);
    bool operator==(const TensorShape &other);
    std::vector<int>::iterator begin();
    std::vector<int>::iterator end();
    // get/set h axis, it the shape has no h, it will raise assertion error
    int &h();
    // get/set w axis, it the shape has no w, it will raise assertion error
    int &w();
    // get/set c axis, it the shape has no c, it will raise assertion error
    int &c();
    // get/set n axis, it the shape has no n, it will raise assertion error
    int &n();
    // to a string like [1,2,3,4]
    const std::string to_string() const;
    // get the product of all dim
    size_t size();
    // get dimension's number
    int dim();
    // just set the layout
    void set_layout(DataLayout lo);
    // change the layout from one to other, the shape also change
    void change_layout(DataLayout layout);
    // get layout
    DataLayout layout();
};
```

