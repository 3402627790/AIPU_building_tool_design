# AIPU Parser Architecture

* Overview design and concept
* Parser process flow and design philosophy
* `Graph` and `Node` design
* Example: how to write a pass for parser 


## Overview design and concept
The AIPU parser is design for converting mutil-framework models to our internal representation. 


## Parser process flow and design philosophy
The main objective of parser is to convert a trained model to a internal representation (IR) feeding to AutoQuantizationTool(AQT). Below is the parser architecture:  

1. a model will be feeded in by a unified configuration file
2. the entry point: configuration reader will parse the config file, and dispatch the job to a supported reader
3. one of supported reader will take over the input model,  there are two steps for read the model:
    * parse the model file(for example: protobuf/flattenbuf/json or private format) and build a raw graph for the model
    * converting the raw graph nodes to unified nodes, for example: 
        * merging several tensorflow nodes to one GRUSeq node
        * converting caffe `detectionoutput` node to `detectbox` and `nms`
        * etc.
4. the reader will generate a unified graph, then will pass to front-end optimizer
5. the front-end optimizer will operate on unified graph, it will merge or eliminate some nodes for AQT, for example:
    * merge conv and add to one node
    * merge conv/fc and bn
    * eliminate useless node: a transpose node with permutation in order, etc.

6. after optimization, then we will do once shape inference, for getting all tensor shape.

7. do some additional passes: add post-process nodes for some models

8. serialization to file.

![](../images/parser_arch.svg)






In this design and process flow, we keep some ideas in mind:

* we one has once shape inference, and it just for output a shape fixed IR for AQT.
    * p.s. if the AQT and backend support various shape model, the only one shape inference may be removed.
* after shape inference, there is not any graph operations, i.e. merging/splitting/elimination/..
    * this is because of any graph operations may change the topology of the graph, and the shape also may be changed.
    * if some parameters can be only known after known shape, that put the parameters process in shape inference stage.
* optimization passes only support unified graph, so put all framework depended passes in the model reader part, because these passes can not be used for other frameworks.

## `Graph` and `Node` design

In this parser, we use (Graph,Node) to represent models just like common framework. We use linked list type to represent the graph. The `Graph` only keep all nodes, and the topology info will store in `Node` by linking other `Node`.

In this arch, we keep thing simple, we do not introduce other object, such as `Operation`,`Layer`, etc. the `Node` is represent the IR's layer, it can be serialized simply by `serialize` method.

### `Graph` design

The Graph is simple in this arch, the main role for Graph is just a container for nodes. Below is the simplified graph definition: it has some general description information, such as name, data-layout,precision, etc. The most important member is the nodes, here we use a OrderedDict for store all nodes, the reason using OrderedDict is that:

* we can access node by its name via hash dict.  
* if we have do topological sort, the OrderedDict can keep its order.

```Python
class Graph(object):

    def __init__(self, name, data_layout="NCHW", data_precision="fp32",...):
        self.name = name
        self.layout = data_layout
        self.precision = data_precision
        self.nodes = OrderedDict()
        self.dumps = set()
        ...
```

We introduce the `Graph` for some utils in it that we can simplify some graph operations.
below listed some commonly use method:

* add_node: add a node to the graph.
* remove_node: remove a node from the graph.
* add_dump: add a dump node for the graph.
* add_outputs: add output node for the graph.
* to_png/to_svg: convert the graph to a png/svg image for visualization, it can help MUCH when debug or develop.

* from_nodes: construct a graph from many nodes.
* to_def: serialize the graph to human readable file.
* parser: deserialize a file to graph.
* print_info: print nodes information, for debug.
* as_node: convert a graph to a node, when merging a complex pattern, it can simplify much your job.
* remove_inplace_node: can simplify some simple pattern merging: [conv,add].

### `Node` design
The `Node` plays an important role in this arch, it represent a operation in a compute graph, and all input/outputs tensors are also can be find in a node.

The node definition is as below shown:

```python
class Node(object):
    graph = None

    def __init__(self, id, name, type):
        self.id = id
        self.name = str(name)
        self.type = type
        self.parents = []
        self.outputs = []
        self.inputs = []
        self.children = []
        ...
```

The three basic members of the node is:

* id: the id of the node
* name: the name of the node
* type: the operator type of the node

The `Node` uses parents/children to maintain the relationship of network.

* parents: a list of parents nodes
* children: a list of children nodes
* inputs: a list for operation input tensors
* outputs: the output tensors list


We do NOT encourage you directly manipulate the parents/children, but you can directly access them.  For adding/removing edge: we provide two methods:

* add_child/add_parent: add a child/parent for this node
* remove_child/remove_parent: remove a child/parent for this node

We override some magic functions, especially the hash function and the eq function. So the the hash is just the node's name's hash, so please pay attention when you use some hash function, for example set(). Another is the eq function, when a node==str, we just compare node's name and the string, if node==node, we check name and type. 

```python
def __hash__(self):
        return self.name.__hash__()

def __eq__(self, other):
    if isinstance(other, Node):
        return self.type == other.type and self.name == other.name
    elif isinstance(other, str):
        return self.name == other
    return False
```


## Example: how to write a pass for parser

 Please ref to the Parser pass dev guide.