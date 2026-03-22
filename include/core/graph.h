//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-02-09 12:15:11
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-04 10:39:29
 */
#ifndef CORE_GRAPH_H
#define CORE_GRAPH_H

#include "any.h"
#include "tensor_shape.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace tongue
{
class Symbol;
class Expr;
} // namespace tongue
namespace aipubt
{
class Node;
class Tensor;
class SubGraph;
class IRChecker;
class Graph;
using TensorPtr = std::shared_ptr<Tensor>;
using NodePtr = std::shared_ptr<Node>;
using GraphPtr = std::shared_ptr<Graph>;
using GraphRef = std::weak_ptr<Graph>;
using SubGraphPtr = std::shared_ptr<SubGraph>;

struct SymbolWrap
{
    SymbolWrap(tongue::Expr sym, tongue::Symbol::Value val);
    SymbolWrap(tongue::Expr sym) : SymbolWrap(sym, 0) {}
    SymbolWrap(const tongue::Symbol &sym) : SymbolWrap(tongue::Expr(sym), 0) {}
    SymbolWrap(const std::string &sym) : SymbolWrap(tongue::Symbol(sym), 0) {}
    SymbolWrap(const SymbolWrap &sym);
    SymbolWrap() = default;

    tongue::Expr expr;
    uint32_t type = 0; // 0: global, 1: data aware
    std::vector<int> dims;
    tongue::Symbol::Value value = 0;
};

/**
 * @brief a graph object for the NN graph
 *
 */
class Graph : public std::enable_shared_from_this<Graph>
{
public:
    static GraphPtr create(std::string name = "Unnamed", DataLayout layout = DataLayout_NHWC);
    virtual ~Graph();
    /**
     * @brief parse a graph for a IR file
     *
     * @param filename  the file name of IR file
     * @param g the target graph, if null, a new graph would be created
     * @param check_ir if check the IR format is correct.
     * @return GraphPtr  the graph of the IR
     */
    static GraphPtr parse(const std::string &filename, GraphPtr g = nullptr, bool check_ir = true);

    /**
     * @brief parse a graph for a IR file
     *
     * @param ifs  IR stream
     * @param g the target graph, if null, a new graph would be created
     * @param check_ir if check the IR format is correct.
     * @return GraphPtr  the graph of the IR
     */
    static GraphPtr parse(std::istream &ifs, GraphPtr g = nullptr, bool check_ir = true);

    /**
     * @brief parse a IR from file
     *
     * @param filename IR file name
     * @param check_ir  if check the IR format
     */
    void parse_from(const std::string &filename, bool check_ir = true);

    /**
     * @brief get the graph's name.
     *
     * @return const std::string&
     */
    const std::string &name() const;

    /**
     * @brief Set the graph name
     *
     * @param name
     */
    void set_name(const std::string &name);

    /**
     * @brief Set a attribute of the graph
     *
     * @param key the attribute key
     * @param v  the attribute value
     */
    void set_attr(const std::string &key, Any v);

    /**
     * @brief Get a graph attribute by key
     *
     * @param key the attribute's key, a Any() will be retrun if the key not exists.
     * @return Any
     */
    Any get_attr(const std::string &key) const;

    /**
     * @brief get the attributes map
     *
     * @return std::map<std::string,Any>& : the attributes map
     */
    std::map<std::string, Any> &attrs();

    /**
     * @brief get the global tensor map, a global tensor is a graph shared tensor, such as workspace.
     *
     * @return std::map<std::string, TensorPtr>&
     */
    std::map<std::string, TensorPtr> &global_tensors();

    /**
     * @brief get the node list of the graph
     *
     * @return const std::vector<NodePtr>&
     */
    const std::vector<NodePtr> &nodes() const;
    std::vector<NodePtr>::iterator begin();
    std::vector<NodePtr>::iterator end();

    /**
     * @brief check if a tensor in the graph
     *
     * @param t the tensor to be checked
     */
    bool has_tensor(TensorPtr t);

    /**
     * @brief add a node to the graph
     *
     * @param n the node to be added
     */
    virtual void add_node(NodePtr n);

    /**
     * @brief remove a node from the graph, please note that, the node's relation would not be remove, please make sure the node is clean.
     *
     * @param n the node to be removed.
     * @param clean_edge  clean inputs and output before remove from graph.
     */
    void remove_node(NodePtr n, bool clean_edge = true);

    // suggestion colors "chartreuse", "beige", "gold", "greenyellow", "purple"
    /**
     * @brief genereate a dot file of the graph for graphviz
     *
     * @param filename the dot filename
     * @param colors the colors for drawing nodes.
     */
    void to_dot(const std::string &filename = "graph.dot", const std::vector<std::string> &colors =
                                                               {"antiquewhite",
                                                                "azure",
                                                                "beige",
                                                                "chartreuse",
                                                                "coral",
                                                                "cyan2",
                                                                "darkolivegreen1",
                                                                "gold",
                                                                "gray",
                                                                "greenyellow",
                                                                "lightblue1",
                                                                "purple",
                                                                "rosybrown1"

                                                               }) const;

    /**
     * @brief generate a png image by using graphviz, please make sure the graphviz is installed properly. and `dot` command is in the PATH.
     *
     * @param filename
     *
     * @return true: build successfully
     *         false: failed to build
     */
    bool to_png(const std::string &filename = "graph.png") const;

    /**
     * @brief generate a svg image by using graphviz, please make sure the graphviz is installed properly. and `dot` command is in the PATH.
     *
     * @param filename
     *
     * @return true: build successfully
     *         false: failed to build
     */
    bool to_svg(const std::string &filename) const;

    /**
     * @brief generate a json string description of the graph.
     *
     * @return const std::string
     */
    const std::string json();

    /**
     * @brief generate a dot string for graphviz.
     *
     * @param colors the colors for drawing nodes.
     * @param with_header if with the dot header
     * @return const std::string
     */
    const std::string dot(const std::vector<std::string> &colors = {}, bool with_header = true) const;

    /**
     * @brief serialize the graph to IR
     *
     * @param def_file the IR file name
     * @param weight_file the binary weights filename
     */
    void serialize(const std::string &def_file = "graph.def", const std::string &weight_file = "weight.bin");

    /**
     * @brief serialize the graph to IR
     *
     * @param oss the IR out stream
     * @param wt_oss the binary weights out stream
     * @param sg_fmt serialize to subgraph format
     */
    void serialize(std::ostream &oss, std::ostream &wt_oss, bool sg_fmt = false);

    /**
     * @brief reset the layer_id of each node
     */
    void reset_layer_id();

    /**
     * @brief check if the graph is topological sorted
     *
     * @return True if the nodes is on topological order or vice versa.
     */
    bool is_topological_sorted() const;

    /**
     * @brief sort the node in topology
     *
     * @param reserve_orders for reserver orders nodes, these nodes order would not be changed when sorting.
     */
    void topological_sort(const std::vector<std::vector<NodePtr>> &reserve_orders = std::vector<std::vector<NodePtr>>());

    /**
     * @brief sort the nodes in topology by using start/end nodes to determinate which subgraph to be sorted.
     *
     * @param begin_nodes start nodes to be sorted
     * @param end_nodes end nodes to sorted
     * @param reserve_orders  for reserver orders nodes, these nodes order would not be changed when sorting.
     * @return std::vector<NodePtr>
     */
    static std::vector<NodePtr> topological_sort(const std::vector<NodePtr> &begin_nodes, const std::vector<NodePtr> &end_nodes, const std::vector<std::vector<NodePtr>> &reserve_orders = std::vector<std::vector<NodePtr>>());

    /**
     * @brief sort a node list in topology
     *
     * @param nodes the node list to be sorted
     * @param reserve_orders  for reserver orders nodes, these nodes order would not be changed when sorting.
     * @return std::vector<NodePtr>
     */
    static std::vector<NodePtr> topological_sort(const std::vector<NodePtr> &nodes, const std::vector<std::vector<NodePtr>> &reserve_orders = std::vector<std::vector<NodePtr>>());

    /**
     * @brief sort the node by specific order
     *
     * @param ordered_nodes the ordered node list
     */
    bool sort_by_ordered(const std::vector<NodePtr> &ordered_nodes);

    // replace a node by other, input and output tensors will be move from the replaced to new one
    // edges will be automaticaly setup
    /**
     * @brief replace a node by other, the tensors and relation will be replace with new node.
     *
     * @param replaced the node to be replaced
     * @param by the node replaces other
     */
    void replace_node_by(NodePtr replaced, NodePtr by);

    /**
     * @brief eliminate a node, and make ling with it's input to it's output,
     *  if the input tensors's size equals to the output tensors' size,
     *  then will perform the change, and return true, else return false.
     *  if n does exist in graph, nothing will change, return false.
     *  if the node's output(s) is a graph output tensors, nothing will change.
     *
     * @param n the node to be eliminated.
     * @return true
     * @return false
     */
    bool eliminate_node(NodePtr n);

    /**
     * @brief eliminate a node, and make ling with it's input to it's output,
     *  if the input tensors's size equals to the output tensors' size,
     *  then will perform the change, and return true, else return false.
     *  if n does exist in graph, nothing will change, return false.
     *  if the node's output(s) is a graph output tensors,
     *  the graph output tensor will be replaced by node's input tensor.
     *
     * @param n the node to be eliminated.
     * @return true
     * @return false
     */
    bool eliminate_node_aware_out(NodePtr n);
    /**
     * @brief load the weight
     *
     * @param filename weight filename
     */
    void load_weight(const std::string &filename = "");

    /**
     * @brief load the weight
     *
     * @param infile weight stream
     */
    void load_weight(std::istream &infile);
    /**
     * @brief get node by name
     *
     * @param name node name
     * @return NodePtr
     */
    NodePtr operator[](const std::string &name);

    /**
     * @brief Get the bottom nodes object, where bottom node is a node without parent.
     *
     * @return std::vector<NodePtr>
     */
    std::vector<NodePtr> get_bottom_nodes() const;

    /**
     * @brief Get the input nodes object, where input node is all node with a OpType_Input.
     *
     * @return std::vector<NodePtr>
     */
    std::vector<NodePtr> get_input_nodes() const;

    /**
     * @brief Get the output nodes object
     *
     * @return std::vector<NodePtr>
     */
    std::vector<NodePtr> get_output_nodes() const;

    /**
     * @brief Get the perf nodes object
     *
     * @return std::vector<NodePtr>
     */
    std::vector<NodePtr> get_perf_nodes() const;

    /**
     * @brief Get the input tensors object
     *
     * @return std::vector<TensorPtr>
     */
    std::vector<TensorPtr> get_input_tensors();

    /**
     * @brief Get the output tensors object
     *
     * @return std::vector<TensorPtr>
     */
    std::vector<TensorPtr> get_output_tensors();

    /**
     * @brief record the input tensors
     *
     * @param input_tensors the input tensor list, if empty we will use the input node's tensor as input tensors.
     */
    void record_input_tensors(const std::vector<TensorPtr> &input_tensors = std::vector<TensorPtr>());

    /**
     * @brief record the output tensors
     *
     * @param output_tensors the output tensor list, if empty we will inference the output tensors.
     */
    void record_output_tensors(const std::vector<TensorPtr> &output_tensors = std::vector<TensorPtr>());

    // get all nodes count, if a node is a subgraph, we will count it's subnodes.
    /**
     * @brief Get the total nodes num
     *
     * @return uint32_t
     */
    uint32_t get_total_nodes_num() const;

    /**
     * @brief get the global layout of this graph
     *
     * @return DataLayout
     */
    DataLayout layout() const
    {
        return m_layout;
    }

    /**
     * @brief Set the layout object
     *
     * @param lo the layout to be set
     */
    void set_layout(DataLayout lo)
    {
        m_layout = lo;
    }

    /**
     * @brief Get the tensor object by name
     *
     * @param name the tensor name, if the tensor is not exists, nullptr will be return.
     * @return TensorPtr
     */
    TensorPtr get_tensor(const std::string &name)
    {
        auto t = m_tensors[name];
        if (!t)
        {
            update_tensor_map();
            t = m_tensors[name];
        }
        return t;
    }

    /**
     * @brief Get the tensors map of this graph
     *
     * @return const std::map<std::string,TensorPtr>&  the tensor map.
     *
     */
    const std::map<std::string, TensorPtr> &tensors() const
    {
        return m_tensors;
    }

    /**
     * @brief get subgraphs
     *
     * @return std::vector<SubGraphPtr>&
     */
    std::vector<SubGraphPtr> &subgraphs()
    {
        return m_subgraphs;
    }

    // NodePtr get_node_by_tensor(const std::string& tname) ;
    // NodePtr get_node_by_tensor(const TensorPtr t) ;
    /**
     * @brief Get the valid tensor name object
     *
     * @param name base name
     * @return std::string
     */
    std::string get_valid_tensor_name(const std::string &name = "");
    /**
     * @brief Get the valid node name object
     *
     * @param name base name
     * @return std::string
     */
    std::string get_valid_node_name(const std::string &name = "") const;

    /**
     * @brief clone this graph
     *
     * @param g the return graph, if nullptr, a new graph will be created.
     * @return GraphPtr
     */
    GraphPtr clone(GraphPtr g = nullptr) const;

    /**
     * @brief clone this graph with begin nodes and end nodes
     * clone a subgraph with given begin nodes and end nodes
     *
     * @param begin_nodes a node ptr list for begin node
     * @param end_nodes a node ptr list for end node
     * @param g the return graph, if nullptr given, a new graph will be created
     * @param insert_input  true: insert input node before begin nodes if they are not input node, false: not insert.
     * @return GraphPtr
     */
    GraphPtr clone(const std::vector<NodePtr> &begin_nodes, const std::vector<NodePtr> &end_nodes, GraphPtr g = nullptr, bool insert_input = false) const;

    /**
     * @brief clone this graph with begin tensors and end tensors
     * clone a subgraph with given begin tensors and end tensors
     *
     * @param begin_tensors a tensors ptr list for begin node
     * @param end_tensors a tensors ptr list for end node
     * @param g the return graph, if nullptr given, a new graph will be created
     * @param insert_input  true: insert input node before begin nodes if they are not input node, false: not insert.
     * @return GraphPtr
     */
    GraphPtr clone(const std::vector<TensorPtr> &begin_tensors, const std::vector<TensorPtr> &end_tensors, GraphPtr g = nullptr, bool insert_input = false) const;

    /**
     * @brief clone this graph with begin nodes and end nodes
     * clone a subgraph with given begin nodes and end nodes
     *
     * @param begin_nodes a node name list for begin node
     * @param end_nodes a node name list for end node
     * @param g the return graph, if nullptr given, a new graph will be created
     * @param insert_input  true: insert input node before begin nodes if they are not input node, false: not insert.
     * @return GraphPtr
     */
    GraphPtr clone(const std::vector<std::string> &begin_nodes, const std::vector<std::string> &end_nodes, GraphPtr g = nullptr, bool insert_input = false) const;

    /**
     * @brief Get the all graphs of this main graphs recursively.
     *
     * @return const std::vector<GraphPtr>  all subgraphs including self
     */
    const std::vector<GraphPtr> get_all_graphs() const;

    /**
     * @brief walk on node from top to down
     *
     */
    void walk_on_node(std::function<void(NodePtr)> f, bool recursive = true, bool top2down = true);

    // extend the graph by searching input node's
    // void extend(const std::vector<TensorPtr>& out_tensors);

    bool show_graph(int depth = 0);
    bool check_valid_name() const;
    bool check_sg() const;
    bool check_valid_io() const;
    std::string update_node_dot_name(NodePtr node, const std::string &name);

    /**
     * @brief check whether the graph is dynamically shaped.
     *
     * @return true
     * @return false
     */
    bool is_dynamic() const;

    void set_symbol(const SymbolWrap &s);
    void erase_symbol(const std::string &name);
    void refresh_symbols();

    /**
     * @brief get global symbols
     * 
     * @return return global symbol table of graph
     */
    const std::map<std::string, SymbolWrap> &symbols();

    /**
     * @brief replace graph symbols according to upper input tensors
     * 
     * @return true
     * @return false
     */
    bool replace_symbols(const std::vector<TensorPtr> &upper_inputs);

protected:
    Graph(std::string name = "Unnamed", DataLayout layout = DataLayout_NHWC);
    template <typename TensorType>
    static GraphPtr parse(const std::string &filename, GraphPtr g = nullptr, bool check_ir = true);
    template <typename NodeType, typename TensorType>
    static GraphPtr parse(const std::string &filename, GraphPtr g = nullptr, bool check_ir = true);
    void update_tensor_map();
    GraphPtr clone(const std::vector<NodePtr> &all_nodes, GraphPtr g = nullptr, bool insert_input = false) const;

    std::string m_name;
    DataLayout m_layout;
    std::map<std::string, NodePtr> m_nodes_map;
    std::vector<NodePtr> m_nodes;
    std::map<std::string, TensorPtr> m_global_tensors;
    std::map<std::string, TensorPtr> m_tensors;
    std::vector<TensorPtr> m_in_tensors;
    std::vector<TensorPtr> m_out_tensors;
    std::vector<SubGraphPtr> m_subgraphs;
    std::map<std::string, NodePtr> m_node_dot_name_map;
    std::map<std::string, Any> m_attrs;
    std::map<std::string, SymbolWrap> m_symbols;
    friend IRChecker;
    friend Node;
    friend Tensor;
};

GraphPtr get_default_graph();
} // namespace aipubt

#endif //! CORE_GRAPH_H