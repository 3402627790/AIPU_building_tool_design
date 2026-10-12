//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang 
 * @Date: 2020-02-09 12:15:57 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-21 16:55:58
 */
#ifndef CORE_SUBGRAPH_H
#define CORE_SUBGRAPH_H

#include <map>
#include <memory>
#include <set>
#include <vector>
namespace aipubt
{
class Node;
class Graph;

class SubGraphNode;
class LoopNode;
class IfNode;
class ProxyTensor;
using NodePtr = std::shared_ptr<Node>;
using NodeRef = std::weak_ptr<Node>;
using SubGraphNodePtr = std::shared_ptr<SubGraphNode>;
using SubGraphNodeRef = std::weak_ptr<SubGraphNode>;
using ProxyTensorPtr = std::shared_ptr<ProxyTensor>;

class SubGraph : public Graph
{
public:
    static SubGraphPtr create(GraphPtr g);
    static SubGraphPtr create(GraphPtr g, const std::vector<NodePtr> &nodes);
    static SubGraphPtr create(GraphPtr g, const std::vector<NodePtr> &begin_nodes, const std::vector<NodePtr> &end_nodes);

    SubGraphNodePtr as_node(bool replace = true, bool add_sg_input = true);
    void update_global_tensors();
    void update_real_nodes();
    static void to_node(SubGraphPtr sg, SubGraphNodePtr sgn, bool add_sg_input = true, const std::set<TensorPtr> &free_out_wlist = {});
    GraphPtr origin_graph();
    SubGraphNodePtr subgraphnode() const;

protected:
    SubGraph(GraphPtr g);
    // SubGraph(GraphPtr g, const std::vector<NodePtr> &nodes);
    // SubGraph(GraphPtr g, const std::vector<NodePtr> &begin_nodes, const std::vector<NodePtr> &end_nodes);

protected:
    SubGraphPtr get_shared_ptr()
    {
        return std::static_pointer_cast<SubGraph>(Graph::shared_from_this());
    }
    std::set<TensorPtr> update_internal_tensors();
    GraphRef ori_graph = GraphRef();
    bool replace = false;
    std::vector<NodePtr> m_real_nodes;
    std::set<TensorPtr> m_internal_tensors;
    SubGraphNodeRef m_sgn;
    friend class SubGraphNode;
    friend class LoopNode;
    friend class IfNode;
};

class SubGraphNode : public Node
{
public:
    static SubGraphNodePtr create(SubGraphPtr subg, const std::string &name_ = "");
    static SubGraphNodePtr from_nodes(GraphPtr g, std::vector<NodePtr> &nodes, const std::string &name_ = "");
    ~SubGraphNode();
    static bool is_subgraph_node(BaseNodePtr n);
    static SubGraphNodePtr from_node(BaseNodePtr n);

    bool has_plugin() const;

    virtual void add_output(TensorPtr t, int idx = -1) override;
    virtual TensorPtr remove_output(int idx) override;
    virtual void add_input(TensorPtr t, int idx = -1) override;
    virtual TensorPtr remove_input(int idx) override;

    virtual bool equal(const NodePtr other) const override;

    virtual void insert_output(TensorPtr t, int idx = -1) override;
    virtual TensorPtr erase_output(int idx) override;
    virtual void insert_input(TensorPtr t, int idx = -1) override;
    virtual TensorPtr erase_input(int idx) override;

    SubGraphNodePtr get_shared_ptr()
    {
        return std::dynamic_pointer_cast<SubGraphNode>(shared_from_this());
    }
    GraphPtr subgraph() const;
    NodePtrList get_real_node()
    {
        return NodePtrList(m_sg->m_real_nodes.begin(), m_sg->m_real_nodes.end());
    }

    NodePtr clone() const override;

    // if the dot view is unified, then all subgraph node would be present as a node,
    // if not, all subnode in the subgraph would be drew.
    void set_unified_view(bool unified);
    bool unified_view()
    {
        return m_unified_view;
    }

    /**
     * @brief update attributes since the internal nodes' attributes has been changed.
     * 
     */
    void update_attrs();

    /**
     * @brief  flat subgraph node which is not unified view for simple topological structure.
     * 
     */
    void flat(bool recurrently = true);

    /**
     * @brief unfold this subgraph node, this is a inverse operation of `as_node`.
     *   the subgraph of the subgraph node will merge to the upper graph, and this node will be removed.
     * 
     */
    std::vector<NodePtr> unfold();
    const std::string dot(const std::string &color, const std::string &prefix) override;
    const std::string json() override;
    GraphPtr extract_graph(const std::string &graph_name, const std::vector<std::string> &in_names = {}, const std::vector<std::string> &out_names = {}) const;
    virtual void serialize(std::map<char *, uint32_t> &data_map, std::ostream &oss, std::ostream &wt_oss, std::ostream &sg_oss) const override;

protected:
    SubGraphNode(SubGraphPtr subg, const std::string &name_ = "");

    void update_input_tensor_map();
    void update_output_tensor_map();
    virtual void add_child_helper(NodePtr child) override;
    virtual void add_parent_helper(NodePtr parent) override;
    virtual void remove_child_helper(NodePtr child) override;
    virtual void remove_parent_helper(NodePtr parent) override;
    virtual void set_graph(GraphPtr g) override;

    NodePtr get_input_node_by_tensor(TensorPtr t, bool view);
    std::vector<NodePtr> get_output_nodes_by_tensor(TensorPtr t, bool view);
    void get_output_nodes_by_tensor(TensorPtr t, std::vector<NodePtr> &nodes, bool view);

    SubGraphPtr m_sg = nullptr;
    bool m_unified_view = false;
    std::vector<ProxyTensorPtr> m_in_tensors, m_out_tensors;
    friend class SubGraph;
};

} // namespace aipubt

#endif //!CORE_SUBGRAPH_H
