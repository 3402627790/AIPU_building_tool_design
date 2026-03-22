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
 * @Date: 2020-02-09 12:15:26
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-10 14:07:14
 */
#ifndef CORE_NODE_H
#define CORE_NODE_H

#include "core/any.h"
#include "core/basenode.h"
#include "core/location.h"
#include "core/optype.h"
#include "core/tensor.h"
#include <algorithm>
#include <map>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
namespace aipubt
{
class Node;
class Graph;
class GraphMatcher;
using GraphPtr = std::shared_ptr<Graph>;
using GraphRef = std::weak_ptr<Graph>;
using NodePtr = std::shared_ptr<Node>;
using NodeRef = std::weak_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
using StringMap = std::map<std::string, std::string>;
using StringMapPtr = std::shared_ptr<StringMap>;

template <typename Container>
bool inline contains(Container const &c, typename Container::const_reference v)
{
    return std::find(c.begin(), c.end(), v) != c.end();
}

template <typename Container, class UnaryPredicate>
bool inline all_of(Container const &c, UnaryPredicate p)
{
    return std::all_of(c.begin(), c.end(), p);
}

class Node : public BaseNode
{
    friend std::ostream &operator<<(std::ostream &os, const NodePtr &node);

public:
    static NodeParamValue parse_param_value(const std::string &s, bool has_expr = false);

    static NodePtr create(const std::string &name = "unamed", OpType optype = OpType_NoOp);
    Node(const Node &) = delete;
    Node &operator=(const Node &) = delete;
    struct NodeParamRef
    {
        NodeParamRef(std::map<std::string, NodeParamValue> &m_map, const std::string &key) : m_map(m_map), key(key) {}
        operator NodeParamValue();
        NodeParamValue &operator=(NodeParamValue v);
        std::string &operator=(std::string s);
        std::map<std::string, NodeParamValue> &m_map;
        const std::string &key;
    };
    virtual ~Node();
    template <typename T>
    std::shared_ptr<T> get_shared_ptr()
    {
        return std::static_pointer_cast<T>(shared_from_this());
    }
    NodePtr get_shaped_ptr()
    {
        return std::static_pointer_cast<Node>(shared_from_this());
    }
    OpType type() const override;
    const std::string &name() const override;
    void set_name(const std::string &name) override;
    size_t size() const override;
    /**
     * @brief add an output tensor, the tensor's owner will change to this node
     * 
     * @param t the tensor to be added
     * @param idx index of the output
     */
    virtual void add_output(TensorPtr t, int idx = -1);

    /**
     * @brief remove an output tensor by a tensor pointer
     * 
     * @param t the tensor to be removed
     * @return idx index of the output, it the tensor is not an output tensor of the node, a INT32_MAX will return.
     */
    virtual int remove_output(TensorPtr t);

    /**
     * @brief remove an output tensor by index
     * 
     * @param idx index of the output, if the idx is negative, it will remove idx+outputs.size().
     * @return t the tensor to be removed, if the idx is ileagal(>output.size() or < -outputs.size()), a nullptr will return.
     * 
     */
    virtual TensorPtr remove_output(int idx);

    /**
     * @brief add an input tensor, this node will be added to/as the tensor's consumers.
     * 
     * @param t the tensor to be added
     * @param idx index of the input, if the idx is illegal(>inputs.size() or <0), the idx will be treated as -1
     */
    virtual void add_input(TensorPtr t, int idx = -1);

    /**
     * @brief remove an input tensor by a tensor pointer
     * 
     * @param t the tensor to be removed
     * @return idx index of the input, it the tensor is not an input tensor of the node, a INT32_MAX will return.
     */
    virtual int remove_input(TensorPtr t);

    /**
     * @brief replace the input tensor with the specified tensor
     * 
     * @param old_tensor the original tensor to be replaced
     * @param new_tensor the replaced tensor
     */
    virtual void replace_input_by(TensorPtr old_tensor, TensorPtr new_tensor);

    /**
     * @brief replace the output tensor with the specified tensor
     * 
     * @param old_tensor the original tensor to be replaced
     * @param new_tensor the replaced tensor
     */
    virtual void replace_output_by(TensorPtr old_tensor, TensorPtr new_tensor);

    /**
     * @brief remove an input tensor by index
     * 
     * @param idx index of the input, if the idx is negative, it will remove idx+input.size().
     * @return t the tensor to be removed, if the idx is ileagal(>input.size() or < -input.size()), a nullptr will return.
     * 
     */
    virtual TensorPtr remove_input(int idx);

    /**
     * @brief add input tensors, all tensors will be added by order with idx -1
     * 
     * @param inps  tensors to be added
     */
    virtual void add_input(const std::vector<TensorPtr> &inps);

    /**
     * @brief add outputs tensors, all tensors will be added by order with idx -1
     * 
     * @param inps  tensors to be added
     */
    virtual void add_output(const std::vector<TensorPtr> &outs);

    /**
     * @brief remove all inputs tensors.
     * 
     */
    virtual void clear_inputs();

    /**
     * @brief remove all outputs tensors.
     * 
     */
    virtual void clear_outputs();

    /**
     * @brief compare a node is equal to other
     *         the compare content only limits inside a node, i.e. type, parameters, constants
     *         and compare the input/outputs tensors' dtype, shape, value(if has)
     *          
     * 
     * @param other 
     * @return true if equal 
     * @return false if not equal
     */
    virtual bool equal(const NodePtr other) const;

    /**
     * @brief Clone a node, the graph connections will not be cloned, below items will be cloned.
     *     1. inputs/outputs tensors
     *     2. all placeholders/constants tensor will be used copy contructor to clone
     *     3. params will be cloned
     *     4. all attributes will be use copy contructor to clone
     *     5. basic properties: type,name,shape,tag... will be copied.
     *     6. Graph, parents,children ... will NOT be cloned.
     * 
     *  a Cloned Node should equal to original one.
     *  
     * @return NodePtr 
     */
    virtual NodePtr clone() const;

    /**
     * @brief get the parents of the node
     * 
     * @return const std::vector<NodePtr>, parents of the node
     */
    const std::vector<NodePtr> parents() const;

    /**
     * @brief get the children of the node
     * 
     * @return const std::vector<NodePtr>, children of the node.
     */
    const std::vector<NodePtr> children() const;

    const std::vector<TensorPtr> &outputs() const override;
    const std::vector<TensorPtr> &inputs() const override;
    std::vector<TensorPtr> &placeholders() override;
    NodeParamRef operator[](const std::string &key);

    /**
     * @brief Get the free outputs: the free tensor is which has no consumer.
     * 
     * @return std::vector<TensorPtr> 
     */
    std::vector<TensorPtr> get_free_outputs();

    std::map<std::string, NodeParamValue> &params() override;
    std::map<std::string, Any> &attrs() override;
    const std::map<std::string, Any> &attrs() const override;
    void set_attr(const std::string &key, Any v) override;
    Any get_attr(const std::string &key) const override;
    void erase_attr(const std::string &key) override;
    void copy_attrs(NodePtr n)
    {
        m_attrs = n->m_attrs;
    }
    template <typename T>
    inline T get_attr(const std::string &key) const
    {
        if (m_attrs.count(key))
        {
            return cast_any<T>(get_attr(key));
        }
        else
        {
            return T();
        }
    }

    /**
     * @brief Get the attributes by key, 
     * if the node is SubGraphNode and the key is not presented in the attributes, 
     * if will try to find the attributes of subnode, once the key match, it will return immediately.
     * if not found, an empty Any return.
     * 
     * @param key the attribute's key
     * @return Any the attribute value
     */
    Any get_attr_deep(const std::string &key) const;

    /**
     * @brief Get the attributes by key, 
     * if the node is SubGraphNode and the key is not presented in the attributes, 
     * if will try to find the attributes of subnode, once the key match, it will return immediately.
     * if not found, an empty Any return.
     * 
     * @param key the attribute's key
     * @return Any the attribute value
     */
    template <typename T>
    inline T get_attr_deep(const std::string &key) const
    {

        return cast_any<T>(get_attr_deep(key));
    }

    void erase_attr_deep(const std::string &key, bool only_first = true);

    std::map<std::string, TensorPtr> &constants() override;
    uint32_t tag() const override;
    void set_tag(uint32_t tag) override;

    // for other node description in dot image
    virtual const std::string dot(const std::string &color = std::string(), const std::string &prefix = std::string()) override;
    virtual const std::string json() override;
    virtual const std::string to_string() const override;
    virtual void serialize(std::ostream &oss, std::ostream &wt_oss) const override;
    virtual void serialize(std::map<char *, uint32_t> &data_map, std::ostream &oss, std::ostream &wt_oss, std::ostream &sg_oss) const;

    LocationPtr &loc();
    void set_loc(LocationPtr loc);

    std::vector<TensorShape> infer_shape(bool only_infer_symbol = false);

    /**
     * @brief check whether the node is dynamically shaped.
     * 
     * @return true 
     * @return false 
     */
    bool is_dynamic();

    GraphPtr graph() const;
    std::string get_node_dot_name();

    void add_dot_section(const std::string &key, const std::string &value);
    bool has_attr_deep(const std::string &key) const;
    uint64_t get_constant_size() const;
    uint64_t get_output_size() const;

    int get_layer_id() const;

    bool replace_symbols(const std::map<std::string, tongue::ExprPtr> &symbols);

    void update_constants();

protected:
    Node(const std::string &name = "unamed", OpType optype = OpType_NoOp);
    virtual void add_child_helper(NodePtr child);
    virtual void add_parent_helper(NodePtr parent);
    virtual void remove_child_helper(NodePtr child);
    virtual void remove_parent_helper(NodePtr parent);
    virtual void set_graph(GraphPtr g);

    std::string m_name;
    OpType m_type;
    uint32_t m_tag = 0;
    std::vector<NodeRef> m_parents;
    std::vector<NodeRef> m_children;
    std::vector<TensorPtr> m_outputs;
    std::vector<TensorPtr> m_inputs;
    std::map<std::string, TensorPtr> m_constants;
    std::vector<TensorPtr> m_placeholders;
    std::map<std::string, NodeParamValue> m_params;
    std::map<std::string, Any> m_attrs;
    LocationPtr m_loc = nullptr;
    GraphRef m_graph;
    friend Graph;
    friend Tensor;
    friend GraphMatcher;
};

// void move_inputs(NodePtr from, NodePtr to);
// void copy_inputs(NodePtr from, NodePtr to);
// void swap_inputs(NodePtr a, NodePtr b);
} // namespace aipubt

#endif //!CORE_NODE_H
