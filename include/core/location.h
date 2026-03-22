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
 * @Author: Alex Zheng
 * @Date: 2022-07-25 11:44:32
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-07-25 11:44:32
 */

#include "core/node.h"
#include <memory>
#include <string>
#include <vector>

#ifndef CORE_LOCATION_H
#define CORE_LOCATION_H

#define ADD_LOCATION_BY_NODE_WITH_NAME(name, n, ...)  \
    {                                                 \
        if (!n->loc())                                \
            n->set_loc(std::make_shared<Location>()); \
        n->loc()->add_roots({__VA_ARGS__});           \
        n->loc()->set_type(n->type());                \
        n->loc()->set_name(#name);                    \
    }

#define ADD_LOCATION_BY_NODE(n, ...)                  \
    {                                                 \
        if (!n->loc())                                \
            n->set_loc(std::make_shared<Location>()); \
        n->loc()->add_roots({__VA_ARGS__});           \
        n->loc()->set_type(n->type());                \
        n->loc()->set_name(__func__);                 \
    }

#define MODIFY_NODE(n)                                \
    {                                                 \
        if (!n->loc())                                \
            n->set_loc(std::make_shared<Location>()); \
        auto loc = n->loc();                          \
        n->set_loc(std::make_shared<Location>());     \
        n->loc()->add_roots({loc});                   \
        n->loc()->set_type(n->type());                \
        n->loc()->set_name(__func__);                 \
    }
#define ADD_LOCATION_BY_TENSOR(t, ...) \
    ADD_LOCATION_BY_NODE(t->node(), __VA_ARGS__)

namespace aipubt
{
class Tensor;
class Node;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorPtrList = std::vector<TensorPtr>;
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;

class Location;
using LocationPtr = std::shared_ptr<Location>;
using LocationList = std::vector<LocationPtr>;

class Location
{
public:
    Location() {}
    Location(const std::string name, OpType type = OpType_NoOp, bool is_original = false)
        : m_name(name),
          m_type(type),
          m_is_original(is_original)
    {
    }
    Location(const std::string name, LocationPtr root, OpType type = OpType_NoOp, bool is_original = false)
        : m_name(name),
          m_type(type),
          m_is_original(is_original)
    {
        m_roots.clear();
        m_roots.push_back(root);
    }

    Location(const std::string name, LocationList roots, OpType type = OpType_NoOp, bool is_original = false)
        : m_name(name), m_roots(roots), m_type(type), m_is_original(is_original) {}

    /**
     * @brief get/set the name.
     *
     * @return name.
     */
    std::string &name();

    /**
     * @brief get the location name.
     *
     * @return name.
     */
    std::string name() const;

    /**
     * @brief set the location name.
     *
     * @return none.
     */
    void set_name(std::string name);

    /**
     * @brief get/set the location list.
     *
     * @return location list.
     */
    LocationList &roots();

    /**
     * @brief get the location list.
     *
     * @return location list
     */
    LocationList roots() const;

    /**
     * @brief set the location list.
     *
     * @return none.
     */
    void set_roots(LocationList roots);

    /**
     * @brief set the location list from location list.
     *
     * @return none.
     */
    void add_roots(LocationList roots);

    /**
     * @brief set the location list from tensor.
     *
     * @return none.
     */
    void add_roots(TensorPtrList tensors);

    /**
     * @brief set the location list from tensor.
     *
     * @return none.
     */
    void add_roots(NodePtrList nodes);

    /**
     * @brief Set the node type of location
     *
     * @return none.
     */
    void set_type(OpType type);

    /**
     * @brief return the node type of location
     *
     * @return node type.
     */
    OpType &type();

    /**
     * @brief Set param m_is_original.
     *        true: From the original file.
     *        false: Not from the original file.
     * @return none.
     */
    void set_original(bool is_original);

    /**
     * @brief Returns whether this location is from the original file.
     *
     * @return true/false.
     */
    bool is_original();

    /**
     * @brief check the trace location whether contains a pass
     * 
     * @param pass_name the pass name to check
     * @return true 
     * @return false 
     */
    bool through_pass(const std::string &pass_name);
    std::string to_string(LocationPtr loc);
    std::string to_string();
    std::string serialize();
    std::vector<std::string> all();

private:
    std::string m_name;
    LocationList m_roots;
    OpType m_type;
    bool m_is_original = false;
    void collect_names(LocationPtr loc, std::vector<std::string> &names);
};

}; // namespace aipubt
#endif