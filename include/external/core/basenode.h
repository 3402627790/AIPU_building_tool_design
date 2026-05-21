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
 * @Last Modified time: 2022-04-19 16:15:51
 */
#ifndef CORE_BASENODE_H
#define CORE_BASENODE_H

#include "core/any.h"
#include "core/optype.h"
#include "core/tensor.h"
#include "tongue/symbolic.h"
#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>
namespace aipubt
{
class BaseNode;
class ComputeTensor;
using BaseNodePtr = std::shared_ptr<BaseNode>;
using BaseNodeRef = std::weak_ptr<BaseNode>;
struct ReadOnlyParam;

/**
 * @brief a union class for store value of Node Param in IR
 * 
 */
struct NodeParamValue
{
    /**
     * @brief the Node paramerter type
     * 
     */
    enum NodeParamType
    {
        NodeParamType_bool = 0, //for bool type
        NodeParamType_dtype,    //for data type
        NodeParamType_int,
        NodeParamType_uint,
        NodeParamType_float,

        NodeParamType_string,
        NodeParamType_shape,
        NodeParamType_void, // for empty param, which will not serialize to IR
        NodeParamType_expr, // for dynamic shape expression

        NodeParamType_arr = 0x7f,
        NodeParamType_arr_bool = 0x80,
        NodeParamType_arr_dtype,
        NodeParamType_arr_int,
        NodeParamType_arr_uint,
        NodeParamType_arr_float,
        NodeParamType_arr_string,
        NodeParamType_arr_shape,
        NodeParamType_arr_expr,
    };

    /**
     * @brief the basic value of Node Parameter
     * 
     */
    union V {
        bool b;
        Tensor::Dtype t;
        long long i;
        unsigned long long ui;
        double f; //use double for all float data
    };

    /**
     * @brief the basic value
     * 
     */
    union V v = {.b = 0};

    /**
     * @brief the string value
     * 
     */
    std::string s;

    /**
     * @brief the expression value
     * 
     */
    tongue::Expr expr;

    /**
     * @brief for a nested array value
     * 
     */
    std::vector<NodeParamValue> arr;

    /**
     * @brief the paramter type
     * 
     */
    NodeParamType t = NodeParamType_bool;

    NodeParamValue();
    NodeParamValue(const std::string &s);
    NodeParamValue(const char *s);
    NodeParamValue(int i);
    NodeParamValue(long int i);
    NodeParamValue(long long i);
    NodeParamValue(bool b);
    NodeParamValue(Tensor::Dtype t);
    NodeParamValue(unsigned int i);
    NodeParamValue(unsigned long int i);
    NodeParamValue(unsigned long long i);
    NodeParamValue(float f);
    NodeParamValue(double f);
    NodeParamValue(TensorShape f);
    NodeParamValue(tongue::Expr o);
    NodeParamValue(const ReadOnlyParam &ro);
    NodeParamValue(const NodeParamValue &other);
    NodeParamValue(const ComputeTensor t);
    template <typename T>
    NodeParamValue(std::initializer_list<T> l)
    {
        for (auto &&i : l)
        {
            push_back(NodeParamValue(i));
        }
        t = NodeParamType_arr;
    }
    template <typename T>
    NodeParamValue(std::vector<T> l)
    {
        for (auto &&i : l)
        {
            push_back(NodeParamValue(i));
        }
        t = NodeParamType_arr;
    }
    template <typename T>
    NodeParamValue(std::initializer_list<std::initializer_list<T>> l)
    {
        for (auto &&i : l)
        {
            push_back(NodeParamValue(i));
        }
        t = NodeParamType_arr;
    }
    NodeParamValue &operator=(const NodeParamValue &other);

    /**
     * @brief push a value a the (nested) list node parameter
     * 
     * @param b 
     */
    void push_back(NodeParamValue b);

    /**
     * @brief push a value a the (nested) list node parameter
     * 
     * @param b 
     */
    void push_back(bool b);

    /**
     * @brief push a value a the (nested) list node parameter
     * 
     * @param b 
     */
    void push_back(Tensor::Dtype b);

    /**
     * @brief push a value a the (nested) list node parameter
     * 
     * @param i 
     */
    void push_back(unsigned int i);

    /**
     * @brief push a value a the (nested) list node parameter
     * 
     * @param i 
     */
    void push_back(int i);

    /**
     * @brief push a value a the (nested) list node parameter
     * 
     * @param f 
     */
    void push_back(float f);

    /**
     * @brief push a value a the (nested) list node parameter
     * 
     * @param f 
     */
    void push_back(double f);

    operator bool() const;
    operator int() const;
    operator long int() const;
    operator long long() const;
    operator unsigned int() const;
    operator unsigned long int() const;
    operator unsigned long long() const;
    operator float() const;
    operator half() const;
    operator bfp16() const;
    operator double() const;
    operator std::string() const;
    operator TensorShape() const;
    operator Tensor::Dtype() const;
    operator tongue::Expr() const;
    template <typename T>
    operator std::vector<T>() const
    {
        std::vector<T> ret;
        if (arr.size())
        {
            for (auto i : arr)
            {
                ret.push_back(i);
            }
        }
        else if (t == aipubt::NodeParamValue::NodeParamType_string && (s == "" || s == "[]"))
        {
            return {};
        }
        else if (t == aipubt::NodeParamValue::NodeParamType_arr && arr.empty())
        {
            return {};
        }
        else
        {
            NodeParamValue val = *this;
            ret.push_back(val);
        }
        return ret;
    }

    template <typename T>
    T operator+(const T &other)
    {
        return ((T)(*this)) + other;
    }
    template <typename T>
    friend T operator+(const T &other, NodeParamValue &v)
    {
        return ((T)(v)) + other;
    }
    template <typename T>
    T operator-(const T &other)
    {
        return ((T)(*this)) - other;
    }
    template <typename T>
    friend T operator-(const T &other, NodeParamValue &v)
    {
        return other - ((T)(v));
    }
    template <typename T>
    T operator*(const T &other)
    {
        return ((T)(*this)) * other;
    }
    template <typename T>
    friend T operator*(const T &other, NodeParamValue &v)
    {
        return other * ((T)(v));
    }
    template <typename T>
    T operator/(const T &other)
    {
        return ((T)(*this)) / other;
    }
    template <typename T>
    friend T operator/(const T &other, NodeParamValue &v)
    {
        return other / ((T)(v));
    }
    bool operator==(const NodeParamValue &other) const;

    bool operator==(const char *other) const
    {
        return ((std::string)(*this)) == std::string(other);
    }

    template <typename T>
    bool operator==(const T &other) const
    {
        return ((T)(*this)) == (other);
    }
    template <typename T>
    friend bool operator==(const T &other, NodeParamValue &v)
    {
        return v.operator==(other);
    }
    template <typename T>
    bool operator!=(const T &other) const
    {
        return !operator==(other);
    }
    template <typename T>
    friend bool operator!=(const T &other, NodeParamValue &v)
    {
        return !v.operator==(other);
    }
    template <typename T>
    bool operator<(const T &other) const
    {
        return ((T)(*this)) < other;
    }
    template <typename T>
    friend bool operator<(const T &other, NodeParamValue &v)
    {
        return v.operator>=(other);
    }
    template <typename T>
    bool operator<=(const T &other) const
    {
        return ((T)(*this)) <= other;
    }
    template <typename T>
    friend bool operator<=(const T &other, NodeParamValue &v)
    {
        return v.operator>(other);
    }
    template <typename T>
    bool operator>(const T &other) const
    {
        return ((T)(*this)) > other;
    }
    template <typename T>
    friend bool operator>(const T &other, NodeParamValue &v)
    {
        return v.operator<=(other);
    }
    template <typename T>
    bool operator>=(const T &other) const
    {
        return ((T)(*this)) >= other;
    }
    template <typename T>
    friend bool operator>=(const T &other, NodeParamValue &v)
    {
        return v.operator<(other);
    }
    NodeParamValue &operator[](int i);
    const std::string to_string() const;
};
std::ostream &operator<<(std::ostream &os, const NodeParamValue &v);

/**
 * @brief The base virtual node class for the node in graph.
 *          A node is a operation in a NN graph.
 * 
 */
class BaseNode : public std::enable_shared_from_this<BaseNode>
{
public:
    /**
     * @brief a tag for runtime time node attributes
     * 
     */
    enum NodeTag : uint32_t
    {
        None = 0b0,
        Inplace = 0b10,
        SegMMU = 0b01,
    };

    friend std::ostream &operator<<(std::ostream &os, const BaseNode &node);
    friend std::ostream &operator<<(std::ostream &os, const BaseNodePtr &node);

    // virtual BaseNode(std::string name = "unamed", OpType optype = OpType_NoOp) = 0;
    virtual ~BaseNode() = default;

    /**
     * @brief get the operation type of this node.
     * 
     * @return const OpType 
     */
    virtual OpType type() const = 0;

    /**
     * @brief get the name of this node.
     * 
     * @return const std::string& 
     */
    virtual const std::string &name() const = 0;

    /**
     * @brief Set the name of this node
     * 
     * @param name 
     */
    virtual void set_name(const std::string &name) = 0;

    /**
     * @brief get the all output memory size needed by this node.
     * 
     * @return const size_t 
     */
    virtual size_t size() const = 0;

    /**
     * @brief get the output tensors of this node.
     * 
     * @return const std::vector<TensorPtr>& 
     */
    virtual const std::vector<TensorPtr> &outputs() const = 0;

    /**
     * @brief get the input tensors
     * 
     * @return const std::vector<TensorPtr>& 
     */
    virtual const std::vector<TensorPtr> &inputs() const = 0;

    /**
     * @brief get some placeholders tensors, thess tensors will NOT use in AIPU.
     * 
     * @return std::vector<TensorPtr>& 
     */
    virtual std::vector<TensorPtr> &placeholders() = 0;

    /**
     * @brief get the parameters map of the node.
     *         The map's key is the IR's field, and the value is a NodeParamValue.
     * 
     * @return std::map<std::string, NodeParamValue>& 
     */
    virtual std::map<std::string, NodeParamValue> &params() = 0;

    /**
     * @brief get the node attributes map, a node attribute is a attribute for compiling time, which not effacts the runtime.
     * 
     * @return std::map<std::string, Any>& 
     */
    virtual std::map<std::string, Any> &attrs() = 0;
    /**
     * @brief get the node attributes map, a node attribute is a attribute for compiling time, which not effacts the runtime.
     * 
     * @return const std::map<std::string, Any>& 
     */
    virtual const std::map<std::string, Any> &attrs() const = 0;
    /**
     * @brief Set the a node attribute, a node attribute is a attribute for compiling time, which not effacts the runtime.
     * 
     * @param key the attribute's name
     * @param v the attribute's value
     */
    virtual void set_attr(const std::string &key, Any v) = 0;

    /**
     * @brief Get a node attribute with a key, a node attribute is a attribute for compiling time, which not effacts the runtime.
     * 
     * 
     * @param key the attribute key, if not exists, a Any() will be return.
     * @return Any 
     */
    virtual Any get_attr(const std::string &key) const = 0;

    /**
     * @brief erase a attributes of node
     * 
     * @param key 
     */
    virtual void erase_attr(const std::string &key) = 0;
    /**
     * @brief Get a node attribute with a key, a node attribute is a attribute for compiling time, which not effacts the runtime.
     * 
     * @tparam T the attribute type
     * @param key the attribute key, if not exists, a T() will be return.
     * @return T 
     */
    template <typename T>
    inline T get_attr(const std::string &key) const
    {
        if (attrs().count(key))
        {
            return cast_any<T>(attrs().at(key));
        }
        else
        {
            return T();
        }
    }

    /**
     * @brief get the constants tensor map, a constant tensor is a tensor which not changed in inference time.
     * 
     * @return std::map<std::string, TensorPtr>& 
     */
    virtual std::map<std::string, TensorPtr> &constants() = 0;

    /**
     * @brief get the tag of the node.
     * 
     * @return uint32_t 
     */
    virtual uint32_t tag() const = 0;

    /**
     * @brief Set a tag to this node
     * 
     * @param tag 
     */
    virtual void set_tag(uint32_t tag) = 0;

    // for other node description in dot image
    /**
     * @brief geneterat a dot string for graphviz
     * 
     * @param color the color string of this node
     * @param prefix the prefix string which would show before the node name
     * @return const std::string the dot string
     */
    virtual const std::string dot(const std::string &color = std::string(), const std::string &prefix = std::string()) = 0;

    /**
     * @brief generate a json string for a simply description of the node
     * 
     * @return const std::string 
     */
    virtual const std::string json() = 0;

    /**
     * @brief serialize the node to IR style string
     * 
     * @return const std::string 
     */
    virtual const std::string to_string() const = 0;

    /**
     * @brief serialize the node to IR
     * 
     * @param def_os the IR out stream
     * @param weight_os the binary data(for weights) out stream
     * @param id the node ID for this node
     */
    virtual void serialize(std::ostream &def_os, std::ostream &weight_os) const = 0;
};

}; // namespace aipubt

#endif //!CORE_BASENODE_H