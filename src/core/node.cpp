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
 * @Date: 2020-01-29 14:28:12
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-07-22 13:59:08
 */
#include "builder/gop_plugin.h"
#include "builder/plugin.h"
#include "builder/profiler.h"
#include "builder/readonly_param.h"
#include "checker/plugin.h"
#include "core.h"
#include "core/config.h"
#include "core/plugin_loader.h"
#include "core/quantization.h"
#include "core/subgraph.h"
#include "tongue/infershape/infershape.h"
#include "utils/utils.h"
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>

namespace aipubt
{

static std::set<OpType> INPLACE_NODE_TYPE = {
    OpType_Reshape,
    // OpType_Const,
    OpType_InternalSpliter,
    OpType_InternalConcatenated,
    OpType_InternalCrop,
    OpType_InternalSlice,
    OpType_InternalPad,
    OpType_Dummy,
};

static std::set<OpType> VIRTUAL_MEMORY_TYPE = {
    OpType_Reshape,
    OpType_InternalConcatenated,
    OpType_Dummy,
};

static std::set<OpType> VIEW_OP_TYPE = {
    OpType_Reshape,
    OpType_InternalConcatenated,
    OpType_InternalSpliter,
    OpType_InternalCrop,
    OpType_InternalSlice,
    OpType_Crop,
    OpType_DataStride,
    OpType_DepthToBatch,
    OpType_DepthToSpace,
    OpType_LayoutConvertor,
    OpType_SpaceToBatch,
    OpType_SpaceToBatchND,
    OpType_SpaceToChannel,
    OpType_SpaceToDepth,
    OpType_Split,
    OpType_Transpose,
    OpType_Dummy,
};

static std::set<OpType> LAYOUT_AWARE_TYPE = {
    OpType_Convolution3D,
    OpType_Convolution,
    OpType_ConvTranspose3D,
    OpType_ConvTranspose,
    OpType_CropAndResize,
    OpType_DataStride,
    OpType_DepthToBatch,
    OpType_DepthToSpace,
    OpType_DepthwiseConv,
    OpType_Dilation,
    OpType_Erosion,
    OpType_MaxPooling3D,
    OpType_MaxPoolingWithArgMax,
    OpType_MaxRoiPool,
    OpType_Pooling3D,
    OpType_Pooling,
    OpType_Resize,
    OpType_RoiAlign29,
    OpType_RoiAlign,
    OpType_ROIPooling,
    OpType_SpaceToBatch,
    OpType_SpaceToChannel,
    OpType_SpaceToDepth,

    OpType_Reshape,
    OpType_Dummy,
};

static std::set<OpType> ELEMENT_WISE_TYPE =
    {
        OpType_Abs,
        OpType_BNLL,
        OpType_Cast,
        OpType_Ceil,
        OpType_Clip,
        OpType_Cosine,
        OpType_Div,
        OpType_Eltwise,
        OpType_Elu,
        OpType_Exp,
        OpType_Floor,
        OpType_HardSigmoid,
        OpType_HardSwish,
        OpType_LeakyRELU,
        OpType_Log,
        OpType_Logical,
        OpType_LRN,
        OpType_LUT,
        OpType_Mod,
        OpType_Negative,
        OpType_Pow,
        OpType_RELU,
        OpType_RELU1,
        OpType_RELU6,
        OpType_Rsqrt,
        OpType_Sign,
        OpType_Silu,
        OpType_Sine,
        OpType_Sqrt,
        OpType_Square,
};

/**
 *  TODO Do we need remove the dependency of Plugin, to re-implement the infer shape feature?
 */
static const std::map<OpType, CheckerOpPluginInfo *> &get_shape_infer_plugins()
{
    static std::map<OpType, CheckerOpPluginInfo *> plugins;
    if (plugins.size())
    {
        return plugins;
    }
    auto disable_plugins = BuilderConfig::get<std::vector<std::string>>("disable-plugins");
    PluginLoader *m_plugin_loader = PluginLoader::get_instance();
    auto all_plugins = (*m_plugin_loader)[PluginType_IRCheckerOp];
    for (auto plugin_ : all_plugins)
    {
        CheckerOpPluginInfo *plugin = reinterpret_cast<CheckerOpPluginInfo *>(plugin_);
        auto opt = plugin->init_handle();
        auto op_type = opt->get_op_type();
        if (plugins.count(op_type) == 0)
        {
            plugins[opt->get_op_type()] = plugin;
        }
    }
    return plugins;
}

inline bool has_only_digits(const std::string &s)
{
    return s.find_first_not_of("+-0123456789") == std::string::npos;
}

inline bool is_float(const std::string &s)
{
    std::istringstream iss(s);
    float f;
    iss >> std::noskipws >> f; // noskipws considers leading whitespace invalid
    // Check the entire string was consumed and if either failbit or badbit is set
    return iss.eof() && !iss.fail();
}

Node::NodeParamRef::operator NodeParamValue()
{
    return m_map[key];
}
NodeParamValue &Node::NodeParamRef::operator=(NodeParamValue v)
{
    return m_map[key] = v;
}
std::string &Node::NodeParamRef::operator=(std::string s)
{
    m_map[key] = s;
    return m_map[key].s;
}
NodeParamValue::NodeParamValue()
{
    t = NodeParamType_void;
}
NodeParamValue::NodeParamValue(const std::string &s) : s(s), t(NodeParamType_string)
{
}
NodeParamValue::NodeParamValue(const char *s) : s(s), t(NodeParamType_string)
{
}

NodeParamValue::NodeParamValue(long int i)
{
    v.i = i;
    t = NodeParamType_int;
}
NodeParamValue::NodeParamValue(long long i)
{
    v.i = i;
    t = NodeParamType_int;
}
NodeParamValue::NodeParamValue(int i)
{
    v.i = i;
    t = NodeParamType_int;
}
NodeParamValue::NodeParamValue(bool b)
{
    v.b = b;
    t = NodeParamType_bool;
}
NodeParamValue::NodeParamValue(Tensor::Dtype t)
{
    v.t = t;
    this->t = NodeParamType_dtype;
}
NodeParamValue::NodeParamValue(unsigned int i)
{
    v.ui = i;
    t = NodeParamType_uint;
}
NodeParamValue::NodeParamValue(unsigned long int i)
{
    v.ui = i;
    t = NodeParamType_uint;
}
NodeParamValue::NodeParamValue(unsigned long long i)
{
    v.ui = i;
    t = NodeParamType_uint;
}
NodeParamValue::NodeParamValue(float f)
{
    v.f = f;
    t = NodeParamType_float;
}
NodeParamValue::NodeParamValue(double f)
{
    v.f = f;
    t = NodeParamType_float;
}
NodeParamValue::NodeParamValue(TensorShape f)
{
    for (auto i : f)
    {
        push_back(i);
    }
    t = NodeParamType_shape;
}
NodeParamValue::NodeParamValue(tongue::Expr o)
{
    expr = o;
    t = NodeParamType_expr;
}
NodeParamValue::NodeParamValue(const ReadOnlyParam &ro)
{
    if (ro.expr)
    {
        expr = ro.expr;
        t = NodeParamType_expr;
    }
    else
    {
        v.ui = ro.value;
        t = NodeParamType_uint;
    }
}

NodeParamValue::NodeParamValue(const ComputeTensor t)
{
    if (t)
    {
        auto tp = t->type();
        if (Tensor::is_int(tp))
        {
            auto c = t.astype(Tensor::INT32);
            auto data = c->data<int32_t>();
            for (auto i = 0u; i < c.size(); ++i)
            {
                push_back(data[i]);
            }
        }
        else
        {
            auto c = t.astype(Tensor::FP64);
            auto data = c->data<double>();
            for (auto i = 0u; i < c.size(); ++i)
            {
                push_back(data[i]);
            }
        }
    }
}

NodeParamValue::NodeParamValue(const NodeParamValue &other)
{
    t = other.t;
    v = other.v;
    s = other.s;
    expr = other.expr;
    arr = other.arr;
}
NodeParamValue &NodeParamValue::operator=(const NodeParamValue &other)
{
    t = other.t;
    v = other.v;
    s = other.s;
    expr = other.expr;
    arr = other.arr;
    return *this;
}

void NodeParamValue::push_back(NodeParamValue b)
{
    arr.push_back(b);
    t = NodeParamType_arr;
}
void NodeParamValue::push_back(bool b)
{
    arr.push_back(b);
    t = NodeParamType_arr;
}
void NodeParamValue::push_back(Tensor::Dtype b)
{
    arr.push_back(b);
    t = NodeParamType_arr;
}
void NodeParamValue::push_back(unsigned int i)
{
    arr.push_back(i);
    t = NodeParamType_arr;
}
void NodeParamValue::push_back(int i)
{
    arr.push_back(i);
    t = NodeParamType_arr;
}
void NodeParamValue::push_back(float f)
{
    arr.push_back(f);
    t = NodeParamType_arr;
}
void NodeParamValue::push_back(double f)
{
    arr.push_back(f);
    t = NodeParamType_arr;
}
NodeParamValue::operator bool() const
{
    if (t == NodeParamType_int || t == NodeParamType_bool || t == NodeParamType_uint)
        return v.b;
    if (t == NodeParamType_float)
        return v.f;
    return 0;
}
NodeParamValue::operator int() const
{
    if (t == NodeParamType_int || t == NodeParamType_bool || t == NodeParamType_uint)
        return v.i;
    if (t == NodeParamType_float)
        return v.f;
    return 0;
}
NodeParamValue::operator long int() const
{
    if (t == NodeParamType_int || t == NodeParamType_bool || t == NodeParamType_uint)
        return v.i;
    if (t == NodeParamType_float)
        return v.f;
    return 0;
}
NodeParamValue::operator long long() const
{
    if (t == NodeParamType_int || t == NodeParamType_bool || t == NodeParamType_uint)
        return v.i;
    if (t == NodeParamType_float)
        return v.f;
    return 0;
}

NodeParamValue::operator unsigned int() const
{
    if (t == NodeParamType_int || t == NodeParamType_bool || t == NodeParamType_uint)
        return v.ui;
    if (t == NodeParamType_float)
        return v.f;
    return 0;
}
NodeParamValue::operator unsigned long int() const
{
    if (t == NodeParamType_int || t == NodeParamType_bool || t == NodeParamType_uint)
        return v.ui;
    if (t == NodeParamType_float)
        return v.f;
    return 0;
}

NodeParamValue::operator unsigned long long() const
{
    if (t == NodeParamType_int || t == NodeParamType_bool || t == NodeParamType_uint)
        return v.ui;
    if (t == NodeParamType_float)
        return v.f;
    return 0;
}
NodeParamValue::operator double() const
{
    if (t == NodeParamType_float)
        return v.f;
    else if (t == NodeParamType_int)
        return v.i;
    else if (t == NodeParamType_uint)
        return v.ui;
    else if (t == NodeParamType_bool)
        return v.ui;
    return 0;
}

NodeParamValue::operator float() const
{
    if (t == NodeParamType_float)
        return v.f;
    else if (t == NodeParamType_int)
        return v.i;
    else if (t == NodeParamType_uint)
        return v.ui;
    else if (t == NodeParamType_bool)
        return v.ui;
    return 0;
}

NodeParamValue::operator half() const
{
    return (float)(*this);
}
NodeParamValue::operator bfp16() const
{
    return (float)(*this);
}

static const std::string node_param_value_to_string(NodeParamValue::V v, NodeParamValue::NodeParamType t);
NodeParamValue::operator std::string() const
{
    if (t == NodeParamType_string)
        return s;
    else
        return node_param_value_to_string(v, t);
}
NodeParamValue::operator TensorShape() const
{
    bool all_element_is_int = false;
    if (t >= NodeParamType_arr)
    {
        all_element_is_int = true;
        for (auto i : arr)
        {
            if (i.t != NodeParamType_int && i.t != NodeParamType_uint)
            {
                all_element_is_int = false;
                break;
            }
        }
    }
    if (arr.size() == 0)
    {
        return TensorShape();
    }
    else if (t == NodeParamType_shape || all_element_is_int || t == NodeParamType_int || t == NodeParamType_uint)
    {
        std::vector<int> shape = *this;
        return shape;
    }
    return TensorShape();
}
NodeParamValue::operator Tensor::Dtype() const
{
    if (t == NodeParamType_dtype)
        return v.t;
    return Tensor::Dtype::VOID;
}
NodeParamValue::operator tongue::Expr() const
{
    if (t == NodeParamType_expr)
    {
        return expr;
    }
    if (t == NodeParamType_int || t == NodeParamType_uint || t == NodeParamType_bool)
    {
        return Expr(v.i);
    }
    if (t == NodeParamType_float)
    {
        return Expr(v.f);
    }
    return tongue::Expr();
}

bool NodeParamValue::operator==(const NodeParamValue &other) const
{
    if (t != other.t)
    {
        return false;
    }
    if (t >= NodeParamType_arr || t == NodeParamType_shape)
    {
        if (arr.size() != other.arr.size())
        {
            return false;
        }
        bool ret = true;
        for (unsigned int i = 0; i < arr.size(); ++i)
        {
            ret = ret && (arr[i] == other.arr[i]);
        }
        return ret;
    }
    else if (t == NodeParamType_string)
    {
        return s == other.s;
    }
    else if (t == NodeParamType_expr)
    {
        if (expr && other.expr)
        {
            return static_cast<NodeParamValue>(other).expr->equal(expr);
        }
        if (!expr && !other.expr)
        {
            return true;
        }
        return false;
    }
    return v.i == other.v.i;
}

NodeParamValue &NodeParamValue::operator[](int i)
{
    return arr[i];
}

static const std::string node_param_value_to_string(NodeParamValue::V v, NodeParamValue::NodeParamType t)
{
    if (t == NodeParamValue::NodeParamType_bool)
    {
        return v.b ? "true" : "false";
    }
    else if (t == NodeParamValue::NodeParamType_dtype)
    {
        return Tensor::dtype_to_string(v.t);
    }
    else if (t == NodeParamValue::NodeParamType_uint || t == NodeParamValue::NodeParamType_int)
    {
        return std::to_string(v.i);
    }
    else if (t == NodeParamValue::NodeParamType_float)
    {
        // INFO()<<"flaot value is "<<v.f<<" string is:"<< std::to_string(v.f);
        std::stringstream oss;
        oss.precision(16);
        if ((int)v.f == v.f)
        {
            oss.precision(1);
        }
        oss << std::fixed;
        oss << v.f;
        return oss.str();
    }
    return std::string();
}
const std::string NodeParamValue::to_string() const
{
    auto ret = node_param_value_to_string(v, t);
    if (ret.size())
    {
        return ret;
    }
    else if (t == NodeParamValue::NodeParamType_shape)
    {
        TensorShape shape = *this;
        return shape.to_string();
    }
    else if (t == NodeParamType_string)
    {
        return s;
    }
    else if (t == NodeParamType_expr)
    {
        if (expr)
        {
            return expr->to_string();
        }
        return std::string();
    }
    else if (t == NodeParamType_arr)
    {
        return "[" + utils::join_string(",", arr) + "]";
    }
    else if (t == NodeParamType_void)
    {
        return std::string();
    }
    else if (t >= NodeParamType_arr_bool)
    {
        std::ostringstream oss;
        oss << "[";
        if (arr.size())
        {
            for (unsigned int i = 0; i < arr.size() - 1; ++i)
            {
                oss << node_param_value_to_string(arr[i].v, static_cast<NodeParamType>(t - NodeParamType_arr_bool)) << ",";
            }
            oss << node_param_value_to_string(arr[arr.size() - 1].v, static_cast<NodeParamType>(t - NodeParamType_arr_bool));
        }
        oss << "]";
        return oss.str();
    }
    WARN() << "Unknown node param value type!";
    return std::string();
}
std::ostream &operator<<(std::ostream &os, const NodeParamValue &v)
{
    return (os << v.to_string());
}

NodeParamValue Node::parse_param_value(const std::string &s, bool has_expr)
{
    if (s == "")
    {
        return NodeParamValue("");
    }
    else if (s == "[]")
    {
        if (has_expr)
        {
            return std::vector<Expr>();
        }
        return TensorShape();
    }
    else if (s == "True" || s == "true" || s == "TRUE")
    {
        return NodeParamValue(true);
    }
    else if (s == "False" || s == "false" || s == "FALSE")
    {
        return NodeParamValue(false);
    }
    else if (get_dtype_dict().count(s))
    {
        return NodeParamValue(get_dtype_dict().at(s));
    }
    else
    {
        NodeParamValue v;
        // parse a nest list
        if (s.find_first_of("[") != std::string::npos)
        {
            std::vector<NodeParamValue *> nested_stack;
            NodeParamValue *current_v = &v;
            auto ptr = s.begin();
            auto value_begin = s.begin();
            while (ptr != s.end())
            {
                if (*ptr == '[')
                {
                    value_begin = ptr + 1;
                    nested_stack.push_back(current_v);
                    current_v->push_back(NodeParamValue(std::vector<int>()));
                    current_v = &(current_v->arr[current_v->arr.size() - 1]);
                }
                else if (*ptr == ']')
                {
                    auto subs = s.substr(value_begin - s.begin(), ptr - value_begin);
                    subs = utils::trim(subs);
                    if (subs.size())
                    {
                        current_v->push_back(parse_param_value(subs, has_expr));
                    }
                    value_begin = ptr + 1;
                    current_v = nested_stack[nested_stack.size() - 1];
                    nested_stack.pop_back();
                }
                else if (*ptr == ',')
                {
                    auto subs = s.substr(value_begin - s.begin(), ptr - value_begin);
                    subs = utils::trim(subs);
                    bool is_value_complete = true;
                    if (has_expr && std::count(subs.begin(), subs.end(), '(') != std::count(subs.begin(), subs.end(), ')'))
                    {
                        is_value_complete = false;
                    }
                    if (is_value_complete)
                    {
                        if (subs.size())
                        {
                            current_v->push_back(parse_param_value(subs, has_expr));
                        }
                        value_begin = ptr + 1;
                    }
                }
                ptr++;
            }
            auto subs = s.substr(value_begin - s.begin(), ptr - value_begin);
            subs = utils::trim(subs);
            if (subs.size())
            {
                current_v->push_back(parse_param_value(subs, has_expr));
            }
            // if this is only a shape
            if (s[0] == '[' && v.arr.size() == 1)
            {
                NodeParamValue t = v.arr[0];
                v = t;
            }
        }
        else if (has_expr)
        {
            v.expr = Expr(s);
            v.t = NodeParamValue::NodeParamType_expr;
        }
        else
        {
            std::istringstream iss(s);
            if (has_only_digits(s))
            {
                iss >> v.v.i;
                v.t = NodeParamValue::NodeParamType_int;
            }
            else if (utils::is_number(s))
            {
                iss >> v.v.f;
                v.t = NodeParamValue::NodeParamType_float;
            }
            else
            {
                iss >> v.s;
                v.t = NodeParamValue::NodeParamType_string;
            }
        }
        return v;
    }
}
NodePtr Node::create(const std::string &name, OpType optype)
{
    return NodePtr(new Node(name, optype));
}

Node::Node(const std::string &name, OpType optype)
{
    m_name = name;
    m_type = optype;
    if (INPLACE_NODE_TYPE.count(optype))
    {
        m_tag |= Node::Inplace;
    }
    if (LAYOUT_AWARE_TYPE.count(optype))
    {
        m_attrs[Constants::layout_aware] = true;
    }
    if (ELEMENT_WISE_TYPE.count(optype))
    {
        m_attrs[Constants::elementwise_type] = true;
    }
    if (VIRTUAL_MEMORY_TYPE.count(optype))
    {
        m_attrs[Constants::virtual_mem] = true;
    }
    if (VIEW_OP_TYPE.count(optype))
    {
        m_attrs[Constants::view_op] = true;
    }
    // initial default attributes
    m_attrs[Constants::dot_sections] = std::make_shared<StringMap>();
    m_attrs[Constants::workspace] = std::make_shared<Tensor>(m_name + "/ws", TensorShape(0));
}
// Node::Node(const Node &n)
// {
// }

Node::~Node()
{
}

OpType Node::type() const
{
    return m_type;
}

const std::string &Node::name() const
{
    return m_name;
}

void Node::set_name(const std::string &name)
{
    auto g = m_graph.lock();
    if (g)
    {
        // update node map
        g->m_nodes_map.erase(m_name);
        g->m_nodes_map[name] = get_shared_ptr<Node>();
    }
    m_name = name;
}

size_t Node::size() const
{
    size_t s = 0;
    for (auto out : m_outputs)
    {
        s += out->size();
    }
    return s;
}

std::vector<TensorPtr> Node::get_free_outputs()
{
    std::set<TensorPtr> consumers;
    std::vector<TensorPtr> ret;
    for (auto c : children())
    {
        consumers.insert(c->inputs().begin(), c->inputs().end());
    }
    for (auto t : m_outputs)
    {
        if (!contains(consumers, t))
        {
            ret.push_back(t);
        }
    }
    return ret;
}

void Node::add_child_helper(NodePtr child)
{
    if (!child)
    {
        return;
    }
    if (!contains(children(), child))
    {
        m_children.push_back(child);
    }
    if (!contains(child->parents(), std::static_pointer_cast<Node>(shared_from_this())))
    {
        child->add_parent_helper(std::static_pointer_cast<Node>(shared_from_this()));
    }
}

void Node::remove_child_helper(NodePtr child)
{
    if (!child)
    {
        return;
    }

    auto it = std::find_if(m_children.begin(), m_children.end(), [child](auto ref) { return ref.lock() == child; });
    if (it != m_children.end())
    {
        m_children.erase(it);
    }
    if (contains(child->parents(), std::static_pointer_cast<Node>(shared_from_this())))
    {
        child->remove_parent_helper(std::static_pointer_cast<Node>(shared_from_this()));
    }
}

void Node::add_parent_helper(NodePtr parent)
{
    if (!parent)
    {
        return;
    }
    if (!contains(parents(), parent))
    {
        m_parents.push_back(parent);
    }
    if (!contains(parent->children(), std::static_pointer_cast<Node>(shared_from_this())))
    {
        parent->add_child_helper(std::static_pointer_cast<Node>(shared_from_this()));
    }
}

void Node::remove_parent_helper(NodePtr parent)
{
    if (!parent)
    {
        return;
    }
    auto it = std::find_if(m_parents.begin(), m_parents.end(), [parent](auto ref) { return ref.lock() == parent; });
    if (it != m_parents.end())
    {
        m_parents.erase(it);
    }
    if (contains(parent->children(), std::static_pointer_cast<Node>(shared_from_this())))
    {
        parent->remove_child_helper(std::static_pointer_cast<Node>(shared_from_this()));
    }
}

void Node::add_output(TensorPtr t, int idx)
{
    if (t != nullptr && !contains(m_outputs, t))
    {
        if (idx < 0 || idx > (int)m_outputs.size())
        {
            idx = m_outputs.size();
        }
        m_outputs.insert(m_outputs.begin() + idx, t);
        t->m_node = get_shaped_ptr();
        if (m_graph.lock())
        {
            m_graph.lock()->m_tensors[t->name()] = t;
        }
        for (auto n : t->consumers())
        {
            add_child_helper(n); // TODO remove after consumer model well setup
        }
    }
}

int Node::remove_output(TensorPtr t)
{
    int ret = INT32_MAX;
    auto it = std::find(m_outputs.begin(), m_outputs.end(), t);
    if (it != m_outputs.end())
    {
        ret = it - m_outputs.begin();
        remove_output(ret);
    }
    return ret;
}

TensorPtr Node::remove_output(int idx)
{
    if (idx < 0)
    {
        idx += m_outputs.size();
    }
    if (idx >= (int)m_outputs.size() || idx < 0)
    {
        return nullptr;
    }
    auto t = m_outputs[idx];
    m_outputs.erase(m_outputs.begin() + idx);
    for (auto child : children())
    {
        if (contains(child->inputs(), t) &&
            (std::all_of(m_outputs.begin(), m_outputs.end(),
                         [child](TensorPtr o) { return !contains(child->inputs(), o); })))
        {
            remove_child_helper(child);
        }
    }
    if (t->m_node.lock().get() == this && t->consumers().size() == 0 && m_graph.lock()) // mantain the graph's tensor map
    {
        m_graph.lock()->m_tensors.erase(t->name());
    }
    if (t->m_node.lock().get() == this)
    {
        t->m_node = NodeRef();
    }
    return t;
}

void Node::add_input(TensorPtr t, int idx)
{
    if (t != nullptr)
    {
        if (idx < 0 || idx > (int)m_inputs.size())
        {
            idx = m_inputs.size();
        }
        m_inputs.insert(m_inputs.begin() + idx, t);
        if (m_graph.lock())
        {
            m_graph.lock()->m_tensors[t->name()] = t;
        }
        t->add_consumer(get_shaped_ptr());
        add_parent_helper(t->node());
    }
}
int Node::remove_input(TensorPtr t)
{
    int ret = INT32_MAX;
    auto it = std::find(m_inputs.begin(), m_inputs.end(), t);
    if (it != m_inputs.end())
    {
        ret = it - m_inputs.begin();
        remove_input(ret);
    }
    return ret;
}

void Node::replace_input_by(TensorPtr old_tensor, TensorPtr new_tensor)
{
    auto it = m_inputs.begin();
    while (it != m_inputs.end())
    {
        it = std::find(it, m_inputs.end(), old_tensor);
        if (it != m_inputs.end())
        {
            auto idx = it - m_inputs.begin();
            remove_input(idx);
            add_input(new_tensor, idx);
        }
    }
}

void Node::replace_output_by(TensorPtr old_tensor, TensorPtr new_tensor)
{
    auto it = m_outputs.begin();
    while (it != m_outputs.end())
    {
        it = std::find(it, m_outputs.end(), old_tensor);
        if (it != m_outputs.end())
        {
            auto idx = it - m_outputs.begin();
            remove_output(idx);
            add_output(new_tensor, idx);
        }
    }
}

TensorPtr Node::remove_input(int idx)
{
    if (idx < 0)
    {
        idx += m_inputs.size();
    }
    if (idx >= (int)m_inputs.size() || idx < 0)
    {
        return nullptr;
    }
    auto t = m_inputs[idx];
    m_inputs.erase(m_inputs.begin() + idx);
    if (t->node())
    {
        if (std::all_of(m_inputs.begin(), m_inputs.end(),
                        [t](TensorPtr o) { return t->node() != o->node(); }))
        {
            remove_parent_helper(t->node());
        }
    }
    if (!contains(m_inputs, t))
    {
        t->remove_consumer(get_shared_ptr<Node>());
    }
    if (!t->node() && t->consumers().size() == 0 && m_graph.lock()) // the tensor does not belong any node'input and output, remove it from graph tensor map
    {
        m_graph.lock()->m_tensors.erase(t->name());
    }
    return t;
}

void Node::add_input(const std::vector<TensorPtr> &inps)
{
    for (auto t : inps)
    {
        add_input(t);
    }
}
void Node::add_output(const std::vector<TensorPtr> &outs)
{
    for (auto t : outs)
    {
        add_output(t);
    }
}
void Node::clear_inputs()
{
    uint32_t s = m_inputs.size();
    for (uint32_t i = 0; i < s; ++i)
    {
        remove_input(0);
    }
}
void Node::clear_outputs()
{
    uint32_t s = m_outputs.size();
    for (uint32_t i = 0; i < s; ++i)
    {
        remove_output(0);
    }
}
const std::vector<NodePtr> Node::parents() const
{
    std::vector<NodePtr> ret;
    // for(auto t: m_inputs)
    // {
    //     if(!contains(ret,t->node()))
    //     {
    //         ret.push_back(t->node());
    //     }
    // }
    for (auto n : m_parents)
    {
        ret.push_back(n.lock());
    }
    return ret;
}
const std::vector<NodePtr> Node::children() const
{
    std::vector<NodePtr> ret;
    for (auto n : m_children)
    {
        ret.push_back(n.lock());
    }
    return ret;
}
const std::vector<TensorPtr> &Node::outputs() const
{
    return m_outputs;
}
const std::vector<TensorPtr> &Node::inputs() const
{
    return m_inputs;
}
std::vector<TensorPtr> &Node::placeholders()
{
    return m_placeholders;
}
Node::NodeParamRef Node::operator[](const std::string &key)
{
    return Node::NodeParamRef(m_params, key);
}

std::map<std::string, NodeParamValue> &Node::params()
{
    return m_params;
}

std::map<std::string, TensorPtr> &Node::constants()
{
    return m_constants;
}

bool Node::equal(const NodePtr other) const
{
    bool ret = other->m_type == m_type;
    ret = ret && other->m_inputs.size() == m_inputs.size();
    ret = ret && other->m_outputs.size() == m_outputs.size();
    if (!ret)
    {
        return ret;
    }
    for (uint32_t i = 0; i < m_inputs.size(); ++i)
    {
        if (*(m_inputs[i]) != *(other->m_inputs[i]))
        {
            return false;
        }
    }

    for (uint32_t i = 0; i < m_outputs.size(); ++i)
    {
        if (*(m_outputs[i]) != *(other->m_outputs[i]))
        {
            return false;
        }
    }

    for (auto kv : m_params)
    {
        if (other->m_params.count(kv.first))
        {
            const NodeParamValue p0 = other->m_params.at(kv.first);
            const NodeParamValue p1 = kv.second;
            if (p0 != p1)
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
    for (auto kv : m_constants)
    {
        if (other->m_constants.count(kv.first) == 0 || *(other->m_constants.at(kv.first)) != *kv.second)
        {
            return false;
        }
    }
    return ret;
}

NodePtr Node::clone() const
{
    NodePtr node = Node::create(m_name, m_type);
    node->m_tag = m_tag;
    for (auto kv : m_params)
    {
        node->m_params[kv.first] = kv.second;
    }
    for (auto kv : m_attrs)
    {
        // TODO fixme!!! the attr may be  tensor,
        node->m_attrs[kv.first] = kv.second;
    }
    for (auto kv : m_constants)
    {
        if (kv.second)
        {
            node->m_constants[kv.first] = std::make_shared<Tensor>(*kv.second);
        }
        else
        {
            WARN() << "clone the constant " << kv.first << " of node " << m_name << " error, which is a nullptr.";
        }
    }
    for (auto t : m_placeholders)
    {
        if (t)
        {
            node->m_placeholders.push_back(std::make_shared<Tensor>(*t));
        }
    }
    for (auto t : m_inputs)
    {
        if (t)
        {
            auto in = std::make_shared<Tensor>(*t);
            if (t->is_io_tensor())
            {
                in->set_io_tensor(true);
            }
            node->add_input(in);
        }
        else
        {
            WARN() << "clone input tensor of node " << m_name << " error, which is a nullptr.";
        }
    }
    for (auto t : m_outputs)
    {
        if (t)
        {
            auto out = std::make_shared<Tensor>(*t);
            if (t->is_io_tensor())
            {
                out->set_io_tensor(true);
            }
            node->add_output(out);
        }
        else
        {
            WARN() << "clone output tensor of node " << m_name << " error, which is a nullptr.";
        }
    }
    if (m_loc != nullptr)
    {
        ADD_LOCATION_BY_NODE(node, m_loc);
        node->loc()->set_original(m_loc->is_original());
    }

    return node;
}

uint32_t Node::tag() const
{

    return m_tag;
}
void Node::set_tag(uint32_t tag)
{
    m_tag = tag;
}

std::string Node::get_node_dot_name()
{
    auto node_name = name();
    std::string graph_name = "";
    if (m_graph.lock())
    {
        graph_name = m_graph.lock()->name();
    }
    if (!isalpha(graph_name[0]))
    {
        graph_name = "graph_" + graph_name;
    }
    auto name = utils::verify_name(graph_name + node_name);
    name = m_graph.lock()->update_node_dot_name(get_shaped_ptr(), name);
    return name;
}

const std::string Node::dot(const std::string &color, const std::string &prefix)
{
    // the attr is on priority
    const std::vector<std::string> &colors =
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
         "rosybrown1"};
    auto m_color = color;
    if (m_attrs.count(".dot_color"))
    {
        m_color = get_attr<std::string>(".dot_color");
    }
    if (color.empty())
    {
        int cid = (int)type();
        m_color = colors[cid % colors.size()];
    }
    std::ostringstream oss, tmp, tmp2;
    std::string name = m_name;

    name = get_node_dot_name();
    oss << name << " [style=\"filled\","
        << "color=\"black\","
        << "fillcolor=\"" << m_color << "\"";
    oss << "];\n";

    std::string layer_ids;
    std::string real_node_names;
    if (auto sgn = SubGraphNode::from_node(shared_from_this()))
    {
        for (unsigned int i = 0; i < sgn->get_real_node().size(); ++i)
        {
            auto real_node = sgn->get_real_node()[i];
            if (i != 0)
            {
                layer_ids += ",";
                real_node_names += ",";
            }
            layer_ids += std::to_string(this->get_attr<int32_t>("layer_id") +
                                        real_node->get_attr<int32_t>("sub_layer_id"));
            auto real_node_type = sgn->get_real_node()[i]->type();
            bool real_node_is_dwconv = sgn->get_real_node()[i]->get_attr<bool>("is_dwconv");
            if (real_node_type == OpType_Convolution && real_node_is_dwconv)
            {
                real_node_type = OpType_DepthwiseConv;
            }
            real_node_names += optype_to_string(real_node_type);
        }
        if (sgn->attrs().count(Constants::plugin_attrs_name))
        {
            auto plugin_info = sgn->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
            auto plugin_name = plugin_info->name;
            // plugin_name = utils::replace(plugin_name, "aipubt::", "");
            real_node_names += " (" + plugin_name + ")";
        }
        else if (sgn->attrs().count(Constants::plugin_impl))
        {
            real_node_names += " (";
            if (sgn->attrs().count(Constants::kernel_infos))
            {
                auto kernel_infos = sgn->get_attr<KernelInfos>(Constants::kernel_infos);
                for (size_t i = 0; i < kernel_infos.size(); ++i)
                {
                    real_node_names += kernel_infos[i].kernel_name;
                    if (i < kernel_infos.size() - 1)
                    {
                        real_node_names += ",";
                    }
                }
            }
            real_node_names += ")";
        }
        else
        {
            real_node_names += " (subgraph)";
        }
    }
    else
    {
        layer_ids = std::to_string(this->get_attr<int32_t>("layer_id"));
        real_node_names = optype_to_string(m_type);
    }
    oss << name << " [label=\"{"
        << layer_ids << " "
        << m_name
        << "|";
    if (m_attrs.count(Constants::operator_id))
    {
        std::vector<uint32_t> operator_ids = get_attr<std::vector<uint32_t>>(Constants::operator_id);

        if (operator_ids.size() > 0)
        {
            int level = -1;
            int gid = -1;
            if (has_attr_deep(Constants::schedule_level))
            {
                level = get_attr_deep<int>(Constants::schedule_level);
            }
            if (has_attr_deep(Constants::schedule_gid))
            {
                gid = get_attr_deep<int>(Constants::schedule_gid);
            }

            oss << "Operator ID: " << utils::join_string(",", operator_ids);
            if (level != -1 || gid != -1)
            {
                oss << ":" << level << ":" << gid;
            }

            auto max_wt_size = aipubt::BuilderConfig::get<uint64_t>(Constants::max_wt_size);
            auto max_act_size = aipubt::BuilderConfig::get<uint64_t>(Constants::max_act_size);
            if (max_wt_size || max_act_size)
            {
                oss << " Bss ID: " << get_attr<uint32_t>(Constants::bss_id);
            }

            oss << "|";
        }
    }

    oss << utils::wrap_text(real_node_names, 100);
    ;

    oss << "|";
    if (m_inputs.size())
    {
        for (unsigned int i = 0; i < m_inputs.size(); ++i)
        {
            auto inp = m_inputs[i];
            if (i != 0)
            {
                tmp2 << ",";
                tmp << ",";
            }
            tmp2 << inp->shape().to_string();
            tmp << inp->shape().layout() << ":" << inp->type();
        }
        tmp2 << " \\-\\> ";
        tmp << " \\-\\> ";
    }
    if (m_outputs.size())
    {
        for (unsigned int i = 0; i < m_outputs.size(); ++i)
        {
            auto out = m_outputs[i];
            if (i != 0)
            {
                tmp2 << ",";
                tmp << ",";
            }
            tmp2 << out->shape().to_string();
            tmp << out->shape().layout() << ":" << out->type();
        }
    }
    oss << utils::wrap_text(tmp2.str(), 100);
    if (m_type != OpType_Input)
    {
        oss << "|" << utils::wrap_text(tmp.str(), 100);
    }
    tmp.clear();
    tmp2.clear();
    if (m_attrs.count(Constants::dot_sections))
    {
        auto dot_label = get_attr<StringMapPtr>(Constants::dot_sections);
        for (auto &label : *dot_label)
        {
            oss << "|" << label.second;
        }
    }
    oss << "}\"];\n";

    for (auto p : children())
    {
        bool add_edge = true;
        if (auto sgn = SubGraphNode::from_node(p))
        {
            add_edge = sgn->unified_view();
        }
        if (add_edge)
        {
            oss << name << " -> {" << p->get_node_dot_name() << "};\n";
        }
    }
    return oss.str();
}

void Node::add_dot_section(const std::string &key, const std::string &value)
{
    auto dot_sections = get_attr<StringMapPtr>(Constants::dot_sections);
    (*dot_sections)[key] = value;
}

const std::string Node::json()
{
    return AIPUBuilder::get_profiler()->json_from_node(this);
}

const std::string Node::to_string() const
{
    std::ostringstream oss, weightos;
    serialize(oss, weightos);
    return oss.str();
}

void Node::serialize(std::ostream &oss, std::ostream &wt_oss) const
{
    std::map<char *, uint32_t> data_map;
    std::ostringstream sg_oss;
    serialize(data_map, oss, wt_oss, sg_oss);
}

void Node::serialize(std::map<char *, uint32_t> &data_map, std::ostream &oss, std::ostream &wt_oss, std::ostream &sg_oss) const
{
    int id = -1;
    if (m_attrs.count("layer_id"))
    {
        id = get_attr<int>("layer_id");
    }
    std::string lid = std::to_string(id);
    oss << "layer_id=" << lid << "\n";
    oss << "layer_name=" << m_name;
    if (m_type == OpType_Convolution && get_attr<bool>("is_dwconv"))
    {
        oss << "\nlayer_type=" << optype_to_string(OpType_DepthwiseConv);
    }
    else if (m_type == OpType_SubGraph && get_attr<bool>("is_call"))
    {
        oss << "\nlayer_type=" << optype_to_string(OpType_Call);
    }
    else
    {
        oss << "\nlayer_type=" << optype_to_string(m_type);
    }
    oss << "\nlayer_bottom=[";
    if (m_inputs.size())
    {
        for (unsigned int i = 0; i < m_inputs.size() - 1; ++i)
        {
            oss << m_inputs[i]->name() << ",";
        }
        oss << m_inputs[m_inputs.size() - 1]->name();
    }
    oss << "]";
    oss << "\nlayer_bottom_shape=[";
    if (m_inputs.size())
    {
        for (unsigned int i = 0; i < m_inputs.size() - 1; ++i)
        {
            oss << m_inputs[i]->shape().to_string() << ",";
        }
        oss << m_inputs[m_inputs.size() - 1]->shape().to_string();
    }
    oss << "]";
    oss << "\nlayer_bottom_type=[";
    if (m_inputs.size())
    {
        for (unsigned int i = 0; i < m_inputs.size() - 1; ++i)
        {
            oss << m_inputs[i]->type_s() << ",";
        }
        oss << m_inputs[m_inputs.size() - 1]->type_s();
    }
    oss << "]";
    oss << "\nlayer_top=[";
    if (m_outputs.size())
    {
        for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
        {
            oss << m_outputs[i]->name() << ",";
        }
        oss << m_outputs[m_outputs.size() - 1]->name();
    }
    oss << "]";
    oss << "\nlayer_top_shape=[";
    if (m_outputs.size())
    {
        for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
        {
            oss << m_outputs[i]->shape().to_string() << ",";
        }
        oss << m_outputs[m_outputs.size() - 1]->shape().to_string();
    }
    oss << "]";
    oss << "\nlayer_top_type=[";
    if (m_outputs.size())
    {
        for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
        {
            oss << m_outputs[i]->type_s() << ",";
        }
        oss << m_outputs[m_outputs.size() - 1]->type_s();
    }
    oss << "]";
    if (BuilderConfig::get<bool>(Constants::dump_debug_IR))
    {
        oss << "\nlayer_top_datalayout=[";
        if (m_outputs.size())
        {
            for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
            {
                oss << m_outputs[i]->shape().layout() << ",";
            }
            oss << m_outputs[m_outputs.size() - 1]->shape().layout();
        }
        oss << "]";
    }
    if (BuilderConfig::get<bool>("show_ds_shape"))
    {
        oss << "\nlayer_top_ds_shape=[";
        if (m_outputs.size())
        {
            for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
            {
                oss << m_outputs[i]->shape().symbol().to_string() << ",";
            }
            oss << m_outputs[m_outputs.size() - 1]->shape().symbol().to_string();
        }
        oss << "]";
    }
    if (BuilderConfig::get<bool>(Constants::serialize_scale_zp))
    {
        auto scale_are_scalar = [&]() {
            for (auto t : m_outputs)
            {
                if (t->quantization().scales().size() > 1) // scale is scalar if size is 0 or 1
                {
                    return false;
                }
            }
            return true;
        };
        auto zp_are_scalar = [&]() {
            for (auto t : m_outputs)
            {
                if (t->quantization().offsets().size() > 1) // zp is scalar if size is 0 or 1
                {
                    return false;
                }
            }
            return true;
        };

        auto has_top_range = [&]() {
            for (auto t : m_outputs)
            {
                if (t->quantization().mins().size() == 0 || t->quantization().maxs().size() == 0)
                {
                    return false;
                }
            }
            return true;
        };
        auto range_are_scalar = [&]() {
            for (auto t : m_outputs)
            {
                if (t->quantization().mins().size() != 1 || t->quantization().maxs().size() != 1)
                {
                    return false;
                }
            }
            return true;
        };
        auto float_to_string = [](float f) {
            std::stringstream oss;
            oss.precision(10);
            oss << f;
            std::string text = oss.str();
            if ((text.find("e-") == std::string::npos) && (text.find(".") == std::string::npos))
            {
                text += ".0";
            }
            return text;
        };
        if (scale_are_scalar())
        {
            oss << "\nlayer_top_scale=[";
            if (m_outputs.size())
            {
                for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
                {
                    oss << float_to_string(m_outputs[i]->quantization().scale()) << ",";
                }
                oss << float_to_string(m_outputs[m_outputs.size() - 1]->quantization().scale());
            }
            oss << "]";
        }
        else
        {
            int32_t output_number = m_outputs.size();
            oss << "\nlayer_top_scale_type=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << Tensor::dtype_to_string(Tensor::FP32);
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
            }
            oss << "]";
            std::vector<int32_t> s_mem_size;
            std::vector<int32_t> s_shape_size;
            oss << "\nlayer_top_scale_offset=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << wt_oss.tellp();
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
                std::vector<float> layer_top_scales = m_outputs[oi]->quantization().scales();
                int32_t s_s_size = layer_top_scales.size();
                int32_t s_m_size = sizeof(float) * s_s_size;
                wt_oss.write((char *)(layer_top_scales.data()), s_m_size);
                s_mem_size.push_back(s_m_size);
                s_shape_size.push_back(s_s_size);
            }
            oss << "]";
            oss << "\nlayer_top_scale_size=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << s_mem_size[oi];
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
            }
            oss << "]";
            oss << "\nlayer_top_scale_shape=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << "[" << s_shape_size[oi] << "]";
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
            }
            oss << "]";
        }
        if (zp_are_scalar())
        {
            oss << "\nlayer_top_zp=[";
            if (m_outputs.size())
            {
                for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
                {
                    oss << m_outputs[i]->quantization().offset() << ",";
                }
                oss << m_outputs[m_outputs.size() - 1]->quantization().offset();
            }
            oss << "]" << std::resetiosflags(std::ios_base::floatfield);
        }
        else
        {
            int32_t output_number = m_outputs.size();
            oss << "\nlayer_top_zp_type=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << Tensor::dtype_to_string(Tensor::INT32);
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
            }
            oss << "]";
            std::vector<int32_t> z_mem_size;
            std::vector<int32_t> z_shape_size;
            oss << "\nlayer_top_zp_offset=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << wt_oss.tellp();
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
                std::vector<int32_t> layer_top_zps = m_outputs[oi]->quantization().offsets();
                int32_t z_s_size = layer_top_zps.size();
                int32_t z_m_size = sizeof(int32_t) * z_s_size;
                wt_oss.write((char *)(layer_top_zps.data()), z_m_size);
                z_mem_size.push_back(z_m_size);
                z_shape_size.push_back(z_s_size);
            }
            oss << "]";
            oss << "\nlayer_top_zp_size=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << z_mem_size[oi];
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
            }
            oss << "]";
            oss << "\nlayer_top_zp_shape=[";
            for (int32_t oi = 0; oi < output_number; oi++)
            {
                oss << "[" << z_shape_size[oi] << "]";
                if (oi < output_number - 1)
                {
                    oss << ",";
                }
            }
            oss << "]";
        }
        if (has_top_range())
        {
            if (range_are_scalar())
            {
                oss << "\nlayer_top_range=[";
                if (m_outputs.size())
                {
                    for (unsigned int i = 0; i < m_outputs.size() - 1; ++i)
                    {
                        oss << "[" << float_to_string(m_outputs[i]->quantization().mins()[0]) << ","
                            << float_to_string(m_outputs[i]->quantization().maxs()[0]) << "],";
                    }
                    oss << "[" << float_to_string(m_outputs[m_outputs.size() - 1]->quantization().mins()[0]) << ","
                        << float_to_string(m_outputs[m_outputs.size() - 1]->quantization().maxs()[0]) << "]";
                }
                oss << "]" << std::resetiosflags(std::ios_base::floatfield);
            }
            else
            {
                int32_t output_number = m_outputs.size();
                oss << "\nlayer_top_range_type=[";
                for (int32_t oi = 0; oi < output_number; oi++)
                {
                    oss << Tensor::dtype_to_string(Tensor::FP32);
                    if (oi < output_number - 1)
                    {
                        oss << ",";
                    }
                }
                oss << "]";
                std::vector<int32_t> range_mem_size;
                std::vector<TensorShape> range_shape_size;
                oss << "\nlayer_top_range_offset=[";
                for (int32_t oi = 0; oi < output_number; oi++)
                {
                    oss << wt_oss.tellp();
                    if (oi < output_number - 1)
                    {
                        oss << ",";
                    }
                    std::vector<float> layer_top_mins = m_outputs[oi]->quantization().mins();
                    auto layer_top_maxs = m_outputs[oi]->quantization().maxs();
                    int32_t mins_size = layer_top_mins.size();
                    int32_t maxs_size = layer_top_maxs.size();
                    if (mins_size != maxs_size)
                    {
                        ERROR() << m_outputs[oi] << " has the different length of min and max statistic values, and mins.size() = "
                                << mins_size << ", and maxs.size() = " << maxs_size;
                    }
                    int32_t mins_maxs_m_size = sizeof(float) * mins_size * 2;
                    layer_top_mins.insert(layer_top_mins.end(), layer_top_maxs.begin(), layer_top_maxs.end());
                    wt_oss.write((char *)layer_top_mins.data(), mins_maxs_m_size);
                    range_mem_size.push_back(mins_maxs_m_size);
                    range_shape_size.push_back(TensorShape(2, mins_size));
                }
                oss << "]";
                oss << "\nlayer_top_range_size=[";
                for (int32_t oi = 0; oi < output_number; oi++)
                {
                    oss << range_mem_size[oi];
                    if (oi < output_number - 1)
                    {
                        oss << ",";
                    }
                }
                oss << "]";
                oss << "\nlayer_top_range_shape=[";
                for (int32_t oi = 0; oi < output_number; oi++)
                {
                    oss << "[" << range_shape_size[oi] << "]";
                    if (oi < output_number - 1)
                    {
                        oss << ",";
                    }
                }
                oss << "]";
            }
        }
    }
    oss << "\n";
    bool is_call_node = m_attrs.count("is_call") && m_attrs.at("is_call").cast<bool>();
    if (!is_call_node && m_type != OpType_Loop && m_type != OpType_If)
    {
        for (auto kv : m_constants)
        {
            if (data_map.count(kv.second->data<char>()))
            {
                oss << kv.first << "_type=" << kv.second->type_s() << "\n";
                oss << kv.first << "_offset=" << data_map[kv.second->data<char>()] << "\n";
                oss << kv.first << "_size=" << kv.second->mem_size() << "\n";
                oss << kv.first << "_shape=" << kv.second->shape().to_string() << "\n";
            }
            else
            {
                oss << kv.first << "_type=" << kv.second->type_s() << "\n";
                oss << kv.first << "_offset=" << wt_oss.tellp() << "\n";
                oss << kv.first << "_size=" << kv.second->mem_size() << "\n";
                oss << kv.first << "_shape=" << kv.second->shape().to_string() << "\n";
                if (kv.second->data<char>())
                {
                    data_map[kv.second->data<char>()] = wt_oss.tellp();
                    wt_oss.write(kv.second->data<char>(), kv.second->mem_size());
                }
            }
            if (BuilderConfig::get<bool>(Constants::serialize_scale_zp))
            {
                auto float_to_string = [](float f) {
                    std::stringstream oss;
                    oss.precision(10);
                    oss << f;
                    std::string text = oss.str();
                    if ((text.find("e-") == std::string::npos) && (text.find(".") == std::string::npos))
                    {
                        text += ".0";
                    }
                    return text;
                };

                auto do_quantization_serialize = [&](Quantization &quantization, std::string suffix = "") {
                    auto scales = quantization.scales();
                    auto offsets = quantization.offsets();

                    // 1. serialize scales
                    auto s_size = scales.size();
                    bool is_params = false;
                    bool is_offset = false;
                    if (s_size == 1)
                    {
                        oss << kv.first << "_scale" << suffix << "=" << float_to_string(scales[0]) << "\n";
                        is_params = true;
                    }
                    else if (s_size > 1)
                    {
                        oss << kv.first << "_scale" << suffix << "_type=" << scales->type_s() << "\n";
                        oss << kv.first << "_scale" << suffix << "_offset=" << wt_oss.tellp() << "\n";
                        oss << kv.first << "_scale" << suffix << "_size=" << scales->mem_size() << "\n";
                        oss << kv.first << "_scale" << suffix << "_shape=" << scales->shape().to_string() << "\n";
                        wt_oss.write(scales->data<char>(), scales->mem_size());
                        is_offset = true;
                    }

                    // 2. serialize zp
                    auto z_size = offsets.size();
                    if (is_params && z_size == 1)
                    {
                        oss << kv.first << "_zp" << suffix << "=" << offsets[0] << "\n";
                    }
                    else if (is_offset && z_size >= 1)
                    {
                        oss << kv.first << "_zp" << suffix << "_type=" << offsets->type_s() << "\n";
                        oss << kv.first << "_zp" << suffix << "_offset=" << wt_oss.tellp() << "\n";
                        oss << kv.first << "_zp" << suffix << "_size=" << offsets->mem_size() << "\n";
                        oss << kv.first << "_zp" << suffix << "_shape=" << offsets->shape().to_string() << "\n";
                        wt_oss.write(offsets->data<char>(), offsets->mem_size());
                    }

                    // 3. serialize range min/max
                    auto range_min_size = quantization.mins().size();
                    auto range_max_size = quantization.maxs().size();
                    if (range_min_size == 1)
                    {
                        auto min_v = quantization.mins()[0];
                        auto max_v = quantization.maxs()[0];
                        oss << kv.first << "_range" << suffix << "=[" << min_v << "," << max_v << "]\n";
                    }
                    else if (range_min_size > 1 && range_max_size == range_min_size)
                    {
                        std::vector<float> mins_maxs_v = quantization.mins();
                        auto maxs_v = quantization.maxs();
                        mins_maxs_v.insert(mins_maxs_v.end(), maxs_v.begin(), maxs_v.end());
                        auto s_m_size = mins_maxs_v.size() * sizeof(float);
                        oss << kv.first << "_range" << suffix << "_type=" << Tensor::dtype_to_string(Tensor::FP32) << "\n";
                        oss << kv.first << "_range" << suffix << "_offset=" << wt_oss.tellp() << "\n";
                        oss << kv.first << "_range" << suffix << "_size=" << s_m_size << "\n";
                        oss << kv.first << "_range" << suffix << "_shape="
                            << "[2," << range_max_size << "]\n";
                        wt_oss.write((char *)mins_maxs_v.data(), s_m_size);
                    }
                };

                // weights/biases/negative_slope
                auto scales = kv.second->quantization().scales();
                if (!scales.contain_quant())
                {
                    do_quantization_serialize(kv.second->quantization());
                }
                else
                {
                    int q_level = 0;
                    auto quantization = kv.second->quantization();
                    do
                    {
                        do_quantization_serialize(quantization, std::to_string(q_level++));
                        scales = quantization.scales();
                        if (scales.contain_quant())
                            quantization = scales->quantization();
                    } while (scales.contain_quant());
                }
            }
        }
    }

    const std::set<std::string> required_layer_params = {
        "layer_id",
        "layer_name",
        "layer_type",
        "layer_bottom",
        "layer_bottom_shape",
        "layer_bottom_type",
        "layer_top",
        "layer_top_shape",
        "layer_top_type",
    };
    // avoid redundant required_layer_params
    for (auto p : m_params)
    {
        if (required_layer_params.count(p.first) == 0)
        {
            if (p.second.t == NodeParamValue::NodeParamType_void)
            {
                DEBUG() << "a void param appears in node's params, key:" << p.first << ", node: " << m_name;
            }
            else
            {
                oss << p.first << "=" << p.second << "\n";
            }
        }
    }
    if (!is_call_node && m_loc != nullptr)
    {
        oss << m_loc->serialize();
    }
}

std::map<std::string, Any> &Node::attrs()
{
    return m_attrs;
}

const std::map<std::string, Any> &Node::attrs() const
{
    return m_attrs;
}

void Node::erase_attr(const std::string &key)
{
    m_attrs.erase(key);
}

void Node::set_attr(const std::string &key, Any v)
{
    m_attrs[key] = v;
}
Any Node::get_attr(const std::string &key) const
{
    if (m_attrs.count(key))
    {
        return m_attrs.at(key);
    }
    return Any();
}

Any Node::get_attr_deep(const std::string &key) const
{
    if (m_attrs.count(key))
    {
        return m_attrs.at(key);
    }
    if (!(m_type >= OpType_SubGraph && m_type < OpType_SubGraphMaxCount))
    {
        return Any();
    }
    const SubGraphNode *sgn = dynamic_cast<const SubGraphNode *>(this);
    for (auto sn : sgn->subgraph()->nodes())
    {
        if (sn->has_attr_deep(key))
        {
            return sn->get_attr_deep(key);
        }
    }
    return Any();
}

void Node::erase_attr_deep(const std::string &key, bool only_first)
{
    m_attrs.erase(key);
    if (!SubGraphNode::is_subgraph_node(shared_from_this()))
    {
        return;
    }
    const SubGraphNode *sgn = dynamic_cast<const SubGraphNode *>(this);
    for (auto sn : sgn->subgraph()->nodes())
    {
        if (only_first)
        {
            if (sn->has_attr_deep(key))
            {
                sn->erase_attr_deep(key);
                return;
            }
        }
        else
        {
            sn->erase_attr_deep(key, false);
        }
    }
}

bool Node::has_attr_deep(const std::string &key) const
{
    if (m_attrs.count(key))
    {
        return true;
    }
    if (!(m_type >= OpType_SubGraph && m_type < OpType_SubGraphMaxCount))
    {
        return false;
    }

    const SubGraphNode *sgn = dynamic_cast<const SubGraphNode *>(this);
    for (auto sn : sgn->subgraph()->nodes())
    {
        if (sn->has_attr_deep(key))
        {
            return true;
        }
    }
    return false;
}

static bool infer_symbol_shape(BaseNodePtr n)
{
    std::vector<SymbolShape> in_shapes;
    for (auto &&inp : n->inputs())
    {
        in_shapes.push_back(inp->shape().symbol());
    }
    auto node = std::dynamic_pointer_cast<Node>(n);
    auto out_shapes = tongue::infer_shape(node, in_shapes);

    if (out_shapes.size() != n->outputs().size())
    {
        return false;
    }
    std::vector<SymbolShape> new_shapes;
    for (size_t i = 0; i < out_shapes.size(); ++i)
    {
        auto s = n->outputs()[i]->shape();
        if (out_shapes[i].dim() != s.dim())
        {
            return false;
        }
        std::vector<Expr> ss;
        for (int d = 0; d < out_shapes[i].dim(); ++d)
        {
            ss.push_back(out_shapes[i][d]->simplify());
        }
        new_shapes.push_back(SymbolShape(ss, s.layout(), s.indexing()));
    }

    for (size_t i = 0; i < new_shapes.size(); ++i)
    {
        n->outputs()[i]->shape().symbol() = new_shapes[i];
    }
    if (n->attrs().count(Constants::is_dynamic))
    {
        n->erase_attr(Constants::is_dynamic);
    }
    return true;
}

std::vector<TensorShape> Node::infer_shape(bool only_infer_symbol)
{
    if (!infer_symbol_shape(shared_from_this()))
    {
        if (BuilderConfig::get<bool>(Constants::enable_ds) && is_dynamic())
        {
            WARN() << "Infer symbolic shape failed for node:\n"
                   << to_string();
        }
    }
    if (!only_infer_symbol)
    {
        auto plugins = get_shape_infer_plugins();
        if (plugins.count(m_type))
        {
            auto plugin = plugins[m_type];
            auto opt = plugin->init_handle();
            auto shapes = opt->infer_shape(((BaseNode *)this)->shared_from_this());
            if (shapes.size() == m_outputs.size())
            {
                for (size_t i = 0; i < shapes.size(); ++i)
                {
                    auto s = TensorShape(shapes[i], m_outputs[i]->shape().layout());
                    s.symbol() = m_outputs[i]->shape().symbol();
                    shapes[i] = s;
                }
            }
            return shapes;
        }
    }
    return std::vector<TensorShape>();
}

bool Node::is_dynamic()
{
    NodePtr self = std::dynamic_pointer_cast<Node>(shared_from_this());
    auto sgn = SubGraphNode::from_node(self);
    if (sgn)
    {
        return sgn->subgraph()->is_dynamic();
    }
    if (m_attrs.count(Constants::is_dynamic))
    {
        return m_attrs[Constants::is_dynamic];
    }
    auto has_ds_param = [](NodePtr n) -> bool {
        for (auto &kv : n->params())
        {
            if (utils::startswith(kv.first, "ds_"))
            {
                return true;
            }
        }
        return false;
    };
    auto check_node = [&has_ds_param](NodePtr n) -> bool {
        for (auto inp : n->inputs())
        {
            for (int d = 0; d < inp->shape().symbol().dim(); ++d)
            {
                if (!inp->shape().symbol()[d]->is_constant())
                {
                    return true;
                }
            }
        }
        if (n->type() == OpType_NonZero ||
            n->type() == OpType_CacheUpdate ||
            n->type() == OpType_ConstantOfShape ||
            (n->type() == OpType_Slice && n->inputs().size() == 5) ||
            n->type() == OpType_ForRangeEnder ||
            n->type() == OpType_IfEnder ||
            (n->type() == OpType_TopK && n->inputs().size() == 2) ||
            has_ds_param(n))
        {
            infer_symbol_shape(n);
        }
        for (auto out : n->outputs())
        {
            for (int d = 0; d < out->shape().symbol().dim(); ++d)
            {
                if (!out->shape().symbol()[d]->is_constant())
                {
                    return true;
                }
            }
        }
        return false;
    };
    if (m_type == OpType_Constant ||
        m_type == OpType_InternalConstant)
    {
        for (auto c : children())
        {
            if (c->m_attrs.count(Constants::is_dynamic))
            {
                if (c->get_attr<bool>(Constants::is_dynamic))
                {
                    return true;
                }
            }
            else if (check_node(c))
            {
                return true;
            }
        }
        return false;
    }
    bool is_dynamic = check_node(self);
    m_attrs[Constants::is_dynamic] = is_dynamic;
    return is_dynamic;
}

GraphPtr Node::graph() const
{
    return m_graph.lock();
}

void Node::set_graph(GraphPtr g)
{
    m_graph = g;
}

LocationPtr &Node::loc()
{
    return m_loc;
}

void Node::set_loc(LocationPtr loc)
{
    m_loc = loc;
}

std::ostream &operator<<(std::ostream &os, const BaseNode &node)
{
    return (os << node.name() << "(" << node.type() << ")");
}
std::ostream &operator<<(std::ostream &os, const BaseNodePtr &node)
{
    return (os << *node);
}
std::ostream &operator<<(std::ostream &os, const NodePtr &node)
{
    return (os << *node);
}

void move_inputs(NodePtr from, NodePtr to)
{
    for (auto t : from->inputs())
    {
        from->remove_input(t);
        to->add_input(t);
    }
}
void copy_inputs(NodePtr from, NodePtr to)
{
    for (auto t : from->inputs())
    {
        to->add_input(t);
    }
}
void swap_inputs(NodePtr a, NodePtr b)
{
    auto ainput = a->inputs();
    auto binput = b->inputs();
    for (auto t : ainput)
    {
        a->remove_input(t);
        b->add_input(t);
    }
    for (auto t : binput)
    {
        b->remove_input(t);
        a->add_input(t);
    }
}

uint64_t Node::get_output_size() const
{
    uint64_t total_size = 0;
    for (auto t : outputs())
    {
        total_size += t->mem_size();
    }
    return total_size;
}

uint64_t Node::get_constant_size() const
{
    auto aligned = [](uint64_t s, uint64_t align) -> uint64_t {
        return (s + align - 1) & ~(align - 1);
    };

    if (m_attrs.count(Constants::virtual_core_idx) &&
        get_attr<int>(Constants::virtual_core_idx) != 0)
    {
        return 0; // virtual multi-core plugins share the same constants
    }

    uint64_t total_constant_size = 0;
    for (auto cts : m_constants)
    {
        auto t = std::make_shared<Tensor>(*cts.second);
        if (t->attrs().count(Constants::zerocopy_cst))
        {
            continue;
        }
        if (t->type() == Tensor::ALIGNED_INT4 || t->type() == Tensor::ALIGNED_UINT4 || t->type() == Tensor::FP4E2M1FN)
        {
            total_constant_size += t->memory_offset()->size / 2; // will compress to half
        }
        else
        {
            total_constant_size += t->memory_offset()->size;
        }
        total_constant_size = aligned(total_constant_size, t->memory_offset()->get_base_object()->align);
    }
    return total_constant_size;
}

int Node::get_layer_id() const
{
    int layer_id = -1;
    if (m_attrs.count(Constants::plugin_attrs_name))
    {
        auto nodes = ((SubGraphNode *)this)->get_real_node();
        if (!nodes.empty() && nodes[0]->attrs().count("layer_id"))
        {
            layer_id = nodes[0]->get_attr<int>("layer_id");
        }
    }
    else if (m_attrs.count("layer_id"))
    {
        layer_id = get_attr<int>("layer_id");
    }
    return layer_id;
}

bool Node::replace_symbols(const std::map<std::string, ExprPtr> &symbols)
{
    for (auto &param : m_params)
    {
        auto param_name = param.first;
        if (!utils::startswith(param_name, "ds_"))
        {
            continue;
        }
        std::vector<std::vector<Expr>> exprs;
        if (param_name == "ds_output_shape" || param_name == "ds_crops" || param_name == "ds_pads")
        {
            exprs = param.second;
        }
        else
        {
            std::vector<Expr> s = param.second;
            exprs.push_back(s);
        }
        for (size_t i = 0; i < exprs.size(); ++i)
        {
            for (size_t j = 0; j < exprs[i].size(); ++j)
            {
                auto expr = exprs[i][j];
                expr = ExprImpl::replace_symbol(expr, symbols);
                exprs[i][j] = expr;
            }
        }
        if (param_name == "ds_output_shape" || param_name == "ds_crops" || param_name == "ds_pads")
        {
            param.second = exprs;
        }
        else
        {
            param.second = exprs[0];
        }
    }
    for (auto &&out : m_outputs)
    {
        std::vector<Expr> exprs = out->shape().symbol();
        for (size_t i = 0; i < exprs.size(); ++i)
        {
            auto expr = exprs[i];
            expr = ExprImpl::replace_symbol(expr, symbols);
            exprs[i] = expr;
        }
        out->shape().symbol() = SymbolShape(exprs, out->shape().layout(), out->shape().indexing());
    }
    return true;
}

void Node::update_constants()
{
    auto self = shared_from_this();
    auto sgn = SubGraphNode::from_node(self);
    if (sgn)
    {
        auto sg = sgn->subgraph();
        auto &nodes = sg->nodes();
        self->constants().clear();
        for (auto n : nodes)
        {
            n->update_constants();
            auto &constants = n->constants();
            for (auto &p : constants)
            {
                auto t = std::make_shared<Tensor>(*p.second);
                p.second->memory_offset()->relative_offset = 0;
                sgn->constants()[n->name() + "/" + p.first] = t;
            }
        }
    }
}

} // namespace aipubt
