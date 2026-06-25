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
 * @Date: 2020-01-29 14:27:57
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-03-29 15:02:14
 */
#include "builder/profiler.h"
#include "builder/target.h"
#include "checker/checker.h"
#include "checker/plugin.h"
#include "core.h"
#include "core/any.h"
#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/plugin_loader.h"
#include "core/subgraph.h"
#include "tongue/symbolic.h"
#include "utils/string_utils.h"
#include "utils/utils.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace aipubt
{
SymbolWrap::SymbolWrap(tongue::Expr sym, tongue::Symbol::Value val)
{
    if (sym && sym->op() == ExprImpl::Op::Constant)
    {
        expr = sym;
        value = val;
    }
}
SymbolWrap::SymbolWrap(const SymbolWrap &sym)
{
    expr = sym.expr;
    type = sym.type;
    dims = sym.dims;
    value = sym.value;
}

GraphPtr get_default_graph()
{
    static auto g = Graph::create("DefaultGraph");
    return g;
}
GraphPtr Graph::create(std::string name, DataLayout layout)
{
    return GraphPtr(new Graph(name, layout));
}

Graph::Graph(std::string name, DataLayout layout)
{
    m_name = name;
    m_layout = layout;
    m_global_tensors[Constants::workspace] = std::make_shared<Tensor>(name + "/ws", TensorShape(0));
    m_global_tensors[Constants::parambase] = std::make_shared<Tensor>(name + "/parambase/", TensorShape(0));
    m_global_tensors[Constants::descriptorbase] = std::make_shared<Tensor>(name + "/descriptorbase/", TensorShape(0));
    m_global_tensors[Constants::random_reg] = std::make_shared<Tensor>(name + "/rg", TensorShape(0));
    uint64_t fifo_size = aipubt::BuilderConfig::get<uint64_t>("printf_fifo_size");
    m_global_tensors[Constants::printf_fifo] = std::make_shared<Tensor>(name + "/printfifo", TensorShape(fifo_size));
    m_global_tensors[Constants::op_cnt] = std::make_shared<Tensor>(name + "/op_cnt", TensorShape(4));
    m_global_tensors[Constants::error_code] = std::make_shared<Tensor>(name + "/error_code", TensorShape(4));
    m_global_tensors[Constants::temp_param_base] = std::make_shared<Tensor>(name + "/tp_base", TensorShape(0));
    m_global_tensors[Constants::tensor_shape_base] = std::make_shared<Tensor>(name + "/ts_base", TensorShape(0));
}
Graph::~Graph()
{
    m_global_tensors.clear();
    m_nodes.clear();
}

void Graph::parse_from(const std::string &filename, bool check_ir)
{
    Graph::parse(filename, shared_from_this(), check_ir);
}

const std::string &Graph::name() const
{
    return m_name;
}
void Graph::set_name(const std::string &name)
{
    m_name = name;
}

void Graph::set_attr(const std::string &key, Any v)
{
    m_attrs[key] = v;
}

Any Graph::get_attr(const std::string &key) const
{
    if (m_attrs.count(key))
    {
        return m_attrs.at(key);
    }
    return Any();
}

std::map<std::string, Any> &Graph::attrs()
{
    return m_attrs;
}

std::map<std::string, TensorPtr> &Graph::global_tensors()
{
    return m_global_tensors;
}

const std::vector<NodePtr> &Graph::nodes() const
{
    return m_nodes;
}
std::vector<NodePtr>::iterator Graph::begin()
{
    return m_nodes.begin();
}
std::vector<NodePtr>::iterator Graph::end()
{
    return m_nodes.end();
}

NodePtr Graph::operator[](const std::string &name)
{
    if (m_nodes_map.count(name))
    {
        return m_nodes_map[name];
    }
    else
    {
        return nullptr;
    }
}

bool Graph::has_tensor(TensorPtr t)
{
    for (auto tensor : m_tensors)
    {
        if (tensor.second == t)
        {
            return true;
        }
    }
    return false;
}

void Graph::add_node(NodePtr n)
{
    if (contains(m_nodes, n))
    {
        return;
    }
    n->set_graph(shared_from_this());
    m_nodes_map[n->name()] = n;
    m_nodes.push_back(n);
    for (auto t : n->outputs())
    {
        // if (m_tensors.count(t->name()))
        // {
        //     t->set_name(get_valid_tensor_name(t->name()));
        // }
        m_tensors[t->name()] = t;
    }
    for (auto kv : n->constants())
    {
        auto t = kv.second;
        m_tensors[n->name() + "/" + t->name()] = t;
    }
    for (auto t : n->placeholders())
    {
        m_tensors[t->name()] = t;
    }
    if (n->type() == OpType_Input)
    {
        for (auto t : n->outputs())
        {
            t->set_io_tensor(true);
            if (!contains(m_in_tensors, t))
            {
                m_in_tensors.push_back(t);
            }
        }
    }
}
void Graph::update_tensor_map()
{
    m_tensors.clear();
    for (auto n : m_nodes)
    {
        for (auto t : n->outputs())
        {
            m_tensors[t->name()] = t;
        }
        for (auto kv : n->constants())
        {
            auto t = kv.second;
            m_tensors[n->name() + "/" + t->name()] = t;
        }
        for (auto t : n->placeholders())
        {
            m_tensors[t->name()] = t;
        }
    }
}

std::string Graph::update_node_dot_name(NodePtr node, const std::string &name)
{
    std::string new_name;
    if (!m_node_dot_name_map.count(name))
    {
        new_name = name;
        m_node_dot_name_map[name] = node;
    }
    else
    {
        if (node == m_node_dot_name_map[name])
        {
            new_name = name;
        }
        else
        {
            new_name = update_node_dot_name(node, name + "_");
        }
    }
    return new_name;
}

bool Graph::is_dynamic() const
{
    for (auto n : m_nodes)
    {
        if (n->is_dynamic())
        {
            return true;
        }
    }
    return false;
}

void Graph::set_symbol(const SymbolWrap &s)
{
    if (s.expr && s.expr->op() == ExprImpl::Op::Constant)
    {
        m_symbols[s.expr->symbol().name()] = s;
    }
}

void Graph::erase_symbol(const std::string &name)
{
    if (m_symbols.count(name))
    {
        m_symbols.erase(name);
    }
}

void Graph::refresh_symbols()
{
    m_symbols.clear();
    auto inputs = get_input_tensors();
    for (auto &&inp : inputs)
    {
        auto shape = get_base_shape(inp->shape());
        for (int i = 0; i < shape.dim(); ++i)
        {
            auto expr = shape.symbol()[i];
            if (expr->op() == ExprImpl::Op::Constant)
            {
                m_symbols[expr->symbol().name()] = SymbolWrap(expr, shape[i]);
            }
        }
    }
}

const std::map<std::string, SymbolWrap> &Graph::symbols()
{
    if (m_symbols.empty())
    {
        refresh_symbols();
    }
    return m_symbols;
}

bool Graph::replace_symbols(const std::vector<TensorPtr> &upper_inputs)
{
    auto tensors = get_input_tensors();
    if (tensors.size() != upper_inputs.size())
    {
        return false;
    }
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        if (tensors[i]->shape().dim() != upper_inputs[i]->shape().dim())
        {
            return false;
        }
    }
    std::map<std::string, ExprPtr> symbols;
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        auto t0 = tensors[i];
        auto t1 = upper_inputs[i];
        for (int j = 0; j < t0->shape().dim(); ++j)
        {
            auto expr = t0->shape().symbol()[j];
            if (expr->op() == ExprImpl::Op::Constant)
            {
                symbols[expr->symbol().name()] = t1->shape().symbol()[j];
            }
        }
    }
    if (symbols.size())
    {
        walk_on_node([&symbols](NodePtr n) { n->replace_symbols(symbols); });
        refresh_symbols();
    }
    return true;
}

void Graph::remove_node(NodePtr n, bool clean_edge)
{
    if (!n || !contains(m_nodes, n))
    {
        return;
    }
    if (clean_edge)
    {
        n->clear_inputs();
        n->clear_outputs();
    }
    m_nodes.erase(std::find(m_nodes.begin(), m_nodes.end(), n));
    if (m_nodes_map[n->name()] == n)
    {
        m_nodes_map.erase(n->name());
    }
    for (auto out : n->outputs())
    {
        if (out->node() == n)
        {
            m_tensors.erase(out->name());
            if (n->type() == OpType_Input && contains(m_in_tensors, out))
            {
                m_in_tensors.erase(std::find(m_in_tensors.begin(), m_in_tensors.end(), out));
            }
        }
    }
    for (auto w : n->constants())
    {
        m_tensors.erase(n->name() + "/" + w.second->name());
    }
    for (auto t : n->placeholders())
    {
        m_tensors.erase(t->name());
    }
    if (n->m_graph.lock() == shared_from_this())
    {
        n->m_graph = GraphRef();
    }
}

void Graph::replace_node_by(NodePtr replaced, NodePtr by)
{
    auto inputs = replaced->inputs();
    auto outputs = replaced->outputs();
    by->clear_inputs();
    by->clear_outputs();
    for (auto t : inputs)
    {
        by->add_input(t);
        replaced->remove_input(t);
    }
    for (auto t : outputs)
    {
        by->add_output(t);
        replaced->remove_output(t);
    }
    remove_node(replaced, false);
    add_node(by);
}

bool Graph::eliminate_node(NodePtr n)
{
    if (n && n->inputs().size() == n->outputs().size() && contains(m_nodes, n))
    {
        auto inputs = n->inputs();
        auto outputs = n->outputs();
        for (auto o : outputs)
        {
            if (contains(get_output_tensors(), o) || o->is_io_tensor())
            {
                return false;
            }
        }
        for (uint32_t i = 0; i < outputs.size(); ++i)
        {
            auto consumers = outputs[i]->consumers();
            for (auto node : consumers)
            {
                for (uint32_t j = 0; j < node->inputs().size(); j++)
                {
                    if (node->inputs()[j] == outputs[i])
                    {
                        node->remove_input(j);
                        node->add_input(inputs[i], j);
                    }
                }
            }
        }

        for (auto inp : inputs)
        {
            n->remove_input(-1);
        }
        remove_node(n, false);
        return true;
    }
    return false;
}

bool Graph::eliminate_node_aware_out(NodePtr n)
{
    if (n && n->inputs().size() == n->outputs().size() && contains(m_nodes, n))
    {
        auto inputs = n->inputs();
        auto outputs = n->outputs();

        for (uint32_t i = 0; i < outputs.size(); ++i)
        {
            auto consumers = outputs[i]->consumers();
            for (auto node : consumers)
            {
                int idx = -1;
                for (uint32_t j = 0; j < node->inputs().size(); j++)
                {
                    if (node->inputs()[j] == outputs[i])
                    {
                        idx = j;
                        break;
                    }
                }
                aipu_assert(idx != -1);
                node->remove_input(idx);
                node->add_input(inputs[i], idx);
            }
        }

        for (auto inp : inputs)
        {
            n->remove_input(-1);
        }
        for (uint32_t i = 0; i < outputs.size(); ++i)
        {
            for (uint32_t j = 0; j < get_output_tensors().size(); ++j)
            {
                if (m_out_tensors[j] == outputs[i])
                {
                    m_out_tensors[j] = inputs[i];
                }
            }
        }
        remove_node(n, false);
        return true;
    }
    return false;
}

const std::string Graph::dot(const std::vector<std::string> &colors, bool with_header) const
{
    std::ostringstream dot;
    if (with_header)
    {
        dot << "digraph \"" << this->m_name << "\" {\n"
            << "graph [nslimit=2 nslimit1=2 maxiter=10000];\n"
            << "node [shape=record];\n";
    }
    int id = 0;
    auto name = utils::verify_name(m_name);
    for (auto node : m_nodes)
    {
        if (node->type() == OpType_SubGraphInput)
        {
            continue;
        }
        std::string color = "white";
        if (colors.size())
        {
            int cid = (int)node->type();
            if (SubGraphNode::is_subgraph_node(node))
            {
                SubGraphNodePtr sgn = std::dynamic_pointer_cast<SubGraphNode>(node);
                if (sgn && sgn->subgraph()->nodes().size() == 1)
                {
                    cid = (int)sgn->subgraph()->nodes()[0]->type();
                    bool is_dwconv = sgn->subgraph()->nodes()[0]->get_attr<bool>("is_dwconv");
                    if (cid == (OpType)OpType_Convolution && is_dwconv)
                    {
                        cid = (int)OpType_DepthwiseConv;
                    }
                }
            }
            color = colors[cid % colors.size()];
        }
        dot << node->dot(color, std::to_string(id) + " ");
        id++;
    }
    dot << "\n\n";

    if (with_header)
    {
        dot << "}\n";
    }

    return dot.str();
}

void Graph::to_dot(const std::string &filename, const std::vector<std::string> &colors) const
{
    std::ofstream dot;
    dot.open(filename);
    dot << this->dot(colors);
    dot.close();
}

bool Graph::to_svg(const std::string &filename) const
{
    auto m_dot = utils::get_temp_dir() + "/temp_" + utils::get_uuid();
    to_dot(m_dot);
    // INFO() << "Drawing svg ...";
    auto retcode = system(("dot " + m_dot + " -Tsvg -o " + filename).c_str());
    utils::remove_file(m_dot);
    if (retcode != 0)
    {
        auto dotf = filename;
        dotf = utils::replace(dotf, ".svg", ".dot");
        utils::copy_file(m_dot, dotf);
        WARN() << "to_svg failed! To png depends on dot! Please make sure graphviz have been installed. We will save the dot file to" << dotf;
        return false;
    }
    return true;
}

bool Graph::to_png(const std::string &filename) const
{
    auto m_dot = utils::get_temp_dir() + "/temp_" + utils::get_uuid();
    to_dot(m_dot);
    // INFO() << "Drawing png ...";
    auto retcode = system(("dot " + m_dot + " -Tpng -o " + filename).c_str());
    utils::remove_file(m_dot);
    if (retcode != 0)
    {
        auto svg = filename;
        svg = utils::replace(svg, ".png", ".svg");
        WARN() << "Failed to generate a png for the graph, trying to generate a svg for it.";
        return to_svg(svg);
    }
    return true;
}

const std::string Graph::json()
{
    return AIPUBuilder::get_profiler()->json_from_graph(shared_from_this());
}

uint32_t Graph::get_total_nodes_num() const
{
    bool dump_debug_IR = BuilderConfig::get("dump_debug_IR");
    auto check_subgraph = [&](NodePtr n) {
        if (dump_debug_IR)
        {
            return SubGraphNode::is_subgraph_node(n);
        }
        return n->type() == OpType_SubGraph;
    };
    uint32_t count = 0;
    for (auto n : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(n);
        if (check_subgraph(n) && sgn)
        {
            count += sgn->subgraph()->get_total_nodes_num();
        }
        else if (n->type() == OpType_SubGraphInput)
        {
            continue;
        }
        else
        {
            count++;
        }
    }
    return count;
}

void Graph::serialize(std::ostream &oss, std::ostream &wt_oss, bool sg_fmt)
{
    if (m_attrs.count(Constants::serialize_scale_zp))
    {
        bool serialize_scale_zp = cast_any<bool>(m_attrs[Constants::serialize_scale_zp]);
        if (serialize_scale_zp)
        {
            BuilderConfig::set(Constants::serialize_scale_zp, serialize_scale_zp);
        }
    }
    if (m_attrs.count(Constants::dump_debug_IR))
    {
        bool dump_debug_IR = cast_any<bool>(m_attrs[Constants::dump_debug_IR]);
        if (dump_debug_IR)
        {
            BuilderConfig::set(Constants::dump_debug_IR, dump_debug_IR);
        }
    }

    int layer_number = get_total_nodes_num();
    // m_subgraphs is virtual and all nodes they contained
    // are in this->m_nodes, no need to count layer number again
    if (sg_fmt)
    {
        oss << "subgraph_name=" << m_name << "\n";
    }
    else
    {
        oss << "model_name=" << m_name << "\n";
    }
    oss << "layer_number=" << layer_number << "\n";
    // oss << "data_format=" << m_layout << "\n";
    // oss << "workspace=" << m_global_tensors[Constants::workspace]->mem_size() << "\n";

    auto in_tensors = get_input_tensors();
    if (in_tensors.size() == 0)
    {
        WARN() << "input_tensors is empty";
    }
    oss << "input_tensors=[" << utils::join_string(",", m_in_tensors) << "]\n";

    auto out_tensors = get_output_tensors();
    if (out_tensors.size() == 0)
    {
        WARN() << "output_tensors is empty";
    }
    oss << "output_tensors=[" << utils::join_string(",", m_out_tensors) << "]\n";

    if (m_attrs.count("model_bin"))
    {
        oss << "model_bin=" << cast_any<std::string>(m_attrs["model_bin"]) << "\n";
    }
    if (m_attrs.count("compat_quantized_model"))
    {
        std::string cqm = m_attrs.at("compat_quantized_model").cast<bool>() ? "true" : "false";
        oss << "compat_quantized_model=" << cqm << "\n";
    }

    int batch = -1;
    for (auto t : in_tensors)
    {
        if (!t)
        {
            continue;
        }
        int shape_d0 = t->shape().dim() == 0 ? 1 : t->shape()[0];
        if (batch == -1)
        {
            batch = shape_d0;
        }
        if (batch != shape_d0)
        {
            batch = 1;
            break;
        }
    }
    // oss << "batch_size=" << batch << "\n";

    std::map<std::string, Expr> m;
    for (auto t : in_tensors)
    {
        if (!t)
        {
            continue;
        }
        auto n = t->node();
        std::vector<Expr> exprs;
        if (n && n->params().count("ds_output_shape"))
        {
            std::vector<std::vector<Expr>> ds_output_shape = n->params()["ds_output_shape"];
            if (ds_output_shape.size() > 0)
                exprs = ds_output_shape[0];
        }
        else
        {
            exprs = t->shape().symbol();
        }
        for (auto expr : exprs)
        {
            if (expr->op() == ExprImpl::Op::Constant)
            {
                m[expr->symbol().name()] = expr;
            }
        }
    }
    if (m.size())
    {
        std::vector<Expr> s;
        for (auto &p : m)
        {
            s.push_back(p.second);
        }
        oss << "dynamic_symbols=" << SymbolShape(s) << "\n";
    }
    oss << "\n";

    std::map<char *, uint32_t> data_map;
    bool noid = false;
    walk_on_node([&noid](NodePtr n) {
        if (noid)
        {
            return;
        }
        if (n->attrs().count("layer_id"))
        {
            if (n->get_attr<int>("layer_id") != -1)
            {
                return;
            }
        }
        noid = true; });
    if (noid)
    {
        reset_layer_id();
    }
    std::ostringstream sg_oss;
    for (auto n : m_nodes)
    {
        n->serialize(data_map, oss, wt_oss, sg_oss);
        oss << "\n";
    }
    auto sg_secs = sg_oss.str();
    if (sg_secs.size())
    {
        oss << "\n";
        oss << sg_secs;
    }
}

void Graph::serialize(const std::string &def_file, const std::string &weight_file)
{
    std::ofstream dfile, wfile;
    dfile.open(def_file);
    wfile.open(weight_file, std::ios::binary);
    m_attrs["model_bin"] = weight_file;
    m_attrs["file_path"] = def_file;
    serialize(dfile, wfile);
}

template <typename NodeType, typename TensorType>
GraphPtr Graph::parse(const std::string &filename, GraphPtr g, bool check_ir)
{
    // load plugin before parse a graph
    PluginLoader::get_instance();
    if (!g)
    {
        g = Graph::create();
    }
    aipubt::IRChecker checker(filename, g);
    bool valid_IR = checker.check_all(check_ir);
    // g->topological_sort();
    return valid_IR ? g : nullptr;
}

template <typename TensorType>
GraphPtr Graph::parse(const std::string &filename, GraphPtr g, bool check_ir)
{
    return Graph::parse<Node, TensorType>(filename, g, check_ir);
}

GraphPtr Graph::parse(const std::string &filename, GraphPtr g, bool check_ir)
{
    return Graph::parse<Node, Tensor>(filename, g, check_ir);
}

GraphPtr Graph::parse(std::istream &ifs, GraphPtr g, bool check_ir)
{
    PluginLoader::get_instance();
    if (!g)
    {
        g = Graph::create();
    }
    aipubt::IRChecker checker(ifs, g);
    bool valid_IR = checker.check_all(check_ir);
    return valid_IR ? g : nullptr;
}

bool Graph::is_topological_sorted() const
{
    std::set<NodePtr> ran_nodes;
    for (auto n : m_nodes)
    {
        for (auto p : n->parents())
        {
            if (!ran_nodes.count(p))
            {
                return false;
            }
        }
        ran_nodes.insert(n);
    }
    return true;
}
bool Graph::check_valid_name() const
{
    std::set<std::string> all_node_names, all_tensor_names;
    for (auto n : m_nodes)
    {
        if (all_node_names.count(n->name()))
        {
            DEBUG() << "duplicate node name: " << n->name();
            return false;
        }
        else
        {
            all_node_names.insert(n->name());
        }
        for (auto t : n->outputs())
        {
            if (all_tensor_names.count(t->name()))
            {
                DEBUG() << "duplicate tensor name: " << t->name();
                return false;
            }
            else
            {
                all_tensor_names.insert(t->name());
            }
        }
        for (auto kv : n->constants())
        {
            if (all_tensor_names.count(kv.second->name()))
            {
                DEBUG() << "duplicate weight tensor name: " << kv.second->name();
                return false;
            }
            // else
            // {
            //     all_tensor_names.insert(kv.second->name());
            // }
        }
    }
    return true;
}
bool Graph::check_sg() const
{
    std::vector<NodePtr> node_vec;
    bool ret = true;
    for (auto n : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn)
        {
            ret = ret & sgn->subgraph()->check_sg();
        }
        if (n->m_graph.lock() == GraphRef().lock())
        {
            DEBUG() << n->name() << " does not exist in this graph";
            ret = false;
        }
        for (auto p : n->parents())
        {
            if (!std::count(m_nodes.begin(), m_nodes.end(), p))
            {
                DEBUG() << n->name() << " parent: " << p->name() << " is not in this graph";
                ret = false;
            }
            if (p == n)
            {
                DEBUG() << n->name() << " parent is its own";
                ret = false;
            }
        }
        for (auto c : n->children())
        {
            if (!std::count(m_nodes.begin(), m_nodes.end(), c))
            {
                DEBUG() << n->name() << " children: " << c->name() << " is not in this graph";
                ret = false;
            }
            if (c == n)
            {
                DEBUG() << n->name() << " children is its own";
                ret = false;
            }
        }
    }
    return ret;
}

bool Graph::check_valid_io() const
{
    for (auto node : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            if (!sgn->subgraph()->check_valid_io())
            {
                return false;
            }
        }
        else
        {

            if (node->type() == OpType_Input ||
                node->type() == OpType_SubGraphInput)
            {
                auto t = node->outputs()[0];
                if (!contains(m_in_tensors, t))
                {
                    ERROR() << "input error " << t->name() << " not in in_tensors";
                    return false;
                }
            }
            else
            {
                for (auto t : node->outputs())
                {
                    if (t->is_io_tensor())
                    {
                        if (!contains(m_out_tensors, t))
                        {
                            ERROR() << "output error " << t->name() << " not in out_tensors";
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool Graph::show_graph(int depth)
{
    depth++;
    for (auto node : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            INFO() << "[sg " << depth << "] " << node->name();
            for (auto t : node->parents())
            {
                INFO() << "--[parents]" << t->name();
            }
            for (auto t : node->children())
            {
                INFO() << "--[children]" << t->name();
            }
            sgn->subgraph()->show_graph(depth);
        }
        else
        {
            INFO() << "[node " << depth << "] " << node->name();
            for (auto t : node->parents())
            {
                INFO() << "--[parents]" << t->name();
            }
            for (auto t : node->children())
            {
                INFO() << "--[children]" << t->name();
            }
        }
        if (depth == 1)
            INFO();
    }
    return true;
}

void Graph::topological_sort(const std::vector<std::vector<NodePtr>> &reserve_orders)
{

    for (auto node : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            sgn->subgraph()->topological_sort(reserve_orders);
        }
    }
    m_nodes = topological_sort(get_bottom_nodes(), get_output_nodes(), reserve_orders);
}

bool check_loop(std::set<NodePtr> ran_nodes, NodePtr n, std::set<NodePtr> &ori_n)
{
    bool is_ready = true;
    for (auto p : n->parents())
    {
        if (!ran_nodes.count(p))
        {
            INFO() << n->name() << " cannot run due to " << p->name() << " is not ready";
            if (ori_n.count(p))
            {
                WARN() << "loop!";
            }
            ori_n.insert(p);
            check_loop(ran_nodes, p, ori_n);

            is_ready = false;
            break;
        }
    }
    return is_ready;
}

// we would NOT check the reserve orders' order is correct!!!!!
std::vector<NodePtr> Graph::topological_sort(const std::vector<NodePtr> &begin_nodes,
                                             const std::vector<NodePtr> &end_nodes,
                                             const std::vector<std::vector<NodePtr>> &reserve_orders)
{
    if (begin_nodes.empty())
    {
        return std::vector<NodePtr>();
    }
    auto inodes = begin_nodes;
    std::reverse(inodes.begin(), inodes.end());
    std::vector<NodePtr> sorted;
    std::set<NodePtr> onodes(end_nodes.begin(), end_nodes.end());
    std::set<NodePtr> ran_nodes;
    NodePtr root = inodes[inodes.size() - 1];
    sorted.push_back(root);
    ran_nodes.insert(root);
    inodes.pop_back();
    auto is_ready_for_run = [&ran_nodes](NodePtr n) -> bool {
        bool is_ready = true;
        for (auto p : n->parents())
        {
            if (!ran_nodes.count(p))
            {

                // INFO() << n->name() << " cannot run due to " << p->name() << " is not ready";
                // std::set<NodePtr> empty;
                // check_loop(ran_nodes, p, empty);
                is_ready = false;
                break;
            }
        }
        return is_ready;
    };
    // 0: normal,
    // 1: continue,
    // 2: break,
    auto update_root_child = [&ran_nodes, &root, &inodes, &sorted, &is_ready_for_run]() -> unsigned int {
        if (root->children().size() == 0)
        {
            if (inodes.size())
            {
                root = inodes[inodes.size() - 1];
                // DEBUG() << "add node:" << root;
                sorted.push_back(root);
                inodes.pop_back();
                ran_nodes.insert(root);
                return 1;
            }
            return 2;
        }

        for (unsigned int i = 0; i < root->children().size(); ++i)
        {
            if (is_ready_for_run(root->children()[i]))
            {
                inodes.push_back(root->children()[i]);
            }
        }
        if (inodes.size())
        {
            root = inodes[inodes.size() - 1];
            // DEBUG() << "add node:" << root;
            sorted.push_back(root);
            inodes.pop_back();
            ran_nodes.insert(root);
        }
        else
        {
            return 2;
        }

        return 0;
    };

    while (onodes.size())
    {
        if (onodes.count(root))
        {
            onodes.erase(root);
            auto ret = update_root_child();
            if (ret == 1)
            {
                continue;
            }
            else if (ret == 2)
            {
                break;
            }
            continue;
        }
        auto ret = update_root_child();
        if (ret == 2)
        {
            break;
        }
    }
    std::vector<NodePtr> ret;
    bool reserve_orders_failed = false;
    // if the reserve order is not correct, we will ignore the order and use the original order.
    if (reserve_orders.size())
    {
        for (auto n : sorted)
        {
            if (!contains(ret, n))
            {
                bool is_in_chain = false;
                for (auto chain : reserve_orders)
                {
                    if (contains(chain, n))
                    {
                        for (auto cn : chain)
                        {
                            // to ensure the node is belong to the graph
                            if (!ran_nodes.count(cn))
                            {
                                reserve_orders_failed = true;
                                WARN() << "Failed to do topological sorting with reserve orders due to the node is not belong to the Graph:"
                                       << cn->name()
                                       << ".\n we will use sorted nodes.";

                                break;
                            }
                            ret.push_back(cn);
                            ran_nodes.erase(cn);
                        }
                        is_in_chain = true;
                        break;
                    }
                }
                if (reserve_orders_failed)
                {
                    break;
                }
                if (!is_in_chain)
                {

                    ret.push_back(n);
                }
            }
        }
    }
    else
    {
        ret = std::move(sorted);
    }
    if (reserve_orders_failed)
    {
        ret = std::move(sorted);
    }

    return ret;
}

bool Graph::sort_by_ordered(const std::vector<NodePtr> &ordered_nodes)
{
    bool ret = true;
    for (auto n : ordered_nodes)
    {
        if (!contains(m_nodes, n))
        {
            WARN() << n << " is not in m_graph";
            ret = false;
        }
    }
    for (auto n : m_nodes)
    {
        if (!contains(ordered_nodes, n))
        {
            WARN() << n << " is not in ordered nodes";
            ret = false;
        }
    }

    if (ret)
    {
        m_nodes = ordered_nodes;
    }
    return ret;
}

void Graph::reset_layer_id()
{
    int id = 0;
    walk_on_node([&](NodePtr n) {
        n->set_attr("layer_id", id);
        if (!SubGraphNode::is_subgraph_node(n) && n->type() != OpType_SubGraphInput)
        {
            id++;
        } });
}

bool is_posterity(NodePtr n, NodePtr child, unsigned int search_depth = 3)
{
    if (n->children().empty() || search_depth == 0)
    {
        return false;
    }
    else
    {
        if (contains(n->children(), child))
        {
            return true;
        }
        else
        {
            search_depth--;
            for (auto c : n->children())
            {
                auto ret = is_posterity(c, child, search_depth);
                if (ret)
                {
                    return true;
                }
            }
        }
    }
    return false;
}
std::vector<NodePtr> Graph::topological_sort(const std::vector<NodePtr> &nodes, const std::vector<std::vector<NodePtr>> &reserve_orders)
{
    std::vector<NodePtr> begin;
    std::vector<NodePtr> end;
    for (auto n : nodes)
    {
        if (n->parents().empty())
        {
            begin.push_back(n);
        }
        else
        {
            for (auto p : n->parents())
            {
                if (!contains(nodes, p))
                {
                    begin.push_back(n);
                    break;
                }
            }
        }
        if (n->children().empty())
        {
            end.push_back(n);
        }
        else
        {
            for (auto child : n->children())
            {
                if (!contains(nodes, child))
                {
                    end.push_back(n);
                    break;
                }
            }
        }
    }
    // remove the begin/end pair if begin is the posterity of end
    std::set<NodePtr> invalid_begin;
    std::set<NodePtr> invalid_end;
    if ((end.size() > 0) && (begin.size() > 0))
    {
        for (auto e : end)
        {
            for (auto b : begin)
            {
                if (is_posterity(e, b))
                {
                    invalid_begin.insert(b);
                    invalid_end.insert(e);
                }
            }
        }
        std::vector<NodePtr> new_begin;
        std::vector<NodePtr> new_end;
        for (auto i : begin)
        {
            if (!invalid_begin.count(i))
            {
                new_begin.push_back(i);
            }
        }
        for (auto i : end)
        {
            if (!invalid_end.count(i))
            {
                new_end.push_back(i);
            }
        }

        // for (uint32_t i = 0; i < new_begin.size(); i++)
        // {
        //     DEBUG() << "new_begin[" << i << "]->name: " << new_begin[i]->name();
        // }
        // for (uint32_t i = 0; i < new_end.size(); i++)
        // {
        //     DEBUG() << "new_end[" << i << "]->name: " << new_end[i]->name();
        // }
        return Graph::topological_sort(new_begin, new_end, reserve_orders);
    }
    else
    {
        return Graph::topological_sort(begin, end, reserve_orders);
    }
}

// NodePtr Graph::get_node_by_tensor(const TensorPtr t)
// {
//     if (!t)
//     {
//         return nullptr;
//     }
//     return get_node_by_tensor(t->name());
// }
template <typename T>
static std::string get_valid_name(const std::string &name, const std::map<std::string, T> &name_map)
{
    if (!name_map.count(name) && name.length())
    {
        return name;
    }
    std::string new_name = name;
    std::string name_base = name;
    if (name_map.count(new_name))
    {
        auto pos = new_name.rfind('_');
        int idx = 0;
        if (pos != std::string::npos)
        {
            size_t count = std::count_if(new_name.begin() + pos, new_name.end(),
                                         [](unsigned char c) { return std::isdigit(c); } // correct
            );
            if (count == new_name.length() - pos - 1)
            {
                name_base = new_name.substr(0, pos);
                idx = std::atoi(new_name.substr(pos + 1).c_str()) + 1;
            }
        }
        new_name = name_base + "_" + std::to_string(idx);
        while (name_map.count(new_name))
        {
            idx++;
            new_name = name_base + "_" + std::to_string(idx);
        }
    }
    return new_name;
}
std::string Graph::get_valid_tensor_name(const std::string &name)
{
    return get_valid_name(name, m_tensors);
}
std::string Graph::get_valid_node_name(const std::string &name) const
{
    return get_valid_name(name, m_nodes_map);
}

std::vector<NodePtr> Graph::get_bottom_nodes() const
{
    std::vector<NodePtr> bottom_nodes;
    for (auto n : m_nodes)
    {
        if (n->parents().empty())
        {
            bottom_nodes.push_back(n);
        }
        else
        {
            for (auto p : n->parents())
            {
                if (!contains(m_nodes, p))
                {
                    bottom_nodes.push_back(n);
                    break;
                }
            }
        }
    }
    return bottom_nodes;
}
std::vector<NodePtr> Graph::get_input_nodes() const
{
    std::vector<NodePtr> inodes;
    for (auto n : m_nodes)
    {
        if (n->inputs().empty())
        {
            inodes.push_back(n);
        }
        else
        {
            for (auto inp : n->inputs())
            {
                if (!inp->node())
                {
                    inodes.push_back(n);
                    break;
                }
                else
                {
                    if (!contains(m_nodes, inp->node()))
                    {
                        inodes.push_back(n);
                        break;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < inodes.size(); ++i)
    {
        auto n = inodes[i];
        if (n->type() == OpType_Constant)
        {
            inodes.erase(inodes.begin() + i);
            i--;
        }
    }
    return inodes;
}

std::vector<NodePtr> Graph::get_output_nodes() const
{
    std::vector<NodePtr> onodes;
    for (auto n : m_nodes)
    {
        if (n->children().empty())
        {
            onodes.push_back(n);
        }
        else if (n->get_free_outputs().size())
        {
            onodes.push_back(n);
        }
        else
        {
            for (auto c : n->children())
            {
                if (!contains(m_nodes, c))
                {
                    onodes.push_back(n);
                    break;
                }
            }
        }
    }
    return onodes;
}

std::vector<NodePtr> Graph::get_perf_nodes() const
{
    std::vector<NodePtr> perf_nodes;
    for (auto n : m_nodes)
    {
        if (n->attrs().count(Constants::node_perf) != 0 && n->get_attr<bool>(Constants::node_perf))
        {
            perf_nodes.push_back(n);
        }
    }
    return perf_nodes;
}

std::vector<TensorPtr> Graph::get_input_tensors()
{
    if (m_in_tensors.size() == 0)
    {
        record_input_tensors();
    }
    return m_in_tensors;
}

std::vector<TensorPtr> Graph::get_output_tensors()
{
    if (m_out_tensors.size() == 0)
    {
        record_output_tensors();
    }
    return m_out_tensors;
}

const std::vector<GraphPtr> Graph::get_all_graphs() const
{
    std::vector<GraphPtr> ret;
    for (auto n : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn)
        {
            auto gs = sgn->subgraph()->get_all_graphs();
            ret.insert(ret.end(), gs.begin(), gs.end());
        }
    }
    ret.push_back(((Graph *)this)->shared_from_this());
    return ret;
}

void Graph::walk_on_node(std::function<void(NodePtr)> f, bool recursive, bool top2down)
{
    std::function<void(NodePtr)> wf = f;
    if (recursive)
    {
        if (top2down)
        {
            wf = [&](NodePtr n) {
                f(n);
                auto sgn = SubGraphNode::from_node(n);
                if (sgn)
                {
                    for (auto sn : sgn->subgraph()->nodes())
                    {
                        wf(sn);
                    }
                }
            };
        }
        else
        {
            wf = [&](NodePtr n) {
                auto sgn = SubGraphNode::from_node(n);
                if (sgn)
                {
                    for (auto sn : sgn->subgraph()->nodes())
                    {
                        wf(sn);
                    }
                }
                f(n);
            };
        }
    }

    for (auto n : m_nodes)
    {
        wf(n);
    }
}

// void Graph::extend(const std::vector<TensorPtr> &out_tensors)
// {
//     std::vector<TensorPtr> outs = out_tensors;
//     while (outs.size())
//     {
//         auto t = outs[0];
//         auto node=t->node();
//         outs.erase(outs.begin());
//         if (node==nullptr)
//         {
//             WARN() << "Unkown tensor's node: " << t->name();
//             continue;
//         }
//         add_node(node);
//         for (auto inp : node->inputs())
//         {
//             outs.push_back(inp);
//         }
//     }
// }

// the node attributes may loss when clone
GraphPtr Graph::clone(GraphPtr g) const
{
    return clone(m_nodes, g);
}

static void dfs_for_extract_subgraph(std::vector<NodePtr> &all_nodes, NodePtr root, const std::vector<NodePtr> &inodes = std::vector<NodePtr>())
{
    for (auto n : root->parents())
    {
        if (!contains(all_nodes, n))
        {
            all_nodes.push_back(n);
            if (!contains(inodes, n))
            {
                dfs_for_extract_subgraph(all_nodes, n, inodes);
            }
        }
    }
}
GraphPtr Graph::clone(const std::vector<NodePtr> &all_nodes, GraphPtr g, bool insert_input) const
{
    if (g == nullptr)
    {
        g = Graph::create();
    }
    auto old_nodes = g->nodes();
    for (auto n : old_nodes)
    {
        g->remove_node(n);
    }
    g->m_name = m_name;
    g->m_layout = m_layout;
    g->m_symbols = m_symbols;

    // build graph
    for (auto n : all_nodes)
    {
        auto node = n->clone();
        node->clear_inputs();
        g->add_node(node);
    }
    for (auto n : all_nodes)
    {
        auto node = g->m_nodes_map[n->name()];
        for (auto t : n->inputs())
        {
            if (!g->m_tensors.count(t->name()) && insert_input)
            {
                auto name = g->get_valid_node_name("input");
                auto inp = Node::create(name, OpType_Input);
                g->add_node(inp);
                auto tensor = std::make_shared<Tensor>(*t);
                inp->add_output(tensor);
            }
            node->add_input(g->m_tensors[t->name()]);
        }
    }
    // copy graph property
    g->m_global_tensors.clear();
    for (auto kv : m_global_tensors)
    {
        auto tensor = std::make_shared<Tensor>(*kv.second);
        g->m_global_tensors[kv.first] = tensor;
    }

    g->m_in_tensors.clear();
    for (auto t : m_in_tensors)
    {
        if (g->m_tensors.count(t->name()))
        {
            g->m_in_tensors.push_back(g->m_tensors.at(t->name()));
        }
    }
    g->m_out_tensors.clear();
    for (auto t : m_out_tensors)
    {
        if (g->m_tensors.count(t->name()))
        {
            g->m_out_tensors.push_back(g->m_tensors.at(t->name()));
        }
    }
    // reconstruct virtual subgraph
    for (auto sg : m_subgraphs)
    {
        NodePtrList nodes;
        for (auto n : sg->nodes())
        {
            nodes.push_back(m_nodes_map.at(n->name()));
        }
        SubGraphPtr subg = SubGraph::create(g, nodes);
        // copy
        subg->set_name(sg->name());
        g->subgraphs().push_back(subg);
    }
    // TODO: forrange and other node
    return g;
}

GraphPtr Graph::clone(const std::vector<NodePtr> &begin_nodes, const std::vector<NodePtr> &end_nodes, GraphPtr g, bool insert_input) const
{
    std::vector<NodePtr> all_nodes = end_nodes;
    for (auto n : end_nodes)
    {
        dfs_for_extract_subgraph(all_nodes, n, begin_nodes);
    }
    std::reverse(all_nodes.begin(), all_nodes.end());
    return clone(all_nodes, g, insert_input);
}
GraphPtr Graph::clone(const std::vector<TensorPtr> &begin_tensors, const std::vector<TensorPtr> &end_tensors, GraphPtr g, bool insert_input) const
{
    std::vector<NodePtr> bnodes, enodes;
    for (auto t : begin_tensors)
    {
        auto n = t->node();
        if (!contains(bnodes, n))
        {
            bnodes.push_back(n);
        }
    }
    for (auto t : end_tensors)
    {
        auto n = t->node();
        if (!contains(enodes, n))
        {
            enodes.push_back(n);
        }
    }
    return clone(bnodes, enodes, g, insert_input);
}

GraphPtr Graph::clone(const std::vector<std::string> &begin_nodes, const std::vector<std::string> &end_nodes, GraphPtr g, bool insert_input) const
{
    std::vector<NodePtr> bnodes, enodes;
    for (auto n : begin_nodes)
    {
        if (m_nodes_map.count(n))
            bnodes.push_back(m_nodes_map.at(n));
    }
    for (auto n : end_nodes)
    {
        if (m_nodes_map.count(n))
            enodes.push_back(m_nodes_map.at(n));
    }
    return clone(bnodes, enodes, g, insert_input);
}

void Graph::record_input_tensors(const std::vector<TensorPtr> &input_tensors)
{
    for (auto t : m_in_tensors)
    {
        t->set_io_tensor(false);
    }
    m_in_tensors.clear();
    if (input_tensors.size())
    {
        bool set_inputs = true;
        for (auto t : input_tensors)
        {
            if (t->node() == nullptr ||
                t->node()->graph() != shared_from_this() ||
                (t->node()->type() != OpType_Input && t->node()->type() != OpType_SubGraphInput))
            {
                set_inputs = false;
                WARN() << "Setting a tensor: " << t->name()
                       << " as input tensor which not belong this graph or the tensor's node is not a Input Type.";
                break;
            }
            t->set_io_tensor(true);
            m_in_tensors.push_back(t);
        }
        if (set_inputs)
        {
            return;
        }
        m_in_tensors.clear();
    }
    for (auto n : get_input_nodes())
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            auto sgn = SubGraphNode::from_node(n);
            for (auto t : sgn->subgraph()->get_input_tensors())
            {
                t->set_io_tensor(true);
                m_in_tensors.push_back(t);
            }
            continue;
        }
        if (n->type() == OpType_Input)
        {
            for (auto t : n->outputs())
            {
                t->set_io_tensor(true);
                m_in_tensors.push_back(t);
            }
        }
        else
        {
            for (auto t : n->inputs())
            {
                t->set_io_tensor(true);
                m_in_tensors.push_back(t);
            }
        }
    }
}

void Graph::record_output_tensors(const std::vector<TensorPtr> &output_tensors)
{
    for (auto t : m_out_tensors)
    {
        t->set_io_tensor(false);
    }
    m_out_tensors.clear();
    if (output_tensors.size())
    {
        bool is_set = true;
        for (auto t : output_tensors)
        {
            if (t->node() == nullptr || t->node()->graph() != shared_from_this())
            {
                is_set = false;
                WARN() << "Setting a tensor: " << t->name()
                       << " as input tensor which not belong this graph.";
                break;
            }
            m_out_tensors.push_back(t);
            t->set_io_tensor(true);
        }
        if (is_set)
        {
            return;
        }
        m_out_tensors.clear();
    }
    for (auto n : get_output_nodes())
    {
        for (auto t : n->get_free_outputs())
        {
            t->set_io_tensor(true);
            m_out_tensors.push_back(t);
        }
    }
}

void Graph::load_weight(const std::string &wf)
{
    std::string filename = wf;
    if (!wf.size() && m_attrs.count("model_bin"))
    {
        filename = cast_any<std::string>(m_attrs["model_bin"]);
    }
    std::ifstream infile(filename, std::ios::binary | std::ios::in);
    infile.seekg(0, std::ios::end);
    auto file_size = infile.tellg();
    INFO() << "loading graph weight: " << filename << " size: 0x" << std::hex << file_size;
    load_weight(infile);
    infile.close();
}

static void restore_offsets(GraphPtr g)
{
    for (auto node : g->nodes())
    {
        if (SubGraphNode::is_subgraph_node(node))
        {
            auto sgn = std::static_pointer_cast<SubGraphNode>(node);
            auto sg = sgn->subgraph();
            for (auto &cts : sgn->constants())
            {
                for (auto n : sg->nodes())
                {
                    bool is_restored = false;
                    for (auto &kv : n->constants())
                    {
                        if (cts.first == n->name() + "/" + kv.first)
                        {
                            auto offset = cts.second->memory_offset()->relative_offset;
                            kv.second->memory_offset()->relative_offset = offset;
                            is_restored = true;
                            break;
                        }
                    }
                    if (is_restored)
                    {
                        break;
                    }
                }
            }
            restore_offsets(sg);
        }
    }
}
static void update_constants(GraphPtr g)
{
    for (auto node : g->nodes())
    {
        node->update_constants();
    }
}
void Graph::load_weight(std::istream &infile)
{
    if (!infile)
    {
        ERROR() << "Cannot read weight files";
        return;
    }
    infile.seekg(0, std::ios::end);
    auto file_size = infile.tellg();
    std::map<uint64_t, std::pair<uint64_t, TensorPtr>> data_map;
    restore_offsets(shared_from_this());
    for (auto node : m_nodes)
    {
        NodePtrList realnodes;
        if (SubGraphNode::is_subgraph_node(node))
        {
            auto sgn = std::static_pointer_cast<SubGraphNode>(node);
            // if the subgraph has it's own weight bin
            if (sgn->attrs().count("own_weight"))
            {
                auto sg = sgn->subgraph();
                std::string bin_path;
                if (sg->attrs().count("model_bin"))
                {
                    bin_path = cast_any<std::string>(sg->attrs()["model_bin"]);
                }
                if (bin_path.size() && utils::path_exists(bin_path))
                {
                    sg->load_weight(bin_path);
                }
            }
            else
            {
                realnodes = sgn->get_real_node();
            }
        }
        else
        {
            realnodes.push_back(node);
        }
        for (auto n : realnodes)
        {
            auto &constants = n->constants();
            auto &params = n->params();
            for (auto kv : constants)
            {
                auto t = kv.second;
                // shared weight
                if (data_map.count(kv.second->memory_offset()->relative_offset))
                {
                    if (data_map[kv.second->memory_offset()->relative_offset].first == kv.second->mem_size())
                    {
                        t->share_data_from(*data_map[kv.second->memory_offset()->relative_offset].second);
                        continue;
                    }
                }
                auto const_offset = kv.second->memory_offset()->relative_offset;
                uint64_t size = kv.second->mem_size();
                if (const_offset + size > (uint64_t)file_size)
                {
                    ERROR() << n->name()
                            << ": Out of range when reading " << kv.first
                            << ". Reading range: [" << const_offset << "," << const_offset + size
                            << "), but length of weight file is: " << file_size;
                    return;
                }
                infile.seekg(const_offset);
                char *data = new char[size];
                infile.read(data, size);
                if (infile.fail())
                {
                    WARN() << "Failed to read weight for Tensor " << kv.first << " of Node " << node->name();
                }
                else
                {
                    t->set_data(data, size);
                    data_map[kv.second->memory_offset()->relative_offset] = {kv.second->mem_size(), t};
                }
            }
            // push scale and zp to quantization
            std::set<std::string> removed_constants;
            std::set<std::string> removed_params;
            for (auto kv : constants)
            {
                if (utils::endswith(kv.first, "_range"))
                {
                    auto key = kv.first;
                    utils::replace(key, "_range", "");
                    if (constants.count(key + "_range"))
                    {
                        TensorPtr w;
                        if (constants.count(key))
                        {
                            w = constants[key];
                        }
                        else
                        {
                            for (auto o : n->outputs())
                            {
                                if (key == o->name())
                                {
                                    w = o;
                                    break;
                                }
                            }
                        }
                        if (w)
                        {
                            removed_constants.insert(kv.first);
                            ComputeTensor range = constants[key + "_range"];
                            auto t = range.astype(Tensor::Dtype::FP32).flatten().reshape(2, -1);
                            std::vector<float> mins = (ComputeTensor)t[0];
                            std::vector<float> maxes = (ComputeTensor)t[1];
                            w->quantization().set_mins(mins);
                            w->quantization().set_maxs(maxes);
                        }
                    }
                }

                auto do_set_quant_info = [&](TensorPtr w, std::string key, std::string suffix = "") {
                    std::string scale_name = key;
                    std::string zp_name = key;
                    if (key.empty())
                    {
                        scale_name += "scale";
                        zp_name += "zp";
                    }
                    else
                    {
                        scale_name += "_scale";
                        zp_name += "_zp";
                    }
                    scale_name += suffix;
                    zp_name += suffix;
                    if (suffix == "0" && !constants.count(scale_name))
                        return;

                    if (w)
                    {
                        if (constants.count(scale_name))
                        {
                            ComputeTensor s = constants[scale_name];
                            w->quantization().set_scales(s);
                            removed_constants.insert(scale_name);
                        }
                        else if (params.count(scale_name))
                        {
                            float scale = params[scale_name];
                            w->quantization().set_scales(std::vector<float>{scale});
                            removed_params.insert(scale_name);
                        }

                        if (constants.count(zp_name))
                        {
                            ComputeTensor z = constants[zp_name];
                            w->quantization().set_offsets(z);
                            removed_constants.insert(zp_name);
                        }
                        else if (params.count(zp_name))
                        {
                            int32_t zp = params[zp_name];
                            w->quantization().set_offsets(ComputeTensor(zp).astype(kv.second->type()));
                            removed_params.insert(zp_name);
                        }
                    }
                };

                if (utils::endswith(kv.first, "_scale0"))
                {
                    // weights_scale0_offset/weights_scale1_offset,...
                    int q_level = 0;
                    std::string key = kv.first;
                    key = utils::replace(key, "_scale0", "");
                    TensorPtr w = nullptr;
                    if (constants.count(key))
                    {
                        w = constants[key];
                    }
                    else
                    {
                        for (auto o : n->outputs())
                        {
                            if (key == o->name())
                            {
                                w = o;
                                break;
                            }
                        }
                    }
                    while (w && constants.count(key + "_scale" + std::to_string(q_level)))
                    {
                        do_set_quant_info(w, key, std::to_string(q_level));
                        if (w->quantization().empty())
                        {
                            break;
                        }
                        w = w->quantization().scales();
                        ++q_level;
                    }
                }
                else if (kv.first == "scale0")
                {
                    // constants['scale0']/.constants['scale1'] set to 'weights' quantization
                    int q_level = 0;
                    std::string key = "weights";
                    TensorPtr w = nullptr;
                    if (constants.count(key))
                    {
                        w = constants[key];
                    }
                    else
                    {
                        for (auto o : n->outputs())
                        {
                            if (key == o->name())
                            {
                                w = o;
                                break;
                            }
                        }
                    }
                    while (w && constants.count("scale" + std::to_string(q_level)))
                    {
                        do_set_quant_info(w, "", std::to_string(q_level));
                        if (w->quantization().empty())
                        {
                            break;
                        }
                        w = w->quantization().scales();
                        ++q_level;
                    }
                }
                else if (utils::endswith(kv.first, "_zp"))
                {
                    std::string key = kv.first;
                    key = utils::replace(key, "_zp", "");
                    TensorPtr w = nullptr;
                    if (constants.count(key))
                    {
                        w = constants[key];
                    }
                    else
                    {
                        for (auto o : n->outputs())
                        {
                            if (key == o->name())
                            {
                                w = o;
                                break;
                            }
                        }
                    }
                    do_set_quant_info(w, key);
                }

                if (params.count(kv.first + "_zp") && params.count(kv.first + "_scale"))
                {
                    int32_t zp = params[kv.first + "_zp"];
                    float scale = params[kv.first + "_scale"];
                    removed_params.insert(kv.first + "_scale");
                    removed_params.insert(kv.first + "_zp");
                    kv.second->quantization().set_offsets(ComputeTensor(zp).astype(kv.second->type()));
                    kv.second->quantization().set_scales(std::vector<float>{scale});
                }
                else if (params.count(kv.first + "_zp0") && params.count(kv.first + "_scale0"))
                {
                    // weights_scale0/weights_scale1
                    int q_level = 0;
                    TensorPtr w = kv.second;
                    while (params.count(kv.first + "_zp" + std::to_string(q_level)) &&
                           params.count(kv.first + "_scale" + std::to_string(q_level)))
                    {
                        std::string scale_name = kv.first + "_scale" + std::to_string(q_level);
                        std::string zp_name = kv.first + "_zp" + std::to_string(q_level);
                        int32_t zp = params[zp_name];
                        float scale = params[scale_name];
                        w->quantization().set_offsets(ComputeTensor(zp).astype(kv.second->type()));
                        w->quantization().set_scales(std::vector<float>{scale});
                        w = w->quantization().scales();
                        removed_params.insert(scale_name);
                        removed_params.insert(zp_name);
                        q_level++;
                    }
                }
                if (params.count(kv.first + "_range"))
                {
                    std::vector<float> ranges = params[kv.first + "_range"];
                    removed_params.insert(kv.first + "_range");
                    kv.second->quantization().set_min(ranges[0]);
                    kv.second->quantization().set_max(ranges[1]);
                }
            }
            for (auto k : removed_constants)
            {
                if (k == "scale0" || k == "scale1" || k == "zp0")
                {
                    continue;
                }
                constants.erase(k);
            }
            for (auto k : removed_params)
            {
                params.erase(k);
            }
        }
    }
    update_constants(shared_from_this());
}
} // namespace aipubt
