
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
 * @Author: Kyle Huang
 * @Date: 2026-01-08 16:50:53
 */
#include "code_emitter.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/proxy_tensor.h"
#include "utils/utils.h"
#include <core/optype.h>
#include <fstream>
#include <queue>
#include <set>
#include <string>

namespace aipubt
{

std::string CodeEmitter::get_type(TensorPtr t)
{
    std::string type = t->type_s();
    if (type == "int8" || type == "uint8")
    {
        type = "char";
    }
    else if (type == "int16" || type == "uint16")
    {
        type = "short";
    }
    else if (type == "float16")
    {
        type = "half";
    }
    else if (type == "int32" || type == "uint32")
    {
        type = "int";
    }
    else if (type == "float32")
    {
        type = "float";
    }
    else
    {
        ERROR() << "unsupported type " << type;
    }

    return type;
}

int CodeEmitter::get_ro_idx(TensorPtr t, const std::vector<std::pair<TensorPtr, std::string>> &args)
{
    for (size_t i = 0; i < args.size(); ++i)
    {
        if (args[i].first == t)
        {
            return i;
        }
    }
    ERROR() << "can not find arg name for " << t->name();
    return -1;
}
static std::string name_to_legal_var_name(std::string name)
{
    std::replace(name.begin(), name.end(), '/', '_');
    return "val_" + name;
}

static std::string name_to_legal_kernel_name(std::string name)
{
    std::replace(name.begin(), name.end(), '/', '_');
    return name;
}

std::stringstream CodeEmitter::emitter_op(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars)
{
    if (n->type() == OpType::OpType_Logical)
    {
        return emitter_logical(n, vars);
    }
    else if (n->type() == OpType::OpType_Cast)
    {
        return emitter_cast(n, vars);
    }
    else if (n->type() == OpType::OpType_InternalConstant)
    {
        return emitter_constant(n, vars);
    }
    else if (n->type() == OpType::OpType_Eltwise)
    {
        return emitter_eltwise(n, vars);
    }
    else if (n->type() == OpType::OpType_Gather)
    {
        return emitter_gather(n, vars);
    }
    else if (n->type() == OpType::OpType_Reshape)
    {
        return emitter_reshape(n, vars);
    }
    else if (n->type() == OpType::OpType_Activation)
    {
        return emitter_activation(n, vars);
    }
    else
    {
        ERROR() << "unsupported op type";
    }
    return std::stringstream{};
}

void CodeEmitter::emit_func(const std::vector<BaseNodePtr> &nodes, BaseNodePtr node, const std::string &name)
{
    std::string kernel_name = name_to_legal_kernel_name(name);
    std::vector<std::pair<TensorPtr, std::string>> args;
    std::map<TensorPtr, std::string> vars; //variable

    std::stringstream code;
    std::vector<TensorPtr> input_t;
    std::vector<TensorPtr> output_t;

    std::vector<BaseNodePtr> input_nodes;
    std::vector<BaseNodePtr> ordered_nodes;

    for (auto n : nodes)
    {
        if (n->type() == OpType::OpType_Gather)
        {
            m_ptrs.insert(n->inputs()[0]);
        }

        bool is_input_n = true;
        for (auto i : n->inputs())
        {
            vars[i] = name_to_legal_var_name(i->name());
            auto input_n = i->node();
            if (input_n && !contains(nodes, input_n))
            {
                input_t.push_back(i);
            }
            else if (!input_n)
            {
                input_t.push_back(i);
            }
            else
            {
                is_input_n = false;
            }
        }
        if (is_input_n)
        {
            input_nodes.push_back(n);
        }

        for (auto o : n->outputs())
        {
            vars[o] = name_to_legal_var_name(o->name());
            if (o->consumers().size() == 0)
            {
                output_t.push_back(o);
            }
            else
            {
                auto o_p = std::dynamic_pointer_cast<ProxyTensor>(o);
                if (o_p)
                {
                    auto proxied_outp = o_p->proxied_tensor();
                    for (auto outp : node->outputs())
                    {
                        if (outp == proxied_outp)
                        {
                            output_t.push_back(o);
                        }
                    }
                }
            }
        }
    }

    std::stable_sort(input_nodes.begin(), input_nodes.end(), [](BaseNodePtr a, BaseNodePtr b) {
        return a->name() < b->name();
    });

    std::queue<BaseNodePtr> q;
    std::set<BaseNodePtr> visited;
    std::map<BaseNodePtr, std::set<BaseNodePtr>> dep;

    for (auto n : nodes)
    {
        for (auto i : n->inputs())
        {
            auto dep_n = i->node();
            if (dep_n && contains(nodes, dep_n))
            {
                dep[n].insert(dep_n);
            }
        }
    }

    for (auto n : input_nodes)
    {
        if (dep[n].size() > 0)
        {
            ERROR();
        }
        q.push(n);
        visited.insert(n);
    }
    while (q.size())
    {
        auto front_n = q.front();
        q.pop();
        ordered_nodes.push_back(front_n);
        // INFO() << "ordered node: " << front_n;
        for (auto o : front_n->outputs())
        {
            for (auto n : o->consumers())
            {
                if (!contains(visited, n) && contains(nodes, n))
                {
                    if (!contains(dep[n], front_n))
                    {
                        ERROR();
                    }
                    dep[n].erase(front_n);
                    if (dep[n].size() == 0)
                    {
                        visited.insert(n);
                        q.push(n);
                    }
                }
            }
        }
    }

    std::stable_sort(input_t.begin(), input_t.end(), [](TensorPtr a, TensorPtr b) {
        return a->name() < b->name();
    });
    for (auto t : input_t)
    {
        // INFO() << "input arg: " << t->name() << " " << vars[t];
        args.push_back(std::make_pair(t, vars[t]));
    }

    std::stable_sort(output_t.begin(), output_t.end(), [](TensorPtr a, TensorPtr b) {
        return a->name() < b->name();
    });
    for (auto t : output_t)
    {
        // INFO() << "output arg: " << t->name() << " " << vars[t];
        args.push_back(std::make_pair(t, vars[t]));
    }

    //cannot have a tensor in both input and output

    for (auto n : ordered_nodes)
    {
        code << emitter_op(n, vars).str();
    }

    std::vector<std::pair<TensorPtr, std::string>> sorted_vars(vars.begin(), vars.end());
    std::sort(sorted_vars.begin(), sorted_vars.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });
    if (aipubt::BuilderConfig::get<bool>(Constants::print_emitter_var))
    {
        code << "printf(\"kernel: " << kernel_name << "\\n\");\n";
        for (auto var : sorted_vars)
        {
            auto t = var.first;

            std::string type = get_type(t) == "half" || get_type(t) == "float" ? "f" : "d";
            auto name = var.second;
            if (contains(m_ptrs, t))
            {
                code << "printf(\"" << name << "[0] = %" << type << "  %"
                     << "x"
                     << "\\n\"," << name << "[0]," << name << ");\n";
            }
            else
            {
                code << "printf(\"" << name << " = %" << type << "  %"
                     << "x"
                     << "\\n\"," << name << "," << name << ");\n";
            }
        }
    }

    // INFO() << "emitter code: ";
    // INFO() << code.str();

    std::stringstream ss;
    ss << "#include \"aipu.h\"\n";
    ss << "#include \"clc/clc.h\"\n";
    ss << "void " << kernel_name << "()\n";
    ss << "{\n";
    ss << "    uint reg = 0;\n";
    ss << "    __asm__(\"{mfctrl0 %[regs],0x14;}\"\n";
    ss << "        : [regs] \"=&r\"(reg)\n";
    ss << "        :);\n";
    ss << "    uint *tcb_ptr = (uint *)reg;\n";
    ss << "    uint *pp = (uint *)tcb_ptr[14];\n";
    ss << "    uint *arg = (uint *)pp[0];\n";
    ss << "    uint tec_id = tcb_ptr[20] >> 16;\n";
    ss << "    if (tec_id == 0)\n";
    ss << "    {\n";

    // ss << "printf(\"ro: %" << "x\\n\", arg);\n";
    for (auto var : sorted_vars)
    {
        auto t = var.first;
        auto name = var.second;
        if (contains(input_t, t))
        {
            //init input
            auto idx = get_ro_idx(t, args);
            std::string type = get_type(t);
            if (contains(m_ptrs, t))
            {
                ss << "        " << type << "* " << var.second << " = (" << type << "*)arg[" << idx << "];\n";
            }
            else
            {
                ss << "        " << type << " " << var.second << " = *(" << type << "*)arg[" << idx << "];\n";
            }
        }
        else
        {
            //init output
            std::string type = get_type(t);
            ss << "        " << type << " " << var.second << ";\n";
        }
    }

    ss << code.str();

    //write output
    for (auto t : output_t)
    {
        auto idx = get_ro_idx(t, args);
        std::string type = get_type(t);
        auto name = vars[t];
        ss << "        *(" << type << "*)arg[" << idx << "] = " << name << ";\n";
    }
    ss << "        __builtin_aipu_wfe(0xffffffdf, 0);\n";
    for (size_t idx = 0; idx < args.size(); ++idx)
    {
        auto t = args[idx].first;

        if (contains(m_ptrs, t))
        {
            uint32_t cache_chunk = 64;
            auto arg1 = vars.at(t);
            size_t num_chunks = (t->mem_size() + cache_chunk - 1) / cache_chunk;
            for (size_t i = 0; i < num_chunks; ++i)
            {
                ss << "        __cacheop((uint)((uint)arg[" << idx << "] + " << i * cache_chunk << "), 24, 0);\n";
            }
        }
        else
        {
            ss << "        __cacheop((uint)arg[" << idx << "], 24, 0);\n";
        }
    }
    // ss << "    __flush_cache();\n";

    ss << "    }\n";

    ss << "    __builtin_aipu_wfe(0xffffffdf, 0);\n";
    ss << "    __builtin_aipu_wfe(0x7fffffff, 2);\n";
    ss << "}\n";

    // INFO() << "emitter full code: ";
    // INFO() << ss.str();

    std::string workdir;
    if (BuilderConfig::exist(Constants::global_cwd))
    {
        workdir = BuilderConfig::get<std::string>(Constants::global_cwd) + "/ld";
        utils::mkdir(workdir);
    }
    else
    {
        workdir = utils::get_temp_dir() + "/" + utils::get_uuid();
        utils::mkdir(workdir);
    }

    std::string file_name = workdir + "/" + kernel_name + ".cl";
    std::string code_name = workdir + "/" + kernel_name + ".bin";
    std::ofstream file(file_name);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return;
    }
    file << ss.str();
    file.close();
    // INFO() << "CL file name " << file_name;
    // INFO() << "kernel name " << kernel_name;
    // INFO() << "code name " << code_name;

    if (m_info.find(name) != m_info.end())
    {
        ERROR() << "kernel name " << name << " has been used";
    }

    m_info[name] = {kernel_name, code_name, file_name, args};
}

void CodeEmitter::compiler_Driver(const std::string &name)
{
    std::string cmd = "aipuocc -target aipu -O2 -mcpu=" + Target::get_the_target()->name() + " -c " + m_info[name].file_name + " -o " + m_info[name].code_name;
    // INFO() << "compiler driver cmd " << cmd;
    bool ret = system(cmd.c_str());

    if (ret)
    {
        ERROR() << "emitter compile error";
    }
}

std::string CodeEmitter::code_name(const std::string &name)
{
    return m_info[name].code_name;
}

std::string CodeEmitter::kernel_name(const std::string &name)
{
    return m_info[name].kernel_name;
}
std::vector<TensorPtr> CodeEmitter::args(const std::string &name)
{
    auto args = m_info[name].args;
    std::vector<TensorPtr> ret;
    for (auto arg : args)
    {
        ret.push_back(arg.first);
    }
    return ret;
}
} // namespace aipubt