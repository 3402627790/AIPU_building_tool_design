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
 * @Date: 2022-11-15 09:57:24
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-22 17:34:46
 */
#include "tools_funcs.h"
#include <algorithm>
#include <builder/builder.h>
#include <builder/dumper/dumper_x.h>
#include <builder/dumper/dumper_x1.h>
#include <builder/lowering/lowering.h>
#include <builder/mcbuilder.h>
#include <builder/mqbuilder.h>
#include <builder/profiler.h>
#include <builder/target.h>
#include <builder/tools.h>
#include <core/any.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <core/pass_manager.h>
#include <core/plugin_loader.h>
#include <core/stub.h>
#include <dirent.h>
#include <fstream>
#include <gcompiler_api.h>
#include <iostream>
#include <launcher.h>
#include <regex>
#include <set>
#include <statistic/allocator_fp_estimation.h>
#include <statistic/perf_estimator.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <transform/passes.h>
#include <unistd.h>
#include <utils/utils.h>

namespace aipubt
{
static std::string get_simulator_path(const std::string &simulator)
{
    std::string sim_path = simulator;
    auto cfg_dir = utils::get_home_dir() + "/.aipubuilder/";
    std::string cmd = "mkdir -p " + cfg_dir;
    system(cmd.c_str());
    auto cfg = cfg_dir + "config";
    std::string target_name = "ZYV1";
    if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZY)
    {
        target_name = "ZYV1";
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv2)
    {
        target_name = "ZYV2";
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv3)
    {
        target_name = "ZYV3";
        if (aipubt::get_target()->name() == "X1_1204")
        {
            target_name = "ZYX1";
        }
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv5)
    {
        target_name = "ZYX2";
        if (aipubt::get_target()->name() == "Z5_0901")
        {
            target_name = "ZYV5";
        }
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv6)
    {
        target_name = "ZYX3";
        if (aipubt::get_target()->chipinfo().revision == 1)
        {
            target_name = "ZYX3P";
        }
        else if (aipubt::get_target()->chipinfo().revision == 2)
        {
            target_name = "ZYX3S";
        }
    }
    else
    {
        // TODO
    }
    if (!utils::file_exists(cfg))
    {
        std::ofstream ofile(cfg);
        utils::JsonObject json_sim;
        json_sim["simulator"] = utils::JsonObject(utils::JsonObject::OBJ);
        ofile << json_sim.to_string();
        ofile.close();
    }
    std::stringstream buffer;
    std::ifstream ifile(cfg);
    buffer << ifile.rdbuf();
    utils::JsonObject config = utils::JsonObject::parse(buffer.str());
    ifile.close();
    auto &json_sim = config["simulator"];
    if (simulator.empty())
    {
        if (ifile)
        {
            if (!json_sim.exist(target_name))
            {
                if (target_name != "ZYX2")
                {
                    ERROR() << "No target " << target_name << " simulator path found";
                }
            }
            else
            {
                std::string path = json_sim[target_name];
                sim_path = path;
            }
        }
    }
    else
    {
        if (json_sim.exist(target_name))
        {
            std::string sim = json_sim[target_name];
            if (sim == sim_path)
            {
                return sim_path;
            }
        }
        json_sim[target_name] = sim_path;
        std::ofstream ofile(cfg);
        if (ofile)
        {
            ofile << config.to_string();
            ofile.close();
        }
    }
    return sim_path;
}

static bool run_simulator(const std::string &cmd)
{
    INFO() << "run simulator:\n"
           << cmd;
    bool ret = system(cmd.c_str()) == 0;
    INFO() << "Simulator finished.";
    return ret;
}

static bool handle_op_counter(std::string name)
{
    std::ifstream op_cnt_file(name, std::ios::binary | std::ios::in);
    if (!op_cnt_file)
    {
        WARN() << "Cannot read op_counter.bin";
    }
    else
    {
        op_cnt_file.seekg(0, std::ios::end);
        size_t file_size = op_cnt_file.tellg();
        op_cnt_file.seekg(0);
        if (file_size < 4)
        {
            WARN() << "invalid size of op_counter.bin:" << file_size << " (expected 4)";
        }
        else
        {
            uint32_t counter;
            char *counter_ptr = (char *)(&counter);
            op_cnt_file.read(counter_ptr, 4);
            DEBUG() << "op_counter (Operator ID of last finished layer): " << counter;
        }
    }
    op_cnt_file.close();
    return true;
}

static bool handle_error_code(std::string name)
{
    static std::map<uint16_t, std::string> error_code_map = {
        {0x200, "RUNTIME_ERROR"},

        {0x300, "KEYBOARD_INTERRUPT"},
        {0x301, "SYSTEM_ERROR"},

        {0x380, "OUT_OF_SPEC"},
        {0x381, "OUT_OF_AIFF_SPEC"},
        {0x382, "OUT_OF_TPC_SPEC"},
        {0x383, "OUT_OF_DMA_SPEC"},

        {0x400, "OUT_OF_MEM_RANGE"},
        {0x401, "OUT_OF_SRAM_RANGE"},
        {0x402, "OUT_OF_LSRAM0_RANGE"},
        {0x403, "OUT_OF_LSRAM1_RANGE"},
        {0x404, "OUT_OF_LSRAM_RANGE"},
        {0x405, "OUT_OF_GSRAM0_RANGE"},
        {0x406, "OUT_OF_GSRAM1_RANGE"},
        {0x407, "OUT_OF_GSRAM_RANGE"},

        {0x480, "ARITHMETIC_ERROR"},
        {0x481, "FLOAT_POINT_ERROR"},
        {0x482, "UNDERFLOW_ERROR"},
        {0x483, "OVERFLOW_ERROR"},
        {0x484, "NOT_A_NUMBER_ERROR"},
        {0x485, "INFINITY_ERROR"},
        {0x486, "STR_LENGTH_ERROR"},
        {0x487, "ZERO_DIVISION_ERROR"},
    };
    std::ifstream error_code_file(name, std::ios::binary | std::ios::in);
    if (!error_code_file)
    {
        DEBUG() << "Cannot read error_code.bin";
    }
    else
    {
        error_code_file.seekg(0, std::ios::end);
        size_t file_size = error_code_file.tellg();
        error_code_file.seekg(0);
        if (file_size < 4)
        {
            WARN() << "invalid size of error_code.bin:" << file_size << " (expected 4)";
        }
        else
        {
            uint32_t error_code;
            char *error_code_ptr = (char *)(&error_code);
            error_code_file.read(error_code_ptr, 4);
            // error_code = (OpType_Resize << 16) + 0x123;
            if (error_code == 0)
            {
                DEBUG() << "error_code: " << error_code;
            }
            else if (error_code >> 31)
            {
                WARN() << "AIPU teminated with error_code: " << std::hex << error_code << std::dec;
            }
            else
            {
                OpType type = static_cast<OpType>((error_code & ~(1 << 31)) >> 16);
                uint16_t code = static_cast<uint16_t>(error_code & 0xFFFF);
                std::string msg = error_code_map.count(code) > 0 ? error_code_map[code] : "RUNTIME_ERROR";
                WARN() << "AIPU teminated with error_code: 0x" << std::hex << error_code << std::dec
                       << " (OpType_" << optype_to_string(type) << ", " << msg << ").";
            }
        }
    }
    error_code_file.close();
    return true;
}

static bool handle_multi_batch(std::vector<TensorPtr> &output_tensors, uint32_t batch, std::string output_file)
{
    for (uint32_t b = 0; b < batch; ++b)
    {
        std::string name = "";
        if (batch > 1)
        {
            name = "b" + std::to_string(b) + ".";
        }
        for (uint32_t i = 0; i < output_tensors.size(); ++i)
        {
            auto ofile = output_file;
            if (i != 0)
            {
                ofile += std::to_string(i);
            }
            auto out = output_tensors[i];
            out->set_data(new char[out->mem_size()], out->mem_size());
            std::ifstream ifs;
            ifs.open(name + ofile, std::ios::in | std::ios::binary);
            ifs.read(out->data<char>(), out->mem_size());
            ifs.close();
        }
    }
    return true;
}

static void handle_dynamic_shape_config(GraphPtr graph)
{
    if (!BuilderConfig::get<bool>(Constants::enable_ds))
    {
        return;
    }

    // set default ds_output_shape if some input nodes don't have
    bool has_ds_output_shape = false;
    auto in_tensors = graph->get_input_tensors();
    for (auto t : in_tensors)
    {
        auto n = t->node();
        if (n->params().count("ds_output_shape"))
        {
            has_ds_output_shape = true;
            break;
        }
    }
    if (has_ds_output_shape)
    {
        for (auto t : in_tensors)
        {
            auto n = t->node();
            if (!n->params().count("ds_output_shape"))
            {
                std::vector<Expr> s = t->shape().symbol();
                n->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{s};
            }
        }
    }

    // check graph input shape
    for (auto &&opt : {Constants::input_shape,
                       Constants::max_input_shape,
                       Constants::default_input_shape})
    {
        if (!BuilderConfig::exist(opt))
        {
            continue;
        }
        bool check_result = true;
        auto in_shapes = parse_input_shape(graph, opt);
        if (in_tensors.size() != in_shapes.size())
        {
            WARN() << "Failed to check option --" << opt << ": "
                   << "number of input tensors is not equal to the given input shape!";
            check_result = false;
        }
        if (check_result)
        {
            for (uint32_t i = 0; i < in_tensors.size(); ++i)
            {
                auto shape = get_base_shape(in_tensors[i]->shape());
                if (shape.dim() != static_cast<int>(in_shapes[i].size()))
                {
                    WARN() << "Failed to check option --" << opt << ": "
                           << "dimension of input " << i << " is not equal to the given input shape!";
                    check_result = false;
                    break;
                }
            }
        }
        if (check_result)
        {
            std::vector<int> max_shape;
            for (uint32_t i = 0; i < in_shapes.size(); ++i)
            {
                if (opt == Constants::max_input_shape)
                {
                    max_shape = in_shapes[i];
                }
                else
                {
                    max_shape = get_base_shape(in_tensors[i]->shape());
                }
                for (uint32_t j = 0; j < in_shapes[i].size(); ++j)
                {
                    if (in_shapes[i][j] < 0 || in_shapes[i][j] > max_shape[j])
                    {
                        WARN() << "Failed to check option --" << opt << ": "
                               << "the given input shape exceeds the valid range of values!";
                        check_result = false;
                        break;
                    }
                }
            }
        }
        if (!check_result)
        {
            BuilderConfig::erase(opt);
        }
    }

    if (!BuilderConfig::exist(Constants::default_input_shape))
    {
        std::vector<std::vector<int>> max_shape;
        if (BuilderConfig::exist(Constants::max_input_shape))
        {
            max_shape = parse_input_shape(graph, Constants::max_input_shape);
        }
        else
        {
            for (auto t : in_tensors)
            {
                max_shape.push_back(get_base_shape(t->shape()));
            }
        }
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < max_shape.size(); ++i)
        {
            oss << TensorShape(max_shape[i]);
            if (i != max_shape.size() - 1)
            {
                oss << ",";
            }
        }
        oss << "]";
        BuilderConfig::set(Constants::default_input_shape, oss.str());
    }

    // all input dimensions are dynamic by default
    // otherwise parse its ds_output_shape
    if (!BuilderConfig::exist(Constants::dynamic_input_dim))
    {
        std::set<std::string> dims;
        uint32_t index = 0;
        for (auto t : in_tensors)
        {
            auto n = t->node();
            auto shape = get_base_shape(t->shape());
            if (n->params().count("ds_output_shape"))
            {
                std::vector<std::vector<Expr>> ds_output_shape = n->params()["ds_output_shape"];
                if (ds_output_shape.size() == 1 && (int)ds_output_shape[0].size() == shape.dim())
                {
                    for (auto expr : ds_output_shape[0])
                    {
                        if (expr->op() == ExprImpl::Op::Constant)
                        {
                            dims.insert(expr->symbol().name());
                        }
                    }
                }
                else
                {
                    FATAL() << "Unknown ds_output_shape param from Input: " << n->name();
                }
                index += shape.dim();
            }
            else
            {
                for (int i = 0; i < shape.dim(); ++i)
                {
                    dims.insert("d" + std::to_string(index++));
                }
            }
        }
        BuilderConfig::set(Constants::dynamic_input_dim, dims);
    }
}

static bool is_empty_graph(GraphPtr g)
{
    size_t num_real_nodes = 0;
    g->walk_on_node([&num_real_nodes](NodePtr n) {
        if (SubGraphNode::is_subgraph_node(n) &&
            !n->attrs().count(Constants::plugin_attrs_name))
        {
            return;
        }
        if (n->attrs().count(Constants::ro_size) &&
            n->get_attr<uint32_t>(Constants::ro_size))
        {
            num_real_nodes++;
        } });
    return num_real_nodes == 0;
}

static bool eval_output_shape(GraphPtr g,
                              const std::vector<std::vector<int>> &in_shapes,
                              std::vector<int> &res)
{
    bool ret = true;
    std::vector<int> in_dims;
    if (in_shapes.size())
    {
        for (auto &s : in_shapes)
        {
            for (auto &d : s)
            {
                in_dims.push_back(d);
            }
        }
    }
    else
    {
        for (auto inp : g->get_input_tensors())
        {
            for (int d : inp->shape())
            {
                in_dims.push_back(d);
            }
        }
    }
    res.clear();
    auto &symbols = g->symbols();
    for (auto out : g->get_output_tensors())
    {
        for (int i = 0; i < out->shape().dim(); ++i)
        {
            auto expr = out->shape().symbol()[i];
            std::map<Symbol, Symbol::Value> args;
            for (auto &&var : expr->variables())
            {
                if (symbols.count(var.name()))
                {
                    auto &sym = symbols.at(var.name());
                    if (sym.type == 0 && sym.dims.size())
                    {
                        auto dim = sym.dims[0];
                        if (dim < (int)in_dims.size())
                        {
                            args[var] = in_dims[dim];
                        }
                    }
                }
            }
            if (!expr->eval(args))
            {
                WARN() << "Eval expr \"" << expr->to_string() << "\" failed!";
                ret = false;
            }
            res.push_back(expr->value());
        }
    }
    return ret;
}

static void serialize_graph(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map)
{
    BuilderConfig::set("dump_debug_IR", true);
    BuilderConfig::set(Constants::show_location, true);
    bool serialize_scale_zp = g->get_attr("serialize_scale_zp");
    g->set_attr("serialize_scale_zp", true);
    g->serialize(cast_any<std::string>(arg_map["dumpIR"]) + ".def",
                 cast_any<std::string>(arg_map["dumpIR"]) + ".bin");
    g->set_attr("serialize_scale_zp", serialize_scale_zp);
}

GraphBuilder *run_gb(GraphPtr g, std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map)
{
    uint64_t t0, t1;
    t0 = utils::get_time_in_us();
    GraphBuilder *builder;
    std::vector<TensorPtr> outputs;
    std::vector<int32_t> dump_ids;
    if (arg_map.count("weight"))
    {
        auto weight_name = cast_any<std::string>(arg_map["weight"]);
        if (!utils::file_exists(weight_name))
        {
            ERROR() << "Cannot Open file:" << weight_name;
            return nullptr;
        }
        g->load_weight(weight_name);
    }
    if (arg_map.count("dump_tensors"))
    {
        auto dump_tensors = utils::split_string(cast_any<std::string>(arg_map["dump_tensors"]), ',');
        for (auto dump : dump_tensors)
        {
            int32_t id = Node::parse_param_value(dump);
            dump_ids.push_back(id);
        }
    }

    // auto cfg = BuilderConfig::get_instance();
    // INFO() << *cfg;

    if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
    {
        builder = new GraphELFBuilder(g);
    }
    else
    {
        builder = new GraphBuilder(g);
    }
    builder->build_aipu(input_tensors, outputs, dump_ids);
    auto optimized_g = builder->graph();
    t1 = utils::get_time_in_us();
    INFO() << " -  compile time: " << (t1 - t0) / 1000 / 1000. << " s";
    return builder;
}

bool run_simulator(std::string dump_file, std::map<std::string, aipubt::Any> &arg_map)
{
    std::string simulator;
    if (arg_map.count("simulator"))
    {
        simulator = cast_any<std::string>(arg_map["simulator"]);
    }
    simulator = get_simulator_path(simulator);
    bool ret = run_simulator(simulator + " " + dump_file);

    if (arg_map.count("save-simulator"))
    {
        get_simulator_path(simulator);
    }

    return ret;
}

static void get_elf_bss_info(const AIPUELFPtr elf,
                             std::vector<Tensor::Dtype> &in_dtypes,
                             std::vector<uint32_t> &out_sizes,
                             std::vector<uint32_t> &out_dims,
                             std::vector<Tensor::Dtype> &out_dtypes)
{
    for (auto &bss : elf->get_fm_list().fmdesc)
    {
        for (size_t i = 0; i < bss.num_buffer_descriptor; ++i)
        {
            auto &desc = bss.buffer_descriptor[i];
            for (size_t j = 0; j < desc.sub_section_cnt; ++j)
            {
                auto &subdesc = desc.sub_sections[j];
                if (subdesc.type == BssSubSectionDescriptor::input && subdesc.offset_in_ro_cnt != 0)
                {
                    if (in_dtypes.size() <= subdesc.id)
                    {
                        in_dtypes.resize(subdesc.id + 1);
                    }
                    in_dtypes[subdesc.id] = static_cast<Tensor::Dtype>(subdesc.dtype);
                }
                if (subdesc.type == BssSubSectionDescriptor::output && subdesc.offset_in_ro_cnt != 0)
                {
                    if (out_sizes.size() <= subdesc.id)
                    {
                        out_sizes.resize(subdesc.id + 1);
                        out_dtypes.resize(subdesc.id + 1);
                    }
                    out_sizes[subdesc.id] = subdesc.size;
                    out_dtypes[subdesc.id] = static_cast<Tensor::Dtype>(subdesc.dtype);
                }
                if (subdesc.type == BssSubSectionDescriptor::output_tensor_shape && subdesc.offset_in_ro_cnt != 0)
                {
                    if (out_dims.size() <= subdesc.id)
                    {
                        out_dims.resize(subdesc.id + 1);
                    }
                    out_dims[subdesc.id] = subdesc.size / sizeof(uint32_t);
                }
            }
        }
    }
}

static void update_simulator_cfg(const std::string &dump_prefix,
                                 const std::string &output_file,
                                 const std::vector<uint32_t> &output_sizes)
{
    auto dump_cfg = dump_prefix + ".cfg";
    std::ifstream ifs(dump_cfg, std::ios::in);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    ifs.close();
    utils::remove_file(dump_cfg);
    auto cfg_lines = utils::split_string(buffer.str(), "\n");
    for (uint32_t i = 0; i < output_sizes.size(); ++i)
    {
        auto ofile = output_file;
        if (i != 0)
        {
            ofile += std::to_string(i);
        }
        for (uint32_t j = 0; j < cfg_lines.size(); ++j)
        {
            auto line = cfg_lines[j];
            if (!utils::startswith(line, "FILE") || line.find_first_of("=") == std::string::npos)
            {
                continue;
            }
            auto splits = utils::split_string(line, "=");
            if (splits.size() == 2 && splits[1] == ofile)
            {
                auto dump_id = utils::replace(splits[0], "FILE", "");
                cfg_lines[j + 2] = "SIZE" + dump_id + "=" + utils::int_to_hex(output_sizes[i]);
                break;
            }
        }
    }
    std::ofstream ofs(dump_cfg, std::ios::out);
    auto cfg_str = utils::join_string("\n", cfg_lines);
    ofs.write(cfg_str.c_str(), cfg_str.size());
    ofs.close();
}

bool cpsdumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map)
{
    aipubt::rt::Launcher launcher;
    launcher.load_from_file(cast_any<std::string>(arg_map["aipubin"]));

    auto inputs = launcher.get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto &input = inputs[i];
        input.data = input_tensors[i]->data<char>();
        input.length = input_tensors[i]->mem_size();
        input.name = input_tensors[i]->name();
        input.shape = input_tensors[i]->shape();
    }
    launcher.set_inputs(inputs);
    INFO() << "fun with btrt:";
    uint64_t t0, t1;
    t0 = utils::get_time_in_us();
    launcher.run();
    t1 = utils::get_time_in_us();
    INFO() << "simulator done! total time: " << (t1 - t0) / 1000 / 1000. << " s";
    launcher.dump_outputs();

    return true;
}

bool elfdumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map, GraphELFBuilder *builder)
{
    for (auto arg : arg_map)
    {
        std::string str;

        if (arg.second.get_type_id() != GET_ANY_TYPEID<std::string>())
        {
            str = std::to_string(cast_any<int>(arg.second));
        }
        else
        {
            str = cast_any<string>(arg.second);
        }
        DEBUG() << arg.first << " " << str;
    }

    if (builder == nullptr)
    {
        builder = new GraphELFBuilder(cast_any<std::string>(arg_map["aipubin"]));
    }
    int core_num = arg_map.count(Constants::multi_core) ? cast_any<int>(arg_map[Constants::multi_core]) : 1;
    string target = Target::get_name_from_arch_number(builder->get_aipuelf()->compiler_msg().arch);
    if (core_num > 1)
    {
        target = target + "_MP" + std::to_string(core_num);
    }
    INFO() << "target: " << target;
    arg_map["target"] = target;

    auto ds_flag = builder->get_aipuelf()->compiler_msg().flag >> 6 & 0x1;

    if (ds_flag)
    {
        arg_map["enable_ds"] = true;
    }

    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return false;
    }

    std::vector<uint32_t> out_sizes, out_dims;
    std::vector<Tensor::Dtype> in_dtypes, out_dtypes;
    if (ds_flag)
    {
        auto str = BuilderConfig::get<std::string>(Constants::input_shape);
        if (str.find_first_of('=') != std::string::npos)
        {
            FATAL() << "assignment to symbols is not supported";
        }
        std::vector<std::vector<int>> in_shapes;
        for (auto &s : utils::split_string(str, "],", 1))
        {
            in_shapes.push_back(get_base_shape(TensorShape(s)));
        }
        if (in_shapes.size() != input_tensors.size())
        {
            ERROR() << "Number of input tensors " << input_tensors.size() << " is not equal to input shapes " << in_shapes.size();
            return false;
        }
        get_elf_bss_info(builder->get_aipuelf(), in_dtypes, out_sizes, out_dims, out_dtypes);
        for (uint32_t i = 0; i < input_tensors.size(); ++i)
        {
            input_tensors[i]->shape() = TensorShape(in_shapes[i]);
            input_tensors[i]->set_type(in_dtypes[i]);
        }
    }

    // auto cfg = BuilderConfig::get_instance();
    // INFO() << *cfg;

    Dumper *dumper;
    std::string dump_prefix = "temp";
    std::string output_file = cast_any<std::string>(arg_map["output"]);
    dumper = new DumperX(aipubt::BuilderConfig::get_instance(), builder->get_aipuelf(), Target::get_the_target());
    auto ret_dump = dumper->gen_dumper_files(input_tensors, dump_prefix, output_file);

    bool ret_sim = false;
    if (ret_dump && arg_map.count("simulator"))
    {
        ret_sim = run_simulator(dump_prefix + ".cfg", arg_map);
    }

    if (ret_sim && arg_map.count("input_shape"))
    {
        std::ifstream ifs;
        ifs.open("output_shape.bin", std::ios::in | std::ios::binary);
        std::vector<uint32_t> bin_sizes;
        bool update_cfg = false;
        for (uint32_t i = 0; i < out_sizes.size(); ++i)
        {
            TensorShape shape(std::vector<int>(out_dims[i], 0));
            uint32_t dim;
            for (uint32_t j = 0; j < out_dims[i]; ++j)
            {
                ifs.read((char *)(&dim), sizeof(uint32_t));
                shape[j] = dim;
            }

            auto ofile = output_file;
            if (i != 0)
            {
                ofile += std::to_string(i);
            }
            uint32_t bin_size = Tensor::dtype_size(out_dtypes[i]) * shape.size();
            if (bin_size < out_sizes[i])
            {
                auto data = new char[bin_size];
                std::ifstream ifs;
                ifs.open(ofile, std::ios::in | std::ios::binary);
                ifs.read(data, bin_size);
                ifs.close();

                utils::remove_file(ofile);
                std::ofstream ofs;
                ofs.open(ofile, std::ios::out | std::ios::binary);
                ofs.write(data, bin_size);
                ofs.close();
                delete[] data;
                update_cfg = true;
            }
            bin_sizes.push_back(bin_size);
        }
        ifs.close();
        if (update_cfg)
        {
            update_simulator_cfg(dump_prefix, output_file, bin_sizes);
        }
    }

    delete dumper;
    return ret_sim;
}

static bool show_aipubin_info(AIPUBin &aipubin)
{
    string s(aipubin.header.ident);
    if (s == "AIPUBIN_FILE_MAGIC_NUM")
    {
        std::cout << "AIPUBIN"
                  << "\n";
        std::cout << "Bin file version 0x" << std::hex << aipubin.header.version << "\n";
        std::cout << "Built for architecture: 0x" << std::hex << (aipubin.header.arch >> 20)
                  << ", ISA type: " << ((aipubin.header.arch >> 16) & 0xF) + 1
                  << ", target: " << std::dec << (aipubin.header.arch & 0xffff)
                  << "\n";
        std::cout << "Compiler version: "
                  << (aipubin.header.build_version & 0xff)
                  << "."
                  << ((aipubin.header.build_version >> 8) & 0xff)
                  << "."
                  << ((aipubin.header.build_version >> 16) & 0xffff)
                  << "\n";
        if (aipubin.header.flag & 0b11)
        {
            std::cout << "ASID enabled.\n";
        }
        else
        {
            std::cout << "ASID disabled.\n";
        }
    }
    return true;
}

bool aipudumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map)
{
    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return 0;
    }

    Dumper *dumper;
    std::string dump_prefix = "temp";
    std::string output_file = cast_any<std::string>(arg_map["output"]);

    std::map<BssSubSecType, std::vector<TensorPtr>> marked_tensors;
    auto aipubin = GraphBuilder::load(cast_any<std::string>(arg_map["aipubin"]));
    show_aipubin_info(aipubin);
    dumper = new DumperX1(aipubt::BuilderConfig::get_instance(), &aipubin, marked_tensors);

    auto ret_dump = dumper->gen_dumper_files(input_tensors, dump_prefix, output_file);

    // simulator
    bool ret_sim = false;
    if (ret_dump && arg_map.count("simulator"))
    {
        ret_sim = run_simulator(dump_prefix + ".cfg", arg_map);
    }

    if (arg_map.count("DEBUG"))
    {
        handle_op_counter("op_counter.bin");
    }
    handle_error_code("error_code.bin");

    delete dumper;
    return ret_sim;
}

// before_build --> build --> dump --> after_build --> simulator
GraphPtr aipurun(GraphPtr g,
                 std::vector<TensorPtr> &input_tensors,
                 std::map<std::string, aipubt::Any> &arg_map)
{
    // before_build
    auto error_cnt = GCompilerLogger::get_error_count();
    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return nullptr;
    }
    handle_dynamic_shape_config(g);

    std::vector<int> dump_shape_dims;
    std::vector<std::vector<int>> input_shapes;
    if (BuilderConfig::exist(Constants::input_shape))
    {
        input_shapes = parse_input_shape(g, Constants::input_shape);
    }

    GraphPtr optimized_g;
    if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv7)
    {
        if (arg_map.count("weight"))
        {
            auto weight_name = cast_any<std::string>(arg_map["weight"]);
            if (!utils::file_exists(weight_name))
            {
                ERROR() << "Cannot Open file:" << weight_name;
                return nullptr;
            }
            g->load_weight(weight_name);
        }
        IBuilder *builder = new MqBuilder(g);

        builder->build();
        optimized_g = builder->get_graph();

        if (GCompilerLogger::get_error_count() > error_cnt)
        {
            INFO() << "There are some errors in compiling time, Simulator will not run.";
            return nullptr;
        }

        auto elf = builder->get_elf();
        std::stringstream ss;
        elf->save(ss);

        aipubt::rt::Launcher launcher;
        auto str = ss.str();
        launcher.load_from_mem(str.data(), str.size());

        auto slot_addr = launcher.get_slot_addresses();
        optimized_g->set_attr("addr_vec_list", slot_addr);

        actions_after_build(optimized_g, arg_map);

        auto inputs = launcher.get_inputs();
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto t = input_tensors[i];
            if (input_shapes.size())
            {
                t = std::make_shared<Tensor>(*input_tensors[i]);
                auto shape = get_base_shape(t->shape());
                t->shape() = TensorShape(input_shapes[i], shape.layout());
            }
            auto &input = inputs[i];
            input.data = t->data<char>();
            input.length = t->mem_size();
            input.name = t->name();
            input.shape = t->shape();
        }
        launcher.set_inputs(inputs);

        if (arg_map.count("dump"))
        {
            launcher.dump_temp(".");
        }

        if (!arg_map.count("skip_simulator"))
        {
            INFO() << "run with btrt:";
            uint64_t t0, t1;
            t0 = utils::get_time_in_us();
            launcher.run();
            t1 = utils::get_time_in_us();
            INFO() << "simulator done! total time: " << (t1 - t0) / 1000 / 1000. << " s";

            launcher.dump_outputs(cast_any<std::string>(arg_map["output"]));

            if (BuilderConfig::get<bool>("dump_all_tensors"))
            {
                std::set<TensorPtr> dump_tensors;
                std::map<TensorPtr, int> layer_id_map;
                auto tensor_index_map = ((MqBuilder *)builder)->get_tensor_index_map();
                for (auto s : optimized_g->nodes())
                {
                    auto sgn = SubGraphNode::from_node(s);
                    if (sgn)
                    {
                        auto sg = sgn->subgraph();
                        for (auto n : sg->nodes())
                        {
                            if (n->type() == OpType_Input || n->type() == OpType_SubGraphInput)
                            {
                                continue;
                            }
                            for (auto out : n->outputs())
                            {
                                auto tensor_p = std::dynamic_pointer_cast<ProxyTensor>(out);
                                if (tensor_p)
                                {
                                    out = tensor_p->proxied_tensor(true);
                                }
                                dump_tensors.insert(out);
                                layer_id_map[out] = n->get_attr("layer_id");
                            }
                        }
                    }
                }
                std::map<int, std::pair<std::string, int>> dump_map;
                for (auto t : dump_tensors)
                {
                    if (!tensor_index_map.count(t))
                    {
                        ERROR();
                    }
                    auto index = tensor_index_map[t];
                    auto id = layer_id_map[t];
                    auto name = "dumptensor_" + to_string(id) + "_" + utils::verify_name(t->name()) + ".bin";
                    auto size = t->mem_size();
                    dump_map[index] = make_pair(name, size);
                }
                launcher.dump_tensors(dump_map);
            }
        }

        delete builder;
    }
    else
    {
        // build
        auto builder = run_gb(g, input_tensors, arg_map);
        optimized_g = builder->graph();
        if (GCompilerLogger::get_error_count() > error_cnt)
        {
            INFO() << "There are some errors in compiling time, Simulator would not run.";
            return nullptr;
        }

        // dump
        std::string global_cwd = BuilderConfig::get<std::string>(Constants::global_cwd);
        std::string dump_prefix = global_cwd + "/temp_" + utils::get_uuid();

        Dumper *dumper;
        if (arg_map.count("dump"))
        {
            dump_prefix = "temp";
        }
        dump_prefix = utils::get_abspath(dump_prefix);
        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            std::string aipubin = dump_prefix + ".aipuelf";
            builder->write(aipubin);
            dumper = new DumperX(BuilderConfig::get_instance(), ((GraphELFBuilder *)builder)->get_aipuelf(), Target::get_the_target(), builder->graph(), builder->get_ro());
            optimized_g->attrs()["_elf"] = ((GraphELFBuilder *)builder)->get_aipuelf(); // attach elf into graph for later use
        }
        else
        {
            dumper = new DumperX1(BuilderConfig::get_instance(), builder->get_aipubin(), builder->get_marked_tensors());
        }

        if (input_shapes.size())
        {
            for (uint32_t i = 0; i < input_tensors.size(); ++i)
            {
                auto tensor = std::make_shared<Tensor>(*input_tensors[i]);
                auto shape = get_base_shape(tensor->shape());
                tensor->shape() = TensorShape(input_shapes[i], shape.layout());
                input_tensors[i] = tensor;
            }
        }

        auto output_file = cast_any<std::string>(arg_map["output"]);
        if (BuilderConfig::exist(Constants::multi_batch))
        {
            auto batch = BuilderConfig::get<uint32_t>(Constants::multi_batch);
            for (auto t : input_tensors)
            {
                auto input_shape = t->shape();
                input_shape[0] = input_shape[0] * batch;
                t->shape() = TensorShape(input_shape, t->shape().layout());
            }
        }
        auto ret_dump = dumper->gen_dumper_files(input_tensors, dump_prefix, output_file);

        // after_build
        actions_after_build(optimized_g, arg_map);

        if (GCompilerLogger::get_error_count())
        {
            INFO() << "There are some errors in dumper, Simulator would not run.";
            return optimized_g;
        }

        // simulator
        if (ret_dump && !arg_map.count("skip_simulator"))
        {
            if (!run_simulator(dump_prefix + ".cfg", arg_map))
            {
                ERROR() << "run simulator failed!";
            }
            else
            {
                bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
                if (enable_ds)
                {
                    if (is_empty_graph(optimized_g))
                    {
                        eval_output_shape(optimized_g, input_shapes, dump_shape_dims);
                    }
                    else
                    {
                        auto ts_base = optimized_g->global_tensors()[Constants::tensor_shape_base];
                        uint32_t num_dims = ts_base->mem_size() / sizeof(uint32_t);
                        std::ifstream ifs;
                        ifs.open("output_shape.bin", std::ios::in | std::ios::binary);
                        for (uint32_t i = 0; i < num_dims; ++i)
                        {
                            uint32_t dim = 0;
                            ifs.read((char *)(&dim), sizeof(uint32_t));
                            dump_shape_dims.push_back(dim);
                        }
                        ifs.close();
                    }
                }

                bool update_cfg = false;
                std::vector<uint32_t> bin_sizes;
                uint32_t index = 0;
                auto out_tensors = builder->graph()->get_output_tensors();
                for (uint32_t i = 0; i < out_tensors.size(); ++i)
                {
                    auto ofile = output_file;
                    if (i != 0)
                    {
                        ofile += std::to_string(i);
                    }
                    auto out = out_tensors[i];
                    out->set_data(new char[out->mem_size()], out->mem_size());
                    std::ifstream ifs;
                    ifs.open(ofile, std::ios::in | std::ios::binary);
                    ifs.read(out->data<char>(), out->mem_size());
                    ifs.close();

                    if (enable_ds)
                    {
                        uint32_t shape_size = 1;
                        for (int j = 0; j < out->shape().dim(); ++j)
                        {
                            shape_size *= dump_shape_dims[index++];
                        }
                        auto bin_size = shape_size * Tensor::dtype_size(out->type());
                        if (bin_size < out->mem_size())
                        {
                            utils::remove_file(ofile);
                            std::ofstream ofs;
                            ofs.open(ofile, std::ios::out | std::ios::binary);
                            ofs.write(out->data<char>(), bin_size);
                            ofs.close();
                            update_cfg = true;
                        }
                        bin_sizes.push_back(bin_size);
                    }
                }
                if (update_cfg && arg_map.count("dump"))
                {
                    update_simulator_cfg(dump_prefix, output_file, bin_sizes);
                }
            }
        }

        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            if (arg_map.count("multi_batch"))
            {
                auto output_tensors = builder->graph()->get_output_tensors();
                handle_multi_batch(output_tensors, cast_any<uint32_t>(arg_map["multi_batch"]), output_file);
            }
        }
        else
        {
            if (arg_map.count("DEBUG"))
            {
                handle_op_counter("op_counter.bin");
            }
            handle_error_code("error_code.bin");
        }

        // serialize real shape if input shape specified
        if (!arg_map.count("skip_simulator") && arg_map.count("dumpIR") && input_shapes.size())
        {
            if (!infer_real_shape(optimized_g, dump_shape_dims, input_shapes))
            {
                ERROR() << "Infer real shapes of optimized graph failed!";
            }
            serialize_graph(optimized_g, arg_map);

            std::set<std::string> dims;
            if (BuilderConfig::exist(Constants::dynamic_input_dim))
            {
                dims = BuilderConfig::get<std::set<std::string>>(Constants::dynamic_input_dim);
            }
            auto new_g = infershape(g, dims, input_shapes, dump_shape_dims, true);
            std::string prefix = arg_map["dumpIR"];
            arg_map["dumpIR"] = prefix + "_raw";
            serialize_graph(new_g, arg_map);
            arg_map["dumpIR"] = prefix;
        }

        // clean temp files
        if (!arg_map.count("dump"))
        {
            INFO() << "remove global cwd: " << global_cwd;
            utils::remove_dir(global_cwd);
        }
        delete builder;
        delete dumper;
    }

    return optimized_g;
}

// before_build --> build --> after_build
GraphPtr aipugb(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map)
{
    // before_build
    auto error_cnt = GCompilerLogger::get_error_count();
    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return nullptr;
    }
    handle_dynamic_shape_config(g);

    GraphPtr optimized_g;
    if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv7)
    {
        if (arg_map.count("weight"))
        {
            auto weight_name = cast_any<std::string>(arg_map["weight"]);
            if (!utils::file_exists(weight_name))
            {
                ERROR() << "Cannot Open file:" << weight_name;
                return nullptr;
            }
            g->load_weight(weight_name);
        }
        IBuilder *builder = new MqBuilder(g);

        builder->build();
        optimized_g = builder->get_graph();

        actions_after_build(optimized_g, arg_map);

        auto elf = builder->get_elf();
        auto bin_name = cast_any<std::string>(arg_map["output"]);
        elf->save(bin_name);

        std::ofstream ofile(bin_name + ".h");
        ofile << builder->get_header();
        ofile.close();

        delete builder;
    }
    else
    {
        // build
        std::vector<TensorPtr> dummy_inputs = std::vector<TensorPtr>();
        auto builder = run_gb(g, dummy_inputs, arg_map);

        optimized_g = builder->graph();
        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            optimized_g->attrs()["_elf"] = ((GraphELFBuilder *)builder)->get_aipuelf(); // attach elf into graph for later use
        }

        // after_build
        actions_after_build(optimized_g, arg_map);

        if (GCompilerLogger::get_error_count() > error_cnt)
        {
            if (arg_map.count("dump"))
            {
                builder->write(cast_any<std::string>(arg_map["output"]));
            }
            INFO() << "There are some errors in compiling time.";
            return nullptr;
        }

        builder->write(cast_any<std::string>(arg_map["output"]));
        // clean temp files
        if (!arg_map.count("dump"))
        {
            std::string global_cwd = BuilderConfig::get<std::string>(Constants::global_cwd);
            INFO() << "remove global cwd: " << global_cwd;
            utils::remove_dir(global_cwd);
        }

        delete builder;
    }
    return optimized_g;
}

GraphPtr aipuestimator(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map)
{
    if (arg_map.count("svg"))
    {
        arg_map["show_info"] = "perf";
    }

    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return nullptr;
    }

    PerfEstimator perf;
    perf(g);

    actions_after_build(g, arg_map);

    if (GCompilerLogger::get_error_count())
    {
        return nullptr;
    }

    auto global_cwd = BuilderConfig::get<std::string>(Constants::global_cwd);
    INFO() << "remove global cwd: " << global_cwd;
    utils::remove_dir(global_cwd);

    return g;
}

} // namespace aipubt
