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
 * @Author: Kyle Huang
 * @Date: 2026-04-24 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:33
 */
#include "builder/code_generator.h"
#include "builder/ar/ar_reader.h"
#include "builder/assembler.h"
#include "builder/ds_code_generator_cx.h"
#include "builder/linker/linker_cps.h"
#include "builder/linker/toolchain.h"
#include "builder/text_generator.h"
#include "core/config.h"
#include "core/constants.h"
#include "utils/utils.h"

namespace aipubt
{

void CodeGenerator::set_graph(GraphPtr g)
{
    m_graph = g;
}

void CodeGenerator::set_device_lib(const std::map<std::string, std::string> &kernel_lib)
{
    m_device_lib = kernel_lib;
}

void CodeGenerator::set_host_lib(const std::map<std::string, std::string> &kernel_lib)
{
    m_host_lib = kernel_lib;
}

void CodeGenerator::gen_device_code(CpsElfPtr elf)
{
    LinkerCps linker;
    auto workdir = linker.workdir();

    std::set<std::string> libs;

    for (auto &k : m_device_lib)
    {
        DEBUG() << "kernel: " << k.first << " lib: " << k.second;
        bool has_lib = false;
        for (auto lib : libs)
        {
            if (lib == k.second)
            {
                has_lib = true;
                break;
            }

            ARFileReader arf(lib);
            if (arf.is_ar_file() && arf.has_file(k.second))
            {
                has_lib = true;
                break;
            }
        }
        if (has_lib)
        {
            continue;
        }

        if (utils::file_exists(k.second))
        {
            libs.insert(k.second);
        }
        else // find from .a files
        {
            bool found = false;
            for (auto afile : get_all_afiles())
            {
                if (afile.has_file(k.second))
                {
                    libs.insert(afile.name());
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                ERROR() << "cannot find " << k.second;
            }
        }
    }

    std::vector<std::string> objs(libs.begin(), libs.end());

    auto ds_code = get_ds_device_code();
    if (ds_code.size())
    {
        auto c_file = workdir + "/" + "ds_impl.cps";
        auto o_file = workdir + "/" + "ds_impl.o";
        std::ofstream objfile(c_file);
        objfile << ds_code;
        objfile.close();
        if (cpscc(c_file, o_file) != 0)
        {
            FATAL() << "An error occurred while compiling the dynamic shape code!";
        }
        objs.push_back(o_file);
    }

    linker.set_objects(objs);
    std::stringstream oss;
    linker.link(oss);
    elf->load(oss.str());
}

void CodeGenerator::gen_host_code(HybridELFPtr elf)
{
    auto workdir = BuilderConfig::get<std::string>(Constants::global_cwd) + "/ld";
    if (!utils::path_exists(workdir))
    {
        utils::mkdir(workdir);
    }

    gen_host_funcs();

    std::set<std::string> libs;
    auto lib_paths = get_all_lib_paths();
    auto devkit_path = get_devkit_path();
    if (devkit_path.size())
    {
        lib_paths.push_back(devkit_path + "/lib");
    }
    for (auto &kv : m_host_lib)
    {
        DEBUG() << "kernel: " << kv.first << " lib: " << kv.second;
        if (utils::file_exists(kv.second))
        {
            libs.insert(utils::get_abspath(kv.second));
        }
        else // find from lib paths
        {
            bool found = false;
            for (auto path : lib_paths)
            {
                auto full_name = path + "/" + kv.second;
                if (utils::file_exists(full_name))
                {
                    libs.insert(utils::get_abspath(full_name));
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                ERROR() << "cannot find " << kv.second;
            }
        }
    }
    std::vector<std::string> objs(libs.begin(), libs.end());

    auto host_code = get_cpshost_code();
    if (host_code.size())
    {
        auto c_file = workdir + "/" + "cpshost.cpp";
        auto o_file = workdir + "/" + "cpshost.o";
        std::ofstream objfile(c_file);
        objfile << host_code;
        objfile.close();
        if (cpshostcc(c_file, objs, o_file) != 0)
        {
            FATAL() << "An error occurred while compiling the host code!";
        }
        elf->load(o_file);
        elf->set_task_infos(m_host_tasks);
    }
}

static void build_host_task_args(const BuilderParams &params, std::vector<HostTaskArg> &res)
{
    res.clear();
    for (auto &p : params)
    {
        HostTaskArg arg;
        arg.type = 0;
        arg.size = 0;
        if (p.surface)
        {
            ERROR() << "unsupported surface parameter!";
        }
        else if (p.tensor)
        {
            arg.type = 1;
            arg.size = p.tensor->mem_size();
        }
        else if (p.const_tensor)
        {
            arg.type = 2;
            arg.size = p.const_tensor->mem_size();
        }
        else if (p.param_list)
        {
            arg.type = 3;
            arg.size = p.param_list->mem_size();
            build_host_task_args(p.param_list->params(), arg.args);
        }
        res.emplace_back(arg);
    }
}

void CodeGenerator::gen_host_funcs()
{
    auto param_root = m_graph->global_tensors()["param"];
    auto param_root_object = param_root->memory_offset()->get_base_object();

    for (auto s : m_graph->nodes())
    {
        auto sgn = SubGraphNode::from_node(s);
        if (sgn)
        {
            auto sg = sgn->subgraph();
            for (auto n : sg->nodes())
            {
                if (!n->attrs().count(Constants::plugin_impl))
                {
                    continue;
                }

                if (!n->get_attr<bool>(Constants::is_host_impl))
                {
                    continue;
                }

                auto kernel_infos = n->get_attr<KernelInfos>(Constants::kernel_infos);
                auto param_bases = n->get_attr<std::vector<PlaceholderPtr>>(Constants::param_bases);
                auto set_id = n->get_attr<std::vector<uint32_t>>(Constants::barrier_set_id);
                auto set_data = n->get_attr<std::vector<uint32_t>>(Constants::barrier_set_data);
                auto wait_id = n->get_attr<std::vector<uint32_t>>(Constants::barrier_wait_id);
                auto wait_ref = n->get_attr<std::vector<uint32_t>>(Constants::barrier_wait_ref);

                for (size_t i = 0; i < kernel_infos.size(); ++i)
                {
                    auto &kernel_info = kernel_infos[i];
                    auto param_base = param_bases[i];
                    auto param_base_object = param_base->memory_offset();
                    auto param_offset = param_base_object->offset() - param_root_object->offset();
                    param_offset += 8 * 4; // real param

                    HostTaskInfo task_info;
                    task_info.lib_name = m_host_lib.at(kernel_info.kernel_name);
                    task_info.task_name = "aipubt_" + utils::verify_name(n->name()) + "_" + std::to_string(i);
                    task_info.kernel_name = kernel_info.kernel_name;
                    task_info.param_offset = param_offset;
                    task_info.set_id = set_id[i];
                    task_info.set_data = set_data[i];
                    task_info.wait_id = wait_id[i];
                    task_info.wait_ref = wait_ref[i];
                    build_host_task_args(kernel_info.args, task_info.kernel_args);

                    constexpr size_t max_arg_num = 64;
                    if (kernel_info.args.size() > max_arg_num)
                    {
                        ERROR() << "host plugin supports up to " << max_arg_num << " arguments\n"
                                << "node: " << n->name();
                    }
                    m_host_tasks.tasks.emplace_back(task_info);
                }
            }
        }
    }

    auto &oss = m_host_code;
    oss.clear();

    for (auto &info : m_host_tasks.tasks)
    {
        gen_host_task_func(oss, info);
    }

    auto tab = std::string(4, ' ');
    auto launcher = "launcher";
    auto ctrl = "ctrl";
    auto tids = "tids";
    auto args = "args";
    auto events = "events";
    auto param_base = "param_base";
    oss << "void aipu_create_threads(aipubt::rt::ILauncher *" << launcher
        << ", sync_ctrl_t *" << ctrl
        << ", std::vector<pthread_t> &" << tids
        << ", std::vector<thread_arg_t> &" << args
        << ", std::vector<aipubt::rt::IEvent *> &" << events
        << ", uint64_t " << param_base << ")\n{\n";

    auto num_tasks = m_host_tasks.tasks.size();
    if (num_tasks > 0)
    {
        oss << tab << ctrl << "->ready.resize(" << num_tasks << ", false);\n";
        for (size_t i = 0; i < num_tasks; ++i)
        {
            auto &info = m_host_tasks.tasks[i];

            auto wait_attr = "wait_attr" + std::to_string(i);
            auto set_attr = "set_attr" + std::to_string(i);
            auto wait_event = "wait_event" + std::to_string(i);
            auto set_event = "set_event" + std::to_string(i);
            oss << tab << "aipubt::rt::IEvent::attr_t " << wait_attr << ", " << set_attr << ";\n"
                << tab << wait_attr << ".data = " << info.set_data << ";\n"
                << tab << wait_attr << ".mask = 0xf;\n"
                << tab << wait_attr << ".task_id = " << info.set_id << ";\n"
                << tab << set_attr << ".data = " << info.wait_ref << ";\n"
                << tab << set_attr << ".mask = 0xf;\n"
                << tab << set_attr << ".task_id = " << info.wait_id << ";\n"
                << tab << "auto " << wait_event << " = " << launcher << "->make_event(" << wait_attr << ").get();\n"
                << tab << events << ".push_back(" << wait_event << ");\n"
                << tab << "auto " << set_event << " = " << launcher << "->make_event(" << set_attr << ").get();\n"
                << tab << events << ".push_back(" << set_event << ");\n";

            auto arg = "arg" + std::to_string(i);
            auto tid = "tid" + std::to_string(i);
            oss << tab << args << ".push_back(thread_arg_t());\n"
                << tab << "auto " << arg << " = &" << args << "[" << args << ".size() - 1];\n"
                << tab << arg << "->tid = " << i << ";\n"
                << tab << arg << "->param_base = " << param_base << ";\n"
                << tab << arg << "->wait_event = " << wait_event << ";\n"
                << tab << arg << "->set_event = " << set_event << ";\n"
                << tab << arg << "->wait_attr = " << wait_attr << ";\n"
                << tab << arg << "->set_attr = " << set_attr << ";\n"
                << tab << arg << "->task_func = " << info.task_name << ";\n"
                << tab << arg << "->ctrl = " << ctrl << ";\n"
                << tab << tids << ".push_back(pthread_t());\n"
                << tab << "auto " << tid << " = &" << tids << "[" << tids << ".size() - 1];\n"
                << tab << "pthread_create(" << tid << ", NULL, task_thread_func, (void *)" << arg << ");\n";
        }
    }
    oss << "}\n\n";
}

struct BufferInfo
{
    std::string host_addr;
    std::string device_addr;
    size_t size = 0;
    bool write_back = false;
};

static size_t get_var_id()
{
    static size_t var_id = 0;
    return var_id++;
}

static void process_param_list(std::stringstream &oss,
                               const std::string &base,
                               const std::vector<HostTaskArg> &args,
                               std::vector<BufferInfo> &bufs)
{
    auto tab = std::string(4, ' ');
    size_t offset = 0;
    for (auto &karg : args)
    {
        if (karg.type == 1 || karg.type == 2 || karg.type == 3)
        {
            auto var_id = get_var_id();
            auto arg = "arg" + std::to_string(var_id);
            auto dp = "dp" + std::to_string(var_id);
            oss << tab << "char *" << arg << " = new char[" << karg.size << "];\n"
                << tab << "uint64_t " << dp << " = *(uint64_t *)(" << base << " + " << offset << ");\n"
                << tab << "aipubt::rt::Allocator::copy("
                << "(void *)" << arg << ", " << dp << ", " << karg.size << ");\n"
                << tab << "*(uint64_t *)(" << base << " + " << offset << ") = (uint64_t)" << arg << ";\n";
            if (karg.type == 3)
            {
                process_param_list(oss, arg, karg.args, bufs);
            }
            BufferInfo buf;
            buf.host_addr = arg;
            buf.device_addr = dp;
            buf.size = karg.size;
            buf.write_back = karg.type == 1 ? true : false;
            bufs.emplace_back(buf);
            offset += 2 * sizeof(uint32_t);
        }
        else
        {
            offset += sizeof(uint32_t);
        }
    }
}

void CodeGenerator::gen_host_task_func(std::stringstream &oss, const HostTaskInfo &info)
{
    auto tab = std::string(4, ' ');
    size_t num_params = 0;
    oss << "extern \"C\" void " << info.kernel_name << "(";
    for (size_t i = 0; i < info.kernel_args.size(); ++i)
    {
        auto &karg = info.kernel_args[i];
        if (karg.type == 1 || karg.type == 2 || karg.type == 3)
        {
            num_params += 2;
            oss << "void *";
        }
        else
        {
            num_params += 1;
            oss << "int";
        }
        if (i != info.kernel_args.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << ");\n\n";

    auto pb = "pb";
    auto pp = "pp";
    auto tpp = "tpp";
    oss << "void " << info.task_name << "(uint64_t " << pb << ")\n{\n"
        << tab << "uint64_t " << pp << " = " << pb << " + " << info.param_offset << ";\n"
        << tab << "uint32_t *" << tpp << " = new uint32_t[" << num_params << "];\n"
        << tab << "aipubt::rt::Allocator::copy("
        << "(void *)" << tpp << ", " << pp << ", " << num_params * sizeof(uint32_t) << ");\n";

    size_t offset = 0;
    std::vector<std::string> args;
    std::vector<BufferInfo> bufs;
    for (auto &karg : info.kernel_args)
    {
        auto var_id = get_var_id();
        auto arg = "arg" + std::to_string(var_id);
        if (karg.type == 1 || karg.type == 2 || karg.type == 3)
        {
            auto dp = "dp" + std::to_string(var_id);
            oss << tab << "char *" << arg << " = new char[" << karg.size << "];\n"
                << tab << "uint64_t " << dp << " = *(uint64_t *)(" << tpp << " + " << offset << ");\n"
                << tab << "aipubt::rt::Allocator::copy("
                << "(void *)" << arg << ", " << dp << ", " << karg.size << ");\n";
            if (karg.type == 3)
            {
                process_param_list(oss, arg, karg.args, bufs);
            }
            BufferInfo buf;
            buf.host_addr = arg;
            buf.device_addr = dp;
            buf.size = karg.size;
            buf.write_back = karg.type == 1 ? true : false;
            bufs.emplace_back(buf);
            offset += 2;
        }
        else
        {
            oss << tab << "uint32_t " << arg << " = " << tpp << "[" << offset << "];\n";
            offset += 1;
        }
        args.push_back(arg);
    }

    oss << tab << info.kernel_name << "(";
    for (size_t i = 0; i < info.kernel_args.size(); ++i)
    {
        auto &karg = info.kernel_args[i];
        auto &arg = args[i];
        if (karg.type == 1 || karg.type == 2 || karg.type == 3)
        {
            oss << "(void *)" << arg;
        }
        else
        {
            oss << "(int)" << arg;
        }
        if (i != info.kernel_args.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << ");\n";

    for (auto &buf : bufs)
    {
        if (buf.write_back)
        {
            oss << tab << "aipubt::rt::Allocator::copy("
                << buf.device_addr << ", (void *)" << buf.host_addr << ", " << buf.size << ");\n";
        }
        oss << tab << "delete[] " << buf.host_addr << ";\n";
    }
    oss << tab << "delete[] " << tpp << ";\n"
        << "}\n\n";
}

const std::string CodeGenerator::get_cpshost_code()
{
    auto host_code = m_host_code.str();
    auto ds_host_code = get_ds_host_code();
    std::stringstream oss;
    oss << R"(
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#include <allocator.h>
#include <launcher.h>

typedef struct sync_ctrl
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    std::vector<bool> ready;
    bool quit = false;
} sync_ctrl_t;

typedef struct thread_arg
{
    uint32_t tid = 0;
    uint64_t param_base = 0;
    aipubt::rt::IEvent *wait_event = NULL;
    aipubt::rt::IEvent *set_event = NULL;
    aipubt::rt::IEvent::attr_t wait_attr;
    aipubt::rt::IEvent::attr_t set_attr;
    void (*task_func)(uint64_t) = NULL;
    sync_ctrl_t *ctrl = NULL;
} thread_arg_t;

void *task_thread_func(void *arg)
{
    auto p = (thread_arg_t *)arg;
    auto tid = p->tid;
    auto param_base = p->param_base;
    auto wait_event = p->wait_event;
    auto set_event = p->set_event;
    auto &wait_attr = p->wait_attr;
    auto &set_attr = p->set_attr;
    auto task_func = p->task_func;
    auto ctrl = p->ctrl;

    while (true)
    {
        pthread_mutex_lock(&ctrl->mutex);
        while (!ctrl->ready[tid] && !ctrl->quit)
        {
            pthread_cond_wait(&ctrl->cond, &ctrl->mutex);
        }
        if (ctrl->quit)
        {
            pthread_mutex_unlock(&ctrl->mutex);
            break;
        }
        ctrl->ready[tid] = false;
        pthread_mutex_unlock(&ctrl->mutex);

        wait_event->wait(wait_attr);
        task_func(param_base);
        set_event->set(set_attr);
    }
    return NULL;
}

)";
    oss << host_code;
    if (ds_host_code.size())
    {
        oss << "typedef unsigned int uint;\n"
            << "typedef unsigned long long ulonglong;\n\n"
            << ds_host_code;
    }
    oss << R"(
namespace aipubt::rt {

class HybridLauncher : public Launcher
{
public:
    HybridLauncher() : Launcher(false)
    {
        pthread_mutex_init(&m_ctrl.mutex, NULL);
        pthread_cond_init(&m_ctrl.cond, NULL);
    }
    ~HybridLauncher()
    {
        if (m_tids.size() > 0)
        {
            pthread_mutex_lock(&m_ctrl.mutex);
            m_ctrl.quit = true;
            pthread_cond_broadcast(&m_ctrl.cond);
            pthread_mutex_unlock(&m_ctrl.mutex);

            for (auto &tid : m_tids)
            {
                pthread_join(tid, NULL);
            }
        }
        pthread_mutex_destroy(&m_ctrl.mutex);
        pthread_cond_destroy(&m_ctrl.cond);
    }

    int load_from_file(const std::string &elf_filepath) override
    {
        int ret = Launcher::load_from_file(elf_filepath);

        if (ret == 0)
        {
            auto param_base = get_param_base();
            aipu_create_threads(this, &m_ctrl, m_tids, m_args, m_events, param_base);
        }
        return ret;
    }

    int run() override
    {
        if (m_tids.size() > 0)
        {
            pthread_mutex_lock(&m_ctrl.mutex);
            for (size_t i = 0; i < m_tids.size(); ++i)
            {
                m_ctrl.ready[i] = true;
            }
            pthread_cond_broadcast(&m_ctrl.cond);
            pthread_mutex_unlock(&m_ctrl.mutex);
        }
        return Launcher::run();
    }

protected:
    sync_ctrl_t m_ctrl;
    std::vector<pthread_t> m_tids;
    std::vector<thread_arg_t> m_args;
    std::vector<IEvent *> m_events;
};

} // namespace aipubt::rt

extern "C" {

aipubt::rt::ILauncher *create_launcher()
{
    Dl_info info;
    if (dladdr((void *)create_launcher, &info) == 0)
    {
        return NULL;
    }
    if (!info.dli_fname)
    {
        return NULL;
    }
    // printf("Library name: %s\n", info.dli_fname);
    auto launcher = new aipubt::rt::HybridLauncher();
    if (launcher->load_from_file(info.dli_fname) != 0)
    {
        delete launcher;
        return NULL;
    }
    return launcher;
}

void destroy_launcher(aipubt::rt::ILauncher *p)
{
    if (p)
    {
        delete p;
    }
}

}

)";
    auto model_name = m_graph->name();
    INFO() << "register model: " << model_name;
    oss << "REGISTER_MODEL(\"" << model_name << "\", create_launcher, destroy_launcher);\n";
    return oss.str();
}

} // namespace aipubt