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
 * @Author: Lang Hu
 * @Date: 2026-08-10 14:59:25
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-08-10 14:59:25
 */
#pragma once
#include "builder/cps_elf.h"
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <utility>

namespace aipubt
{

struct EventAttr
{
    uint64_t addr;
    uint32_t type;
    uint32_t data;
    uint32_t mask;
    uint16_t task_id;
};

struct RuntimeAPI
{
    void (*host_to_device)(void *dst, const void *src, size_t num_bytes) = 0;
    void (*device_to_host)(void *dst, const void *src, size_t num_bytes) = 0;
    void (*device_to_device)(void *dst, const void *src, size_t num_bytes) = 0;

    void *(*get_event)(EventAttr &attr) = 0;
    int (*wait_event)(void *event, const EventAttr &attr) = 0;
    int (*set_event)(void *event, const EventAttr &attr) = 0;
};

struct KernelArg
{
    KernelArg(int val)
    {
        value = static_cast<size_t>(static_cast<int64_t>(val));
    }
    KernelArg(uint32_t val)
    {
        value = static_cast<size_t>(val);
    }
    KernelArg(void *val)
    {
        value = reinterpret_cast<size_t>(val);
    }

    size_t value = 0;
};

template <size_t MaxArgs>
struct DynamicCaller
{
    static size_t call(void *func_ptr, const std::vector<KernelArg> &args)
    {
        if (args.size() == MaxArgs)
        {
            return invoke(func_ptr, args, std::make_index_sequence<MaxArgs>{});
        }
        return DynamicCaller<MaxArgs - 1>::call(func_ptr, args);
    }

private:
    template <size_t... Idxs>
    static size_t invoke(void *func_ptr,
                         const std::vector<KernelArg> &args,
                         std::index_sequence<Idxs...>)
    {
        using FuncType = void (*)(decltype(args[Idxs].value)...);
        auto kernel_func = reinterpret_cast<FuncType>(func_ptr);
        kernel_func(args[Idxs].value...);
        return 0;
    }
};

// deal with boundary
template <>
struct DynamicCaller<0>
{
    static size_t call(void *func_ptr, const std::vector<KernelArg> &args)
    {
        if (args.size() == 0)
        {
            using FuncType = void (*)();
            auto kernel_func = reinterpret_cast<FuncType>(func_ptr);
            kernel_func();
        }
        return 0;
    }
};

/**
 * @brief An interpreter that executes host code
 * 
 * How to use:
 *   elf = HybridELF()
 *   elf.load("xxx")
 *   task_infos = elf.get_task_infos()
 * 
 *   rt_api = RuntimeAPI()
 *   rt_api = xxx
 *   param_base = xxx
 *   interpreter = Interpreter(task_infos, rt_api, param_base)
 * 
 *   interpreter.run()
 *   launcher.run()
 */
class Interpreter
{
public:
    struct SyncCtrl
    {
        pthread_mutex_t mutex;
        pthread_cond_t cond;
        std::vector<bool> ready;
        bool quit = false;
    };

    struct Event
    {
        void *event = nullptr;
        EventAttr attr;
    };

    struct ThreadArg
    {
        uint32_t tid = 0;
        Event wait_event;
        Event set_event;
        Interpreter *self = nullptr;
    };

    struct Buffer
    {
        char *host_addr = nullptr;
        uint64_t device_addr = 0;
        size_t size = 0;
        bool write_back = false;
    };

    Interpreter(const HostTaskInfos &infos, const RuntimeAPI &rt_api, uint64_t param_base)
    {
        m_task_infos = infos;
        m_rt_api = rt_api;
        m_param_base = param_base;
        init();
    }
    ~Interpreter()
    {
        if (m_task_infos.tasks.size() > 0)
        {
            pthread_mutex_lock(&m_ctrl.mutex);
            m_ctrl.quit = true;
            pthread_cond_broadcast(&m_ctrl.cond);
            pthread_mutex_unlock(&m_ctrl.mutex);

            for (auto &tid : m_tids)
            {
                pthread_join(tid, NULL);
            }
            pthread_mutex_destroy(&m_ctrl.mutex);
            pthread_cond_destroy(&m_ctrl.cond);
            for (auto handle : m_handles)
            {
                if (handle)
                {
                    dlclose(handle);
                }
            }
        }
    }

    int run()
    {
        auto num_tasks = m_task_infos.tasks.size();
        if (num_tasks > 0)
        {
            pthread_mutex_lock(&m_ctrl.mutex);
            for (size_t i = 0; i < num_tasks; ++i)
            {
                m_ctrl.ready[i] = true;
            }
            pthread_cond_broadcast(&m_ctrl.cond);
            pthread_mutex_unlock(&m_ctrl.mutex);
        }
        return 0;
    }

private:
    static void *task_thread_func(void *arg)
    {
        auto p = (ThreadArg *)arg;
        auto tid = p->tid;
        auto &wait_event = p->wait_event;
        auto &set_event = p->set_event;
        auto self = p->self;
        auto &task_info = self->m_task_infos.tasks[tid];
        auto &rt_api = self->m_rt_api;
        auto param_base = self->m_param_base;
        auto ctrl = &self->m_ctrl;
        auto kernel_func = self->m_kernels[tid];

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

            rt_api.wait_event(wait_event.event, wait_event.attr);
            run_task(rt_api, task_info, kernel_func, param_base);
            rt_api.set_event(set_event.event, set_event.attr);
        }
        return NULL;
    }

    static void process_param_list(const RuntimeAPI &rt_api,
                                   char *base,
                                   const std::vector<HostTaskArg> &args,
                                   std::vector<Buffer> &bufs)
    {
        size_t offset = 0;
        for (auto &karg : args)
        {
            if (karg.type == 1 || karg.type == 2 || karg.type == 3)
            {
                auto arg = new char[karg.size];
                uint64_t dp = *(uint64_t *)(base + offset);
                rt_api.device_to_host(arg, (void *)dp, karg.size);
                *(uint64_t *)(base + offset) = (uint64_t)arg;
                if (karg.type == 3)
                {
                    process_param_list(rt_api, arg, karg.args, bufs);
                }
                Buffer buf;
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

    static void run_task(const RuntimeAPI &rt_api,
                         const HostTaskInfo &info,
                         void *kernel_func,
                         uint64_t param_base)
    {
        if (!kernel_func)
        {
            return;
        }
        auto num_params = info.kernel_args.size();
        uint64_t pp = param_base + info.param_offset;
        auto tpp = new uint32_t[num_params];
        rt_api.device_to_host(tpp, (void *)pp, num_params * sizeof(uint32_t));
        size_t offset = 0;
        std::vector<KernelArg> args;
        std::vector<Buffer> bufs;
        for (auto &karg : info.kernel_args)
        {
            if (karg.type == 1 || karg.type == 2 || karg.type == 3)
            {
                auto arg = new char[karg.size];
                uint64_t dp = *(uint64_t *)(tpp + offset);
                rt_api.device_to_host(arg, (void *)dp, karg.size);
                if (karg.type == 3)
                {
                    process_param_list(rt_api, arg, karg.args, bufs);
                }
                Buffer buf;
                buf.host_addr = arg;
                buf.device_addr = dp;
                buf.size = karg.size;
                buf.write_back = karg.type == 1 ? true : false;
                bufs.emplace_back(buf);
                args.push_back(arg);
                offset += 2;
            }
            else
            {
                args.push_back(tpp[offset]);
                offset += 1;
            }
        }
        DynamicCaller<64>::call(kernel_func, args);
        for (auto &buf : bufs)
        {
            if (buf.write_back)
            {
                rt_api.host_to_device((void *)buf.device_addr, buf.host_addr, buf.size);
            }
            delete[] buf.host_addr;
        }
        delete[] tpp;
    }

    void init()
    {
        auto num_tasks = m_task_infos.tasks.size();
        if (num_tasks > 0)
        {
            pthread_mutex_init(&m_ctrl.mutex, NULL);
            pthread_cond_init(&m_ctrl.cond, NULL);
            m_ctrl.ready.resize(num_tasks, false);
            for (size_t i = 0; i < num_tasks; ++i)
            {
                create_thread(i, m_task_infos.tasks[i]);
            }
        }
    }

    void create_thread(size_t tid, const HostTaskInfo &info)
    {
        auto handle = dlopen(info.lib_name.c_str(), RTLD_LAZY);
        if (!handle)
        {
            std::cerr << "Failed to open library: " << info.lib_name << std::endl;
        }
        m_handles.push_back(handle);
        void *kernel_func = nullptr;
        if (handle)
        {
            kernel_func = dlsym(handle, info.kernel_name.c_str());
        }
        if (!kernel_func)
        {
            std::cerr << "Cannot find symbol: " << info.kernel_name << std::endl;
        }
        m_kernels.push_back(kernel_func);

        m_args.push_back(ThreadArg());
        auto arg = &m_args[m_args.size() - 1];
        arg->tid = tid;
        arg->wait_event.attr.data = info.set_data;
        arg->wait_event.attr.mask = 0xf;
        arg->wait_event.attr.task_id = info.set_id;
        arg->wait_event.event = m_rt_api.get_event(arg->wait_event.attr);
        m_events.push_back(arg->wait_event.event);
        arg->set_event.attr.data = info.wait_ref;
        arg->set_event.attr.mask = 0xf;
        arg->set_event.attr.task_id = info.wait_id;
        arg->set_event.event = m_rt_api.get_event(arg->set_event.attr);
        m_events.push_back(arg->set_event.event);
        arg->self = this;

        m_tids.push_back(pthread_t());
        pthread_create(&m_tids[m_tids.size() - 1], NULL, task_thread_func, (void *)arg);
    }

    HostTaskInfos m_task_infos;
    RuntimeAPI m_rt_api;
    uint64_t m_param_base = 0;
    SyncCtrl m_ctrl;
    std::vector<pthread_t> m_tids;
    std::vector<ThreadArg> m_args;
    std::vector<void *> m_events;
    std::vector<void *> m_handles;
    std::vector<void *> m_kernels;
};

} // namespace aipubt
