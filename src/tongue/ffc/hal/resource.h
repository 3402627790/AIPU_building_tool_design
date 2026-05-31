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
 * @Date: 2024-10-24 17:17:01
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-11-24 09:52:48
 */
#pragma once
#include "../tensor.h"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utils/utils.h>

namespace ffc
{
namespace HAL
{
    class AIFFRegs;
    class Resource;
    using ResourcePtr = std::shared_ptr<Resource>;
    using ResourceRef = std::weak_ptr<Resource>;

    enum ResourceStatus
    {
        FREE = 0,
        USED = 1,
    };
    /**
     * @brief design of Resource
     * 
     * Resource is a abstract of HW resource
     * A resource may contains many sub-resources, such as a CPU may contains many cores
     * if a resource has sub-resource, then the status of the resource is decided by its sub-resources.
     * only the leaf level resource' status is valid.
     * 
     */
    class Resource : public std::enable_shared_from_this<Resource>
    {
    public:
        Resource() = default;
        Resource(const std::string &name, const std::string &arch)
        {
            m_name = name;
            m_arch = arch;
        }
        virtual ~Resource() = default;

        bool feed_pipe(Tensor t);
        virtual bool match(Tensor t)
        {
            return false;
        }
        virtual void setup(Tensor t)
        {
            m_status = ResourceStatus::USED;
            m_tensor = t;
        }

        void free()
        {
            if (is_free())
            {
                return;
            }
            if (m_resources.empty())
            {
                m_status = ResourceStatus::FREE;
                m_tensor = Tensor(uOp::None);
            }
            else
            {
                for (auto r : m_resources)
                {
                    r->free();
                }
            }
        }
        bool is_free() const
        {
            return status() == ResourceStatus::FREE;
        }

        ResourceStatus status() const
        {
            // only leaf's status is valid
            if (m_resources.size())
            {
                for (auto r : m_resources)
                {
                    auto s = r->status();
                    if (s != ResourceStatus::FREE)
                    {
                        return s;
                    }
                }
                return ResourceStatus::FREE;
            }
            return m_status;
        }

        void infer_pipe(TensorDesc main_input,
                        const std::map<int, TensorDesc> &other_inputs = {})
        {
            auto fn = [&main_input, &other_inputs](ResourcePtr r) {
                if (r->m_resources.empty())
                {
                    std::vector<TensorDesc> inputs;
                    if (r->m_tensor->inputs.empty())
                    {
                        inputs.push_back(main_input);
                    }
                    else
                    {
                        for (auto t : r->m_tensor->inputs)
                        {
                            if (other_inputs.count(t->id))
                            {
                                inputs.push_back(other_inputs.at(t->id));
                            }
                            else
                            {
                                inputs.push_back(main_input);
                            }
                        }
                    }
                    main_input = r->forward(inputs);
                }
            };
            auto res = get_local_pipe();
            for (auto r : res)
            {
                r->walk(fn);
            }
        }
        virtual TensorDesc forward(const std::vector<TensorDesc> &inputs)
        {
            if (inputs.size())
            {
                m_output = inputs[0];
            }
            return m_output;
        }

        void walk(std::function<void(ResourcePtr)> fn, bool top2down = true)
        {
            std::function<void(ResourcePtr)> dfs;
            if (top2down)
            {
                dfs = [&dfs, &fn](ResourcePtr r) {
                    fn(r);
                    for (auto sr : r->m_resources)
                    {
                        dfs(sr);
                    }
                };
            }
            else
            {
                dfs = [&dfs, &fn](ResourcePtr r) {
                    for (auto sr : r->m_resources)
                    {
                        dfs(sr);
                    }
                    fn(r);
                };
            }
            dfs(shared_from_this());
        }

        virtual ResourcePtr init()
        {
            return shared_from_this();
        }
        void build_pipe(AIFFRegs &regs)
        {
            auto fn = [&regs](ResourcePtr r) {
                r->build(regs);
            };
            auto res = get_local_pipe();
            for (auto r : res)
            {
                r->walk(fn, false);
            }
        }
        virtual void build(AIFFRegs &regs)
        {
        }

        ResourcePtr get_resource(const std::string &xpath)
        {
            ResourcePtr ptr;
            auto pathes = utils::split_string(xpath, ':');
            if (pathes.size())
            {
                for (size_t i = 0; i < pathes.size(); ++i)
                {
                    auto name = pathes[i];
                    auto res = ptr ? ptr->m_resources : m_resources;
                    bool found = false;
                    for (auto r : res)
                    {
                        if (r->m_name == name)
                        {
                            ptr = r;
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        return nullptr;
                    }
                }
            }
            return ptr;
        }

        void add_res(ResourcePtr r)
        {
            if (r && !contains(m_resources, r))
            {
                m_resources.push_back(r);
            }
        }

        std::vector<ResourcePtr> previous() const
        {
            std::vector<ResourcePtr> ret;
            for (auto r : m_previous)
            {
                if (!r.expired())
                {
                    ret.push_back(r.lock());
                }
            }
            return ret;
        }
        std::vector<ResourcePtr> subsequent() const
        {
            std::vector<ResourcePtr> ret;
            for (auto r : m_subsequent)
            {
                if (!r.expired())
                {
                    ret.push_back(r.lock());
                }
            }
            return ret;
        }

        ResourcePtr then(ResourcePtr r)
        {
            auto self = shared_from_this();
            if (r && r != self)
            {
                if (!contains(r->previous(), self))
                {
                    r->m_previous.push_back(self);
                }
                if (!contains(subsequent(), r))
                {
                    m_subsequent.push_back(r);
                }
            }
            return r;
        }
        ResourcePtr add_pipe(ResourcePtr r)
        {
            auto res = r->get_local_pipe();
            for (auto r : res)
            {
                add_res(r);
            }
            return shared_from_this();
        }
        std::vector<ResourcePtr> get_top_pipe()
        {
            auto entry = m_global_entry.lock();
            return entry->get_local_pipe();
        }
        std::vector<ResourcePtr> get_local_pipe()
        {
            std::vector<ResourcePtr> entry;
            std::function<void(ResourcePtr)> dfs;
            dfs = [&dfs, &entry](ResourcePtr r) {
                auto pres = r->previous();
                if (pres.empty())
                {
                    entry.push_back(r);
                }
                else
                {
                    for (auto pr : pres)
                    {
                        if (!contains(entry, pr))
                        {
                            dfs(pr);
                        }
                    }
                }
            };
            dfs(shared_from_this());
            std::vector<ResourcePtr> ret;
            dfs = [&dfs, &ret](ResourcePtr r) {
                ret.push_back(r);
                auto subs = r->subsequent();
                for (auto sr : subs)
                {
                    if (!contains(ret, sr))
                    {
                        dfs(sr);
                    }
                }
            };
            for (auto r : entry)
            {
                dfs(r);
            }
            return ret;
        }
        ResourcePtr get_from_pipe(const std::string &name)
        {
            auto res = get_local_pipe();
            for (auto r : res)
            {
                if (r->m_name == name)
                {
                    return r;
                }
            }
            return nullptr;
        }
        bool is_pipe_free()
        {
            auto res = get_local_pipe();
            for (auto r : res)
            {
                if (!r->is_free())
                {
                    return false;
                }
            }
            return true;
        }

        std::string m_name;
        std::string m_arch;
        ResourceStatus m_status = ResourceStatus::FREE;
        std::vector<ResourcePtr> m_resources;
        std::vector<ResourceRef> m_previous;
        std::vector<ResourceRef> m_subsequent;
        int m_global_pid = 0;
        ResourceRef m_global_entry;
        Tensor m_tensor = Tensor(uOp::None);
        TensorDesc m_output;
    };

    class AIFFResource : public Resource
    {
    public:
        AIFFResource(const std::string &arch);
        static ResourcePtr create(const std::string &arch);

    private:
        static void set_pipe_info(int pid, ResourcePtr entry);
    };

    std::vector<ResourcePtr> get_real_pipe(int pid, ResourcePtr entry);
} // namespace HAL
} // namespace ffc