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
 * @Date: 2024-07-23 16:09:42 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-16 11:06:09
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
namespace utils
{

#define DEFAULT_THREAD_NUM 16
class ThreadPool
{
public:
    ThreadPool(size_t threads = DEFAULT_THREAD_NUM) : m_stop(false)
    {
        for (size_t i = 0; i < threads; ++i)
        {
            m_works.emplace_back([this] {
                while (true)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->m_queue_mutex);
                        this->m_condition.wait(lock, [this] { return this->m_stop || !this->m_tasks.empty(); });
                        if (this->m_stop && this->m_tasks.empty())
                        {
                            return;
                        }
                        task = std::move(this->m_tasks.front());
                        this->m_tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    auto enqueue(F &&f, Args &&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);

            if (m_stop)
            {
                // throw std::runtime_error("enqueue on stopped ThreadPool");
                return res;
            }

            m_tasks.emplace([task]() { (*task)(); });
        }
        m_condition.notify_one();
        return res;
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        m_condition.notify_all();
        for (auto &worker : m_works)
        {
            worker.join();
        }
    }

private:
    std::vector<std::thread> m_works;
    std::queue<std::function<void()>> m_tasks;

    std::mutex m_queue_mutex;
    std::condition_variable m_condition;
    bool m_stop;
};

class Barrier
{
public:
    explicit Barrier(size_t count)
        : threshold(count), count_(count), generation_(0) {}

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        auto gen = generation_;
        if (--count_ == 0)
        {
            generation_++;
            count_ = threshold;
            cv.notify_all();
        }
        else
        {
            cv.wait(lock, [this, gen] { return gen != generation_; });
        }
    }

private:
    std::mutex mutex_;
    std::condition_variable cv;
    size_t threshold;   // 总线程数
    size_t count_;      // 当前剩余等待线程数
    size_t generation_; // 代数标记，用于重置屏障
};
} // namespace utils
