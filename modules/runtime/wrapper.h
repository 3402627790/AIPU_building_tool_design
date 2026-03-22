/**********************************************************************************
 * This file is CONFIDENTIAL and any use by you is subject to the terms of the
 * agreement between you and Arm China or the terms of the agreement between you
 * and the party authorised by Arm China to disclose this file to you.
 * The confidential and proprietary information contained in this file
 * may only be used by a person authorised under and to the extent permitted
 * by a subsisting licensing agreement from Arm China.
 *
 *        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
 *                    All rights reserved.
 *
 * This entire notice must be reproduced on all copies of this file and copies of
 * this file may only be made by a person if such person is permitted to do so
 * under the terms of a subsisting license agreement from Arm China.
 *
 *********************************************************************************/
#pragma once
#include <core/logger.h>
#include <core/quantization.h>
#include <core/stub.h>
#include <core/tensor.h>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utils/path_utils.h>
#include <utils/string_utils.h>
#include <vector>
using namespace std;
namespace aipubt
{
namespace AIPUDriverWrapper
{

    class AIPUContext;
    using AIPUContextPtr = std::shared_ptr<AIPUContext>;
    class AIPUModel;
    using AIPUModelPtr = std::shared_ptr<AIPUModel>;
    class AIPUJob;
    using AIPUJobPtr = std::shared_ptr<AIPUJob>;
    class MultiAIPUModels;
    using MultiAIPUModelsPtr = std::shared_ptr<MultiAIPUModels>;
    class AIPUContext
    {

    public:
        static AIPUContextPtr create();
        static AIPUContextPtr create(void *cfg);
        static AIPUContextPtr get_default();
        const std::string get_error_msg(uint32_t status);

        void *get_handle();
        ~AIPUContext();

        bool has_error(uint32_t status, const std::string &where = "");

    private:
        AIPUContext();
        AIPUContext(void *);

        int ioctl(uint32_t cmd, void *res);

        friend AIPUModel;

    protected:
        void *m_ctx = nullptr;
    };

    class MultiAIPUModels
    {
    public:
        MultiAIPUModels(const std::string &filename, AIPUContextPtr ctx = nullptr);
        ~MultiAIPUModels();

        const std::vector<AIPUModelPtr> &get_models() const
        {
            return m_models;
        }
        AIPUModelPtr operator[](int32_t index);
        size_t size() const
        {
            return m_models.size();
        }
        std::vector<AIPUModelPtr>::iterator begin(void);
        std::vector<AIPUModelPtr>::iterator end(void);
        std::vector<AIPUModelPtr>::const_iterator begin(void) const;
        std::vector<AIPUModelPtr>::const_iterator end(void) const;

        const std::string to_string(void) const;

    private:
        AIPUContextPtr m_ctx;
        bool m_status;
        std::string m_filename;
        std::vector<uint64_t> m_gids;
        std::vector<AIPUModelPtr> m_models;
    };

    class AIPUModel
    {
    public:
        AIPUModel(const std::string &filename, AIPUContextPtr ctx = nullptr);
        AIPUModel(AIPUContextPtr ctx, uint64_t gid) : m_ctx(ctx), m_gid(gid)
        {
            if (!ctx)
            {
                ERROR() << "please give the correct context when init a AIPUModel.";
            }
            m_status = true;
            get_model_basic_info();
        }
        ~AIPUModel();

        AIPUContextPtr context();

        uint64_t gid();
        uint64_t gid() const;
        bool is_dynamic() const
        {
            return m_is_ds;
        }
        const std::vector<TensorShape> get_input_min_shapes() const
        {
            return m_min_inps;
        }
        const std::vector<TensorShape> get_input_max_shapes() const
        {
            return m_max_inps;
        }
        const std::vector<TensorPtr> &get_outputs() const;
        const std::vector<TensorPtr> &get_profdata() const;
        const std::vector<TensorPtr> &get_inputs() const;
        const std::string &get_profiling_info() const;

        const void get_model_basic_info();
        const std::string to_string(void) const;

    protected:
        inline bool has_error(uint32_t status, const std::string &where = "") const;
        friend AIPUJob;
        AIPUContextPtr m_ctx;
        std::string m_filename;
        mutable std::string m_profiling;
        uint64_t m_gid;
        bool m_is_ds;
        bool m_status;
        std::vector<TensorPtr> m_inputs;
        std::vector<TensorPtr> m_outputs;
        std::vector<TensorPtr> m_profdata;
        std::vector<TensorShape> m_min_inps;
        std::vector<TensorShape> m_max_inps;
    };

    class AIPUJob
    {
    public:
        AIPUJob(AIPUModelPtr graph);

        void set_dump(bool d);
        void set_output_prefix(const std::string &p);

        AIPUJob(const std::string &graph_file);
        ~AIPUJob();
        void set_inputs(const std::vector<TensorPtr> &inputs);

        const std::vector<TensorPtr> get_outputs();
        const std::vector<TensorPtr> get_profdata();

        const std::vector<TensorPtr> run(const std::vector<TensorPtr> &inputs);

        void get_outputs(const std::vector<TensorPtr> &outputs);
        void rewrite_ro(uint32_t offset, uint32_t value);
        AIPUModelPtr model() const
        {
            return m_graph;
        }

    protected:
        void init();
        void ds_init_and_set_input(const std::vector<TensorPtr> &inputs);
        void clean_job();
        inline bool has_error(uint32_t status, const std::string &where = "") const;
        // void set_ds_input_shapes()
        uint64_t m_jid;
        AIPUModelPtr m_graph;
        bool is_init = false;
        bool m_dump = false;
        std::string m_out_prefix = "output.bin";
    };

    struct DevMemObj
    {
        int buf_fd = 0;
        uint64_t pa = 0;
        void *va = 0;
        uint64_t size = 0;
        DevMemObj() = default;
        DevMemObj(int fd, uint64_t pa, void *va, uint64_t size) : buf_fd(fd), pa(pa), va(va), size(size){};
        DevMemObj(uint64_t size);
        ~DevMemObj();
    };

    struct AIPUClusterCap
    {
        uint32_t core_cnt;    /* total core count in this cluster */
        uint32_t en_core_cnt; /* currently enabled core count */
        uint32_t tec_cnt;
    };

    struct AIPUPartitionCap
    {

        uint32_t id;
        uint32_t arch;
        uint32_t version;
        uint32_t config;
        uint64_t reg_base;                    /* External register base address (physical) */
        std::vector<AIPUClusterCap> clusters; /* every partition has at maximum 8 clusters */
    };

    struct AIPUCap
    {
        uint32_t partition_cnt;
        uint32_t is_homogeneous;
        uint64_t dtcm_base;
        uint32_t dtcm_size;
        uint32_t gm0_size;
        uint32_t gm1_size;
        std::vector<uint64_t> asid_base;
        AIPUPartitionCap partition_cap;
    };

    /**
         * struct aipu_cluster_status - AIPU cluster working status.
         * @cluster_status:	[kmd] filled by KMD,
         * @core_status:	[kmd] filled by KMD, each bit is the status of one core in the cluster,
         *			0 for busy, 1 for idle
        */
    struct AIPUClusterStatus
    {
        uint32_t cluster_status;
        uint32_t core_status;
    };
    std::string get_device_version();
    const std::string &get_driver_version();
    AIPUClusterStatus get_cluster_status(uint32_t cluster_id = 0);
    std::vector<uint32_t> get_running_job_thread_id();
    uint32_t get_partion_number();
    uint32_t get_cluster_number(uint32_t partion_id);
    uint32_t get_core_number(uint32_t partion_id, uint32_t cluster_id);
    uint32_t get_device_status();
    bool device_soft_reset();
    bool enable_tick_counter();
    bool disable_tick_counter();
    AIPUCap get_aipu_cap();
    DevMemObj dev_allocate(uint64_t size);
    void dev_free(DevMemObj &obj);
    void *dev_mmap(DevMemObj &obj);
    void dev_munmap(DevMemObj &obj);
} // namespace AIPUDriverWrapper

} // namespace aipubt