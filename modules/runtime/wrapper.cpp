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

#include "wrapper.h"
#include "armchina_aipu.h"
#include "internal_api.h"
#include "standard_api.h"
#include <sys/ioctl.h>
#include <sys/mman.h>

using namespace std;

extern "C"
{
    // define the default implementation of the function if it is not provided by the user
    aipu_status_t __attribute__((weak)) aipu_rewrite_rodata(
        const aipu_ctx_handle_t *ctx,
        uint64_t job_id,
        uint32_t offset,
        uint32_t value)
    {
        return AIPU_STATUS_ERROR_UNKNOWN_ERROR;
    }
};

namespace aipubt
{
namespace AIPUDriverWrapper
{

    constexpr char DEV_PATH[] = "/dev/aipu";
    static TensorPtr desc_to_tensor(const aipu_tensor_desc_t &desc)
    {
        auto dtype = (Tensor::Dtype)desc.data_type;
        auto ret = std::make_shared<Tensor>(TensorShape(desc.size / Tensor::dtype_size(dtype)), dtype);
        ret->quantization().set_scale(desc.scale);
        ret->quantization().set_offset(desc.zero_point);
        return ret;
    }
    AIPUContextPtr AIPUContext::create()
    {
        struct make_shared_enabler : public AIPUContext
        {
        };
        return std::make_shared<make_shared_enabler>();
    }

    AIPUContextPtr AIPUContext::create(void *cfg)
    {
        struct make_shared_enabler : public AIPUContext
        {
            make_shared_enabler(void *cfg) : AIPUContext(cfg) {}
        };
        return std::make_shared<make_shared_enabler>(cfg);
    }
    AIPUContextPtr AIPUContext::get_default()
    {
        static AIPUContextPtr default_ = create();
        return default_;
    }
    const std::string AIPUContext::get_error_msg(uint32_t status)
    {
        const char *msg = nullptr;
        aipu_get_error_message((aipu_ctx_handle_t *)m_ctx, (aipu_status_t)status, &msg);
        return msg;
    }

    void *AIPUContext::get_handle()
    {
        return m_ctx;
    }
    AIPUContext::~AIPUContext()
    {
        if (m_ctx)
        {
            auto ret = aipu_deinit_context((aipu_ctx_handle_t *)m_ctx);
            (void)ret;
            // has_error(ret, "deinit_context"); // it may failed due to a static obj is free before the ctx
        }
    }

    bool AIPUContext::has_error(uint32_t status, const std::string &where)
    {
        if (status != (uint32_t)AIPU_STATUS_SUCCESS)
        {
            ERROR() << where << " " << get_error_msg(status);
            return true;
        }
        return false;
    }

    AIPUContext::AIPUContext()
    {
        auto ret = aipu_init_context((aipu_ctx_handle_t **)&m_ctx);
        if (has_error(ret, "init_context"))
        {
            m_ctx = nullptr;
        }
    }
    AIPUContext::AIPUContext(void *cfg) : AIPUContext()
    {
        aipu_global_config_simulation_t conf = *(aipu_global_config_simulation_t *)cfg;
        if (conf.log_level == 0)
        {
            /**
             * @brief  simlog level is:
             *  0: error
             *  1: warning
             *  2: info
             *  3: debug
             *
             * however the bt log is:
             * 10: fatal
             * 9 : error
             * 5 : info
             * 2 : warning
             * 1 : debug
             */

            int level = GCompilerLogger::get_logger_level();
            if (level >= _ERROR_LEVEL)
            {
                conf.log_level = 0;
            }
            else if (level >= _WARN_LEVEL || level >= _INFO_LEVEL)
            {
                conf.log_level = 2;
            }
            else //(level >= _DEBUG_LEVEL)
            {
                conf.log_level = 3;
            }
            INFO() << "Simulator log level:" << level;
        }
        INFO() << "Simulator log level:" << conf.log_level;
        auto ret = aipu_config_global((aipu_ctx_handle_t *)m_ctx, AIPU_CONFIG_TYPE_SIMULATION, &conf);
        has_error(ret, "config_global");
    }

    int AIPUContext::ioctl(uint32_t cmd, void *res)
    {
        // AIPU_IOCTL_SET_PROFILE = 255,
        // AIPU_IOCTL_GET_AIPUBIN_BUILDVERSION,
        // AIPU_IOCTL_GET_DS_NUM,
        // AIPU_IOCTL_GET_DS_DIM_NUM,
        // AIPU_IOCTL_GET_DS_INFO,
        // AIPU_IOCTL_SET_DS_INFO,
        // AIPU_IOCTL_IS_DYNAMIC_SHAPE,
        // AIPU_IOCTL_ALLOC_SHARE_BUF,
        // AIPU_IOCTL_FREE_SHARE_BUF,
        // AIPU_IOCTL_ALLOC_DMABUF,
        // AIPU_IOCTL_FREE_DMABUF,
        // AIPU_IOCTL_WRITE_DMABUF,
        // AIPU_IOCTL_READ_DMABUF,
        // AIPU_IOCTL_ATTACH_DMABUF,
        // AIPU_IOCTL_DETACH_DMABUF,
        // AIPU_IOCTL_GET_VERSION

        return aipu_ioctl((aipu_ctx_handle_t *)m_ctx, cmd, res);
    }

    MultiAIPUModels::MultiAIPUModels(const std::string &filename, AIPUContextPtr ctx)
    {

        if (!ctx)
        {
            ctx = AIPUContext::get_default();
        }
        m_ctx = ctx;
        m_filename = filename;
        auto handle = (aipu_ctx_handle_t *)(m_ctx->get_handle());
        m_status = handle != nullptr;
        aipu_status_t ret = AIPU_STATUS_SUCCESS;
        if (m_status)
        {
            auto _check_load_graph = [&handle, &filename](aipu_status_t ret, bool &m_status) {
                if (ret != AIPU_STATUS_SUCCESS)
                {
                    const char *msg = nullptr;
                    aipu_get_error_message(handle, ret, &msg);
                    ERROR("aipu_load_graph_helper: %s (%s)\n",
                          msg, filename.c_str());
                    m_status = false;
                }
            };
            m_gids.clear();
            if (utils::endswith(filename, ".zip"))
            {
                uint64_t *gids;
                uint32_t gids_cnt;
                // INFO() << filename;
                ret = aipu_load_share_weight_graph(handle, filename.c_str(), &gids, &gids_cnt);
                _check_load_graph(ret, m_status);
                m_gids.assign(gids, gids + gids_cnt);
            }
            else
            {
                uint64_t gid;
                auto driver_extra_weight_cfg = new aipu_load_graph_cfg_t{0};
                std::string extra_wt_path = utils::dir_name(filename);
                driver_extra_weight_cfg->extra_weight_path = extra_wt_path.c_str();
                auto ret = aipu_load_graph(handle, filename.c_str(), &gid, driver_extra_weight_cfg);
                _check_load_graph(ret, m_status);
                m_gids.push_back(gid);
            }

            if (m_gids.size())
            {
                for (auto &gid : m_gids)
                {
                    // INFO() << "generating AIPUModel with gid = " << gid;
                    auto one_model = std::make_shared<AIPUModel>(ctx, gid);
                    m_models.push_back(one_model);
                }
            }
            INFO() << "load graph success  with aipu_bin = " << filename;
        }
    }
    MultiAIPUModels::~MultiAIPUModels()
    {
        m_gids.clear();
        m_models.clear();
    }

    AIPUModelPtr MultiAIPUModels::operator[](int32_t index)
    {
        if (index < 0)
        {
            index = m_models.size() + index;
        }
        aipu_assert(index < static_cast<int>(m_models.size()));
        return m_models[index];
    }

    std::vector<AIPUModelPtr>::iterator MultiAIPUModels::begin()
    {
        return m_models.begin();
    }
    std::vector<AIPUModelPtr>::iterator MultiAIPUModels::end()
    {
        return m_models.end();
    }

    std::vector<AIPUModelPtr>::const_iterator MultiAIPUModels::begin() const
    {
        return m_models.begin();
    }
    std::vector<AIPUModelPtr>::const_iterator MultiAIPUModels::end() const
    {
        return m_models.end();
    }
    const std::string MultiAIPUModels::to_string(void) const
    {
        std::ostringstream oss;
        oss << "MultiAIPUModels's graph info are: \n";
        if (m_models.size() > 0)
        {
            for (uint32_t i = 0; i < m_models.size(); ++i)
            {
                auto m = m_models[i];
                oss << m->to_string();
                oss << "\n";
            }
        }
        return oss.str();
    }

    AIPUModel::AIPUModel(const std::string &filename, AIPUContextPtr ctx)
    {
        if (!ctx)
        {
            ctx = AIPUContext::get_default();
        }
        m_ctx = ctx;
        m_filename = filename;
        auto handle = (aipu_ctx_handle_t *)(m_ctx->get_handle());
        m_status = handle != nullptr;
        m_is_ds = false;
        if (m_status)
        {
            auto driver_extra_weight_cfg = new aipu_load_graph_cfg_t{0};
            std::string extra_wt_path = utils::dir_name(filename);
            driver_extra_weight_cfg->extra_weight_path = extra_wt_path.c_str();
            auto ret = aipu_load_graph(handle, filename.c_str(), &m_gid, driver_extra_weight_cfg);
            if (ret != AIPU_STATUS_SUCCESS)
            {
                const char *msg = nullptr;
                aipu_get_error_message(handle, ret, &msg);
                ERROR("aipu_load_graph_helper: %s (%s)\n",
                      msg, filename.c_str());
                m_status = false;
            }
            // INFO() << "aipu_load_graph success  with aipu_bin = " << filename;
        }

        if (m_status)
        {
            get_model_basic_info();
        }
    }
    const void AIPUModel::get_model_basic_info()
    {
        if (!m_ctx)
        {
            ERROR() << "m_ctx should be not null when get_model_basic_info.";
        }

        // get inputs/outputs
        {
            m_inputs.clear();
            uint32_t input_cnt;
            auto ret = aipu_get_tensor_count((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, AIPU_TENSOR_TYPE_INPUT, &input_cnt);
            has_error(ret, "get_tensor_cnt");
            for (uint32_t i = 0; i < input_cnt; i++)
            {
                aipu_tensor_desc_t desc;
                ret = aipu_get_tensor_descriptor((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, AIPU_TENSOR_TYPE_INPUT, i, &desc);
                has_error(ret, "get_tensor_desc");
                m_inputs.push_back(desc_to_tensor(desc));
            }

            m_outputs.clear();
            uint32_t output_cnt;
            ret = aipu_get_tensor_count((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, AIPU_TENSOR_TYPE_OUTPUT, &output_cnt);
            has_error(ret, "get_tensor_cnt");
            for (uint32_t i = 0; i < output_cnt; i++)
            {
                aipu_tensor_desc_t desc;
                ret = aipu_get_tensor_descriptor((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, AIPU_TENSOR_TYPE_OUTPUT, i, &desc);
                has_error(ret, "get_tensor_desc");
                m_outputs.push_back(desc_to_tensor(desc));
            }
            m_profdata.clear();
            uint32_t prof_cnt;
            ret = aipu_get_tensor_count((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, AIPU_TENSOR_TYPE_PROFILER, &prof_cnt);
            has_error(ret, "get_profile_data_count");
            for (uint32_t i = 0; i < prof_cnt; i++)
            {
                aipu_tensor_desc_t prof;
                ret = aipu_get_tensor_descriptor((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, AIPU_TENSOR_TYPE_PROFILER, i, &prof);
                has_error(ret, "get_profile_data");
                m_profdata.push_back(desc_to_tensor(prof));
            }
        }

        // get shape constrains
        {
            for (auto i = 0u; i < m_inputs.size(); ++i)
            {
                aipu_dynshape_dim_num_t dynshape_dim_num = {0};
                uint32_t dim_num = 0;
                dynshape_dim_num.graph_id = m_gid;
                dynshape_dim_num.ds_idx = i;
                dynshape_dim_num.max_threshhold = true;
                dynshape_dim_num.ds_dim_num = &dim_num;
                auto ret = m_ctx->ioctl(AIPU_IOCTL_GET_DS_DIM_NUM, &dynshape_dim_num);
                has_error(ret, "AIPU_IOCTL_GET_DS_DIM_NUM");
                if (dim_num == 0)
                {
                    m_is_ds = false;
                    break;
                }
                m_is_ds = true;
                std::vector<int32_t> shape(dim_num);
                aipu_dynshape_info_t dynshape_info = {0};
                dynshape_info.graph_id = m_gid;
                dynshape_info.ds_idx = i;
                dynshape_info.max_threshhold = true;
                dynshape_info.ds_data = (uint32_t *)&shape[0];
                ret = m_ctx->ioctl(AIPU_IOCTL_GET_DS_INFO, &dynshape_info);
                has_error(ret, "AIPU_IOCTL_GET_DS_INFO");
                m_max_inps.emplace_back(shape);
                m_inputs[i]->shape().reshape(shape);

                dynshape_info.max_threshhold = false;
                dynshape_info.ds_data = (uint32_t *)&shape[0];
                ret = m_ctx->ioctl(AIPU_IOCTL_GET_DS_INFO, &dynshape_info);
                has_error(ret, "AIPU_IOCTL_GET_DS_INFO");
                m_min_inps.emplace_back(shape);
            }
        }
    }

    AIPUModel::~AIPUModel()
    {
        if (!m_status)
        {
            return;
        }
        const char *msg = nullptr;
        auto ret = aipu_unload_graph((aipu_ctx_handle_t *)(m_ctx->get_handle()), m_gid);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            aipu_get_error_message((aipu_ctx_handle_t *)(m_ctx->get_handle()), ret, &msg);
            ERROR("aipu_unload_graph: %s\n", msg);
        }
    }

    AIPUContextPtr AIPUModel::context()
    {
        return m_ctx;
    }

    uint64_t AIPUModel::gid()
    {
        return m_gid;
    }

    uint64_t AIPUModel::gid() const
    {
        return m_gid;
    }

    const std::vector<TensorPtr> &AIPUModel::get_outputs() const
    {
        return m_outputs;
    }
    const std::vector<TensorPtr> &AIPUModel::get_inputs() const
    {

        return m_inputs;
    }
    const std::string &AIPUModel::get_profiling_info() const
    {
        if (m_profdata.size() && m_profiling.empty())
        {
            uint64_t note_size = 0;

            auto ret = aipu_get_elf_note_size((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, "graphjson", &note_size);
            has_error(ret, "aipu_get_elf_note_size");
            if (note_size)
            {
                char *data = new char[note_size];
                ret = aipu_read_elf_note((aipu_ctx_handle_t *)m_ctx->get_handle(), m_gid, "graphjson", data, &note_size);
                if (has_error(ret, "aipu_read_elf_note"))
                {
                    delete[] data;
                }
                else
                {
                    m_profiling = data;
                    delete[] data;
                }
            }
        }
        return m_profiling;
    }

    const std::vector<TensorPtr> &AIPUModel::get_profdata() const
    {
        return m_profdata;
    }
    inline bool AIPUModel::has_error(uint32_t status, const std::string &where) const
    {
        return m_ctx->has_error(status, where);
    }

    const std::string AIPUModel::to_string(void) const
    {
        std::ostringstream oss;
        oss << "model(gid=" << gid() << "): ";
        oss << "the number of input is " << m_inputs.size();
        oss << ", the number of output is " << m_outputs.size();
        oss << ", and is_dynamic = " << is_dynamic();
        oss << ";";
        return oss.str();
    }

    AIPUJob::AIPUJob(AIPUModelPtr graph) : m_graph(graph) {}

    void AIPUJob::set_dump(bool d)
    {
        m_dump = d;
    }
    void AIPUJob::set_output_prefix(const std::string &p)
    {
        m_out_prefix = p;
    }

    AIPUJob::AIPUJob(const std::string &graph_file)
    {
        m_graph = std::make_shared<AIPUModel>(graph_file);
    }
    AIPUJob::~AIPUJob()
    {
        if (is_init)
        {
            clean_job();
        }
    }

    void AIPUJob::clean_job()
    {
        auto ret = aipu_clean_job((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid);
        has_error(ret, "clean job");
    }

    void AIPUJob::set_inputs(const std::vector<TensorPtr> &inputs)
    {
        auto &m_inputs = m_graph->get_inputs();
        if (!m_graph->is_dynamic() && inputs.size() != m_inputs.size())
        {
            ERROR() << "set inputs failed, the inputs num does not match: expected " << m_inputs.size() << " but got " << inputs.size();
            return;
        }
        for (auto i = 0u; i < inputs.size(); ++i)
        {
            auto inp = inputs[i];
            if (!inp->data<void>() ||
                m_inputs[i]->type() != inp->type() ||
                (!m_graph->is_dynamic() && m_inputs[i]->mem_size() < inp->mem_size()))
            {
                ERROR() << "set inputs failed, the input has no data or dtype/size not match.";
            }
            if (m_inputs[i]->mem_size() == inp->mem_size())
            {
                m_inputs[i]->shape() = inp->shape();
            }
            if (inp->attrs().count("_dev"))
            {
                auto dev = inp->get_attr<DevMemObj>("_dev");
                aipu_shared_tensor_info_t zerocpy;
                zerocpy.tensor_idx = i;
                zerocpy.type = AIPU_TENSOR_TYPE_INPUT;
                zerocpy.dmabuf_fd = dev.buf_fd;
                zerocpy.offset_in_dmabuf = 0; // TODO support shadow buf later
                zerocpy.shared_case_type = AIPU_SHARE_BUF_ATTACH_DMABUF;
                /**
                 * bind dma_buf to IO buffer after the job is created.
                 */
                auto ret = aipu_specify_iobuf((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, &zerocpy);
                has_error(ret, "map_input_tensor");
            }
            else
            {
                auto ret = aipu_load_tensor((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, i, inp->data<void>());
                has_error(ret, "upload_tensor");
            }
        }
    }

    const std::vector<TensorPtr> AIPUJob::get_outputs()
    {
        auto outputs = m_graph->get_outputs();
        std::vector<TensorPtr> ret_outputs;
        bool ret;
        for (uint32_t i = 0; i < outputs.size(); i++)
        {
            auto data_size = outputs[i]->mem_size();
            auto output_shape = outputs[i]->shape();
            if (m_graph->is_dynamic())
            {
                aipu_tensor_desc_t desc;
                ret = aipu_get_tensor_descriptor((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, AIPU_TENSOR_TYPE_OUTPUT, i, &desc);
                has_error(ret, "get_output_tensor_desc_" + std::to_string(i));
                data_size = desc.size;

                ret = aipu_get_tensor_descriptor((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, AIPU_TENSOR_TYPE_OUT_TENSOR_SHAPE, i, &desc);
                has_error(ret, "get_shape_tensor_desc_" + std::to_string(i));
                auto shape_tensor = desc_to_tensor(desc);

                char *shape_data = new char[desc.size];
                // INFO() << "shape tensor size:" << desc.size;
                auto ret = aipu_get_tensor((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, AIPU_TENSOR_TYPE_OUT_TENSOR_SHAPE, i, shape_data);
                has_error(ret, "get_shape_tensor_data_" + std::to_string(i));
                shape_tensor->set_data(shape_data, desc.size);
                shape_tensor->set_type(aipubt::Tensor::INT32);
                shape_tensor->shape() = TensorShape(int(desc.size / sizeof(int32_t)));
                ComputeTensor shape_ct = ComputeTensor(shape_tensor);
                // INFO() << "shape tensor shape:" << shape_ct;
                std::vector<int32_t> shape_vt = shape_ct;
                output_shape = TensorShape(shape_vt);
            }

            char *data = new char[data_size];
            auto s = aipu_get_tensor((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, AIPU_TENSOR_TYPE_OUTPUT, i, data);
            has_error(s, "get_output_tensor_" + std::to_string(i));
            auto out = std::make_shared<Tensor>(*outputs[i]);
            out->set_data(data, data_size);
            out->shape() = output_shape;
            ret_outputs.push_back(out);
        }
        return ret_outputs;
    }

    const std::vector<TensorPtr> AIPUJob::get_profdata()
    {
        auto prof = m_graph->get_profdata();
        std::vector<TensorPtr> ret;
        for (uint32_t i = 0; i < prof.size(); i++)
        {
            char *data = new char[prof[i]->mem_size()];
            auto s = aipu_get_tensor((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, AIPU_TENSOR_TYPE_PROFILER, i, data);
            has_error(s, "get profiler tensor with jod = " + std::to_string(m_jid));
            auto out = std::make_shared<Tensor>(*prof[i]);
            out->set_data(data, prof[i]->mem_size());
            ret.push_back(out);
        }
        return ret;
    }
    const std::vector<TensorPtr> AIPUJob::run(const std::vector<TensorPtr> &inputs)
    {
        if (!m_graph)
        {
            return std::vector<TensorPtr>();
        }
        if (m_graph->is_dynamic())
        {
            ds_init_and_set_input(inputs);
        }
        else
        {
            init();
            set_inputs(inputs);
        }
        auto ret = aipu_finish_job((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, -1);
        has_error(ret, "finish job");
        auto outputs = get_outputs();

        if (m_graph->is_dynamic())
        {
            clean_job();
        }
        return outputs;
    }

    void AIPUJob::get_outputs(const std::vector<TensorPtr> &outputs)
    {
        if (!m_graph)
        {
            return;
        }
        auto outs = get_outputs();
        if (outs.size() != outputs.size())
        {
            ERROR() << "get outputs failed, outputs num does not match.";
        }
        for (auto i = 0u; i < outs.size(); ++i)
        {
            if (outputs[i]->mem_size() != outs[i]->mem_size())
            {
                ERROR() << "get outputs failed, outputs size does not match.";
            }
            outputs[i]->share_data_from(outs[i]);
        }
    }

    void AIPUJob::init()
    {
        if (is_init)
        {
            return;
        }
        is_init = true;
        aipu_create_job_cfg_t cfg = {0};
        auto ret = aipu_create_job((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_graph->gid(), &m_jid, &cfg);
        has_error(ret, "create job");
        if (m_dump)
        {
            aipu_job_config_dump_t mem_dump_config;
            memset(&mem_dump_config, 0, sizeof(mem_dump_config));
            mem_dump_config.dump_dir = ".";
            mem_dump_config.output_prefix = m_out_prefix.c_str();
            ret = aipu_config_job((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, AIPU_JOB_CONFIG_TYPE_DUMP_EMULATION, &mem_dump_config);
            has_error(ret, "config job");
        }
    }
    void AIPUJob::ds_init_and_set_input(const std::vector<TensorPtr> &inputs)
    {
        aipu_create_job_cfg_t cfg = {0};
        if (inputs.size() != (m_graph->get_inputs()).size())
        {
            ERROR() << "set inputs failed, the inputs num does not match: expected " << m_graph->get_inputs().size() << " but got " << inputs.size();
            return;
        }
        aipu_dynshape_param_t dynshape_param;
        dynshape_param.input_shape_cnt = inputs.size();
        dynshape_param.shape_items = new aipu_dynshape_item_t[inputs.size()];
        for (uint32_t i = 0; i < inputs.size(); i++)
        {
            dynshape_param.shape_items[i].ds_idx = i;
            std::vector<uint32_t> inp_shape = inputs[i]->shape();
            uint32_t *shape = new uint32_t[inp_shape.size()];
            memcpy(shape, inp_shape.data(), inp_shape.size() * sizeof(uint32_t));
            dynshape_param.shape_items[i].ds_data = shape;
        }
        cfg.dynshape = &dynshape_param;
        auto ret = aipu_create_job((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_graph->gid(), &m_jid, &cfg);
        has_error(ret, "create aipu job");

        for (size_t i = 0; i < inputs.size(); i++)
        {
            auto ret = aipu_load_tensor((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, i, inputs[i]->data<void>());
            has_error(ret, "upload_tensor " + std::to_string(i));
        }
        // if (m_dump)
        // {
        //     aipu_job_config_dump_t mem_dump_config;
        //     memset(&mem_dump_config, 0, sizeof(mem_dump_config));
        //     mem_dump_config.dump_dir = ".";
        //     mem_dump_config.output_prefix = m_out_prefix.c_str();
        //     ret = aipu_config_job((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, AIPU_JOB_CONFIG_TYPE_DUMP_EMULATION, &mem_dump_config);
        //     has_error(ret, "config job");
        // }
    }
    inline bool AIPUJob::has_error(uint32_t status, const std::string &where) const
    {
        return m_graph->has_error(status, where);
    }

    void AIPUJob::rewrite_ro(uint32_t offset, uint32_t value)
    {
        auto ret = aipu_rewrite_rodata((aipu_ctx_handle_t *)m_graph->context()->get_handle(), m_jid, offset, value);
        has_error(ret, "rewrtie ro");
    }

    static inline bool has_error(uint32_t status, const std::string &where)
    {
        return AIPUContext::get_default()->has_error(status, where);
    }
    std::string get_device_version()
    {
        char target[20] = {'\0'};
        auto ret = aipu_get_target((aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle(), target);
        has_error(ret, "get target");
        return target;
    }
    uint32_t get_partion_number()
    {
        uint32_t partition_count;
        auto ret = aipu_get_partition_count((aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle(), &partition_count);
        has_error(ret, "get partition count");
        return partition_count;
    }

    uint32_t get_cluster_number(uint32_t partion_id)
    {
        uint32_t cluster_count;
        auto ret = aipu_get_cluster_count((aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle(), partion_id, &cluster_count);
        has_error(ret, "get cluster count");
        return cluster_count;
    }

    uint32_t get_core_number(uint32_t partion_id, uint32_t cluster_id)
    {
        uint32_t core_count;
        auto ret = aipu_get_core_count((aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle(), partion_id, cluster_id, &core_count);
        has_error(ret, "get core count");
        return core_count;
    }

    uint32_t get_device_status()
    {
        device_status_t device_status;
        auto ret = aipu_get_device_status((aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle(), &device_status);
        has_error(ret, "get device status");
        return device_status;
    }

    AIPUCap get_aipu_cap()
    {
        auto fd = open(DEV_PATH, O_RDWR | O_SYNC);
        if (fd <= 0)
        {
            ERROR() << "open " << DEV_PATH << " failed, make sure the driver is well installed and the device is connected.";
            return AIPUCap();
        }

        aipu_cap cap;
        auto res = ioctl(fd, AIPU_IOCTL_QUERY_CAP, &cap);
        if (res || (cap.partition_cnt == 0))
        {
            ERROR() << "query capability [fail]";
            return AIPUCap();
        }
        close(fd);
        auto ret = AIPUCap();
        ret.partition_cnt = cap.partition_cnt;
        ret.asid_base.resize(cap.asid_cnt);
        for (auto i = 0u; i < cap.asid_cnt; ++i)
        {
            ret.asid_base[i] = cap.asid_base[i];
        }
        ret.dtcm_base = cap.dtcm_base;
        ret.dtcm_size = cap.dtcm_size;
        ret.gm0_size = cap.gm0_size;
        ret.gm1_size = cap.gm1_size;
        ret.is_homogeneous = cap.is_homogeneous;
        auto part_cap = cap.partition_cap;
        ret.partition_cap.id = part_cap.id;
        ret.partition_cap.arch = part_cap.arch;
        ret.partition_cap.version = part_cap.version;
        ret.partition_cap.config = part_cap.config;
        ret.partition_cap.reg_base = part_cap.info.reg_base;
        ret.partition_cap.clusters.resize(part_cap.cluster_cnt);
        for (auto i = 0u; i < part_cap.cluster_cnt; ++i)
        {
            ret.partition_cap.clusters[i] = {part_cap.clusters[i].core_cnt, part_cap.clusters[i].en_core_cnt, part_cap.clusters[i].tec_cnt};
        }
        return ret;
    }

    bool device_soft_reset()
    {
        auto ctx = (aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle();
        INFO() << "device soft reset";
        auto ret = aipu_ioctl(ctx, AIPU_IOCTL_SW_RESET, nullptr);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            ERROR() << "reset fail";
            return false;
        }
        return true;
    }

    bool enable_tick_counter()
    {
        auto ctx = (aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle();
        auto ret = aipu_ioctl(ctx, AIPU_IOCTL_ENABLE_TICKCOUNTER, nullptr);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            ERROR() << "enable tick counter fail";
            return false;
        }
        return true;
    }

    bool disable_tick_counter()
    {
        auto ctx = (aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle();
        auto ret = aipu_ioctl(ctx, AIPU_IOCTL_DISABLE_TICKCOUNTER, nullptr);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            ERROR() << "disable tick counter fail";
            return false;
        }
        return true;
    }

    const std::string &get_driver_version()
    {
        static std::string version = "Unkown Version";
        static aipu_driver_version_t drv_ver = {0};
        if (drv_ver.kmd_version[0] != 0)
        {
            return version;
        }
        auto ctx = (aipu_ctx_handle_t *)AIPUContext::get_default()->get_handle();
        auto ret = aipu_ioctl(ctx, AIPU_IOCTL_GET_VERSION, &drv_ver);
        if (ret != AIPU_STATUS_SUCCESS)
        {
            ERROR() << "get_driver version fail";
        }
        else
        {
            version = std::string(drv_ver.kmd_version);
        }
        return version;
    }

    AIPUClusterStatus get_cluster_status(uint32_t cluster_id)
    {
        auto fd = open(DEV_PATH, O_RDWR | O_SYNC);
        AIPUClusterStatus ret = {0};
        if (fd <= 0)
        {
            ERROR() << "open " << DEV_PATH << " failed, make sure the driver is well installed and the device is connected.";
            return ret;
        }

        aipu_cluster_status status;
        status.cluster_id = cluster_id;
        auto res = ioctl(fd, AIPU_IOCTL_GET_CLUSTER_STATUS, &status);
        if (res)
        {
            ERROR() << "get cluster status [fail]";
        }
        close(fd);
        ret.cluster_status = status.cluster_status;
        ret.core_status = status.core_status;
        return ret;
    }

    std::vector<uint32_t> get_running_job_thread_id()
    {
        auto fd = open(DEV_PATH, O_RDWR | O_SYNC);
        std::vector<uint32_t> ret;
        if (fd <= 0)
        {
            ERROR() << "open " << DEV_PATH << " failed, make sure the driver is well installed and the device is connected.";
            return ret;
        }

        aipu_running_job_query jobs = {0};
        auto res = ioctl(fd, AIPU_IOCTL_GET_RUNNING_JOB_THREAD_ID, &jobs);
        if (res)
        {
            ERROR() << "get cluster status [fail]";
        }
        close(fd);
        for (auto i = 0; i < 32; ++i)
        {
            if (jobs.thread_id[i] != 0)
            {
                ret.push_back(jobs.thread_id[i]);
            }
        }
        return ret;
    }

    struct aipu_dma_buf_request
    {
        int fd;
        uint64_t bytes;
    };

    struct aipu_dma_buf
    {
        int fd;
        uint64_t pa;
        uint64_t bytes;
    };
    DevMemObj dev_allocate(uint64_t size)
    {
        int ret = 0;
        struct aipu_dma_buf_request dma_buf_req = {0};
        struct aipu_dma_buf dma_buf = {0};
        auto fd = open(DEV_PATH, O_RDWR);
        if (fd < 0)
        {
            ERROR() << "open " << DEV_PATH << " [fail]";
            close(fd);
            return DevMemObj();
        }
        dma_buf_req.bytes = size;
        ret = ioctl(fd, AIPU_IOCTL_ALLOC_DMA_BUF, &dma_buf_req);
        if (ret < 0)
        {
            close(fd);
            ERROR() << "ioctl alloc" << DEV_PATH << " [fail]";
            return DevMemObj();
        }
        dma_buf.fd = dma_buf_req.fd;
        ret = ioctl(fd, AIPU_IOCTL_GET_DMA_BUF_INFO, &dma_buf);
        if (ret < 0)
        {
            close(fd);
            ERROR() << "ioctl get" << DEV_PATH << " [fail]";
            return DevMemObj();
        }
        close(fd);
        return {dma_buf.fd, dma_buf.pa, 0, size};
    }
    void dev_free(DevMemObj &obj)
    {
        auto fd = open(DEV_PATH, O_RDWR);
        if (fd < 0)
        {
            ERROR() << "open " << DEV_PATH << " [fail]\n";
            close(fd);
            return;
        }
        auto ret = ioctl(fd, AIPU_IOCTL_FREE_DMA_BUF, &obj.buf_fd);
        if (ret < 0)
        {
            close(fd);
            ERROR() << "ioctl free" << DEV_PATH << " [fail]\n";
            return;
        }
        close(fd);
        obj.buf_fd = 0;
    }
    void *dev_mmap(DevMemObj &obj)
    {
        obj.va = mmap(NULL, obj.size, PROT_READ | PROT_WRITE, MAP_SHARED, obj.buf_fd, 0);
        if (obj.va == MAP_FAILED)
        {
            ERROR() << "mmap dmabuf [fail]";
            return 0;
        }

        return (void *)obj.va;
    }
    void dev_munmap(DevMemObj &obj)
    {
        munmap((void *)obj.va, obj.size);
        obj.va = 0;
    }

    DevMemObj::DevMemObj(uint64_t size)
    {
        *this = dev_allocate(size);
        dev_mmap(*this);
    }
    DevMemObj::~DevMemObj()
    {
        if (buf_fd != 0)
        {
            if (va != 0)
            {
                dev_munmap(*this);
                va = 0;
            }
            dev_free(*this);
            buf_fd = 0;
            size = 0;
        }
    }

} // namespace AIPUDriverWrapper

} // namespace aipubt