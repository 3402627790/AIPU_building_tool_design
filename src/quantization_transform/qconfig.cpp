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
 * @Author: Samuel Han
 * @Date: 2026-07-08 02:08:39
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2026-07-20 07:34:04
 */

#include "quantization_transform/qconfig.h"

namespace qtlibn
{

QuantizationConfig::~QuantizationConfig() = default;

QuantizationConfig::QuantizationConfig(const QuantizationConfig &other) = default;
QuantizationConfig &QuantizationConfig::operator=(const QuantizationConfig &other) = default;
QuantizationConfig::QuantizationConfig(QuantizationConfig &&other) noexcept = default;
QuantizationConfig &QuantizationConfig::operator=(QuantizationConfig &&other) noexcept = default;

void QuantizationConfig::set_default_activation_quantization(const aipubt::Quantization &q)
{
    m_default_activation = q;
}

const aipubt::Quantization &QuantizationConfig::default_activation_quantization() const
{
    return m_default_activation;
}

void QuantizationConfig::set_default_weight_quantization(const aipubt::Quantization &q)
{
    m_default_weight = q;
}

const aipubt::Quantization &QuantizationConfig::default_weight_quantization() const
{
    return m_default_weight;
}

void QuantizationConfig::set_default_bias_quantization(const aipubt::Quantization &q)
{
    m_default_bias = q;
}

const aipubt::Quantization &QuantizationConfig::default_bias_quantization() const
{
    return m_default_bias;
}

// ---- Per-layer-type customization ----
void QuantizationConfig::set_layer_quantization(
    const std::string &layer_type,
    const aipubt::Quantization &activation,
    const aipubt::Quantization &weight)
{
    if (layer_type.empty())
    {
        ERROR() << "QuantizationConfig::set_layer_quantization: layer_type must not be empty";
    }
    m_layer_configs[layer_type] = LayerQuantization{activation, weight};
}

bool QuantizationConfig::remove_layer_quantization(const std::string &layer_type)
{
    return m_layer_configs.erase(layer_type) > 0;
}

bool QuantizationConfig::has_layer_quantization(const std::string &layer_type) const
{
    return m_layer_configs.find(layer_type) != m_layer_configs.end();
}

// ---- Resolution (lookup) ----
const aipubt::Quantization &QuantizationConfig::get_activation_quantization(const std::string &layer_type) const
{
    auto it = m_layer_configs.find(layer_type);
    if (it != m_layer_configs.end())
    {
        return it->second.activation;
    }
    return m_default_activation;
}

const aipubt::Quantization &QuantizationConfig::get_weight_quantization(const std::string &layer_type) const
{
    auto it = m_layer_configs.find(layer_type);
    if (it != m_layer_configs.end())
    {
        return it->second.weight;
    }
    return m_default_weight;
}

// ---- Node-level filter ----

void QuantizationConfig::set_filter_fn(NodeFilter fn)
{
    m_filter_fn = std::move(fn);
}

const NodeFilter &QuantizationConfig::filter_fn() const
{
    return m_filter_fn;
}

bool QuantizationConfig::should_quantize(const aipubt::NodePtr &node) const
{
    if (!m_filter_fn)
    {
        return true; // no filter → quantize all
    }
    return m_filter_fn(node);
}

bool QuantizationConfig::generate_bias_quantization(aipubt::Quantization act_q)
{
    /* bias quantization is aligned with activation quantization.
    * if activation dtype is int8, bias dtype is int32
    * if activation dtype is int16, bias dtype is int48
    * if activation dtype is float(fp16/bfp16/fp8), bias dtype is fp32
    */
    bool ret = true;

    if (act_q != m_default_activation)
    {
        WARN() << "when generate bias quantization, the configured quantization is not equal to default activation quantization.";
    }
    if (m_default_bias.has_update())
    {
        return ret;
    }

    if (aipubt::Tensor::is_float(act_q.dtype()))
    {
        m_default_bias.set_dtype(aipubt::Tensor::FP32).set_bits(32);
    }
    else if (act_q.dtype() == aipubt::Tensor::INT8)
    {
        m_default_bias.set_dtype(aipubt::Tensor::INT32).set_bits(32);
    }
    else if (act_q.dtype() == aipubt::Tensor::INT16)
    {
        m_default_bias.set_dtype(aipubt::Tensor::INT64).set_bits(48);
    }
    else
    {
        ERROR() << "unknown quantization dtype in configured quantization";
        ret = false;
    }
    return ret;
}

void QuantizationConfig::sync_quantization(aipubt::TensorPtr t)
{
    auto &q = t->quantization();
    if (q.dtype() == aipubt::Tensor::VOID)
    {
        auto t_dtype = t->type();
        // if (is_unquantized_dtype(t_dtype) && (!aipubt::Tensor::is_float(t_dtype)))
        // {
        //     WARN() << "tensor = " << t
        //            << "'s quantization.dtype is VOID when sync_quantization, but tensor.dtype = "
        //            << t_dtype << " which is qinvariant dtype, so using tensor dtype to set quantization dtype.";
        //     // q.set_dtype(t_dtype);
        // }
        if (!is_unquantized_dtype(t_dtype))
        {
            std::ostringstream oss;
            oss << "tensor = '" << t << "'";
            if (t->node())
            {
                oss << "(its node.type = " << t->node()->type() << ")";
            }
            oss << "'s quantization.dtype is VOID when sync_quantization, but tensor.dtype = "
                << t_dtype << ", which is quantized dtype, so please set quantization dtype firstly.";
            WARN() << oss.str();
            return;
        }
        else
        {
            DEBUG() << "tensor = " << t
                    << "'s quantization.dtype is VOID when sync_quantization, and tensor.dtype = "
                    << t_dtype << ", qtlib will automatically deduce quantization dtype.";
            return;
        }
    }

    if (!is_unquantized_dtype(q.dtype()))
    {
        auto bits = q.bits();
        if (bits == -1)
        {
            q.set_bits(aipubt::Tensor::dtype_actual_bits(q.dtype()));
        }

        // TODO: add more default value for quantization.
    }
}

bool QuantizationConfig::apply(aipubt::NodePtr node, const bool overwrite_quantization)
{
    if (m_filter_fn)
    {
        bool ret = filter_fn()(node);
        if (ret)
        {
            return ret;
        }
    }

    // layer type specification configuration, if not find, will return the default_activation
    auto act_q = get_activation_quantization(aipubt::optype_to_string(node->type()));
    // TODO : layer name customized configuration

    for (auto &ot : node->outputs())
    {
        if (ot->quantization().has_update() && !overwrite_quantization)
        {
            sync_quantization(ot);
            continue;
        }

        ot->quantization() = act_q;
        INFO() << "applying QuantizationConfig to tensor = " << ot;
    }

    for (auto &kv : node->constants())
    {
        if (kv.second->quantization().has_update() && !overwrite_quantization)
        {
            sync_quantization(kv.second);
            continue;
        }
        if (kv.first == "weights")
        {
            auto wht_q = get_weight_quantization(aipubt::optype_to_string(node->type()));
            kv.second->quantization() = wht_q;
        }
        else if (kv.first == "biases")
        {
            generate_bias_quantization(act_q);
            auto bs_q = default_bias_quantization();
            kv.second->quantization() = bs_q;
        }
        else
        {
            kv.second->quantization() = act_q;
        }
    }

    for (auto &ot : node->placeholders())
    {
        if (ot->quantization().has_update() && !overwrite_quantization)
        {
            sync_quantization(ot);
            continue;
        }
        ot->quantization() = act_q;
    }

    return true;
}

// ---- Utility ----

std::string QuantizationConfig::to_string() const
{
    std::ostringstream oss;
    oss << "QuantizationConfig {"
        << "\n  default_activation: " << m_default_activation
        << "\n  default_weight:     " << m_default_weight
        << "\n  has_filter:         " << (m_filter_fn ? "true" : "false")
        << "\n  per_layer_configs (" << m_layer_configs.size() << "):\n";

    for (const auto &kv : m_layer_configs)
    {
        oss << "    [" << kv.first << "]\n"
            << "      activation: " << kv.second.activation << "\n"
            << "      weight:     " << kv.second.weight << "\n";
    }
    oss << "}";
    return oss.str();
}

std::ostream &operator<<(std::ostream &os, const QuantizationConfig &qconfig)
{
    os << qconfig.to_string();
    return os;
}

namespace
{

    // Helper: apply override if key exists — uses NodeParamValue's implicit conversions
    template <typename T>
    void get_params(T &target, const std::string &key, const NodeParamDict &ov)
    {
        auto it = ov.find(key);
        if (it != ov.end())
        {
            target = static_cast<T>(it->second);
        }
    }

} // anonymous namespace

#ifdef TEST

std::string qtype_to_string(const QuantType::QType &type)
{
    switch (type)
    {
#define X(name)               \
    case QuantizedType::name: \
        return #name;
        FOR_EACH_QUANTIZED_TYPE(X)
#undef X
    }
    return Constants::unknown;
}
std::string qconfig_type_to_string(const QConfigType &type)
{
    switch (type)
    {
#define X(name)                      \
    case QConfigType::name##Qconfig: \
        return #name "Qconfig";
        FOR_EACH_QUANTIZED_TYPE(X)
#undef X
    }
    return Constants::unknown;
}

QType string_to_qtype(const std::string &s)
{
    std::string lower;
    lower.reserve(s.size());
    std::transform(s.begin(), s.end(), std::back_inserter(lower),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Match member names directly (case-insensitive)
#define X(name)         \
    if (lower == #name) \
        return QType::name;
    FOR_EACH_QUANTIZED_TYPE(X)
#undef X
    ERROR() << "Unknown QType: " << s;
    return QType::NONE;
}

QConfigType string_to_qconfig_type(const std::string &s)
{
    std::string lower;
    lower.reserve(s.size());
    std::transform(s.begin(), s.end(), std::back_inserter(lower),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Match member names directly (case-insensitive)
#define X(name)         \
    if (lower == #name) \
        return QConfigType::name;
    FOR_EACH_QUANTIZED_TYPE(X)
#undef X

    ERROR() << "Unknown QConfigType: " << s;
    return QConfigType::NONEQConfig;
}
#endif

// ============================================================================
// Factory implementation
// ============================================================================

void make_fp16_activation(aipubt::Quantization &act, std::string dtype, bool is_perf_mode)
{
    act.set_bits(16).set_dtype(dtype).set_lut_items_bits(9);
    act.extra_info()["is_perf_mode"] = is_perf_mode;
}
void make_fp_quantization(
    QuantizationConfig &cfg,
    const NodeParamDict &ov,
    const QuantType::QType qtype)
{
    aipubt::Quantization act_q;
    bool is_perf_mode = true;
    std::string act_dtype = QuantType::activation_dtype(qtype);
    get_params(is_perf_mode, "is_perf_mode", ov);

    aipubt::Quantization wht_q;
    std::string wht_dtype = QuantType::weight_dtype(qtype);

    make_fp16_activation(act_q, act_dtype, is_perf_mode);
    make_fp16_activation(wht_q, wht_dtype, false);
    cfg.set_default_activation_quantization(act_q);
    cfg.set_default_weight_quantization(wht_q);
}

void make_weight_only_quantization(
    QuantizationConfig &cfg,
    const NodeParamDict &ov,
    const QuantType::QType qtype)
{
    aipubt::Quantization act_q;
    bool is_perf_mode = true;
    std::string act_dtype = QuantType::activation_dtype(qtype);
    get_params(is_perf_mode, "is_perf_mode", ov);

    aipubt::Quantization wht_q;
    int32_t wht_bits = QuantType::weight_bits(qtype);
    std::string wht_dtype = QuantType::weight_dtype(qtype);

    // w8
    int32_t wht_block_size = 1;
    std::string wht_scheme = "symmetric";
    std::string wht_granularity = "per_axis";
    // w4 or wfp4
    if (wht_bits < 8)
    {
        wht_block_size = 128;
        wht_scheme = "asymmetric";
        wht_granularity = "per_block";
    }
    get_params(wht_block_size, "block_size", ov);
    get_params(wht_scheme, "weight_scheme", ov);
    get_params(wht_granularity, "weight_granularity", ov);

    make_fp16_activation(act_q, act_dtype, is_perf_mode);
    wht_q.set_bits(wht_bits)
        .set_dtype(wht_dtype)
        .set_block_size(wht_block_size)
        .set_scheme(wht_scheme)
        .set_axis(-1)
        .set_granularity(wht_granularity);
    cfg.set_default_activation_quantization(act_q);
    cfg.set_default_weight_quantization(wht_q);
}

void make_int_ptq_quantization(
    QuantizationConfig &cfg,
    const NodeParamDict &ov,
    const QuantType::QType qtype)
{
    aipubt::Quantization act_q;
    int32_t act_bits = QuantType::activation_bits(qtype);
    std::string act_dtype = QuantType::activation_dtype(qtype);

    aipubt::Quantization wht_q;
    int32_t wht_bits = QuantType::weight_bits(qtype);
    std::string wht_dtype = QuantType::weight_dtype(qtype);

    // a8
    std::string act_scheme = "asymmetric";
    std::string act_granularity = "per_tensor";
    get_params(act_scheme, "activation_scheme", ov);
    get_params(act_granularity, "activation_granularity", ov);
    // w8
    std::string wht_scheme = "symmetric";
    std::string wht_granularity = "per_axis";
    get_params(wht_scheme, "weight_scheme", ov);
    get_params(wht_granularity, "weight_granularity", ov);

    if (act_bits >= 16 && act_scheme == "asymmetric")
    {
        act_scheme = "symmetric";
        WARN() << "when generating int_ptq_quantization config, act_bits >=16, and scheme is asymmetric, we change the asymmetric to symmetric.";
    }

    if (wht_scheme == "asymmetric")
    {
        wht_scheme = "symmetric";
        WARN() << "when generating int_ptq_quantization config, weight scheme is asymmetric, we change the asymmetric to symmetric.";
    }

    act_q.set_bits(act_bits)
        // .set_dtype(act_dtype) // act dtype is automatically deduced by its ops quantization algo
        .set_scheme(act_scheme)
        .set_granularity(act_granularity);
    wht_q.set_bits(wht_bits)
        // .set_dtype(wht_dtype)
        .set_scheme(wht_scheme)
        .set_axis(0)
        .set_granularity(wht_granularity);
    cfg.set_default_activation_quantization(act_q);
    cfg.set_default_weight_quantization(wht_q);
}

QuantizationConfig QConfigFactory::create(QuantType::QType qtype)
{
    return create(qtype, NodeParamDict{});
}

QuantizationConfig QConfigFactory::create(QuantType::QType qtype, NodeParamDict ov)
{
    QuantizationConfig cfg;

    switch (qtype)
    {
        // ---- Weight-only presets -------------------------------------------

    case QuantType::QType::W4:
    case QuantType::QType::W4AFP16:
    case QuantType::QType::W4ABFP16:
    case QuantType::QType::WFP4:
    case QuantType::QType::WFP4AFP16:
    case QuantType::QType::WFP4ABFP16:
    case QuantType::QType::W8:
    case QuantType::QType::W8AFP16:
    case QuantType::QType::W8ABFP16:
    case QuantType::QType::WFP8:
    case QuantType::QType::WFP8AFP16:
    {
        make_weight_only_quantization(cfg, ov, qtype);
        break;
    }

    // ---- PTQ formats ---------------------------------------------------
    case QuantType::QType::W4AFP8E5M2:
    case QuantType::QType::W4AFP8E4M3:
    case QuantType::QType::WFP4AFP8E4M3:
    case QuantType::QType::WFP8E4M3AFP8E4M3:
    case QuantType::QType::WFP8E5M2AFP8E5M2:
    {
        // TODO: flesh out PTQ configs
        INFO() << qtype;
        break;
    }

    // // ---- Float formats -------------------------------------------------
    case QuantType::QType::FP16:
    case QuantType::QType::BFP16:
    case QuantType::QType::WBFP16ABFP16:
    case QuantType::QType::WFP16AFP16:
    {
        make_fp_quantization(cfg, ov, qtype);
        break;
    }

    // // ---- Full quantization ---------------------------------------------
    case QuantType::QType::W8A8:
    case QuantType::QType::W16A16:
    {
        make_int_ptq_quantization(cfg, ov, qtype);
        break;
    }

    // // ---- GPTQ / QAT ----------------------------------------------------
    case QuantType::QType::AWQQAT:
    case QuantType::QType::GPTQQAT:
    {
        INFO() << qtype;
        break;
    }

        // case QType::AWQQAT:
        //     // TODO: flesh out AWQ QAT config
        //     break;
        // }
    }

    return cfg;
}

QuantizationConfig QConfigFactory::create(const std::string &preset_name)
{
    return create(QuantType::parse(preset_name));
}

QuantizationConfig QConfigFactory::create(const std::string &preset_name, const NodeParamDict &overrides)
{
    return create(QuantType::parse(preset_name), overrides);
}

QuantizationConfig QConfigFactory::w4_qconfig(int block_size)
{
    NodeParamDict ov;
    ov["block_size"] = block_size;
    return create(QuantType::W4AFP16, ov);
}

QuantizationConfig QConfigFactory::wfp4_qconfig(int block_size)
{
    NodeParamDict ov;
    ov["block_size"] = block_size;
    return create(QuantType::WFP4AFP16, ov);
}

QuantizationConfig QConfigFactory::w8_qconfig()
{
    return create(QuantType::W8AFP16);
}

QuantizationConfig QConfigFactory::w8a8_qconfig(const std::string &act_scheme,
                                                const std::string &act_granu,
                                                const std::string &wht_scheme,
                                                const std::string &wht_granu)
{
    NodeParamDict ov;
    ov["weight_granularity"] = wht_granu;
    ov["weight_scheme"] = wht_scheme;
    ov["activation_granularity"] = act_granu;
    ov["activation_scheme"] = act_scheme;
    return create(QuantType::W8A8, ov);
}

QuantizationConfig QConfigFactory::w16a16_qconfig(const std::string &act_granu, const std::string &wht_granu)
{
    NodeParamDict ov;
    ov["weight_granularity"] = wht_granu;
    ov["activation_granularity"] = act_granu;
    return create(QuantType::W16A16, ov);
}

QuantizationConfig QConfigFactory::fp16_qconfig(const bool is_perf_mode)
{

    NodeParamDict ov;
    ov["is_perf_mode"] = is_perf_mode;
    return create(QuantType::WFP16AFP16, ov);
}

QuantizationConfig QConfigFactory::bfp16_qconfig(const bool is_perf_mode)
{
    NodeParamDict ov;
    ov["is_perf_mode"] = is_perf_mode;
    return create(QuantType::WBFP16ABFP16, ov);
}

} // namespace qtlibn
