#pragma once

#include "cps_plugin.h"

namespace aipubt
{
// usage of CpsPlugins see comment in cps_plugin.h
namespace CpsPlugins
{

};

// IsNanBaseImpl: one obj has many kernels
class IsNanBaseImpl : public GopBaseImpl
{
public:
    IsNanBaseImpl(const NodePtrList& nodes, const GPattern& pattern, const std::string& target)
        : GopBaseImpl(nodes, pattern, target)
    {
        _objname = "isnan";
    }

    KernelInfos kernel_infos() override = 0;

    /// optional: use for turning performance
    void update_prefer_cfg(const std::string& key, const Any& value) override;
    /// optional: implement score
    uint32_t score() const override;
    /// optional:  kernel cps bin file name
    const std::string kernel_cpsbin() override;
    /// optional : implement flops
    uint64_t flops() const override;

protected:
    const int nidx = 0;   ///< node index in nodes
    const int itidx = 0;  ///< input index in nodes[nidx]->inputs
    const int otidx = 0;  ///< output index in nodes[nidx]->outputs

    /// select kerne name
    Tensor::Dtype dtype_map();
};

// xxxImpl: each kernel
class IsNanNaiveImpl : public IsNanBaseImpl
{
public:
    IsNanNaiveImpl(const NodePtrList& nodes, const GPattern& pattern, const std::string& target)
        : IsNanBaseImpl(nodes, pattern, target)
    {
    }

    KernelInfos kernel_infos() override;
};

// xxxPlugin(GopBasePlugin): one plugin has many implements
class IsNanPlugin : public GopBasePlugin
{
public:
    IsNanPlugin();
    virtual ~IsNanPlugin();

    GopImplPtr get_impl(const NodePtrList& nodes, const GPattern& pattern = {}) override;
    const std::vector<bt_impl_dtypes> dtype_cap() override;

protected:
    bool check_params(const NodePtrList& nodes, const GPattern& pattern);
    // bool is_high_precision(const NodePtrList& nodes, const GPattern& pattern);
};

class IsNanDSPlugin : public IsNanPlugin
{
};

#define CPS_ISNAN_MAJOR 1
#define CPS_ISNAN_MINOR 0
#define CPS_ISNAN_VERSION (((CPS_ISNAN_MAJOR) << 24) + ((CPS_ISNAN_MINOR) << 16) + CPS_PLUGIN_BUILDNUM)

/// register static plugin
REGISTER_PLUGIN(IsNanPlugin, CPS_ISNAN_VERSION, GOPPLUGIN_TYPE);
/// register dynamic plugin when --enable_ds and some dimensions are symbolic
REGISTER_PLUGIN(IsNanDSPlugin, CPS_ISNAN_VERSION, GOPPLUGIN_TYPE | GBPlugin_DS);

};  // namespace aipubt
