#include "isnan_plugin.h"

namespace aipubt
{
namespace CpsPlugins
{

};

void IsNanBaseImpl::update_prefer_cfg(const std::string& key, const Any& value) { CPS_OBJECT_STARTUP; }

uint32_t IsNanBaseImpl::score() const
{
    CPS_OBJECT_STARTUP;

    return 10;
}

Tensor::Dtype IsNanBaseImpl::dtype_map()
{
    Tensor::Dtype ret = otype(nidx, otidx);
    switch (ret)
    {
        case Tensor::UINT32:
            ret = Tensor::INT32;
            break;
        case Tensor::UINT16:
            ret = Tensor::INT16;
            break;
        case Tensor::UINT8:
            ret = Tensor::INT8;
            break;
        default:
            break;
    }
    return ret;
}

const std::string IsNanBaseImpl::kernel_cpsbin()
{
    CPS_OBJECT_STARTUP;

    std::vector<std::string> cps_objname{utils::tolower(_objname), utils::tolower(_target), "kernel_obj"};
    std::string name = utils::join_string("_", cps_objname);

    DEBUG() << "object_name: " << name;
    return name + CpsConst::OBJ_EXTENSION;
}

uint64_t IsNanBaseImpl::flops() const
{
    CPS_OBJECT_STARTUP;

    return 0;
}

KernelInfos IsNanNaiveImpl::kernel_infos()
{
    CPS_OBJECT_STARTUP;

    KernelInfos infos;

    tongue::Expr n = ishape_symbol(nidx, itidx).size();  // dynamic
    tongue::Expr block = static_cast<size_t>(CpsUtils::maxThreadsPerBlock);
    tongue::Expr elements_per_block = block * static_cast<size_t>(CpsUtils::maxElementsPerThread);
    tongue::Expr grid = CpsCommon::CeilDiv(n, elements_per_block);
    DEBUG() << "n: " << n->to_string();
    DEBUG() << "block: " << block->to_string();
    DEBUG() << "elements_per_block: " << elements_per_block->to_string();
    DEBUG() << "grid: " << grid->to_string();

    std::string component_name = "naive";
    std::vector<std::string> kernel_name{
        utils::tolower(_objname),
        utils::tolower(component_name),
        Tensor::dtype_to_string(input(nidx, itidx)->type()),
        Tensor::dtype_to_string(dtype_map()),
    };
    std::string name = utils::join_string("_", kernel_name);

    size_t sharemem_size = 0;

    // // How to set constant tensor: just put aaaprt into args
    // auto t = constants(ninx)["aaa"];
    // // or
    // auto t = std::make_shared<Tensor>("unique name", TensorShape{1,2,3},
    // Tensor::FP32); auto d = new float[t->shape().size()]; for (int i = 0; i <
    // t->shape().size(); ++i) {
    //     d[i] = static_cast<float>(i);
    // }
    // t->set_data((void *)d, t->shape().size() * sizeof(float));
    // auto aaaprt = ConstTensorPtr(t);

    // // How to set workspace tensor: just put ws into args
    // auto ws = Tensor::workspace(0x10000);

    PlaceholderPtr inptr = input(nidx, itidx);
    PlaceholderPtr outptr = output(nidx, otidx);
    // tongue::Expr i_offset = xxxx;
    // auto in_addr = inptr + i_offset;

    DEBUG() << "kernel_name: " << name;
    infos.push_back({name, grid, block, sharemem_size, {inptr, outptr, n}});

    return infos;
}

IsNanPlugin::IsNanPlugin()
{
    _plugin_name = "isnan";

    _target_support_list.clear();
    _target_support_list = std::vector<std::string>(CpsCommon::all_targets.begin(), CpsCommon::all_targets.end());
    //// some targets do not support
    // for (auto target : all_targets)
    // {
    //     if (aipu_chip_compare(get_chip_name(target), "xxx") >= 0)
    //     {
    //         _target_support_list.push_back(target);
    //     }
    // }

    bt_nodes_type nodes;
    bt_edges_type edges;
    // pattern_native
    nodes.clear();
    edges.clear();
    nodes.emplace_back("n0", OpType_IsNaN);
    edges = {};
    _patterns.emplace_back(nodes, edges);
    //// pattern_epilogue
    // nodes.clear();
    // edges.clear(
    // nodes.emplace_back("0", OpType_Convolution);
    // nodes.emplace_back("1", OpType_Eltwise);
    // edges.emplace_back("0", "1");
    // _patterns.emplace_back(nodes, edges);

    DEBUG() << "Plugin(" << _plugin_name << ") has constructed done!";
}

IsNanPlugin::~IsNanPlugin() {}

const std::vector<bt_impl_dtypes> IsNanPlugin::dtype_cap()
{
    CPS_PLUGIN_STARTUP;

    std::vector<bt_impl_dtypes> ret;
    ret.push_back(bt_impl_dtypes{
        {{Tensor::FP16}, {Tensor::INT8}},
        {{Tensor::BFP16}, {Tensor::INT8}},
        {{Tensor::FP32}, {Tensor::INT8}},

        {{Tensor::FP16}, {Tensor::UINT8}},
        {{Tensor::BFP16}, {Tensor::UINT8}},
        {{Tensor::FP32}, {Tensor::UINT8}},

        {{Tensor::FP16}, {Tensor::INT16}},
        {{Tensor::BFP16}, {Tensor::INT16}},
        {{Tensor::FP32}, {Tensor::INT16}},

        {{Tensor::FP16}, {Tensor::UINT16}},
        {{Tensor::BFP16}, {Tensor::UINT16}},
        {{Tensor::FP32}, {Tensor::UINT16}},

        {{Tensor::FP16}, {Tensor::INT32}},
        {{Tensor::BFP16}, {Tensor::INT32}},
        {{Tensor::FP32}, {Tensor::INT32}},

        {{Tensor::FP16}, {Tensor::UINT32}},
        {{Tensor::BFP16}, {Tensor::UINT32}},
        {{Tensor::FP32}, {Tensor::UINT32}},
    });

    return ret;
}

bool IsNanPlugin::check_params(const NodePtrList& nodes, const GPattern& pattern)
{
    auto node_name = nodes[0]->name();
    auto input_size = nodes[0]->inputs()[0]->shape().size();
    auto output_size = nodes[0]->inputs()[0]->shape().size();
    auto input_dtype = nodes[0]->inputs()[0]->type();
    auto output_dtype = nodes[0]->outputs()[0]->type();

    // bool ret = true;
    // ret = ret && Tensor::is_float(nodes[0]->inputs()[0]->type());
    // ret = ret && Tensor::is_int(nodes[0]->outputs()[0]->type());
    // return ret;

    CPS_ERROR_IF_TRUE((input_size != output_size), node_name, CpsConst::DNS, "different input/output elements size");
    CPS_ERROR_IF_TRUE(Tensor::is_int(input_dtype), node_name, CpsConst::DNS, "int input");
    CPS_ERROR_IF_TRUE(CpsUtils::none_of_eq(input_dtype, Tensor::FP16, Tensor::BFP16, Tensor::FP32), node_name,
                      CpsConst::DNS, "this input dtype: ", Tensor::dtype_to_string(input_dtype));
    CPS_ERROR_IF_TRUE(Tensor::is_float(output_dtype), node_name, CpsConst::DNS, "float output");
    CPS_ERROR_IF_TRUE(CpsUtils::none_of_eq(output_dtype, Tensor::INT8, Tensor::UINT8, Tensor::INT16, Tensor::UINT16,
                                           Tensor::INT32, Tensor::UINT32),
                      node_name, CpsConst::DNS, "this output dtype: ", Tensor::dtype_to_string(output_dtype));
    return true;
}

GopImplPtr IsNanPlugin::get_impl(const NodePtrList& nodes, const GPattern& pattern)
{
    CPS_PLUGIN_STARTUP;

    if (!check_params(nodes, pattern))
    {
        return nullptr;
    }

    _target = get_target()->name();
    return std::make_shared<IsNanNaiveImpl>(nodes, pattern, _target);
}
};  // namespace aipubt
