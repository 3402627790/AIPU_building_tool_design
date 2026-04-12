#include "transpose_plugin.h"
#include <algorithm>
#include <climits>

namespace aipubt
{
const GraphMatcherPattern TransposePlugin::get_graph_pattern()
{
    return {{{"transpose", OpType_Transpose}}, {}};
}

const std::string TransposePlugin::get_layer_type_name()
{
    return VAR_NAME_TO_STR(OpType_Transpose);
}

uint32_t TransposePlugin::get_score()
{
    return 10;
}

bool TransposePlugin::_special_check_params(BaseNodePtrList nodes)
{
    // check_info info;
    // info.dim_size = INT_MAX;
    // info.shape_size = UINT_MAX;
    // info.param_list = {"perm"};
    // info.type_list = {Tensor::INT8, Tensor::UINT8, Tensor::FP8E4M3FN, Tensor::FP8E5M2,
    //                   Tensor::INT16, Tensor::UINT16, Tensor::FP16, Tensor::BFP16,
    //                   Tensor::INT32, Tensor::UINT32};
    // if (!auto_check(nodes[0], info))
    // {
    //     return false;
    // }

    return true;
}

bool TransposePlugin::set_target(const char *target)
{
    m_target_name = target;
    if (!check_target("transpose", m_target_name))
    {
        return false;
    }

    return true;
}

void TransposePlugin::setup(BaseNodePtr node, BaseNodePtrList match_list)
{
    node->set_attr("keeping_layout", false);
    LayoutCombinations valid_in;
    LayoutCombinations valid_out;
    bool trans_c_dim = check_trans_c_dim(node, match_list[0]);
    DEBUG() << "trans_c_dim:" << trans_c_dim;
    if (aipu_get_options(OpConstants::transpose_not_fuse_layoutconvertor) || trans_c_dim)
    {
        valid_in = {
            {DataLayout_NHWC},
            {DataLayout_NDHWC},
            {DataLayout_Flat},
        };
        valid_out = {
            {DataLayout_NHWC},
            {DataLayout_NDHWC},
            {DataLayout_Flat},
        };
    }
    else
    {
        valid_in = {
            {DataLayout_NHWC},
            {DataLayout_NDHWC},
            {DataLayout_NCHWC32},
            {DataLayout_NCHWC16},
            {DataLayout_Flat},
        };
        valid_out = {
            {DataLayout_NHWC},
            {DataLayout_NDHWC},
            {DataLayout_NCHWC32},
            {DataLayout_NCHWC16},
            {DataLayout_Flat},
        };
    }
    node->set_attr("valid_input_layouts", valid_in);
    node->set_attr("valid_output_layouts", valid_out);
    std::map<std::vector<DataLayout>, std::vector<std::vector<DataLayout>>> forbidden_layout_comb =
        {
            // CTL-5985: may note perf issue
            // CS-733: add NDWHC->Flat transpose
            // {{DataLayout_NHWC}, {{DataLayout_Flat}}},
            // {{DataLayout_NDHWC}, {{DataLayout_Flat}}},
            // {{DataLayout_Flat}, {{DataLayout_NCHWC32}}},
            // {{DataLayout_Flat}, {{DataLayout_NCHWC16}}},
            // {{DataLayout_Flat}, {{DataLayout_NDCHWC32}}},
            // {{DataLayout_Flat}, {{DataLayout_NDCHWC16}}},
            // {{DataLayout_NCHWC32}, {{DataLayout_Flat}}},
            // {{DataLayout_NCHWC16}, {{DataLayout_Flat}}},
            // {{DataLayout_NDCHWC32}, {{DataLayout_Flat}}},
            // {{DataLayout_NDCHWC16}, {{DataLayout_Flat}}},
            {{DataLayout_NCHWC32}, {{DataLayout_NCHWC32}}},
            {{DataLayout_NCHWC16}, {{DataLayout_NCHWC16}}},
            {{DataLayout_NCHWC8}, {{DataLayout_NCHWC8}}},
            {{DataLayout_NCHWC4}, {{DataLayout_NCHWC4}}},
            {{DataLayout_NDCHWC32}, {{DataLayout_NDCHWC32}}},
            {{DataLayout_NDHWC}, {{DataLayout_NDCHWC32}}},
        };
    node->set_attr("deny_layouts_comb", forbidden_layout_comb);
    //set node kernel_dtype_size attr
    // auto match_node = match_list[0];
    // int dim;
    // std::vector<int> temp_shape,temp_perm;
    // get_new_shape_perm_dim(node, match_node, temp_shape, temp_perm, dim);
    // int kernel_dtype_size = merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(node->inputs()[0]->type()));
    // node->set_attr("kernel_dtype_size", kernel_dtype_size);
}

bool TransposePlugin::check_trans_c_dim(BaseNodePtr sgnode, BaseNodePtr &node)
{
    std::vector<int> perm = node->params()["perm"];
    int last_perm_axis = sgnode->inputs()[0]->shape().dim() - 1;
    return perm[last_perm_axis] != last_perm_axis;
}

int TransposePlugin::merge_dma_data_unit(std::vector<int> &shape, std::vector<int> &perm, int &dim, int dtype_size)
{
    if (shape.empty() || (int)shape.size() < dim ||
        perm.empty() || (int)perm.size() < dim)
    {
        return dtype_size;
    }
    // bool valid_target = (m_target_name == "x2_1204") || (m_target_name == "x3p_1304");
    int last_dim = dim - 1;
    if (perm[last_dim] == last_dim)
    {
        if (shape[last_dim] * dtype_size == 4)
        {
            dim = dim - 1;
            return 4;
        }
        if (shape[last_dim] * dtype_size == 2)
        {
            dim = dim - 1;
            return 2;
        }
    }
    return dtype_size;
}

void TransposePlugin::get_new_shape_perm_dim(BaseNodePtr sgnode, BaseNodePtr &node, std::vector<int> &temp_shape, std::vector<int> &temp_perm, int &dim)
{
    auto in_shape = sgnode->inputs()[0]->shape();
    dim = in_shape.dim();
    auto from = in_shape.layout();
    auto to = sgnode->outputs()[0]->shape().layout();
    std::vector<int> perm = node->params()["perm"];
    temp_perm.clear();
    temp_shape.clear();
    std::set<DataLayout> NCHWc = {
        DataLayout_NCHWC4,
        DataLayout_NCHWC8,
        DataLayout_NCHWC16,
        DataLayout_NCHWC32};

    // preprocess perm if perm axis is negative
    for (uint i = 0; i < perm.size(); i++)
    {
        if (perm[i] < 0)
        {
            perm[i] += perm.size();
        }
    }

    // next time deal with NDHWC
    if (from == DataLayout_NDCHWC32 && to == DataLayout_NDHWC)
    {
        DEBUG() << "layout:" << from << "; "
                << "shape: " << in_shape << "; ";
        temp_shape = {in_shape[0], in_shape[1], in_shape[2], in_shape[3], in_shape[4], in_shape[5]};
        // TODO(Yu): dev merge(perm, perm) to replace hard coding
        for (uint i = 0; i < perm.size(); i++)
        {
            if (perm[i] == 0 || perm[i] == 1)
            {
                temp_perm.push_back(perm[i]);
            }
            if (perm[i] >= 2 && perm[i] <= 3)
            {
                temp_perm.push_back(perm[i] + 1);
            }
            if (perm[i] == 4)
            {
                temp_perm.push_back(2);
                temp_perm.push_back(5);
            }
        }
    }
    else
    {
        // convert to NHWC
        if (NCHWc.count(from) && !NCHWc.count(to))
        {
            aipu_assert(dim == 5);
            int trans[5] = {0, 2, 3, 1, 4};
            for (int i = 0; i < 4; i++)
            {
                if (perm[i] == 3)
                {
                    temp_perm.push_back(perm[i]);
                    temp_perm.push_back(4);
                }
                else
                {
                    temp_perm.push_back(perm[i]);
                }
            }
            for (int i = 0; i < 5; i++)
            {
                temp_perm[i] = trans[temp_perm[i]];
                temp_shape.push_back(in_shape[i]);
            }
        }

        if (NCHWc.count(to) && !NCHWc.count(from))
        {
            if (from != DataLayout_NHWC)
            {
                WARN() << "In layout: " << from << "; Out layout: " << to;
                ERROR() << "It may not be a valid transpose, please check transpose IR.";
                // NOTE: ro is not right here for transpose; to not crash the program
            }
            aipu_assert(sgnode->outputs()[0]->shape().dim() == 5);
            aipu_assert(perm.size() == 4);
            for (int i = 0; i < 4; i++)
            {
                if (i == 1)
                {
                    temp_perm.push_back(perm[3]);
                }
                if (perm[i] >= perm[3])
                {
                    temp_perm.push_back(perm[i] + 1);
                }
                else
                {
                    temp_perm.push_back(perm[i]);
                }
            }
            int c = 4 << ((int)(to - DataLayout_NCHWC4));
            aipu_assert(in_shape[perm[3]] % c == 0);
            for (int i = 0; i < 4; i++)
            {
                if (i == (int)perm[3])
                {
                    temp_shape.push_back(in_shape[i] / c);
                    temp_shape.push_back(c);
                }
                else
                {
                    temp_shape.push_back(in_shape[i]);
                }
            }
            // after perm&shape reorg, perm&shape dim will be 5
            aipu_assert(temp_perm.size() == 5);
            aipu_assert(temp_shape.size() == 5);
            dim = reduce_1_axes(temp_shape, temp_perm, 5);
        }

        if ((from == DataLayout_NHWC && to == DataLayout_NHWC) ||
            (from == DataLayout_NHWC && to == DataLayout_Flat) ||
            (from == DataLayout_Flat && to == DataLayout_NHWC) ||
            (from == DataLayout_Flat && to == DataLayout_Flat) ||
            (from == DataLayout_NDHWC && to == DataLayout_Flat) ||
            (from == DataLayout_Flat && to == DataLayout_NDHWC) ||
            (from == DataLayout_NDHWC && to == DataLayout_NDHWC))
        {
            aipu_assert(sgnode->outputs()[0]->shape().dim() == sgnode->inputs()[0]->shape().dim());
            for (int i = 0; i < dim; i++)
            {
                temp_perm.push_back(perm[i]);
                temp_shape.push_back(in_shape[i]);
            }
        }

        if (temp_perm.size() == 0 || temp_shape.size() == 0)
        {
            INFO() << "Transpose input layout: " << from;
            INFO() << "Transpose output layout: " << to;
            ERROR() << "It may not be a valid transpose, please check transpose IR.";
        }
    }

    dim = reduce_1_axes(temp_shape, temp_perm, dim);
    dim = fuse_axes(temp_shape, temp_perm, dim);
    if (dim > 6)
    {
        ERROR() << " Transpose can't support real > 6d permution."
                << " please use tranpose recursively";
    }
}

BuilderParams TransposePlugin::generate_params(BaseNodePtr sgnode,
                                               std::vector<BaseNodePtr> &nodes)
{
    auto node = nodes[0];
    BuilderParams ro;

    int dim;
    std::vector<int> temp_shape, temp_perm;
    get_new_shape_perm_dim(sgnode, node, temp_shape, temp_perm, dim);
    merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(node->inputs()[0]->type()));
    push_transpose_param(ro, temp_shape, dim);
    push_transpose_param(ro, temp_perm, dim);
    ro.push_back(dim);

    ro.push_back(sgnode->inputs()[0]);
    ro.push_back(sgnode->outputs()[0]);
    bool need_workspace = true;
    int trans_count = 0;
    for (int i = 0; i < dim; i++)
    {
        if (temp_perm[i] != i)
        {
            trans_count++;
        }
    }
    if ((temp_perm[dim - 1] == (dim - 1)) || (trans_count == 2 && temp_perm[dim - 1] == (dim - 2)))
    {
        need_workspace = false;
    }
    DEBUG() << "need_workspace: " << need_workspace;
    if (need_workspace)
    {
        // NOTE: GB allow to set_ws in generate_params
        int node_size = node->inputs()[0]->shape().size();
        auto dtype = node->inputs()[0]->type();
        auto dtype_size = Tensor::dtype_size(dtype);
        node_size *= dtype_size;
        node->set_attr("workspace_size", node_size);
        auto workspace = sgnode->get_attr<TensorPtr>("workspace");
        ro.push_back(workspace);
    }
    else
    {
        ro.push_back(0); // 0 is nullptr here;
    }

    return ro;
}
const std::string TransposePlugin::generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
{
    // int kernel_dtype_size = sgnode->get_attr<bool>("kernel_dtype_size");
    auto node = nodes[0];
    int dim;
    std::vector<int> temp_shape, temp_perm;
    get_new_shape_perm_dim(sgnode, node, temp_shape, temp_perm, dim);
    int kernel_dtype_size = merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(node->inputs()[0]->type()));

    std::string code_name;
    if (kernel_dtype_size == 1)
    {
        code_name = "tpcc_transpose_i8u8_" + targets_map[m_target_name] + ".bin";
    }
    else if (kernel_dtype_size == 2)
    {
        code_name = "tpcc_transpose_i16u16_" + targets_map[m_target_name] + ".bin";
    }
    else
    {
        code_name = "tpcc_transpose_i32u32_" + targets_map[m_target_name] + ".bin";
    }
    return code_name;
}
const std::string TransposePlugin::generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes)
{
    // int kernel_dtype_size = sgnode->get_attr<bool>("kernel_dtype_size");
    auto node = nodes[0];
    int dim;
    std::vector<int> temp_shape, temp_perm;
    get_new_shape_perm_dim(sgnode, node, temp_shape, temp_perm, dim);
    int kernel_dtype_size = merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(node->inputs()[0]->type()));

    if (kernel_dtype_size == 1)
    {
        return "Transpose8Bit";
    }
    else if (kernel_dtype_size == 2)
    {
        return "Transpose16Bit";
    }
    else
    {
        return "Transpose32Bit";
    }
}

REGISTER_PLUGIN(TransposePlugin, VERSION_NUMBER, BUILDERPLUGIN_TYPE);

}; // namespace aipubt
