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
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include <builder/plugin.h>
#include <core/graph.h>
#include <core/node.h>
#include <utils/utils.h>

#include <cctype>
#include <map>
#include <string>

namespace aipubt
{

namespace
{

// Push transpose parameters to fixed 6 dimensions
void push_transpose_param(BuilderParams &ro, const std::vector<int> &param, int dim)
{
    for (int i = 0; i < 6; i++)
    {
        if (i < dim)
        {
            ro.push_back(param[i]);
        }
        else
        {
            ro.push_back(1);
        }
    }
}

/// Same semantics as TransposePlugin::merge_dma_data_unit (inlined for internal codegen).
int merge_dma_data_unit(std::vector<int> &shape, std::vector<int> &perm, int &dim, int dtype_size)
{
    if (shape.empty() || (int)shape.size() < dim || perm.empty() || (int)perm.size() < dim)
    {
        return dtype_size;
    }
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

/// Check if axis is in axes array
static bool check_in_axes(const int *axes, int axis, int ndim)
{
    for (int32_t i = 0; i < ndim; i++)
    {
        if (axis == axes[i])
        {
            return 1;
        }
    }
    return 0;
}

// reduce all axis which is equal to 1.
int reduce_1_axes(std::vector<int> &in_shape, std::vector<int> &perm, int ndim)
{
    int32_t axes[6];
    int32_t cnt_axe = 0, cnt_shape = 0, cnt_perm = 0;
    int32_t temp_perm[6];
    int32_t temp_in_shape[6];
    for (int32_t i = 0; i < ndim; i++)
    {
        if (in_shape[perm[i]] != 1)
        {
            temp_perm[cnt_perm++] = perm[i];
        }
        if (in_shape[i] == 1)
        {
            axes[cnt_axe++] = i;
        }
        else
        {
            temp_in_shape[cnt_shape++] = in_shape[i];
        }
    }

    for (int32_t i = cnt_axe - 1; i >= 0; i--)
    {
        for (int32_t j = 0; j < cnt_perm; j++)
        {
            if (temp_perm[j] >= axes[i])
            {
                temp_perm[j]--;
            }
        }
    }

    for (int32_t i = 0; i < cnt_shape; i++)
    {
        in_shape[i] = temp_in_shape[i];
        perm[i] = temp_perm[i];
    }

    return cnt_shape;
}

/// Fuse adjacent axes in permutation
int fuse_axes(std::vector<int> &in_shape, std::vector<int> &perm, int ndim)
{
    int32_t fuse_axes_num = 0;
    int32_t fuse_axes[6];
    int32_t temp_in_shape[6];
    int32_t temp_perm[6];
    for (int32_t i = 0; i < ndim; i++)
    {
        temp_in_shape[i] = in_shape[i];
        temp_perm[i] = perm[i];
    }

    for (int32_t i = 1; i < ndim; i++)
    {
        if (perm[i] - 1 == perm[i - 1])
        {
            temp_perm[i] = temp_perm[i - 1];
            temp_in_shape[perm[i]] *= temp_in_shape[perm[i - 1]];
            fuse_axes[fuse_axes_num] = perm[i - 1];
            fuse_axes_num += 1;
        }
    }

    for (int32_t i = 0; i < fuse_axes_num; i++)
    {
        for (int32_t j = 0; j < ndim; j++)
        {
            if (temp_perm[j] > fuse_axes[i])
            {
                temp_perm[j] -= 1;
            }
        }
    }

    // Calculate fused input shape
    uint32_t count = 0;
    for (int32_t i = 0; i < ndim; i++)
    {
        if (!check_in_axes(fuse_axes, i, fuse_axes_num))
        {
            in_shape[count] = temp_in_shape[i];
            count += 1;
        }
    }

    // Calculate fused permutation
    count = 1;
    perm[0] = temp_perm[0];
    for (int32_t i = 1; i < ndim; i++)
    {
        if (temp_perm[i] != temp_perm[i - 1])
        {
            perm[count] = temp_perm[i];
            count += 1;
        }
    }

    return ndim - fuse_axes_num;
}

// Get shape and permutation after axis fusion
void get_new_shape_perm_dim(BaseNodePtr sgnode, BaseNodePtr &node, 
                            std::vector<int> &temp_shape, 
                            std::vector<int> &temp_perm, int &dim)
{
    auto in_ts = sgnode->inputs()[0]->shape();
    auto out_ts = sgnode->outputs()[0]->shape();
    dim = in_ts.dim();
    aipu_assert(out_ts.dim() == dim);
    
    // Get permutation parameter
    std::vector<int> perm = node->params()["perm"];
    aipu_assert((int)perm.size() == dim);
    
    // Handle negative indices
    for (size_t i = 0; i < perm.size(); i++)
    {
        if (perm[i] < 0)
        {
            perm[i] += (int)perm.size();
        }
    }
    
    // Get input shape
    std::vector<int> in_shape(dim);
    for (int i = 0; i < dim; i++)
    {
        in_shape[i] = in_ts[i];
    }
    
    temp_shape = in_shape;
    temp_perm = perm;
    
    // Apply axis fusion optimization
    dim = reduce_1_axes(temp_shape, temp_perm, dim);
    dim = fuse_axes(temp_shape, temp_perm, dim);
    
    if (dim > 6)
    {
        ERROR() << "[InternalLayoutReshapeLayout] Transpose can't support real > 6d permutation.";
    }
}

/// Target architecture mapping for transpose kernels
static const std::map<std::string, std::string> &transpose_targets_map()
{
    static const std::map<std::string, std::string> k_map = {
        {"z1_0904", "z1_0904"},
        {"z1_0701", "z1_0701"},
        {"z1_0701_p", "z1_0701"},
        {"z2_1104", "z2_1104"},
        {"z2_1002", "z2_1002"},
        {"z2_0901", "z2_0901"},
        {"z3_1204", "z2_1104"},
        {"z3_1104", "z2_1104"},
        {"z3_1002", "z2_1002"},
        {"z3_0901", "z2_0901"},
        {"x1_1204", "x1_1204"},
        {"z5_0901", "x2"},
        {"z5_1204", "x2"},
        {"x2_1204", "x2"},
        {"x3_1204", "x3"},
        {"x3_1202", "x3"},
        {"x3_1304", "x3"},
        {"x3_1302", "x3"},
        {"x3p_1204", "x3p"},
        {"x3p_1202", "x3p"},
        {"x3p_1304", "x3p"},
        {"x3p_1302", "x3p"},
        {"x3s_1204", "x3s"},
        {"x3s_1202", "x3s"},
        {"x3s_1304", "x3s"},
        {"x3s_1302", "x3s"},
    };
    return k_map;
}

// Get target architecture suffix for binary file names
static std::string tpcc_transpose_target_suffix(const std::string &target_name)
{
    std::string key;
    key.reserve(target_name.size());
    for (unsigned char c : target_name)
    {
        key.push_back((char)std::tolower(c));
    }
    const auto &m = transpose_targets_map();
    auto it = m.find(key);
    if (it != m.end())
    {
        return it->second;
    }
    WARN() << "[InternalLayoutReshapeLayout] transpose targets_map: unknown target \"" << key
           << "\", using lowercased name as bin token";
    return key;
}

} // namespace

class InternalLayoutReshapeLayout : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern() override
    {
        GraphMatcherPattern pattern = {
            {{"n0", OpType_LayoutConvertor}, {"n1", OpType_Reshape}, {"n2", OpType_LayoutConvertor}},
            {{"n0", "n1"}, {"n1", "n2"}}};
        return pattern;
    }

    uint32_t get_score() override
    {
        return 100;
    }

    bool set_target(const char *target) override
    {
        return true;
    };

    /// Validate whether the LayoutConvertor-Reshape-LayoutConvertor pattern can be
    /// optimized. The pattern is accepted when all of the following hold:
    ///   1. Input/output layouts are the same NCHWC16 or NCHWC32.
    ///   2. The reshape operates in NHWC and is static.
    ///   3. Both shapes are 5-D with matching C_out and C_in.
    ///   4. The spatial element count N*H*W is preserved.
    ///   5. If a transpose is needed (N changes while C_out > 1), the case must be
    ///      a spatial axis swap expressible as a single 5-D transpose:
    ///        - N-H swap (H_in == N_out), or
    ///        - N-W swap (W_in == N_out).
    ///      After the swap, the remaining spatial dims are flat-reinterpreted
    ///      within each C_out block to reach the final [H_out, W_out].
    bool check_params(BaseNodePtrList nodes) override
    {
        auto inp = nodes[0]->inputs()[0];
        auto outp = nodes[2]->outputs()[0];
        auto in_layout = inp->shape().layout();
        auto out_layout = outp->shape().layout();

        if (in_layout != out_layout)
        {
            return false;
        }

        if (in_layout != DataLayout_NCHWC16 && in_layout != DataLayout_NCHWC32)
        {
            return false;
        }

        auto reshape_in_layout = nodes[1]->inputs()[0]->shape().layout();
        auto reshape_out_layout = nodes[1]->outputs()[0]->shape().layout();
        if (reshape_in_layout != DataLayout_NHWC ||
            reshape_out_layout != DataLayout_NHWC)
        {
            return false;
        }

        NodePtr real_reshape = get_real_node(nodes[1]->inputs()[0]->consumers()[0]);
        if (real_reshape->is_dynamic())
        {
            return false;
        }

        std::vector<int> in_shape = inp->shape();
        std::vector<int> out_shape = outp->shape();

        if (in_shape.size() != 5 || out_shape.size() != 5)
        {
            return false;
        }

        if (in_shape[1] != out_shape[1] || in_shape[4] != out_shape[4])
        {
            return false;
        }

        long long in_spatial_prod = (long long)in_shape[0] * in_shape[2] * in_shape[3];
        long long out_spatial_prod = (long long)out_shape[0] * out_shape[2] * out_shape[3];
        if (in_spatial_prod != out_spatial_prod)
        {
            return false;
        }

        // In NCHWC memory (N->C_out->H->W->C_in), C_out separates N from H,W.
        // Changing N when C_out > 1 crosses that barrier and requires a transpose.
        // The transpose swaps N with one spatial axis so that N_out lands in dim 0.
        // The remaining spatial dims are then a flat reinterpretation within each
        // C_out block. Supported patterns:
        //   - N-H swap: H_in == N_out  (remaining [N_in, W] reshapes to [H_out, W_out])
        //   - N-W swap: W_in == N_out  (remaining [H, N_in] reshapes to [H_out, W_out])
        // (H-W reshape does not cross the C_out barrier so no transpose is needed.)
        bool crosses_cout = (in_shape[0] != out_shape[0]) && (in_shape[1] > 1);
        if (crosses_cout)
        {
            bool is_nh_swap = (in_shape[2] == out_shape[0]);
            bool is_nw_swap = (in_shape[3] == out_shape[0]);
            if (!is_nh_swap && !is_nw_swap)
            {
                return false;
            }
        }

        return true;
    }

    // Setup attributes for the fused node
    void setup(BaseNodePtr sgnode, BaseNodePtrList match_list) override
    {
        std::vector<int> in_shape = match_list[0]->inputs()[0]->shape();
        std::vector<int> out_shape = match_list[2]->outputs()[0]->shape();

        // In NCHWC memory (N->C_out->H->W->C_in), C_out separates N from H,W.
        // Changing N when C_out > 1 crosses that barrier and requires a transpose.
        // When N is unchanged (or C_out==1), any spatial [H,W] reshape is a flat
        // reinterpretation within each (n, c_out) block — no transpose needed.
        bool need_transpose = (in_shape[0] != out_shape[0]) && (in_shape[1] > 1);
        sgnode->set_attr("need_transpose", need_transpose);

        if (need_transpose)
        {
            bool is_nw_swap = (in_shape[3] == out_shape[0]);
            if (is_nw_swap)
            {
                // N-W swap: transpose produces [W, C_out, H, N, C_in].
                // Remaining [H, N_in] is flat-reinterpreted as [H_out, W_out]
                // within each C_out block.
                sgnode->params()["perm"] = NodeParamValue({3, 1, 2, 0, 4});
                std::vector<int> trans_out_shape = {
                    in_shape[3], in_shape[1], in_shape[2], in_shape[0], in_shape[4]};
                sgnode->set_attr("transpose_out_shape", trans_out_shape);
            }
            else
            {
                // N-H swap: transpose produces [H, C_out, N, W, C_in].
                // Remaining [N_in, W] is flat-reinterpreted as [H_out, W_out]
                // within each C_out block.
                sgnode->params()["perm"] = NodeParamValue({2, 1, 0, 3, 4});
                std::vector<int> trans_out_shape = {
                    in_shape[2], in_shape[1], in_shape[0], in_shape[3], in_shape[4]};
                sgnode->set_attr("transpose_out_shape", trans_out_shape);
            }
        }
    }

    // Generate builder parameters for transpose operation
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            return BuilderParams();
        }

        auto node = sgnode;
        BuilderParams ro;
        
        int dim = 0;
        std::vector<int> temp_shape, temp_perm;
        
        get_new_shape_perm_dim(sgnode, node, temp_shape, temp_perm, dim);
        
        int kernel_dtype_size = Tensor::dtype_size(node->inputs()[0]->type());
        kernel_dtype_size = merge_dma_data_unit(temp_shape, temp_perm, dim, kernel_dtype_size);
        
        push_transpose_param(ro, temp_shape, dim);
        push_transpose_param(ro, temp_perm, dim);
        ro.push_back(dim);
        
        ro.push_back(sgnode->inputs()[0]);
        ro.push_back(sgnode->outputs()[0]);
        ro.push_back(0);  // workspace
        
        return ro;
    }

    // Generate binary code name for transpose kernel
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            return "";
        }

        auto node = sgnode;
        auto inp = sgnode->inputs()[0];
        auto outp = sgnode->outputs()[0];
        auto saved_in_layout = inp->shape().layout();
        auto trans_out_shape = sgnode->get_attr<std::vector<int>>("transpose_out_shape");

        // Temporarily set to Flat layout for dimension calculation
        inp->shape().set_layout(DataLayout_Flat);
        outp->shape() = TensorShape(trans_out_shape);
        outp->shape().set_layout(DataLayout_Flat);

        int dim = 0;
        std::vector<int> temp_shape, temp_perm;
        get_new_shape_perm_dim(sgnode, node, temp_shape, temp_perm, dim);
        int kernel_dtype_size =
            merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(node->inputs()[0]->type()));

        // Restore original layout
        inp->shape().set_layout(saved_in_layout);
        outp->shape().set_layout(saved_in_layout);

        const std::string arch = tpcc_transpose_target_suffix(get_target()->name());
        std::string code_name;
        if (kernel_dtype_size == 1)
        {
            code_name = "tpcc_transpose_i8u8_" + arch + ".bin";
        }
        else if (kernel_dtype_size == 2)
        {
            code_name = "tpcc_transpose_i16u16_" + arch + ".bin";
        }
        else
        {
            code_name = "tpcc_transpose_i32u32_" + arch + ".bin";
        }

        return code_name;
    }

    // Generate kernel name for transpose operation
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            return "";
        }
        auto inp = sgnode->inputs()[0];
        auto outp = sgnode->outputs()[0];
        auto saved_in_layout = inp->shape().layout();
        auto trans_out_shape = sgnode->get_attr<std::vector<int>>("transpose_out_shape");
        
        // Temporarily set to Flat layout for dimension calculation
        inp->shape().set_layout(DataLayout_Flat);
        outp->shape() = TensorShape(trans_out_shape);
        outp->shape().set_layout(DataLayout_Flat);
        
        int dim = 0;
        std::vector<int> temp_shape, temp_perm;
        get_new_shape_perm_dim(sgnode, sgnode, temp_shape, temp_perm, dim);
        int kernel_dtype_size =
            merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(sgnode->inputs()[0]->type()));
        
        // Restore original layout
        inp->shape().set_layout(saved_in_layout);
        outp->shape().set_layout(saved_in_layout);
        
        const char *kname = "Transpose32Bit";
        if (kernel_dtype_size == 1)
        {
            kname = "Transpose8Bit";
        }
        else if (kernel_dtype_size == 2)
        {
            kname = "Transpose16Bit";
        }
        return kname;
    }

    // Memory remapping optimization
    void memory_remapping(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            // If no transpose needed, share memory between input and output
            auto out = sgnode->outputs()[0];
            auto inp = sgnode->inputs()[0];
            out->memory_offset()->relative_offset = 0;
            out->memory_offset()->set_base_offset(inp->memory_offset());
        }
    }
};

REGISTER_PLUGIN(InternalLayoutReshapeLayout, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_Internal);
} // namespace aipubt