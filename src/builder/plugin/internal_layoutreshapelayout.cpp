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

#include <algorithm>
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

static std::vector<int> integer_factorize(int n)
{
    std::vector<int> ret;
    int f = 2;
    while (f * f <= n)
    {
        while (n % f == 0)
        {
            ret.push_back(f);
            n = n / f;
        }
        f += 1;
    }
    if (n > 1)
    {
        ret.push_back(n);
    }
    return ret;
}

static int prod(TensorShape &s)
{
    int p = 1;
    for (auto i : s)
    {
        p *= i;
    }
    return p;
}

/**
 * @brief Determines if a reshape operation requires a physical memory reordering (transpose) 
 * by analyzing dimension factor decomposition.
 * * This function attempts to decompose the dimensions of shape0 and shape1 into a common 
 * sequence of "atomic" factors (the 'split' vector). If the dimensions align such that 
 * they share the same underlying factors in a way the compiler can map, it returns true.
 * * @param shape0 The input tensor shape.
 * @param shape1 The target output tensor shape.
 * @param split [Output] A vector that stores the synchronized dimension factors found during matching.
 * @return true If a valid factor-based decomposition is found (enabling Reshape+Transpose lowering).
 * @return false If dimension boundaries are misaligned or factors cannot be matched.
 */
static bool is_strides_changed(TensorShape &shape0, TensorShape &shape1, std::vector<int> &split)
{
    std::vector<std::vector<int>> a, b;
    for (auto i : shape0)
    {
        a.push_back(integer_factorize(i));
    }
    for (auto i : shape1)
    {
        b.push_back(integer_factorize(i));
    }
    auto p = prod(shape0);
    int s = 1;
    while (a.size() > 0 && b.size() > 0 && s != p)
    {
        int v = 1;
        auto a0 = a[0];
        for (auto i : a0)
        {
            if (contains(b[0], i))
            {
                v *= i;
                s *= i;
                b[0].erase(std::find(b[0].begin(), b[0].end(), i));
                a[0].erase(std::find(a[0].begin(), a[0].end(), i));
            }
        }
        if (v != 1)
        {
            split.push_back(v);
        }
        bool res = false;
        if (a[0].size() == 0)
        {
            a.erase(a.begin());
            res = true;
        }
        if (b[0].size() == 0)
        {
            b.erase(b.begin());
            res = true;
        }
        if (!res)
        {
            return false;
        }
    }
    return true;
}

// split a shape into vector of vector
std::vector<std::vector<int>> split_shape(const std::vector<int> &split, TensorShape &shape)
{
    uint32_t idx = 0;
    std::vector<std::vector<int>> ret;
    for (auto i : shape)
    {
        if (i == 1)
        {
            ret.push_back({1});
            continue;
        }
        int p = 1;
        std::vector<int> spl;
        for (; idx < split.size();)
        {
            p *= split[idx];
            spl.push_back(split[idx]);
            ++idx;
            if (p == i)
            {
                ret.push_back(spl);
                break;
            }
        }
    }
    return ret;
}

/**
 * @brief Returns true if the permutation is the identity (perm[i] == i for all i).
 *        An identity permutation means the transpose is a no-op and can be elided.
 *
 * @param perm  The permutation vector to check (int or uint32_t values).
 * @return true  All indices are in-place; no data movement required.
 * @return false At least one index is out of place; a real transpose is required.
 */
static bool is_perm_consistent(const std::vector<int> &perm)
{
    for (size_t i = 0; i < perm.size(); i++)
    {
        if ((int)i != perm[i])
        {
            return false;
        }
    }
    return true;
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

        // Channel-group and inner-channel dimensions must be invariant across the reshape.
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

        // -----------------------------------------------------------------------
        // Determine whether the fused LC+Reshape+LC pattern actually requires an
        // explicit Transpose op after lowering.
        //
        // The two LC nodes are equivalent to fixed transposes:
        //   LC0 (NCHWC -> NHWCG): perm0 = [0, 2, 3, 1, 4]
        //   LC2 (NHWCG -> NCHWC): perm1 = [0, 3, 1, 2, 4]
        //
        // We analyse the pattern as-if it were Transpose(perm0) + Reshape + Transpose(perm1)
        // and apply the same factor-decomposition logic used by fuse_transpose_with_reshape
        // to obtain the effective residual permutation (new_perm) on the expanded shape.
        //
        // Two conditions must BOTH be true for a transpose to be necessary:
        //   1. is_strides_changed() returns true  -- the reshape crosses dimension
        //      boundaries in a way that changes memory strides.
        //   2. new_perm is NOT the identity        -- the residual permutation is
        //      non-trivial; an identity perm would be a no-op transpose.
        // -----------------------------------------------------------------------

        // Build the 5-D "NHWCG" shapes that the LC transposes see.
        // NCHWC[N, Cg, H, W, c] after perm0=[0,2,3,1,4] becomes NHWCG[N, H, W, Cg, c].
        std::vector<int> s5d_in  = {in_shape[0],  in_shape[2],  in_shape[3],  in_shape[1],  in_shape[4]};
        std::vector<int> s5d_out = {out_shape[0], out_shape[2], out_shape[3], out_shape[1], out_shape[4]};
        TensorShape ts5d_in(s5d_in);
        TensorShape ts5d_out(s5d_out);

        // Check 1: do the strides actually change across the reshape?
        std::vector<int> split;
        bool is_strides_ok = is_strides_changed(ts5d_in, ts5d_out, split);
        if (!is_strides_ok)
        {
            // No stride change: the reshape is a pure logical relabelling.
            // No transpose is needed; memory_remapping() will alias the buffers.
            m_need_transpose = false;
            return true;
        }

        // Check 2: compute the effective permutation (new_perm) that would remain
        // after fusing the two LC transposes with the reshape, following the same
        // algorithm as fuse_transpose_with_reshape().

        const std::vector<int> perm0 = {0, 2, 3, 1, 4}; // LC0: NCHWC -> NHWCG
        const std::vector<int> perm1 = {0, 3, 1, 2, 4}; // LC2: NHWCG -> NCHWC

        // Step A: split tr_shape (= ts5d_in, the reshape's logical input) according
        //         to the common factor decomposition obtained above.
        auto tr_shape_split = split_shape(split, ts5d_in);

        // Step B: undo perm0 to recover the per-dim factor groups in the original
        //         NCHWC input space (before_trans).
        //         Then build:
        //           new_perm0       -- flat index sequence for the expanded NCHWC shape
        //           after_trans_split -- dimension sizes in perm0-traversal order (NHWCG)
        std::vector<std::vector<int>> before_trans(perm0.size());
        for (uint32_t i = 0; i < (uint32_t)perm0.size(); ++i)
        {
            before_trans[perm0[i]] = tr_shape_split[i];
        }

        int acc = 0;
        std::vector<int> accumulate_size;
        for (auto &grp : before_trans)
        {
            acc = (int)grp.size() + acc - 1;
            accumulate_size.push_back(acc);
        }

        std::vector<int> new_perm0;
        std::vector<int> after_trans_split;
        for (uint32_t i = 0; i < (uint32_t)perm0.size(); ++i)
        {
            int pi = perm0[i];
            for (uint32_t j = 0; j < (uint32_t)before_trans[pi].size(); ++j)
            {
                new_perm0.push_back(
                    pi + accumulate_size[pi] + (int)j - (int)before_trans[pi].size() + 1);
            }
            for (auto v : before_trans[pi])
            {
                after_trans_split.push_back(v);
            }
        }

        // Step C: map each dimension of the reshape output shape back to one or
        //         more indices in new_perm0, producing after_reshape.
        uint32_t aidx = 0;
        std::vector<std::vector<int>> after_reshape;
        for (int s : ts5d_out)
        {
            std::vector<int> temp;
            if (aidx < (uint32_t)after_trans_split.size() && s < after_trans_split[aidx])
            {
                after_reshape.push_back(temp);
                continue;
            }
            int prod_val = 1;
            for (; aidx < (uint32_t)after_trans_split.size(); ++aidx)
            {
                if (prod_val == s)
                {
                    if (temp.empty())
                    {
                        temp.push_back(new_perm0[aidx]);
                        aidx++;
                    }
                    break;
                }
                prod_val *= after_trans_split[aidx];
                temp.push_back(new_perm0[aidx]);
            }
            after_reshape.push_back(temp);
        }

        // Step D: apply perm1 over after_reshape to obtain the final effective
        //         permutation on the expanded (pre-LC0 NCHWC) shape.
        std::vector<int> new_perm;
        for (uint32_t i = 0; i < (uint32_t)perm1.size(); ++i)
        {
            int p1i = perm1[i];
            if (p1i < (int)after_reshape.size())
            {
                for (auto v : after_reshape[p1i])
                {
                    new_perm.push_back(v);
                }
            }
        }

        // Dimensions that were size-1 may have been omitted; append them at the end
        // so that is_perm_consistent() sees a complete permutation vector.
        if (new_perm.size() != new_perm0.size())
        {
            for (uint32_t i = 0; i < (uint32_t)new_perm0.size(); ++i)
            {
                if (!contains(new_perm, i))
                {
                    new_perm.push_back(i);
                }
            }
        }

        // A permutation where every index stays in place (perm[i] == i) is the
        // identity: the effective transpose is a no-op and can be elided.
        m_need_transpose = !is_perm_consistent(new_perm);

        // -----------------------------------------------------------------------
        // Save the computed permutation and the expanded input shape of the
        // transpose kernel so that setup() can publish them as node attributes.
        //
        // m_new_reshape0  -- flat expansion of before_trans (indexed in NCHWC axis
        //                    order, possibly with sub-dimensions when a single NCHWC
        //                    axis is split across the reshape boundary).
        //                    This is the shape the transpose kernel actually reads.
        // m_new_perm      -- effective permutation over m_new_reshape0.
        //                    trans_out_shape[i] = m_new_reshape0[m_new_perm[i]].
        //
        // Using the expanded shape (rather than a hardcoded NCHWC-space perm) makes
        // setup() correct for every possible axis permutation -- not just N↔H.
        // -----------------------------------------------------------------------
        m_new_perm = new_perm;

        m_new_reshape0.clear();
        for (auto &grp : before_trans)
        {
            for (auto v : grp)
            {
                m_new_reshape0.push_back(v);
            }
        }

        return true;
    }

    // Setup attributes for the fused node.
    // Translates the perm/shape data computed in check_params into node attributes
    // consumed by generate_params(), generate_code_name(), generate_kernel(), and
    // memory_remapping().
    void setup(BaseNodePtr sgnode, BaseNodePtrList match_list) override
    {
        sgnode->set_attr("need_transpose", m_need_transpose);

        if (!m_need_transpose)
        {
            return;
        }

        // -----------------------------------------------------------------------
        // m_new_perm and m_new_reshape0 may describe an EXPANDED shape where one
        // NCHWC axis has been split into multiple sub-axes (e.g. W=1024 → [4, 256]).
        // Using them directly (rather than hardcoding a 5-D NCHWC-space perm) makes
        // this path correct for N↔W, N↔H, H↔W, and any split-axis permutation.
        //
        // The transpose kernel operates on:
        //   input  shape = m_new_reshape0          ("transpose_in_shape")
        //   output shape = m_new_reshape0 reordered by m_new_perm ("transpose_out_shape")
        //   perm         = m_new_perm              (stored in node params["perm"])
        // -----------------------------------------------------------------------

        // Compute the transpose output shape by applying m_new_perm to m_new_reshape0.
        // trans_out_shape[i] = m_new_reshape0[m_new_perm[i]]  (output[i] = input[perm[i]])
        std::vector<int> trans_out_shape(m_new_perm.size());
        for (size_t i = 0; i < m_new_perm.size(); ++i)
        {
            trans_out_shape[i] = m_new_reshape0[m_new_perm[i]];
        }

        sgnode->params()["perm"]             = NodeParamValue(m_new_perm);
        sgnode->set_attr("transpose_in_shape",  m_new_reshape0);
        sgnode->set_attr("transpose_out_shape", trans_out_shape);
    }

    // Generate builder parameters for the transpose kernel.
    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            return BuilderParams();
        }

        auto node = sgnode;
        auto inp  = sgnode->inputs()[0];
        BuilderParams ro;

        // Temporarily substitute the tensor's shape with the expanded transpose-input
        // shape so that get_new_shape_perm_dim() sees the correct per-axis sizes and
        // can apply reduce_1_axes / fuse_axes for kernel parameter computation.
        auto trans_in_shape  = sgnode->get_attr<std::vector<int>>("transpose_in_shape");
        auto trans_out_shape = sgnode->get_attr<std::vector<int>>("transpose_out_shape");
        auto saved_in_ts     = inp->shape();                  // full TensorShape (layout + dims)
        // auto saved_in_layout = inp->shape().layout();

        inp->shape() = TensorShape(trans_in_shape);
        inp->shape().set_layout(DataLayout_Flat);

        auto outp            = sgnode->outputs()[0];
        auto saved_out_ts    = outp->shape();
        outp->shape()        = TensorShape(trans_out_shape);
        outp->shape().set_layout(DataLayout_Flat);

        int dim = 0;
        std::vector<int> temp_shape, temp_perm;
        get_new_shape_perm_dim(sgnode, node, temp_shape, temp_perm, dim);

        int kernel_dtype_size = Tensor::dtype_size(node->inputs()[0]->type());
        kernel_dtype_size = merge_dma_data_unit(temp_shape, temp_perm, dim, kernel_dtype_size);

        push_transpose_param(ro, temp_shape, dim);
        push_transpose_param(ro, temp_perm, dim);
        ro.push_back(dim);

        // Restore original shapes before appending tensor references.
        inp->shape()  = saved_in_ts;
        outp->shape() = saved_out_ts;

        ro.push_back(sgnode->inputs()[0]);
        ro.push_back(sgnode->outputs()[0]);
        ro.push_back(0);  // workspace

        return ro;
    }

    // Generate binary code name for the transpose kernel.
    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            return "";
        }

        auto node            = sgnode;
        auto inp             = sgnode->inputs()[0];
        auto outp            = sgnode->outputs()[0];
        auto saved_in_ts     = inp->shape();
        auto saved_out_ts    = outp->shape();
        // auto saved_in_layout = inp->shape().layout();
        auto trans_in_shape  = sgnode->get_attr<std::vector<int>>("transpose_in_shape");
        auto trans_out_shape = sgnode->get_attr<std::vector<int>>("transpose_out_shape");

        // Temporarily replace tensor shapes with the expanded transpose shapes
        // (Flat layout, no NCHWC/NHWC reinterpretation) so that get_new_shape_perm_dim
        // works purely on the kernel's logical axes.
        inp->shape()  = TensorShape(trans_in_shape);
        inp->shape().set_layout(DataLayout_Flat);
        outp->shape() = TensorShape(trans_out_shape);
        outp->shape().set_layout(DataLayout_Flat);

        int dim = 0;
        std::vector<int> temp_shape, temp_perm;
        get_new_shape_perm_dim(sgnode, node, temp_shape, temp_perm, dim);
        int kernel_dtype_size =
            merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(node->inputs()[0]->type()));

        // Restore original shapes.
        inp->shape()  = saved_in_ts;
        outp->shape() = saved_out_ts;

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

    // Generate kernel function name for the transpose operation.
    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            return "";
        }

        auto inp             = sgnode->inputs()[0];
        auto outp            = sgnode->outputs()[0];
        auto saved_in_ts     = inp->shape();
        auto saved_out_ts    = outp->shape();
        auto trans_in_shape  = sgnode->get_attr<std::vector<int>>("transpose_in_shape");
        auto trans_out_shape = sgnode->get_attr<std::vector<int>>("transpose_out_shape");

        // Same shape substitution as generate_code_name.
        inp->shape()  = TensorShape(trans_in_shape);
        inp->shape().set_layout(DataLayout_Flat);
        outp->shape() = TensorShape(trans_out_shape);
        outp->shape().set_layout(DataLayout_Flat);

        int dim = 0;
        std::vector<int> temp_shape, temp_perm;
        get_new_shape_perm_dim(sgnode, sgnode, temp_shape, temp_perm, dim);
        int kernel_dtype_size =
            merge_dma_data_unit(temp_shape, temp_perm, dim, Tensor::dtype_size(sgnode->inputs()[0]->type()));

        // Restore original shapes.
        inp->shape()  = saved_in_ts;
        outp->shape() = saved_out_ts;

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

    // Memory remapping: when no transpose is needed the output can share the
    // input buffer (the LC+Reshape+LC is a pure logical relabelling).
    void memory_remapping(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        bool need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        if (!need_transpose)
        {
            auto out = sgnode->outputs()[0];
            auto inp = sgnode->inputs()[0];
            out->memory_offset()->relative_offset = 0;
            out->memory_offset()->set_base_offset(inp->memory_offset());
        }
    }

private:
    // -----------------------------------------------------------------------
    // State carried from check_params() into setup().
    //
    // check_params() runs before the fused sgnode exists, so intermediate
    // results must be stashed here and then published as node attributes in
    // setup() for the downstream code-generation methods to consume.
    // -----------------------------------------------------------------------

    /// Whether the fused LC+Reshape+LC requires a real transpose kernel.
    bool m_need_transpose = false;

    /// Effective permutation over m_new_reshape0 computed in check_params().
    /// trans_out_shape[i] = m_new_reshape0[m_new_perm[i]].
    /// May be larger than 5 when a single NCHWC axis is split across the reshape.
    std::vector<int> m_new_perm;

    /// Expanded input shape for the transpose kernel.
    /// Derived by flattening before_trans (the per-NCHWC-axis factor groups).
    /// When no axis is split, equals the original NCHWC in_shape; otherwise
    /// some axes are replaced by multiple sub-axis sizes.
    std::vector<int> m_new_reshape0;
};

REGISTER_PLUGIN(InternalLayoutReshapeLayout, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_Internal);
} // namespace aipubt
