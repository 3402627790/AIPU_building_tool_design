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
#include "builder/optimizer.h"
#include "builder/target.h"
#include <builder/plugin.h>
#include <core/graph.h>
#include <core/node.h>
#include <utils/utils.h>

#include <algorithm>
#include <string>

namespace aipubt
{

namespace
{

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

        // Check 1: do the strides actually change across the reshape
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

    void setup(BaseNodePtr sgnode, BaseNodePtrList match_list) override
    {
        sgnode->set_attr("need_transpose", m_need_transpose);

        if (!m_need_transpose)
        {
            return;
        }

        // Compute the transpose output shape by applying m_new_perm to m_new_reshape0.
        // trans_out_shape[i] = m_new_reshape0[m_new_perm[i]]  (output[i] = input[perm[i]])
        std::vector<int> trans_out_shape(m_new_perm.size());
        for (size_t i = 0; i < m_new_perm.size(); ++i)
        {
            trans_out_shape[i] = m_new_reshape0[m_new_perm[i]];
        }

        sgnode->params()["perm"] = NodeParamValue(m_new_perm);
        // Store the flat input shape that the transpose kernel will actually read.
        sgnode->set_attr("transpose_in_shape", m_new_reshape0);
    }

    // -----------------------------------------------------------------------
    // 1. Temporarily reshape sgnode's tensors to the flat layout
    // 2. Invoke the callback with the plugin and the node list
    // 3. Restore the shapes
    // -----------------------------------------------------------------------
    template<typename Fn>
    auto invoke_as_flat_transpose(BaseNodePtr sgnode, Fn &&fn)
        -> decltype(fn(std::declval<BuilderOpPlugin *>(),
                       std::declval<BaseNodePtrList &>()))
    {
        using R = decltype(fn(std::declval<BuilderOpPlugin *>(),
                              std::declval<BaseNodePtrList &>()));

        // ── Step 1: select and validate the plugin BEFORE patching shapes.
        uint32_t flag = 0;
        auto plugin = BuilderOptimizer::get_single_op_plugin(OpType_Transpose, flag);
        if (!plugin)
        {
            ERROR() << "[InternalLayoutReshapeLayout] Cannot obtain TransposePlugin for "
                    << sgnode->name();
            return R{};
        }
        BaseNodePtrList m = {sgnode};
        bool res = plugin->check_params(m);
        res &= plugin->set_target(get_target()->name().c_str());

        if (!res)
        {
            ERROR() << "[InternalLayoutReshapeLayout] TransposePlugin check_params/"
                       "set_target failed for " << sgnode->name();
            return R{};
        }

        // ── Step 2: patch tensor shapes to flat so TransposePlugin's
        // get_new_shape_perm_dim hits the Flat→Flat branch.
        auto flat_in_vec = sgnode->get_attr<std::vector<int>>("transpose_in_shape");
        std::vector<int> perm_vec = sgnode->params()["perm"];

        std::vector<int> flat_out_vec(perm_vec.size());
        for (size_t i = 0; i < perm_vec.size(); ++i)
            flat_out_vec[i] = flat_in_vec[perm_vec[i]];

        // Save dtype and current shapes before patching
        auto dtype = sgnode->inputs()[0]->type();
        auto orig_in_shape = sgnode->inputs()[0]->shape();
        auto orig_out_shape = sgnode->outputs()[0]->shape();

        TensorShape flat_in_shape(flat_in_vec, DataLayout_Flat);
        TensorShape flat_out_shape(flat_out_vec, DataLayout_Flat);

        sgnode->inputs()[0]->shape() = flat_in_shape;
        sgnode->outputs()[0]->shape() = flat_out_shape;

        // ── Step 3: invoke the caller's lambda with the validated plugin.
        auto result = fn(plugin, m);

        // ── Step 4: restore original shapes and dtype.
        sgnode->inputs()[0]->shape()  = orig_in_shape;
        sgnode->outputs()[0]->shape() = orig_out_shape;
        sgnode->inputs()[0]->set_type(dtype);
        sgnode->outputs()[0]->set_type(dtype);

        return result;
    }

    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        if (!cast_any<bool>(sgnode->get_attr("need_transpose")))
            return BuilderParams();

        return invoke_as_flat_transpose(sgnode, [&](BuilderOpPlugin *plugin, BaseNodePtrList &m) {
            return plugin->generate_params(sgnode, m);
        });
    }

    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        if (!cast_any<bool>(sgnode->get_attr("need_transpose")))
            return "";

        return invoke_as_flat_transpose(sgnode, [&](BuilderOpPlugin *plugin, BaseNodePtrList &m) {
            return plugin->generate_code_name(sgnode, m);
        });
    }

    const std::string generate_kernel(BaseNodePtr sgnode, BaseNodePtrList &nodes) override
    {
        if (!cast_any<bool>(sgnode->get_attr("need_transpose")))
            return "";

        return invoke_as_flat_transpose(sgnode, [&](BuilderOpPlugin *plugin, BaseNodePtrList &m) {
            return plugin->generate_kernel(sgnode, m);
        });
    }

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
    bool m_need_transpose = false;       // true if a real Transpose kernel is required
    std::vector<int> m_new_perm;         // effective perm on the expanded flat shape
    std::vector<int> m_new_reshape0;     // expanded flat input sizes the transpose kernel reads
};

REGISTER_PLUGIN(InternalLayoutReshapeLayout, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_Internal);
} // namespace aipubt