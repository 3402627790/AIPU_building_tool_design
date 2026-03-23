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
#include <builder/plugin.h>
#include <core/graph.h>
#include <core/location.h>
#include <core/node.h>
#include <core/subgraph.h>
#include <gcompiler_api.h>
#include <utils/utils.h>

namespace aipubt
{

class InternalLayoutReshapeLayout : public BuilderOpPlugin
{
public:
    const GraphMatcherPattern get_graph_pattern()
    {
        GraphMatcherPattern pattern = {
            {{"n0", OpType_LayoutConvertor}, {"n1", OpType_Reshape}, {"n2", OpType_LayoutConvertor}},
            {{"n0", "n1"}, {"n1", "n2"}}};
        return pattern;
    }

    uint32_t get_score()
    {
        return 100;
    }

    bool set_target(const char *target)
    {
        return true;
    };

    bool check_params(BaseNodePtrList nodes)
    {
        auto inp = nodes[0]->inputs()[0];
        auto outp = nodes[2]->outputs()[0];
        auto in_layout = inp->shape().layout();
        auto out_layout = outp->shape().layout();

        INFO() << "[InternalLayoutReshapeLayout] check_params:"
               << " n0=" << nodes[0]->name()
               << " n1=" << nodes[1]->name()
               << " n2=" << nodes[2]->name();
        INFO() << "[InternalLayoutReshapeLayout]   inp: shape=" << inp->shape().to_string()
               << " layout=" << in_layout;
        INFO() << "[InternalLayoutReshapeLayout]   outp: shape=" << outp->shape().to_string()
               << " layout=" << out_layout;

        if (in_layout != out_layout)
        {
            INFO() << "[InternalLayoutReshapeLayout]   REJECTED: input layout(" << in_layout
                   << ") != output layout(" << out_layout << ")";
            return false;
        }

        if (in_layout != DataLayout_NCHWC16 && in_layout != DataLayout_NCHWC32)
        {
            INFO() << "[InternalLayoutReshapeLayout]   REJECTED: not NCHWC16/NCHWC32 layout";
            return false;
        }

        auto reshape_in_layout = nodes[1]->inputs()[0]->shape().layout();
        auto reshape_out_layout = nodes[1]->outputs()[0]->shape().layout();
        if (reshape_in_layout != DataLayout_NHWC ||
            reshape_out_layout != DataLayout_NHWC)
        {
            INFO() << "[InternalLayoutReshapeLayout]   REJECTED: reshape not in NHWC"
                   << " (in=" << reshape_in_layout << " out=" << reshape_out_layout << ")";
            return false;
        }

        NodePtr real_reshape = get_real_node(nodes[1]->inputs()[0]->consumers()[0]);
        if (real_reshape->is_dynamic())
        {
            INFO() << "[InternalLayoutReshapeLayout]   REJECTED: dynamic reshape";
            return false;
        }

        std::vector<int> in_shape = inp->shape();
        std::vector<int> out_shape = outp->shape();

        if (in_shape.size() != 5 || out_shape.size() != 5)
        {
            INFO() << "[InternalLayoutReshapeLayout]   REJECTED: shape dim != 5"
                   << " (in_dim=" << in_shape.size() << " out_dim=" << out_shape.size() << ")";
            return false;
        }

        // C/x and x must be the same
        if (in_shape[1] != out_shape[1] || in_shape[4] != out_shape[4])
        {
            INFO() << "[InternalLayoutReshapeLayout]   REJECTED: channel mismatch"
                   << " C_in=" << in_shape[1] << "x" << in_shape[4]
                   << " C_out=" << out_shape[1] << "x" << out_shape[4];
            return false;
        }

        long long in_spatial = (long long)in_shape[0] * in_shape[2] * in_shape[3];
        long long out_spatial = (long long)out_shape[0] * out_shape[2] * out_shape[3];
        if (in_spatial != out_spatial)
        {
            INFO() << "[InternalLayoutReshapeLayout]   REJECTED: N*H*W mismatch"
                   << " (" << in_spatial << " vs " << out_spatial << ")";
            return false;
        }

        bool need_transpose = (in_shape[0] != out_shape[0]);
        if (need_transpose)
        {
            // For the swap-N-H approach: require H_in == N_out
            if (in_shape[2] != out_shape[0])
            {
                INFO() << "[InternalLayoutReshapeLayout]   REJECTED: transpose case but H_in("
                       << in_shape[2] << ") != N_out(" << out_shape[0]
                       << "), unsupported transpose pattern";
                return false;
            }
        }

        INFO() << "[InternalLayoutReshapeLayout]   ACCEPTED: need_transpose=" << need_transpose;
        return true;
    }

    void setup(BaseNodePtr sgnode, BaseNodePtrList match_list)
    {
        // Capture shapes before unfold (vector<int> copies are safe)
        std::vector<int> in_shape = match_list[0]->inputs()[0]->shape();
        std::vector<int> out_shape = match_list[2]->outputs()[0]->shape();

        bool need_transpose = (in_shape[0] != out_shape[0]);
        sgnode->set_attr("need_transpose", need_transpose);

        if (!need_transpose)
        {
            INFO() << "[InternalLayoutReshapeLayout] setup: no transpose needed, will use memory_remapping";
            return;
        }

        INFO() << "[InternalLayoutReshapeLayout] setup: transpose needed, modifying graph...";
        INFO() << "[InternalLayoutReshapeLayout]   in_shape=[" << in_shape[0] << "," << in_shape[1]
               << "," << in_shape[2] << "," << in_shape[3] << "," << in_shape[4] << "]";
        INFO() << "[InternalLayoutReshapeLayout]   out_shape=[" << out_shape[0] << "," << out_shape[1]
               << "," << out_shape[2] << "," << out_shape[3] << "," << out_shape[4] << "]";

        // Unfold subgraph first — this restores original nodes to the parent graph
        // and invalidates any ProxyTensor references from the subgraph
        auto sgn = SubGraphNode::from_node(sgnode);
        auto unfolded = sgn->unfold();

        auto n0 = std::dynamic_pointer_cast<Node>(match_list[0]);
        auto n1 = std::dynamic_pointer_cast<Node>(match_list[1]);
        auto n2 = std::dynamic_pointer_cast<Node>(match_list[2]);

        auto graph = n1->graph();
        if (!graph)
        {
            INFO() << "[InternalLayoutReshapeLayout]   ERROR: cannot get graph from reshape node";
            return;
        }

        // Re-capture all tensor references AFTER unfold — now they are real tensors
        auto inp_tensor = n0->inputs()[0];
        auto outp_tensor = n2->outputs()[0];
        auto nhwc_tensor_1 = n0->outputs()[0];
        auto nhwc_tensor_2 = n1->outputs()[0];

        INFO() << "[InternalLayoutReshapeLayout]   After unfold: inp=" << inp_tensor->name()
               << " outp=" << outp_tensor->name();

        // Transpose: swap N(dim0) and H(dim2) in NCHWC16 [N,C/x,H,W,x]
        // perm = [2, 1, 0, 3, 4]
        std::vector<int> trans_out_shape = {in_shape[2], in_shape[1], in_shape[0], in_shape[3], in_shape[4]};

        std::string trans_name = graph->get_valid_node_name(n0->name() + "/c16_transpose");
        NodePtr trans_node = Node::create(trans_name, OpType_Transpose);
        ADD_LOCATION_BY_NODE(trans_node, n0);
        graph->add_node(trans_node);

        trans_node->params()["perm"] = NodeParamValue({2, 1, 0, 3, 4});
        trans_node->add_input(inp_tensor);

        TensorShape trans_out_ts(trans_out_shape, inp_tensor->shape().layout());
        TensorPtr trans_out = std::make_shared<Tensor>(trans_name + "_out", trans_out_ts, inp_tensor->type());
        trans_out->shape().set_layout(inp_tensor->shape().layout());
        trans_out->quantization() = inp_tensor->quantization();
        trans_node->add_output(trans_out);

        INFO() << "[InternalLayoutReshapeLayout]   Created transpose: " << trans_node->name()
               << " shape=[" << trans_out_shape[0] << "," << trans_out_shape[1]
               << "," << trans_out_shape[2] << "," << trans_out_shape[3]
               << "," << trans_out_shape[4] << "]";

        // Reconnect reshape: replace NHWC input with C16 transpose output
        n1->replace_input_by(nhwc_tensor_1, trans_out);

        // Disconnect n2 from n1's old output
        n2->remove_input(nhwc_tensor_2);

        // Transfer output tensor from n2 to n1
        n2->remove_output(outp_tensor);
        n1->remove_output(nhwc_tensor_2);
        n1->add_output(outp_tensor);

        // Update reshape params to reflect C16 shape
        NodeParamValue shape_val;
        for (int i = 0; i < (int)out_shape.size(); ++i)
        {
            shape_val.push_back(NodeParamValue(out_shape[i]));
        }
        n1->params()["shape"] = shape_val;

        graph->remove_node(n0);
        graph->remove_node(n2);

        INFO() << "[InternalLayoutReshapeLayout]   Graph modified: "
               << trans_node->name() << " -> " << n1->name()
               << " -> downstream consumers of " << outp_tensor->name();
    }

    BuilderParams generate_params(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        return BuilderParams();
    }

    const std::string generate_code_name(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        return "";
    }

    void memory_remapping(BaseNodePtr sgnode, BaseNodePtrList &nodes)
    {
        bool need_transpose = false;
        if (sgnode->attrs().count("need_transpose"))
        {
            need_transpose = cast_any<bool>(sgnode->get_attr("need_transpose"));
        }

        if (!need_transpose)
        {
            INFO() << "[InternalLayoutReshapeLayout] memory_remapping: no-op reshape in C16";
            auto out = sgnode->outputs()[0];
            auto inp = sgnode->inputs()[0];
            out->memory_offset()->relative_offset = 0;
            out->memory_offset()->set_base_offset(inp->memory_offset());
        }
        else
        {
            INFO() << "[InternalLayoutReshapeLayout] memory_remapping: transpose case "
                   << "(subgraph was unfolded in setup, this should not be reached)";
        }
    }
};

REGISTER_PLUGIN(InternalLayoutReshapeLayout, 0x01010000, BUILDERPLUGIN_TYPE | GBPlugin_Internal);
} // namespace aipubt
