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
/*
 * @Author: Lang Hu
 * @Date: 2026-03-26 11:21:13
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-03-26 11:21:13
 */
 #include "builder/lowering/lowering.h"
 #include "builder/target.h"
 #include "core/config.h"
 #include "core/core.h"
 #include "core/ops/ops.h"
 
 namespace aipubt
 {
 static inline bool criteria(NodePtr n)
 {
     if (n->attrs().count(Constants::virtual_multi_core))
     {
         return n->get_attr<bool>(Constants::virtual_multi_core);
     }
     return false;
 }
 
 void virtual_core_adaptor(GraphPtr g)
 {
     auto nodes = g->nodes();
     for (auto &&node : nodes)
     {
         if (!criteria(node))
         {
             continue;
         }
 
         int core_num = get_target()->core_num();
         if (core_num < 2)
         {
             node->set_attr(Constants::virtual_core_num, 1);
             node->set_attr(Constants::virtual_core_idx, 0);
         }
         else
         {
             bool is_dynamic = node->is_dynamic();
             auto final_outputs = node->outputs();
             node->clear_outputs();
 
             std::vector<std::vector<TensorPtr>> switch_inputs;
             for (int i = 0; i < core_num; ++i)
             {
                 auto n = node;
                 if (i == 0)
                 {
                     for (size_t j = 0; j < final_outputs.size(); ++j)
                     {
                         auto t = std::make_shared<Tensor>(*final_outputs[j]);
                         t->set_io_tensor(false);
                         t->set_name(g->get_valid_tensor_name(t->name() + "/" + std::to_string(j)));
                         n->add_output(t);
                     }
                     MODIFY_NODE(n);
                 }
                 else
                 switch_node->add_output(final_outputs[i]);
                 if (is_dynamic)
                 {
                     std::vector<Expr> s = final_outputs[i]->shape().symbol();
                     switch_node->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{s};
                 }
                 switch_node->infer_shape(true);
                 g->add_node(switch_node);
                 ADD_LOCATION_BY_NODE(switch_node, node);
             }
         }
     }
 }
 } // namespace aipubt
 