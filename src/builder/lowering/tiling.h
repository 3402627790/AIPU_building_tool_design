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
 * @Author: Tom Gao
 * @Date: 2022-11-23 10:25:29
 * @Last Modified by: Ryan Lai
 * @Last Modified time: 2024-04-09 10:48:47
 */
#include "core/basepass.h"
#include "core/core.h"
#include "core/optype.h"
#include "core/tensor.h"
#include "utils/json.h"
#include <functional>
#include <memory>

namespace aipubt
{
struct TSlice
{
    uint32_t offset;
    uint32_t size;
    uint32_t share; // share tail(overlap) of the slice
                    // from this slice to next slice
};
struct MultiDimTile
{
    uint32_t axis;
    uint32_t num;
};
struct CompareNodeByName {
    bool operator ()(NodePtr a, NodePtr b) const {
        if (a == nullptr && b == nullptr) return false;
        if (a == nullptr) return true;
        if (b == nullptr) return false;
        return a->name() < b->name();
    }
};
using TileInfo = std::vector<TSlice>;
using TilePlan = std::map<NodePtr, TileInfo, CompareNodeByName>;
using TileNumMap = std::map<NodePtr, uint32_t, CompareNodeByName>;
using TileNumMapMultiDim = std::map<NodePtr, std::vector<MultiDimTile>, CompareNodeByName>;
using TileSubGraphMap = std::map<NodePtr, NodePtrList, CompareNodeByName>;
using NodeTensorInfoMap = std::map<NodePtr, std::map<TensorPtr, uint32_t>>;

/**
 * tile_graph_by_gm_size: to tile the input graph @g under the constraint:
 * if gm_size = -1
 * tiled_output_size < gm_size/core_num
 * else as @tile_num specified
 */
bool tile_graph_by_size(GraphPtr g, uint32_t axis, uint32_t tile_num, uint32_t gm_size);
bool tile_graph_by_size(GraphPtr g, uint32_t axis, uint32_t tile_num, uint32_t gm_size, TileSubGraphMap &mp_bp_map, TileNumMap &max_num_map, double threshold = 1.6);
bool is_supported_op(const NodePtr &n, int32_t axis = 1, bool is_supported_op = false);

bool is_special_eltwise(const NodePtr &node);

void tiling_set_constant(NodePtr node, NodePtr subnode, std::string name, int start, int size, int axis = 0);

/* It's a trick way to make a bairrer between core (CP-20953)
 *
 * Each slice has a flag: "keep_isolate_when_scheduling" to let scheduler know: "don't merge me"
 * Now the flag exist when user using tiling cfg only (Not for auto tiling)
 */
bool keep_isolate_when_scheduling(const NodePtr &n);

bool check_manual_tile_n(NodePtr n);
bool check_auto_tile_n(NodePtr n);
bool check_batched_graph_tile_n(GraphPtr g);
NodePtrList apply_tile_n(NodePtr n, std::vector<int> &batches);
void apply_tile_batched_graph(GraphPtr g, std::string tile_cfg, bool dump_tile_cfg);

void tiling_update_params(NodePtr n, std::string param, uint32_t axis,
                          int32_t size_0 = 0, int32_t size_1 = 0);
void dump_cfg_to_json(utils::JsonObject &json,
                      std::string sec_name,
                      const TileSubGraphMap &mp_bp_map,
                      const TileNumMap &max_num_map);

}; // namespace aipubt
