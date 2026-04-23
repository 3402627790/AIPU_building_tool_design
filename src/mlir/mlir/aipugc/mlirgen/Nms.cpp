//===- Nms.cpp - MLIR Generation function 'mlirGen' for NMS OP.
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements MLIR generation function 'mlirGen' for NMS.
//
//===----------------------------------------------------------------------===//
#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenNMS : public MLIRGenBase {
public:
  MLIRGenNMS() = default;
  ~MLIRGenNMS() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    if (node->inputs().size() < 4) {
      llvm::errs() << "NMS requires 4 inputs, got " << node->inputs().size()
                   << "\n";
      return {};
    }
    if (node->outputs().size() < 4) {
      llvm::errs() << "NMS requires 4 outputs, got " << node->outputs().size()
                   << "\n";
      return {};
    }

    // Inputs: box_coor, box_num_per_class, class_num_per_batch, score
    auto ts_box_coor = node->inputs()[0];
    auto ts_box_num_per_class = node->inputs()[1];
    auto ts_class_num_per_batch = node->inputs()[2];
    auto ts_score = node->inputs()[3];

    mlir::Value v_box_coor = getsymbol(symbolTable, ts_box_coor->name());
    mlir::Value v_box_num_per_class =
        getsymbol(symbolTable, ts_box_num_per_class->name());
    mlir::Value v_class_num_per_batch =
        getsymbol(symbolTable, ts_class_num_per_batch->name());
    mlir::Value v_score = getsymbol(symbolTable, ts_score->name());

    if (!v_box_coor || !v_box_num_per_class || !v_class_num_per_batch ||
        !v_score) {
      llvm::errs() << "NMS missing input value for node: " << node->name()
                   << "\n";
      return {};
    }

    // Outputs: selected_coordinates, selected_box_number,
    //          selected_score, selected_indices
    auto selCoorType = utils::getType(builder, node->outputs()[0]);
    auto selBoxNumType = utils::getType(builder, node->outputs()[1]);
    auto selScoreType = utils::getType(builder, node->outputs()[2]);
    auto selIndicesType = utils::getType(builder, node->outputs()[3]);

    auto location = loc(builder, node);
    auto &params = node->params();

    // score_threshold / iou_threshold may be float or quantized integer.
    auto readFloatLike = [&](const std::string &key,
                             float defaultVal) -> float {
      auto it = params.find(key);
      if (it == params.end())
        return defaultVal;
      const aipubt::NodeParamValue &v = it->second;
      switch (v.t) {
      case aipubt::NodeParamValue::NodeParamType_float:
        return static_cast<float>(v.v.f);
      case aipubt::NodeParamValue::NodeParamType_int:
        return static_cast<float>(v.v.i);
      case aipubt::NodeParamValue::NodeParamType_uint:
        return static_cast<float>(v.v.ui);
      default:
        return defaultVal;
      }
    };

    float score_threshold = readFloatLike("score_threshold", 0.0f);
    float iou_threshold = readFloatLike("iou_threshold", 0.0f);
    float soft_nms_sigma = readFloatLike("soft_nms_sigma", 0.0f);

    int32_t max_output_size = 0;
    {
      auto it = params.find("max_output_size");
      if (it != params.end()) {
        const aipubt::NodeParamValue &v = it->second;
        if (v.t == aipubt::NodeParamValue::NodeParamType_int)
          max_output_size = static_cast<int32_t>(v.v.i);
        else if (v.t == aipubt::NodeParamValue::NodeParamType_uint)
          max_output_size = static_cast<int32_t>(v.v.ui);
      }
    }

    bool center_point_box = false;
    {
      auto it = params.find("center_point_box");
      if (it != params.end()) {
        center_point_box = it->second;
      }
    }

    std::string method = "HARD";
    {
      auto it = params.find("method");
      if (it != params.end()) {
        std::string m = it->second;
        if (!m.empty())
          method = m;
      }
    }

    // Create MLIR attributes.
    auto scoreThresholdAttr =
        builder.getFloatAttr(builder.getF32Type(), score_threshold);
    auto iouThresholdAttr =
        builder.getFloatAttr(builder.getF32Type(), iou_threshold);
    auto maxOutputSizeAttr = builder.getI32IntegerAttr(max_output_size);
    auto softNmsSigmaAttr =
        builder.getFloatAttr(builder.getF32Type(), soft_nms_sigma);
    auto centerPointBoxAttr = builder.getBoolAttr(center_point_box);
    auto methodAttr = builder.getStringAttr(method);

    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    auto nmsOp = builder.create<NmsOp>(
        location,
        selCoorType,
        selBoxNumType,
        selScoreType,
        selIndicesType,
        v_box_coor,
        v_box_num_per_class,
        v_class_num_per_batch,
        v_score,
        scoreThresholdAttr,
        iouThresholdAttr,
        maxOutputSizeAttr,
        softNmsSigmaAttr,
        centerPointBoxAttr,
        methodAttr,
        opAttrs);

    return {
        nmsOp.getSelectedCoordinates(),
        nmsOp.getSelectedBoxNumber(),
        nmsOp.getSelectedScore(),
        nmsOp.getSelectedIndices()};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_NMS, NMS);

} // namespace aipugc
} // namespace mlir
