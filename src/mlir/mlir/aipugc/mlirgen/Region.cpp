#include "aipugc/MLIRGenRegister.h"
#include "aipugc/utils.h"

namespace mlir {
namespace aipugc {

class MLIRGenRegion : public MLIRGenBase {
public:
  MLIRGenRegion() = default;
  ~MLIRGenRegion() = default;

  virtual std::vector<mlir::Value>
  mlirGen(mlir::OpBuilder &builder, aipubt::NodePtr node,
          std::map<std::string, mlir::Value> &symbolTable) override {
    auto ts_input = node->inputs()[0];
    auto ts_scores = node->outputs()[0];
    auto ts_coordinates = node->outputs()[1];
    auto ts_boxes_number = node->outputs()[2];
    auto ts_valid_classes = node->outputs()[3];
    auto ts_total_class = node->outputs()[4];

    auto scoresType = utils::getType(builder, ts_scores);
    auto coordinatesType = utils::getType(builder, ts_coordinates);
    auto boxesNumberType = utils::getType(builder, ts_boxes_number);
    auto validClassesType = utils::getType(builder, ts_valid_classes);
    auto totalClassType = utils::getType(builder, ts_total_class);

    auto location = loc(builder, node);

    mlir::Value mlir_input = getsymbol(symbolTable, ts_input->name());
    if (!mlir_input) {
      llvm::errs() << "Region missing input: " << ts_input->name() << "\n";
      return {};
    }

    // Get parameters from node
    aipubt::NodeParamValue obj_thresh_param = node->params()["obj_thresh"];
    float obj_thresh = 0.5f;
    if (obj_thresh_param.t == aipubt::NodeParamValue::NodeParamType_float) {
      obj_thresh = static_cast<float>(obj_thresh_param.v.f);
    }

    int32_t max_box_num = node->params()["max_box_num"];
    int32_t box_per_grid = node->params()["box_per_grid"];
    int32_t class_num = node->params()["class_num"];
    int32_t grid_width = node->params()["grid_width"];
    int32_t grid_height = node->params()["grid_height"];
    bool grid_compensate = node->params()["grid_compensate"];

    // Float anchors in params (same as core region.cpp default). Text IR often
    // stores anchor LUT as qanchors_lut in weights only; params["anchors"] may
    // be empty or wrong length — RegionOp::verify requires 2*box_per_grid
    // values, so normalize here.
    static const float kDefaultYoloV2Anchors[] = {
        1.3221f,  1.73145f, 3.19275f, 4.00944f, 5.05587f, 8.09892f,
        9.47112f, 4.84053f, 11.2364f, 10.0071f};
    constexpr size_t kNumDefaultAnchors =
        sizeof(kDefaultYoloV2Anchors) / sizeof(kDefaultYoloV2Anchors[0]);
    std::vector<float> anchors = node->params()["anchors"];
    const size_t needAnchors = static_cast<size_t>(box_per_grid) * 2;
    if (anchors.size() != needAnchors) {
      anchors.assign(kDefaultYoloV2Anchors,
                     kDefaultYoloV2Anchors + kNumDefaultAnchors);
      if (anchors.size() > needAnchors)
        anchors.resize(needAnchors);
      while (anchors.size() < needAnchors) {
        anchors.push_back(1.0f);
        anchors.push_back(1.0f);
      }
    }

    // Create attributes
    auto objThreshAttr = builder.getFloatAttr(builder.getF32Type(), obj_thresh);
    auto maxBoxNumAttr = builder.getIntegerAttr(builder.getIntegerType(32), max_box_num);
    auto boxPerGridAttr = builder.getIntegerAttr(builder.getIntegerType(32), box_per_grid);
    auto classNumAttr = builder.getIntegerAttr(builder.getIntegerType(32), class_num);
    auto gridWidthAttr = builder.getIntegerAttr(builder.getIntegerType(32), grid_width);
    auto gridHeightAttr = builder.getIntegerAttr(builder.getIntegerType(32), grid_height);
    auto gridCompensateAttr = builder.getBoolAttr(grid_compensate);

    // Convert anchors to DenseF64ArrayAttr
    SmallVector<double, 10> anchorsDouble(anchors.begin(), anchors.end());
    auto anchorsAttr = DenseF64ArrayAttr::get(builder.getContext(), anchorsDouble);

    // Get dictionary attributes (quantization info, etc.)
    DictionaryAttr opAttrs = ParamsToMlirAttr(builder, node);

    // Create RegionOp
    auto regionOp = builder.create<RegionOp>(
        location,
        scoresType,
        coordinatesType,
        boxesNumberType,
        validClassesType,
        totalClassType,
        mlir_input,
        objThreshAttr,
        maxBoxNumAttr,
        boxPerGridAttr,
        classNumAttr,
        gridWidthAttr,
        gridHeightAttr,
        gridCompensateAttr,
        anchorsAttr,
        opAttrs);

    // Return all 5 outputs
    return {
        regionOp.getScores(),
        regionOp.getCoordinates(),
        regionOp.getBoxesNumber(),
        regionOp.getValidClasses(),
        regionOp.getTotalClass()};
  }
};

REGISTER_OP_MLIRGEN(aipubt::OpType_Region, Region);

} // namespace aipugc
} // namespace mlir
