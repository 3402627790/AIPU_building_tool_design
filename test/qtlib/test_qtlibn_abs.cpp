#include "test_qtlib.h"

static void init_asym_param(aipubt::NodePtr node)
{
    node->inputs()[0]->quantization().set_scale(1.0);
    node->inputs()[0]->quantization().set_offset(0);
    node->inputs()[0]->quantization().set_aipu_dtype(aipubt::Tensor::UINT8);
    node->inputs()[0]->quantization().set_bits(8);
    node->inputs()[0]->quantization().set_granularity(aipubt::Quantization::Granularity::Per_Tensor);
    node->inputs()[0]->quantization().set_scheme(aipubt::Quantization::Symmetric);
    // node->inputs()[0]->quantization().set_scheme(aipubt::Quantization::Asymmetric);
    node->inputs()[0]->quantization().set_quantized_range(aipubt::Quantization::Restricted_Range);

    node->outputs()[0]->quantization().set_scale(0.5);
    node->outputs()[0]->quantization().set_offset(0);
    node->outputs()[0]->quantization().set_aipu_dtype(aipubt::Tensor::UINT8);
    node->outputs()[0]->quantization().set_bits(8);
    node->outputs()[0]->quantization().set_granularity(aipubt::Quantization::Granularity::Per_Tensor);
    node->outputs()[0]->quantization().set_scheme(aipubt::Quantization::Symmetric);
    // node->outputs()[0]->quantization().set_scheme(aipubt::Quantization::Asymmetric);
    node->outputs()[0]->quantization().set_quantized_range(aipubt::Quantization::Restricted_Range);
}
static void test_abs(void)
{
    aipubt::GraphPtr abs_graph = aipubt::Graph::create("abs_graph");
    aipubt::OpType op_type = aipubt::OpType_Abs;
    aipubt::NodePtr input_node = aipubt::Node::create("input_op", aipubt::OpType_Input);
    aipubt::TensorPtr input_output_t = std::make_shared<aipubt::Tensor>("input_output_tensor", aipubt::TensorShape(1, 2, 3), aipubt::Tensor::UINT8);
    input_node->add_output(input_output_t);
    aipubt::NodePtr abs_node0 = aipubt::Node::create("abs_0", op_type);
    aipubt::TensorPtr abs_output_t = std::make_shared<aipubt::Tensor>("abs_output_tensor", aipubt::TensorShape(1, 2, 3), aipubt::Tensor::UINT8);
    abs_node0->add_input(input_output_t);
    abs_node0->add_output(abs_output_t);
    init_asym_param(abs_node0);
    DEBUG() << (int)abs_node0->outputs()[0]->quantization().scheme();
    abs_graph->add_node(input_node);
    abs_graph->add_node(abs_node0);

    DEBUG() << "[before:input_asym_scale]: " << abs_node0->inputs()[0]->quantization().scale();
    DEBUG() << "[before:input_asym_offset]: " << (float)abs_node0->inputs()[0]->quantization().offset();
    DEBUG() << "[before:output_asym_scale]: " << abs_node0->outputs()[0]->quantization().scale();
    DEBUG() << "[before:output_asym_offset]: " << (float)abs_node0->outputs()[0]->quantization().offset();
    qtlibn::QuantizationTransform *quantize_abs = new qtlibn::QuantizationTransform(abs_graph, std::string("Sym"), 8);
    quantize_abs->run();
    DEBUG() << "[after:input_sym_scale]: " << abs_node0->inputs()[0]->quantization().scale();
    DEBUG() << "[after:input_sym_offset]: " << (float)abs_node0->inputs()[0]->quantization().offset();
    DEBUG() << "[after:output_sym_scale]: " << abs_node0->outputs()[0]->quantization().scale();
    DEBUG() << "[after:output_sym_offset]: " << (float)abs_node0->outputs()[0]->quantization().offset();
    DEBUG() << "[after:output_min]: " << abs_node0->outputs()[0]->quantization().mins()[0];
    DEBUG() << "[after:output_max]: " << abs_node0->outputs()[0]->quantization().maxs()[0];
}

void test_qtlibn_abs()
{
    test_abs();
}
