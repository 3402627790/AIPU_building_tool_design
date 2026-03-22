#include "test_qtlib.h"

void test_qtlibn_depthwise()
{

    aipubt::GraphPtr dw_graph = aipubt::Graph::create("dw_graph");
    aipubt::NodePtr input_node = aipubt::Node::create("input_op", aipubt::OpType_Input);
    aipubt::TensorPtr input_o_t = std::make_shared<aipubt::Tensor>("input_o_t", aipubt::TensorShape(1, 2, 2, 2), aipubt::Tensor::UINT8);
    input_o_t->quantization().set_scale(0.5);
    input_o_t->quantization().set_offset(0);
    input_node->add_output(input_o_t);

    auto dw_node = aipubt::Node::create("dw_node", aipubt::OpType_DepthwiseConv);
    auto dw_o_t = std::make_shared<aipubt::Tensor>("dw_o_t", aipubt::TensorShape(1, 1, 1, 2), aipubt::Tensor::UINT8);
    dw_o_t->quantization().set_scale(1.0);
    dw_o_t->quantization().set_offset(0);
    dw_node->add_input(input_o_t);
    dw_node->add_output(dw_o_t);

    uint8_t *w_data = new uint8_t[8]{2, 2, 2, 2, 4, 0, 2, 0};
    auto w_t = std::make_shared<aipubt::Tensor>("weight_t", aipubt::TensorShape(2, 2, 2, 1), aipubt::Tensor::UINT8);
    // w_t->quantization().set_scale(0.5);
    // w_t->quantization().set_offset(0);
    w_t->quantization().set_scales({0.5, 0.5});
    w_t->quantization().set_offsets({0, 0});
    w_t->set_data(w_data, 8 * sizeof(uint8_t));

    int32_t *b_data = new int32_t[2]{0, 0};
    auto b_t = std::make_shared<aipubt::Tensor>("bias_t", aipubt::TensorShape(2), aipubt::Tensor::INT32);
    // b_t->quantization().set_scale(0.25);
    // b_t->quantization().set_offset(0);
    b_t->quantization().set_scales({0.25, 0.25});
    b_t->quantization().set_offsets({0, 0});
    b_t->set_data(b_data, 2 * sizeof(int32_t));

    dw_node->constants()["weights"] = w_t;
    dw_node->constants()["biases"] = b_t;

    dw_node->params()["kernel_y"] = 2;
    dw_node->params()["kernel_x"] = 2;
    dw_node->params()["stride_y"] = 1;
    dw_node->params()["stride_x"] = 1;
    dw_node->params()["dilation_y"] = 1;
    dw_node->params()["dilation_x"] = 1;
    dw_node->params()["pad_bottome"] = 0;
    dw_node->params()["pad_top"] = 0;
    dw_node->params()["pad_left"] = 0;
    dw_node->params()["pad_right"] = 0;
    dw_node->params()["group"] = 2;
    dw_node->params()["with_activation"] = "RELU";

    dw_graph->add_node(input_node);
    dw_graph->add_node(dw_node);
    auto *qtransformer = new qtlibn::QuantizationTransform(dw_graph, std::string("Sym"), 8);
    qtransformer->run();
    auto nodes = dw_graph->nodes();
    for (auto n : nodes)
    {
        for (auto ot : n->outputs())
        {
            auto q = ot->quantization();
            DEBUG() << ot->name() << ": [output] scale = " << q.scale() << ", zerop = " << q.offset() << ", mins = " << q.mins()[0] << ", maxs = " << q.maxs()[0];
        }

        for (auto ct : n->constants())
        {
            auto q = ct.second->quantization();
            for (uint32_t m = 0; m < q.mins().size(); m++)
            {
                DEBUG() << ct.second->name() << " [" << m << "]: mins = " << q.mins()[m] << ", maxs = " << q.maxs()[m] << ", scale = " << q.scales()[m] << ", zerop = " << q.offsets()[m];
            }
            if (ct.first == "weights")
            {
                auto data = ct.second->data<int8_t>();
                auto size = ct.second->mem_size() / sizeof(int8_t);
                for (uint32_t i = 0; i < size; i++)
                {
                    DEBUG() << "weights aipu quantize: [" << i << "] = " << (float)data[i];
                }
            }
            else if (ct.first == "biases")
            {
                auto data = ct.second->data<int32_t>();
                auto size = ct.second->mem_size() / sizeof(int32_t);
                for (uint32_t i = 0; i < size; i++)
                {
                    DEBUG() << "biases aipu quantize: [" << i << "] = " << (float)data[i];
                }
            }
        }
    }
}