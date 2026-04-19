
        {
            kernel.push_back(kernel[0]);
        }
        if (stride.size() == 1)
        {
            stride.push_back(stride[0]);
        }
        int32_t new_h = in_h / stride[0] * kernel[0];
        if (in_h % stride[0] >= kernel[0])
        {
            new_h += kernel[0];
        }
        output_shape.h() = new_h;
        int32_t new_w = in_w / stride[1] * kernel[1];
        if (in_w % stride[1] >= kernel[1])
        {
            new_w += kernel[1];
        }
        output_shape.w() = new_w;
        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("data_stride"), output_shape);
        add_tensor_info(input, output);
        output->set_type(input->type());
        output->quantization() = input->quantization();

        auto node = Node::create(g->get_valid_node_name("data_stride"), OpType_DataStride);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);
        auto &params = node->params();
        params["kernel_y"] = kernel[0];
        params["kernel_x"] = kernel[1];
        params["stride_y"] = stride[0];
        params["stride_x"] = stride[1];
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt
