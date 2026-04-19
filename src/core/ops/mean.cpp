
                continue;
            }
            if (std::count(new_axis.begin(), new_axis.end(), it) == 0)
            {
                new_axis.push_back(it);
            }
        }
        if (static_cast<int32_t>(new_axis.size()) > input->shape().dim())
        {
            ERROR() << "axis size is invalid, it should be <= input rank!";
            return nullptr;
        }

        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("mean_variance_normalization"), input->shape());
        add_tensor_info(input, output, quant);
        auto node = Node::create(g->get_valid_node_name("mean_variance_normalization"), OpType_MVN);
        auto &params = node->params();
        params["axis"] = new_axis;
        params["epsilon"] = epsilon;

        node->add_input(input);
        node->add_output(output);
        g->add_node(node);
        quantize(node);

        return output;
    }

}; //namespace ops
}; // namespace aipubt