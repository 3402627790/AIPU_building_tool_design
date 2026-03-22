
                }
                if (n->attrs().count(Constants::ldwfromsm))
                {
                    int sm = n->get_attr<int>("_use_sm");
                    oss << "|";
                    oss << "ld_w_from_sm " << sm;
                }
            }

            if (show_full_mem_info && aipubt::BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
            {
                if (n->attrs().count(Constants::wr_out_to_gm))
                {
                    auto offset = n->get_attr<uint32_t>(Constants::wr_out_to_gm);
                    oss << "|";
                    oss << "wr_out_to_gm address: 0x" << std::hex << gm_addr + offset;
                }

                if (n->attrs().count(Constants::mov_d_to_ddr))
                {
                    auto end_node = n->get_attr<NodeRef>(Constants::mov_d_to_ddr).lock();
                    auto root_node = end_node->get_attr<NodeRef>(Constants::mov_d_to_ddr_wait).lock();
                    oss << "|";
                    oss << "mov_d_to_ddr: " << root_node->name();
                }

                if (n->attrs().count(Constants::mov_d_to_ddr_wait))
                {
                    auto root_node = n->get_attr<NodeRef>(Constants::mov_d_to_ddr_wait).lock();
                    oss << "|";
                    oss << "mov_d_to_ddr_wait: " << root_node->name();
                }
            }

            n->add_dot_section(Constants::dot_address, oss.str());
        }
    }

    return false;
}

bool add_mem_info_to_dot(GraphPtr g)
{
    return _add_mem_info_to_dot(g, g);
}

} // namespace aipubt
