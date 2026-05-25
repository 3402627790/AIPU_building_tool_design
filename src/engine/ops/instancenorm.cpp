        {
            node->outputs()[0]->share_data_from(state_batch);
            if (out_seq.size() > 1)
            {
                node->outputs()[1]->share_data_from(state_last);
            }
        }
    }

    EXEOP_REGISTER(OpType_GRUv1, gru, DTCriteria::FP32_ONLY);
    EXEOP_REGISTER(OpType_GRUv3, gru, DTCriteria::FP32_ONLY);
}
}