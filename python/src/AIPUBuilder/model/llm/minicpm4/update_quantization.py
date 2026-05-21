import os
import torch
from AIPUBuilder.logger import ERROR, INFO, WARN
from AIPUBuilder.core import OpType, Tensor, TensorShape, Dtype, SubGraphNode
from AIPUBuilder.model.register import register_model_quantize
from AIPUBuilder.model.qinfo import OptQuantize, QConf, QuantType
from AIPUBuilder.model.app.pyllmchat import PyLLMChat


@register_model_quantize(['minicpm4_w4afp8e4m3', 'minicpm4_w4afp8e4m3_gptq'])
class minicpm4W4afp8ModelQuantize(OptQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.bits = 4
        self.hf_weights = hf_config.weights
        self.model_quantize_path = os.path.join(os.path.dirname(__file__))
        json_file = os.path.join(self.model_quantize_path, f"minicpm4_w4afp8e4m3_qinfo_experiment.json")
        self.qconfig = QConf(json_file)
        self.optimizer_json = self.qconfig.config
        self.activation_dtype = QuantType.activation_dtype(quantization_type)
        self.weight_dtype = QuantType.weight_dtype(quantization_type)

    def update_qinfo(self):  # TODO streamline
        def _set_node_qinfo(n):
            if n.type in [OpType.FullyConnected]:
                w = n.constants['weights']
                scales = torch.tensor(w.quantization.scales).to(w.betensor.device)
                if len(scales) > 1:
                    w.dtype = Dtype.ALIGNED_INT4 if self.bits < 8 else Dtype.FP32

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = n.subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        for node in self.aipu_g.nodes:
            set_qinfo(node)

    def statistic_opt_graph(self):
        prompts = self.qconfig.get_info('calibration_prompt')
        if prompts is None:
            super().statistic_opt_graph()
        else:
            calibration_prompt_max_gen_length = int(self.qconfig.config.get("calibration_prompt_max_gen_length", 3))
            self.opt_g.current_batch_idx = 0
            use_embedding = False  # TODO
            from AIPUBuilder.model.llm.llmutils import get_embedding_data
            embedding_data = get_embedding_data(self.hf_config, self.model_type, True)
            for prompt in prompts:
                tokens_t, cache_k, cache_v, init_seq_len, iters = self.get_inputs(prompt)
                while True:
                    if iters > calibration_prompt_max_gen_length:
                        break
                    iters += 1
                    input_ids = [(embedding_data.to(self.device))[tokens_t]]  # [1,37]
                    inputs = PyLLMChat()._gen_inputs(input_ids, cache_k, cache_v, init_seq_len, iters,
                                                     self.hf_config.get('kvcache_optimization', False))
                    self.calibration_data.append(inputs)
                    self.opt_g.statistic(inputs, self.optimizer_argv)
                    self.opt_g.current_batch_idx += 1
                    out = []
                    for out_tensor in self.opt_g.output_tensors:
                        out.append(out_tensor.betensor)
                    if self.hf_config.get('kvcache_optimization', False):
                        cache_k = out[1:2]
                        cache_v = out[2:]
                    else:
                        cache_k = out[1: 1 + len(cache_k)]
                        cache_v = out[1 + len(cache_k):]
                    logits = out[0]
                    token = logits[0, -1].argmax()
                    output = self.tokenizer.decode(token)
                    new_tokens_t = torch.tensor(token).int().flatten().reshape([1, -1])
                    # the historical token and the generated token are concatenated here for quantization calibration.
                    tokens_t = torch.concat([tokens_t, new_tokens_t], dim=-1)

    def generate_dataloader(self):
        if len(self.calibration_data):
            from torch.utils.data import DataLoader
            from AIPUBuilder.model.calibration import MultiTaskDataset
            labels = [torch.tensor([0], device=self.calibration_data[0][0].device)
                      for _ in range(len(self.calibration_data))]
            dataset = MultiTaskDataset(self.calibration_data, labels)
            dataloader = DataLoader(dataset, batch_size=None)
            return dataloader
        return None
