import os
import torch
from AIPUBuilder.logger import ERROR, INFO, WARN
from AIPUBuilder.core import OpType, Tensor, TensorShape, Dtype, SubGraphNode
from AIPUBuilder.model.register import register_model_quantize
from AIPUBuilder.model.qinfo import OptQuantize, QConf, QuantType
from AIPUBuilder.model.app.pyllmchat import PyLLMChat


@register_model_quantize(['qwen2_w4afp8e4m3_gptq'])
class Qwen2W4afp8ModelQuantize(OptQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.bits = 4
        self.hf_weights = hf_config.weights
        self.model_quantize_path = os.path.join(os.path.dirname(__file__))
        json_file = os.path.join(self.model_quantize_path, f"qwen2_w4afp8e4m3_qinfo.json")
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
                # scales = torch.tensor(w.quantization.scales).to(w.betensor.device)
                # zeros = torch.tensor(w.quantization.offsets).to(w.betensor.device)
                # if len(scales) > 1:
                #     group_size = self.hf_config.hf_quantization_config.get('group_size', 128)
                #     w.dtype = Dtype.ALIGNED_INT4 if self.bits < 8 else Dtype.FP32
                #     # if w4afp8 in ['float8_e4m3fn', 'float8_e5m2']:
                #     scale0, scale1 = self.convert_gptq_fp16_scales_to_fp8_scales(
                #         w.betensor, scales, zeros, group_size, self.activation_dtype)
                #     n.constants['scale0'] = Tensor.new(scale0)
                #     n.constants['scale1'] = Tensor.new(scale1.to(torch.float16))
                #     n.constants['zp0'] = Tensor.new(zeros.reshape(*list(scale0.shape)))

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

    # def convert_opt_graph_to_aipu_graph(self):
    #     if self.opt_g is None or self.opt_g.quantgraph is None:
    #         self.aipu_g = None
    #     else:
    #         from AIPUBuilder.core import Graph
    #         from AIPUBuilder.Optimizer.utils import timestamp_string
    #         import os
    #         ir_txt = f"{self.opt_g.name}_{timestamp_string()}.txt"
    #         ir_bin = ir_txt.replace('.txt', '.bin')
    #         try:
    #             self.opt_g.quantgraph.serialize(ir_txt, ir_bin)
    #         except OSError as e:
    #             self.aipu_g = None
    #             ERROR(f"generate tmp file failed! because : {e}")
    #         if os.path.exists(ir_txt) and os.path.exists(ir_bin):
    #             try:
    #                 self.aipu_g = Graph.parse(ir_txt, ir_bin)
    #                 os.remove(ir_txt)
    #                 os.remove(ir_bin)
    #             except Exception as e:
    #                 WARN(f"{e}")
    #                 self.aipu_g = None
    #         else:
    #             self.aipu_g = None
