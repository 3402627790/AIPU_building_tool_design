import torch
import os
from AIPUBuilder.logger import ERROR, INFO
from AIPUBuilder.core import OpType, NodeParamValue, Scheme, Granularity, QuantizedRange, Dtype, SubGraphNode
from AIPUBuilder.model.register import register_model_quantize
from AIPUBuilder.model.qinfo import ModelQuantize, DEFAULT_TARGET, OptQuantize, QConf, QuantType
from AIPUBuilder.model.app.vlchat import VLChat


# @register_model_quantize(['qwen2_5_vl_vision_w4'])
class Qwen25VLVisionW4ModelQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)

    def update_qinfo(self):
        '''
        set fp16 op
        '''
        key_name = 'ignore_quantization_of_fc_weight_name'
        # fp16_op_key_name = 'merger,mlp'
        fp16_op_key_name = 'merger'
        if hasattr(self.hf_config, 'aipu_quant_config'):
            if key_name not in self.hf_config.aipu_quant_config:
                self.hf_config.aipu_quant_config[key_name] = fp16_op_key_name
            else:
                ignore_name = self.hf_config.aipu_quant_config.get(key_name)
                ignore_name = ignore_name.replace(' ', '')
                if ignore_name.endswith(','):
                    ignore_name = ignore_name[:-1]
                self.hf_config.aipu_quant_config[key_name] = f'{ignore_name},{fp16_op_key_name}'
        else:
            self.hf_config.update({'aipu_quant_config': {key_name: 'merger'}})

        super().update_qinfo()


@register_model_quantize(['qwen2_5_vl_vision_w4'])
class Qwen25VLVisionW4ModelQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.bits = int(''.join(list(filter(str.isdigit, quantization_type))))
        if self.bits not in [4, 8]:
            ERROR(f"when activation is fp16, weight quantization bits should be in [4, 8], but now is {self.bits}")
        self.hf_weights = hf_config.weights
        self.target = hf_config.get('target', DEFAULT_TARGET)

    def get_quant_config(self):
        lm_head_fp16 = True
        is_perf_mode = True
        if hasattr(self.hf_config, 'aipu_quant_config'):
            qc = self.hf_config.aipu_quant_config
            if 'lmhead_fc_weight_type' in qc:
                wt = qc['lmhead_fc_weight_type']
                if wt == 'fp16':
                    lm_head_fp16 = True
            if 'is_perf_mode' in qc:
                is_perf_mode = qc['is_perf_mode']

        return lm_head_fp16, is_perf_mode

    def update_qinfo(self):
        _, is_perf_mode = self.get_quant_config()
        activation_type = self.hf_config.get('activation_type', 'float16')
        act_dtype = Dtype(activation_type.lower())
        target = DEFAULT_TARGET if self.hf_config.get('target', '') == '' else self.hf_config.target
        try:
            from AIPUBuilder.Optimizer.utils import Target as _PyTarget
            aiff_lut_items_in_bits = _PyTarget.aiff_lut_items_in_bits(target.split('_')[0])
        except:
            aiff_lut_items_in_bits = 9

        def _set_node_qinfo(n):
            if n.type in [OpType.FullyConnected]:
                w = n.constants['weights']
                if 'merger' in w.name:
                    # lm head fc uses fp16 * fp16
                    n.attrs['trigger_float_op'] = NodeParamValue(str(Dtype.FP16))
                elif 'mlp.down_proj.weight' in w.name:
                    n.attrs['trigger_float_op'] = NodeParamValue(str(Dtype.FP16))
                elif 'patch_embed' in w.name:
                    n.attrs['trigger_float_op'] = NodeParamValue(str(Dtype.FP16))
                else:
                    w.quantization.bits = self.bits
                    w.quantization.scheme = Scheme.Asymmetric if self.bits < 8 else Scheme.Symmetric
                    w.quantization.granularity = Granularity.Per_Group
                    w.quantization.quantized_range = QuantizedRange.Full_Range
                    w.quantization.aipu_dtype = Dtype.ALIGNED_INT4 if self.bits < 8 else Dtype.INT8
                    w.quantization.group = w.shape[0] * (w.shape[1] // 32)
                    trigger_mode = f"{activation_type.upper()}_ACT_INT{self.bits}_WHT"
                    n.attrs['trigger_float_op'] = NodeParamValue(trigger_mode)
            else:
                n.attrs['trigger_float_op'] = NodeParamValue(activation_type)

            n.attrs['unquantifiable'] = True
            n.params['unquantifiable'] = True

            if n.type in [OpType.Softmax]:
                n.attrs['min_compatible_zhouyi_target'] = self.target
                n.attrs['approx_params'] = NodeParamValue(int(1))
                n.attrs['lut_items_in_bits'] = aiff_lut_items_in_bits
            if n.type in [OpType.Activation]:
                n.attrs['min_compatible_zhouyi_target'] = NodeParamValue(self.target)
                n.attrs['approx_params'] = NodeParamValue(0)
                if is_perf_mode and n.params['method'].lower() in ['silu', 'gelu']:
                    n.attrs['approx_params'] = NodeParamValue([1])
                    n.attrs['lut_items_in_bits'] = aiff_lut_items_in_bits

            if n.type == OpType.Input:
                if n.outputs[0].dtype.is_float:
                    n.outputs[0].dtype = act_dtype

            for ot in n.outputs:
                if ot.io_tensor and ot.dtype.is_float:
                    ot.dtype = act_dtype

        def set_qinfo(n):
            if SubGraphNode.is_subgraph_node(n):
                sbg = SubGraphNode.from_node(n).subgraph
                for sbgn in sbg.nodes:
                    set_qinfo(sbgn)
            else:
                _set_node_qinfo(n)

        for node in self.aipu_g.nodes:
            set_qinfo(node)


@register_model_quantize(['qwen2_5_vl_llm_w4afp8e4m3_awq'])
class Qwen25VLLLMW4afp8ModelQuantize(OptQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)
        self.bits = 4
        self.hf_weights = hf_config.weights
        self.model_quantize_path = os.path.join(os.path.dirname(__file__))
        json_file = os.path.join(self.model_quantize_path, f"qwen2_5_vl_llm_w4afp8e4m3_qinfo_experiment.json")
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

    def get_inputs(self, vlchat_handler):
        inputs = vlchat_handler._vl_prepare()

        init_seq_len = inputs['input_ids'].shape[-1]
        input_ids = inputs['input_ids']

        cache_shape = self.opt_g.input_tensors[-1].ir_shape
        cache_k, cache_v = [], []
        if self.hf_config.get('kvcache_optimization', False):
            cache_k = [torch.zeros(cache_shape)]
            cache_v = [torch.zeros(cache_shape)]
        else:
            cache_num = len(self.opt_g.input_tensors) - 3
            if cache_num % 2 != 0:
                ERROR(f"Cache length should be double of num_hidden_layer, but now is {cache_num}")

            if cache_num > 0:
                num_hidden_layers = cache_num // 2
                kcache_shape = list(self.opt_g.input_tensors[3].ir_shape)
                vcache_shape = list(self.opt_g.input_tensors[3 + num_hidden_layers].ir_shape)
                cache_k = [torch.zeros([1, kcache_shape[1], 0, kcache_shape[3]]).to(self.device)
                           for _ in range(num_hidden_layers)]
                cache_v = [torch.zeros([1, vcache_shape[1], 0, vcache_shape[3]]).to(self.device)
                           for _ in range(num_hidden_layers)]
        iters = 0
        return (input_ids, cache_k, cache_v, init_seq_len, iters)

    def _gen_inputs(self, input_ids, position_ids=None,  cache_k=[], cache_v=[], init_seq_len=0, iters=1, kvcache_optimization=False):
        def mask_lambda(x): return (x - 1) * 100

        cur_seq_len = input_ids.shape[1]
        if kvcache_optimization:
            cur_past_len = (init_seq_len + iters - 2)if iters > 1 else 0
        else:
            cur_past_len = cache_k[0].shape[2] if len(cache_k) else 0

        ctx_mask = torch.ones([cur_seq_len, cur_seq_len]).tril().to(input_ids.device)
        past_mask = torch.ones([cur_seq_len, cur_past_len]).to(input_ids.device)
        mask = torch.concat([past_mask, ctx_mask], dim=-1)
        mask = mask.reshape([1, 1, cur_seq_len, cur_seq_len + cur_past_len]).float()
        mask = mask_lambda(mask)
        if position_ids is None:
            position_ids = torch.arange(cur_seq_len) + cur_past_len
            position_ids = position_ids.reshape(1, -1).int()

        inputs = [input_ids, mask, position_ids] + cache_k + cache_v
        return inputs

    def statistic_opt_graph(self):
        prompts = self.qconfig.get_info('calibration_prompt')
        if prompts is None:
            super().statistic_opt_graph()
        else:
            calibration_prompt_max_gen_length = int(self.qconfig.config.get("calibration_prompt_max_gen_length", 3))
            self.opt_g.current_batch_idx = 0
            from AIPUBuilder.model.llm.llmutils import get_embedding_data
            embedding_data = get_embedding_data(self.hf_config, self.model_type, True)
            for prompt in prompts:
                vlchat_handler = VLChat(prompt=prompt)
                vlchat_handler.set_configs([self.hf_config])
                input_ids, cache_k, cache_v, init_seq_len, iters = self.get_inputs(vlchat_handler)
                iters = 0
                tokens_t = input_ids
                while True:
                    if iters > calibration_prompt_max_gen_length:
                        break
                    iters += 1
                    input_ids = (embedding_data.to(self.device))[tokens_t]  # [1,37]
                    batch_size, seq_length, _ = input_ids.shape
                    position_ids = torch.arange(seq_length, device=input_ids.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                    inputs = self._gen_inputs(input_ids, position_ids, cache_k, cache_v, init_seq_len, iters,
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
