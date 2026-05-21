from ...register import register_model_quantize
from ...qinfo import ModelQuantize


@register_model_quantize(['chatglm3_w4', 'chatglm3_w8', 'chatglm_w4', 'chatglm_w8'])
class ChatGLM3W4ModelQuantize(ModelQuantize):
    def __init__(self, model, model_type, quantization_type='w4', hf_config=None):
        super().__init__(model, model_type, quantization_type, hf_config)

    def update_qinfo(self):
        '''
        default config: is_perf_mode=false
        '''
        if hasattr(self.hf_config, 'aipu_quant_config'):
            if 'is_perf_mode' not in self.hf_config.aipu_quant_config:
                self.hf_config.aipu_quant_config['is_perf_mode'] = False
        else:
            self.hf_config.update({'aipu_quant_config': {'is_perf_mode': False}})
        super().update_qinfo()
