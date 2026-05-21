import torch
import numpy as np
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.model.app.pyllmchat import PyLLMChat
from AIPUBuilder.model.register import register_app


@register_app()
class PDModeLLMChat(PyLLMChat):
    """PDModeLLMChat is used for LLM model chat based on prefill + decoder mode."""

    def __init__(self, prompt: str = '', sys_prompt: str = '', sampling: str = "NONE", sampling_params: str = "", max_new_tokens: str = "", dump: bool = False):
        """PDModeLLMChat constructor.

        Parameters
        ---------
        prompt: str
            the question discription for the LLM model.

        sys_prompt: str
            the prompt for getting better quality chat.

        sampling: str
            sampling method, now only support: NONE, RANDOM.

        sampling_params: str
            when sampling=RANDOM, can configure the sampling_params, the sampling_params
            can set 3 values, like "1.1,0.95,0"

        dump: Union[str, bool]
            whether dump the input/output tensors of graph when running.

        """
        super().__init__(prompt, sys_prompt, sampling, sampling_params, max_new_tokens, dump)

    @classmethod
    def engines(cls):
        return ["exe", "opt", "npu", "sim"]

    def generate(self, query):

        tokens = self.aipu_tokenizer.encode(query)
        INFO(f"now templated query is: {query}, and its teminators={self.terminators}")
        INFO(f"now tokens of query is: {tokens}")

        tokens_t = torch.tensor(tokens).to(self.device).flatten().reshape([1, -1])
        cache_shape = self.engines[-1].input_shapes[-1]
        is_cacheupdate = self.configs[len(self.configs) - 2].get('kvcache_optimization', False)
        if is_cacheupdate:
            cache_k = [torch.zeros(cache_shape)]
            cache_v = [torch.zeros(cache_shape)]
        else:
            cache_num = len(self.engines[-1].input_shapes) - 3
            if cache_num % 2 != 0:
                ERROR(f"Cache length should be double of num_hidden_layer, but now is {cache_num}")

            num_hidden_layers = cache_num // 2
            INFO(f"num_hidden_layers = {num_hidden_layers}")
            kcache_shape = self.engines[-1].input_shapes[3]
            vcache_shape = self.engines[-1].input_shapes[3 + num_hidden_layers]
            INFO(f"kcache_shape = {kcache_shape}, vcache_shape = {vcache_shape}")
            cache_k = [torch.zeros([1, kcache_shape[1], 0, kcache_shape[3]]).to(self.device)
                       for _ in range(num_hidden_layers)]
            cache_v = [torch.zeros([1, vcache_shape[1], 0, vcache_shape[3]]).to(self.device)
                       for _ in range(num_hidden_layers)]

        with torch.no_grad():
            init_seq_len = len(tokens)
            iters = 0
            while True:
                if init_seq_len > self.max_len:
                    break
                # embedding
                start_idx = 0
                if not self.has_configured_embedding:
                    input_ids = [(self.embedding_data.to(self.device))[tokens_t]]
                else:
                    input_ids = self.engines[0].forward(tokens_t)
                    start_idx = 1
                # inputs = self._gen_inputs(input_ids, cache_k, cache_v)
                inputs = self._gen_inputs(input_ids, cache_k, cache_v, init_seq_len, iters + 1, is_cacheupdate)
                start_idx = start_idx + 1 if iters > 0 else start_idx
                if iters == 0:
                    inputs = inputs if is_cacheupdate else inputs[:3]
                for engine in self.engines[start_idx:start_idx+1]:
                    out = engine.forward(inputs)
                    inputs = out

                if is_cacheupdate:
                    cache_k = out[1:2]
                    cache_v = out[2:]
                else:
                    cache_k = out[1: 1 + self.num_hidden_layers]
                    cache_v = out[1 + self.num_hidden_layers:]
                if iters == 0 and self.configs[0].get('prefill_batch_opt', False) and not is_cacheupdate:
                    cache_k = [ck.transpose(1, 0, 2, 3) for ck in cache_k]
                    cache_v = [cv.transpose(1, 0, 2, 3) for cv in cache_v]
                logits = out[0]
                token = logits[0, -1].argmax()
                tokens_t = torch.tensor(token).int().flatten().reshape([1, -1])
                iters += 1
                yield token
                if token in self.terminators:
                    break
        return self.aipu_tokenizer.eos_token_id

    def forward(self):
        super().forward()
