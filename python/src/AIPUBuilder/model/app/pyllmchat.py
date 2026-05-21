import os
import torch
import numpy as np
from datetime import datetime
from AIPUBuilder.core import Tensor
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.model.app.llmchat import LLMChat
from AIPUBuilder.model.register import register_app, ENGINES


class TemperatureLogitsWarper():
    """
    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores / self.temperature
        return scores_processed


class TopKLogitsWarper():
    """
    [`LogitsProcessor`] that performs top-k, i.e. restricting to the k highest probability elements. Often used
    together with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class TopPLogitsWarper():
    """
    [`LogitsProcessor`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
    Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].
    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


@register_app()
class PyLLMChat(LLMChat):
    """PYLLMChat is used for LLM model chat and chat scripts uses python."""

    def __init__(self, prompt: str = '', sys_prompt: str = '', sampling: str = "NONE", sampling_params: str = "", max_new_tokens: str = "", dump: bool = False, with_chat_template: str = "True"):
        """PYLLMChat constructor.

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

        max_new_tokens: str
            if set, app will terminate at number of tokens, regardless of context status

        dump: Union[str, bool]
            whether dump the input/output tensors of graph when running.

        """
        super().__init__()
        self.prompts = self._gen_prompt(prompt)
        self.sys_prompt = sys_prompt
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.dump = dump
        self.with_chat_template = with_chat_template.lower() != 'false'
        self.max_new_tokens = int(max_new_tokens) if len(max_new_tokens) > 0 else -1
        sampling_params = self.sampling_params
        if sampling.lower() == 'random':
            if sampling_params == "":
                sampling_params = [""] * 3
            else:
                sampling_params = sampling_params.replace(" ", "").split(',')
                if len(sampling_params) < 3:
                    sampling_params += [""] * (3 - len(sampling_params))
                elif len(sampling_params) > 3:
                    sampling_params = sampling_params[:3]
            self.temperatureLogitsWarper = TemperatureLogitsWarper(float(sampling_params[0]))
            self.topKLogitsWarper = TopKLogitsWarper(int(sampling_params[1]))
            self.topPLogitsWarper = TopPLogitsWarper(float(sampling_params[2]))

        if isinstance(dump, str):
            self.dump = True if dump.lower() == 'true' else False

        INFO(f"{self.__class__.__name__} init Done: using devices={self.device}")

    def _gen_prompt(self, prompt):
        if os.path.exists(prompt) and os.path.isfile(prompt):
            with open(prompt, 'r') as f:
                prompts = f.readlines()
        else:
            prompts = [prompt]
        return prompts

    @classmethod
    def engines(cls):
        return ["exe", "opt", "npu", "sim"]

    def generate(self, query):

        tokens = self.aipu_tokenizer.encode(query)
        INFO(f"now templated query is: {query}, and its teminators={self.terminators}")
        INFO(f"now tokens of query is: {tokens}")

        tokens_t = torch.tensor(tokens).to(self.device).flatten().reshape([1, -1])

        cache_shape = self.engines[-1].input_shapes[-1]
        if self.configs[-1].get('kvcache_optimization', False):
            cache_k = [torch.zeros(self.engines[-1].input_shapes[-2])]
            cache_v = [torch.zeros(self.engines[-1].input_shapes[-1])]
        else:
            cache_num = len(self.engines[-1].input_shapes) - 3
            if cache_num % 2 != 0:
                ERROR(f"Cache length should be double of num_hidden_layer, but now is {cache_num}")

            # cache_num = 1
            # for i in range(len(self.engines[-1].input_shapes)-2, -1, -1):
            #     if self.engines[-1].input_shapes[i] == cache_shape:
            #         cache_num += 1
            #     else:
            #         break
            # # cache_len = (self.engines[-1].input_shapes)-3
            # if cache_num % 2 != 0:
            #     ERROR(f"Cache length should be double of num_hidden_layer, but now is {cache_num}")
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
            if init_seq_len > self.max_len:
                tokens_t = tokens_t[:, :self.max_len]
                init_seq_len = self.max_len
            while True:
                if iters > 1 and not self.configs[-1].get('kvcache_optimization', False):
                    cache_k = [c_k[:, :, (1-self.max_len):, :] for c_k in cache_k]
                    cache_v = [c_v[:, :, (1-self.max_len):, :] for c_v in cache_v]
                iters += 1
                # embedding
                start_idx = 0
                if not self.has_configured_embedding:
                    input_ids = [(self.embedding_data.to(self.device))[tokens_t]]
                else:
                    input_ids = self.engines[0].forward(tokens_t)
                    start_idx = 1
                if self.dump:
                    self.dump_io(tokens_t, f'{iters}_{self.engines[0].config.model_name}_input')
                    self.dump_io(input_ids, f'{iters}_{self.engines[0].config.model_name}_output')

                inputs = self._gen_inputs(input_ids, cache_k, cache_v, init_seq_len, iters,
                                          self.configs[-1].get('kvcache_optimization', False))
                for engine in self.engines[start_idx:]:
                    out = engine.forward(inputs)
                    if self.dump:
                        self.dump_io(inputs, f'{iters}_{engine.config.model_name}_input')
                        self.dump_io(out, f'{iters}_{engine.config.model_name}_output')
                    inputs = out

                if self.configs[-1].get('kvcache_optimization', False):
                    cache_k = out[1:2]
                    cache_v = out[2:]
                else:
                    cache_k = out[1: 1 + num_hidden_layers]
                    cache_v = out[1 + num_hidden_layers:]
                logits = out[0]
                if self.sampling == "random":
                    next_token_logits = torch.tensor(logits[0, -1]).to(tokens_t.device).unsqueeze(0)
                    scores = self.temperatureLogitsWarper(inputs, next_token_logits)
                    scores = self.topKLogitsWarper(inputs, scores)
                    next_token_scores = self.topPLogitsWarper(inputs, scores)
                    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
                    token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    token = logits[0, -1].argmax()
                tokens_t = torch.tensor(token).int().flatten().reshape([1, -1])
                yield token
                if token in self.terminators:
                    break
                if self.max_new_tokens > 0 and iters > self.max_new_tokens:
                    break
        return self.aipu_tokenizer.eos_token_id

    def forward(self):
        self._prepare()

        for prompt in self.prompts:
            self._write_answer(f"\n{datetime.now()}\n{self.config_basic_infos}{prompt}\n")
            if self.with_chat_template:
                prompt_after_template = self.aipu_tokenizer.apply_chat_template(prompt, self.sys_prompt)
            else:
                prompt_after_template = prompt
            answer = []
            for token in self.generate(prompt_after_template):
                answer.append(token)
                s = self.aipu_tokenizer.decode(token)
                self._write_answer(s)

            self._write_answer(f"\n")

    def dump_io(self, dump_tensor, dump_name, dump_dir='./dump_io'):
        INFO(f"Dump {dump_name}...")
        os.makedirs(dump_dir, exist_ok=True)

        tensor_list = dump_tensor if isinstance(dump_tensor, list) else [dump_tensor]
        for i, tensor in enumerate(tensor_list):
            file_path = f"{dump_dir}/{dump_name}{i}.npy"
            try:
                c_t = Tensor.new(tensor)
                np.save(file_path, c_t.numpy())
            except Exception as e:
                WARN(f"Dump fail: {dump_name}, and failed msg: {e}")
                continue


@register_app()
class ExeLLMChat(PyLLMChat):
    def __init__(self, prompt='', sys_prompt='', sampling="NONE", sampling_params="", dump=False):
        super().__init__(prompt, sys_prompt, sampling, sampling_params, dump)

    def _gen_engines(self):
        for config in self.configs:
            if config.engine.lower() != 'exe':
                WARN(
                    f"when using exellmchat plugin, its engine should be set to exe, but {config.engine} is set")
                config.engine = 'exe'
            self.engines.append(ENGINES[config.engine](config))

    @classmethod
    def engines(cls):
        return ["exe"]


@register_app()
class OptLLMChat(PyLLMChat):
    def __init__(self, prompt='', sys_prompt='', sampling="NONE", sampling_params="", dump=False):
        super().__init__(prompt, sys_prompt, sampling, sampling_params, dump)

    def _gen_engines(self):
        for config in self.configs:
            if config.engine.lower() != 'opt':
                WARN(
                    f"when using optllmchat plugin, its engine should be set to opt, but {config.engine} is set")
                config.engine = 'opt'
            self.engines.append(ENGINES[config.engine](config))

    @classmethod
    def engines(cls):
        return ["opt"]
