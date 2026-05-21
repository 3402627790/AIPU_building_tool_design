import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from AIPUBuilder.core import use_cuda, Dtype
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.model.app.baseapp import BaseApp
from AIPUBuilder.model.register import ENGINES, CHATTEMPLATE, CAUSALLMS
from AIPUBuilder.model.loadweight import LoadWeight
from AIPUBuilder.model.llm.chat_template import EOT_TOKENS
from AIPUBuilder.model.llm.llmutils import get_embedding_data
from transformers import AutoTokenizer


class AIPUTokenizer(object):

    def __init__(self, config, model_type):
        self.config = config
        model_path = config.model_path
        self.llm = None
        self.model_type = model_type

        self.is_gguf_tokenizer = False
        if os.path.isfile(model_path) and model_path.endswith('.gguf') and len(config.get('tokenizer_path', '')) == 0:
            try:
                from llama_cpp import Llama
                self.llm = Llama(model_path)
                self.is_gguf_tokenizer = True
            except Exception as e:
                ERROR(
                    f"import llama_cpp package failed, please install llama-cpp-python package or set tokenizer_path in config. error msg={e}")
        else:
            model_path = config.tokenizer_path if len(config.get('tokenizer_path', '')) > 0 else model_path
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def encode(self, text):
        if self.is_gguf_tokenizer:
            return self.llm.tokenize(text.encode('utf-8'))
        else:
            return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        if self.is_gguf_tokenizer:
            return self.llm.detokenize(tokens.reshape([-1]).tolist()).decode()
        else:
            return self._tokenizer.decode(tokens)

    @property
    def tokenizer(self):
        if self.is_gguf_tokenizer:
            raise RuntimeError
        else:
            return self._tokenizer

    @property
    def eos_token_id(self):
        if self.is_gguf_tokenizer:
            return self.llm.token_eos()
        else:
            return self._tokenizer.eos_token_id

    def apply_chat_template(self, messages, sys_prompt=''):
        if self.is_gguf_tokenizer:
            raise RuntimeError
        else:
            if self.model_type not in CHATTEMPLATE:
                ERROR(f"{self.__class__} only supports model_type={','.join(CHATTEMPLATE.keys())}, but got {self.model_type}")
                self.chat_template_func = self._tokenizer.apply_chat_template
            else:
                self.chat_template_func = CHATTEMPLATE[self.model_type]

            return self.chat_template_func(self._tokenizer, messages, sys_prompt)


class LLMChat(BaseApp):
    """LLMChat is used for LLM model chat and chat scripts uses python."""

    def __init__(self, *args, **kwargs):
        """LLMChat constructor. """
        super().__init__()

        self.aipu_tokenizer = None
        self.engines = []

        self.device = torch.device('cuda') if use_cuda() else torch.device('cpu')
        # visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    @classmethod
    @abstractmethod
    def engines(cls):
        pass

    def _prepare(self):
        # check use_cache, now only support use_cache=True
        is_use_cache = [c.use_cache for c in self.configs]
        if not all(is_use_cache):
            WARN(f"now only support use_cache=True IR or binary.")
        self.use_cache = is_use_cache[0]

        # get tokenizer
        self.model_paths = [c.model_path for c in self.configs]
        mp = self.model_paths[0]
        for i in range(1, len(self.model_paths)):
            if self.model_paths[i] != mp:
                ERROR(f"{self.__class__} only supports one same model path for now.")
                return

        # get base model_type
        self.model_type = ''
        if len(self.configs):
            mts = [c.model_type for c in self.configs]
            if len(set(mts)) == 1:
                self.model_type = self.configs[0].model_type
            else:
                for mt in mts:
                    if 'emb' not in mt:
                        self.model_type = mt

        if self.model_type == '':
            config_json_path = os.path.join(mp, 'config.json')
            with open(config_json_path, 'r') as fi:
                lines = fi.readlines()
                for l in lines:
                    if 'model_type' in l:
                        self.model_type = l.split(':')[-1].strip().replace(' ', '')[1:-2]

        self.aipu_tokenizer = AIPUTokenizer(self.configs[-1], self.model_type)

        # gen answer txt
        self.answer_txt = f"answer{'_'+self.model_type if len(self.model_type) else ''}.txt"

        self._gen_engines()

        self.embedding_data = None
        self.has_configured_embedding = False
        for idx, c in enumerate(self.configs[:]):
            if 'embedding' in c.model_type:
                self.configs = self.configs[idx:idx+1] + self.configs[:idx] + self.configs[idx+1:]
                self.engines = self.engines[idx:idx+1] + self.engines[:idx] + self.engines[idx+1:]
                self.has_configured_embedding = True
                break
            elif c.get('quantized_ir_path', None) is not None and 'embed' in c.quantized_ir_path:
                self.configs = self.configs[idx:idx+1] + self.configs[:idx] + self.configs[idx+1:]
                self.engines = self.engines[idx:idx+1] + self.engines[:idx] + self.engines[idx+1:]
                self.has_configured_embedding = True
                break

        if not self.has_configured_embedding:
            self.embedding_data = get_embedding_data(self.configs[-1], self.model_type, True)
            if self.embedding_data is None:
                ERROR(f"{self.__class__} can't find embedding data in configs when not providing the emb IR, please check your configs.")

        # any model can get its whole configuration infos, so need to write the special configuration to config
        self.num_hidden_layers = self.configs[-1].num_hidden_layers
        self.hidden_size = self.configs[-1].hidden_size
        self.head_num = self.configs[-1].num_key_value_heads
        self.head_dim = self.configs[-1].get('head_dim',
                                             self.configs[-1].hidden_size // self.configs[-1].num_attention_heads)
        self.logits_dim = self.configs[-1].vocab_size
        self.use_cache = self.configs[-1].use_cache
        self.hidden_layers = self.configs[-1].num_hidden_layers

        # TODO: config configurated past_len != IR past_len, so should getting from IR if info.collector.has_ir
        self.max_len = self.configs[-1].seq_len + self.configs[-1].past_len

        self.terminators = [self.aipu_tokenizer.eos_token_id]
        if self.model_type in EOT_TOKENS:
            eot = EOT_TOKENS[self.model_type]
            self.terminators.append(eot(self.aipu_tokenizer.tokenizer) if callable(eot) else eot)

    def _get_input_shapes(self):
        pass

    def _write_answer(self, s):
        with open(self.answer_txt, 'a+') as fw:
            fw.write(s)
            print(s, end='', flush=True)

    def _gen_engines(self):
        for aipu_config in self.configs:
            engine = aipu_config.engine
            if engine in ENGINES:
                self.engines.append(ENGINES[engine](aipu_config))
            else:
                INFO(f"{self.__class__} meets issue: {engine} is not supported when using {aipu_config.config_name} of {self.model_type} model")

    def _gen_inputs(self, input_ids, cache_k=[], cache_v=[], init_seq_len=0, iters=1, kvcache_optimization=False):
        def mask_lambda(x): return (x - 1) * 100

        input_id = input_ids[0]
        if isinstance(input_id, np.ndarray):
            # support input_id.dtype=bfloat16, if input_id.dtype == bfloat16, torch.from_numpy not support this dtype
            torch_dtype = Dtype.from_np(input_id.dtype).torch
            input_id = torch.from_numpy(input_id.astype(np.float32)).to(torch_dtype).to(self.device)
        input_id = input_id.float()

        cur_seq_len = input_id.shape[1]
        if kvcache_optimization:
            cur_past_len = (init_seq_len + iters - 2) if iters > 1 else 0
        else:
            cur_past_len = cache_k[0].shape[2] if len(cache_k) else 0

        ctx_mask = torch.ones([cur_seq_len, cur_seq_len]).tril().to(input_id.device)
        past_mask = torch.ones([cur_seq_len, cur_past_len]).to(input_id.device)
        mask = torch.concat([past_mask, ctx_mask], dim=-1)
        mask = mask.reshape([1, 1, cur_seq_len, cur_seq_len + cur_past_len]).float()
        mask = mask_lambda(mask)
        position_ids = torch.arange(cur_seq_len) + cur_past_len
        position_ids = position_ids.reshape(1, -1).int()

        inputs = [input_id, mask, position_ids] + cache_k + cache_v
        return inputs

    def forward(self):
        pass
