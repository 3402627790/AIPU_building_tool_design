import os
import torch
import numpy as np
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.model.app.llmchat import LLMChat
from AIPUBuilder.model.register import register_app


@register_app()
class PPLMetric(LLMChat):
    def __init__(self, wiki_text='', max_seq_len='2048', model_name=''):
        """PPLMetric constructor.

        Parameters
        ---------
        wiki_text: str
            the wiki dataset path.

        max_seq_len: str
            the max sequence length of prompt for metric
        """

        super().__init__()
        default_wiki_text = '/project/ai/scratch01/Dataset/LLM/wikitext-2-test-raw.txt'
        wiki_text = wiki_text if len(wiki_text) else default_wiki_text
        if not os.path.exists(wiki_text):
            ERROR(f"{self.__class__} meets issue: {wiki_text} is not exist.")

        with open(wiki_text, 'r') as fr:
            self.raw = fr.read()

        self.max_seq_len = int(max_seq_len)
        self.nlls = []
        self.loss = torch.nn.CrossEntropyLoss()
        self.model_name = model_name

        INFO(f"PPLMetric init Done: using devices={self.device}")

    def _fix_prefix_postfix(self):

        if self.model_name == 'llama2':
            return ([1, 518, 25580, 29962], [29961, 29914, 25580, 29962])
        return None

    def _gen_token_prefix_and_postfix(self, tokenizer, prompt=['', 'AIPU']):

        if self._fix_prefix_postfix() is not None:
            return self._fix_prefix_postfix()

        token_without_prompt = tokenizer.encode(tokenizer.apply_chat_template(prompt[0]))
        token_with_prompt = tokenizer.encode(tokenizer.apply_chat_template(prompt[1]))
        INFO(f"token_without_prompt={token_without_prompt}, token_with_prompt={token_with_prompt}")

        if len(token_without_prompt) >= len(token_with_prompt):
            ERROR(f"{self.__class__} meets issue: {token_with_prompt} is not longer than {token_without_prompt} when using {self.model_type} model")

        start_insert_position = None
        # end_insert_position = None

        for i in range(len(token_with_prompt)):
            if token_with_prompt[i] != token_without_prompt[i]:
                start_insert_position = i
                break
        if start_insert_position is None:
            ERROR("undefined the token prefix and postfix")
            return None, None

        return token_without_prompt[:start_insert_position], token_without_prompt[start_insert_position:]

    def _ppl_prepare(self):

        self._prepare()

        # self.prefix = len(self.aipu_tokenizer.encode(self.aipu_tokenizer.apply_chat_template(messages="")))
        # self.seq_len = self.max_seq_len - self.prefix
        self.template_prefix, self.template_postfix = self._gen_token_prefix_and_postfix(self.aipu_tokenizer)
        self.prefix = len(self.template_postfix) + len(self.template_prefix)
        self.seq_len = self.max_seq_len - self.prefix
        INFO(f"chat template prefix: {self.template_prefix}, postfix: {self.template_postfix}")

        raw_token_id = self.aipu_tokenizer.encode(self.raw)
        self.dataset_len = len(raw_token_id)
        self.dataset = []

        prev_end_loc = 0
        for begin in range(0, self.dataset_len, self.seq_len):
            end_loc = min(begin + self.seq_len, self.dataset_len)
            tgrg_len = end_loc - prev_end_loc
            inputs = raw_token_id[begin:end_loc]
            inputs = self.template_prefix + inputs + self.template_postfix
            if len(inputs) > self.max_seq_len:
                WARN(f"Wikitext dataset error, batch {begin} is too long, {len(inputs)} > {self.max_seq_len}")
                continue
            self.dataset.append(inputs)
        INFO(f"Wikitext dataset loaded, initialized {len(self.dataset)} batches, witch context length {self.seq_len}")

    @classmethod
    def engines(cls):
        return ["opt", "exe", "npu"]

    def forward(self):

        self._ppl_prepare()

        for idx, prompt in enumerate(self.dataset):
            tokens_t = torch.tensor(prompt, device=self.device).reshape(1, -1)
            labels_id = tokens_t  # .clone()
            start_idx = 0
            if not self.has_configured_embedding:
                input_ids = [self.embedding_data.to(self.device)[tokens_t]]
            else:
                input_ids = self.engines[0].forward(tokens_t)
                start_idx = 1
            inputs = self._gen_inputs(input_ids)
            for engine in self.engines[start_idx:]:
                out = engine.forward(inputs)
                inputs = out
            logits = out[0].reshape(-1, self.logits_dim)[:-1]
            logits = torch.tensor(logits, device=self.device)
            labels = labels_id.reshape(-1)[1:]
            nll = self.loss(logits, labels).item()
            self.nlls.append(nll)
            del tokens_t
            del labels_id
            self._write_answer(
                f"Batch {idx} finished, nll {nll}, cumulative ppl {torch.exp(torch.tensor(self.nlls).mean())}\n")

        res = torch.exp(torch.tensor(self.nlls).mean())
        self._write_answer(f"\nFinished all batches, average nll {res}\n")
