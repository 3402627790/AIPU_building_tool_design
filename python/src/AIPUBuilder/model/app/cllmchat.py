import os
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.cruntime import (CompassLlmConfig,
                                  compass_llm_sampling_method,
                                  LLM_init,
                                  LLM_run,
                                  LLM_setSystemPrompt,
                                  LLM_destroy)
from AIPUBuilder.model.app.llmchat import LLMChat
from AIPUBuilder.model.register import register_app


@register_app()
class CLLMChat(LLMChat):
    """CLLMChat is used for LLM model chat running in the npu platform"""

    def __init__(self, prompt='', sys_prompt='', sampling="NONE", sampling_params="", cpu_num=16, dump=False):
        """CLLMChat constructor.

        Parameters
        ---------
        prompt: str
            the question discription for the LLM model.

        sys_prompt: str
            the system prompt for getting better quality chat.

        sampling: str
            sampling method, now only support: NONE, RANDOM.

        sampling_params: str
            when sampling=RANDOM, can configure the sampling_params, the sampling_params
            can set 3 values, like "1.1,0.95,0"

        cpu_num: Union[int, str]
            the cpu core number for calculation when chatting. default is 16.

        dump: Union[str, bool]
            whether dump the input/output tensors of graph when running.

        """
        super().__init__()
        self.prompt = prompt
        self.sys_prompt = sys_prompt
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.cpu_num = int(cpu_num)
        self.dump = dump
        if isinstance(dump, str):
            self.dump = True if dump.lower() == 'true' else False

    @classmethod
    def engines(cls):
        return ["sim"]

    def forward(self):
        self.config = self.configs[0]
        if self.prompt != '' and self.aipu_bins is not None and self.config is not None:
            sampling_params = self.sampling_params
            if sampling_params == "":
                sampling_params = [""] * 3
            else:
                sampling_params = sampling_params.replace(" ", "").split(',')
                if len(sampling_params) < 3:
                    sampling_params += [""] * (3 - len(sampling_params))
                elif len(sampling_params) > 3:
                    sampling_params = sampling_params[:3]
            # INFO(f"sample_paras = {str(sampling_params)}")

            sampling = self.sampling
            if isinstance(sampling, str):
                s = compass_llm_sampling_method.SAMPLING_NONE
                if s == "RANDOM":
                    s = compass_llm_sampling_method.SAMPLING_RANDOM
                sampling = s
            if sampling == compass_llm_sampling_method.SAMPLING_RANDOM and sampling_params == ['', '', '']:
                sampling_params['1.0', '1.0', '']

            INFO(f"aipu_bin = {self.aipu_bins}")
            llm_config = CompassLlmConfig(self.aipu_bins[0],
                                          self.config.use_cache,
                                          self.config.seq_len + self.config.past_len,
                                          sampling,
                                          *sampling_params,
                                          self.cpu_num,
                                          self.dump)
            INFO(f"[LLM] run config = {llm_config.show()}")

            sess_id = LLM_init(llm_config)
            if self.sys_prompt != '':
                INFO(f"\t llm system prompt: {self.sys_prompt}")
                LLM_setSystemPrompt(sess_id, self.sys_prompt)
            INFO(f"\t llm query prompt = {self.prompt}")

            def _show_func(s):
                fn = os.path.join(self.config.output_dir, 'answer_result.txt')
                with open(fn, 'a') as fi:
                    fi.write(s)
                    print(s)

            LLM_run(sess_id, self.prompt, _show_func)
            LLM_destroy(sess_id)
