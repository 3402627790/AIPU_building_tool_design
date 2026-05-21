import numpy as np
from AIPUBuilder.logger import INFO, ERROR, WARN
from AIPUBuilder.model.app.pdmodellmchat import PDModeLLMChat
from AIPUBuilder.model.register import register_app, ENGINES
from AIPUBuilder.core import *
from AIPUBuilder.driver import MultiAIPUModels
from AIPUBuilder.tools.zipper.zip import *


@register_app()
class ShareWtLLMChat(PDModeLLMChat):
    """ShareWtLLMChat is used for LLM model chat based on share weight prefill + decoder mode."""

    def __init__(self, prompt: str = '', sys_prompt: str = '', sampling: str = "NONE", sampling_params: str = "", dump: bool = False):
        """ShareWtLLMChat constructor.

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
        super().__init__(prompt, sys_prompt, sampling, sampling_params, dump)

    @classmethod
    def engines(cls):
        return ["npu"]

    def _gen_engines(self):
        sharewt_idx = 0
        for aipu_config in self.configs:
            engine = aipu_config.engine
            if engine in ENGINES:
                if 'embedding' in aipu_config.model_type:
                    self.engines.append(ENGINES[engine](aipu_config))
                else:
                    self.engines.append(ENGINES[engine](job_or_model=self.sharewt_models[sharewt_idx]))
                    sharewt_idx += 1
            else:
                INFO(f"{self.__class__} meets issue: {engine} is not supported when using {aipu_config.config_name} of {self.model_type} model")
        # only provide aipu.zip in config
        if sharewt_idx == 1:
            self.engines.append(ENGINES[engine](job_or_model=self.sharewt_models[1]))

    def forward(self):
        if self.aipu_bins[0].endswith('.zip'):
            self.sharewt_zip_path = self.aipu_bins[0]
        else:
            sharewt_bins = []
            for idx, bin_item in enumerate(self.aipu_bins):
                if 'embed' not in bin_item:
                    sharewt_bins.append(bin_item)
                    self.sharewt_zip_path = os.path.join(self.configs[idx].output_dir, f"{self.configs[idx].model_name}_aipu.zip")
            compress(sharewt_bins, self.sharewt_zip_path)
        INFO(f"ShareWtLLMChat app running with {self.sharewt_zip_path}")
        self.sharewt_models = MultiAIPUModels(self.sharewt_zip_path)
        super().forward()
