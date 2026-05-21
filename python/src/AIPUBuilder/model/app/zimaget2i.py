import os
import sys
import torch
import numpy as np
import configparser
from datetime import datetime
from AIPUBuilder.core import use_cuda, Target
from AIPUBuilder.CGBuilder import aipudumper, run_simulator
from AIPUBuilder.logger import INFO, ERROR, WARN, DEBUG
from AIPUBuilder.model.app.baseapp import BaseApp
from AIPUBuilder.model.register import register_app, ENGINES
from transformers import AutoTokenizer
from AIPUBuilder.model.scripts.zimage_pipeline import (CompassZImagePipeline,
                                                       VAEWrappper,
                                                       TransformerWrapper,
                                                       EncoderWrapper,
                                                       LATENT_SIZE)


@register_app()
class ZImageT2I(BaseApp):
    '''
    ZImageT2I: get prompt and gen image
    '''

    def __init__(self, prompt="", img_size=1024, display=True, dump=False, dump_dir="", seed=None):
        super().__init__()
        self.prompt = prompt
        self.display = display
        if self.display:
            from matplotlib import pyplot as plt
            self.pltframe = plt
        self.seed = int(seed)
        self.dump = dump
        self.dump_dir = dump_dir
        self.device = torch.device('cuda') if use_cuda() else torch.device('cpu')
        self.img_size = int(img_size)
        LATENT_SIZE.value = self.img_size // 8
        self.engines = []

    @classmethod
    def engines(cls):
        return ['opt', 'exe']

    def forward(self, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs[0].tokenizer_path)
        for config in self.configs:
            self.engines.append(ENGINES[config.engine](config))

        emb_engine, encoder_engine, transformer_engine, vae_engine = self.engines
        encoder_warpper = EncoderWrapper(encoder_engine, emb_engine)
        transformer_warpper = TransformerWrapper(transformer_engine)
        vae_warpper = VAEWrappper(vae_engine)

        pipeline = CompassZImagePipeline(
            encoder_warpper,
            transformer_warpper,
            vae_warpper,
            self.tokenizer
        )

        img = pipeline.generate(self.prompt, seed=self.seed, device=self.device)
        if self.display:
            self.pltframe.imshow(img)
            self.pltframe.show()
        if self.dump:
            np.save(os.path.join(self.dump_dir, "img_output.npy"), img)
