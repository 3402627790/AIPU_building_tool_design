import os, sys, gc
import numpy as np
import torch
import configparser
from datetime import datetime
from AIPUBuilder.core import use_cuda, Target
from AIPUBuilder.CGBuilder import aipudumper, run_simulator
from AIPUBuilder.logger import INFO, ERROR, WARN, DEBUG
from AIPUBuilder.model.app.baseapp import BaseApp
from AIPUBuilder.model.register import register_app, ENGINES
from transformers import AutoTokenizer

from tqdm import tqdm

FIX_WIDTH = 1280
FIX_HEIGHT = 704
FIX_FRAMES = 81

@register_app()
class WANTI2V(BaseApp):

    latents_mean = [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.157,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.123,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.052,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667
    ]
    latents_std = [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.499,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.06,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744
    ]

    def __init__(self,
        img:str,
        positive_prompt:str,
        negative_prompt:str,
        step:int,
        guidance_scale:float,
        seed:int):
        super().__init__()
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.step = int(step)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.device = 'cuda' if use_cuda() else 'cpu'
        self.dtype = torch.bfloat16 if use_cuda() else torch.float32
        self.img = self.load_image(img)
        self.engines = []

        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16

    def load_image(self, path):
        from matplotlib import pyplot as plt
        img = plt.imread(path)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        # N, C, H, W
        img = img.to(self.device).to(self.dtype)
        img = img[:,:3] # crop alpha channel
        img = img / 255.0
        return img

    def get_t5_prompt_embeds(self, emb_engine, llm_engine, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        text_input_ids, mask = text_inputs["input_ids"], text_inputs["attention_mask"]
        seqlen = mask.gt(0).sum().long()
        text_input_ids = text_input_ids.reshape(1,-1)
        hidden = emb_engine.forward([text_input_ids], tensor_type="pt")[0].to(self.dtype).to(self.device)
        mask = torch.ones([512,512], dtype=text_input_ids.dtype, device=text_input_ids.device)
        mask[:, seqlen:] = 0
        mask = ((mask-1) * 100000).reshape(1,1,512,512)
        pos = torch.arange(512, dtype=torch.int32, device=text_input_ids.device).reshape(1,512)
        res = llm_engine.forward([hidden, mask, pos], tensor_type="torch")[0]
        res = res.to(self.device).to(self.dtype)
        res[:, seqlen:] = 0
        return res

    def prepare_latents(self,
        vae_encoder_engine,
        image: torch.Tensor,
        num_channels_latents:int = 16,
        height: int = 480,
        width: int = 640,
        num_frames: int = 81,
        generator: torch.Generator = None):

        assert len(image.shape) == 4, "image must be 4D"

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (1, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)

        image = image.unsqueeze(2) # [1, channels, 1, height, width]

        latents_mean = torch.tensor(WANTI2V.latents_mean).reshape(1, -1, 1, 1, 1).to(self.device).to(self.dtype)
        latents_std = torch.tensor(WANTI2V.latents_std).reshape(1, -1, 1, 1, 1).to(self.device).to(self.dtype)
        latents_std = 1.0 / latents_std

        b,c,t,h,w = image.shape
        image = image.reshape(b,c,t,h//2,2,w//2,2)
        image = image.permute(0,1,6,4,2,3,5).contiguous()
        image = image.reshape(b,c*4,t,h//2,w//2)
        latent_condition = vae_encoder_engine.forward([image], tensor_type="pt")[0]
        latent_condition = latent_condition.to(self.device).to(self.dtype)
        latent_condition = latent_condition[:,:num_channels_latents]
        latent_condition = (latent_condition - latents_mean) * latents_std
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width
        , dtype=self.dtype, device=self.device)
        first_frame_mask[:, :, 0] = 0
        return latents, latent_condition, first_frame_mask

    @classmethod
    def engines(cls):
        return ['opt', 'exec', 'sim']

    def forward(self, *args, **kwargs):
        height = FIX_HEIGHT
        width = FIX_WIDTH
        num_frames = FIX_FRAMES
        from diffusers import UniPCMultistepScheduler
        self.tokenizer = AutoTokenizer.from_pretrained(self.configs[0].tokenizer_path)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(self.configs[0].scheduler_path)
        for config in self.configs:
            self.engines.append(ENGINES[config.engine](config))

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1

        text_emb_engine, prefill_engine, transformer_engine, time_emb_engine, vae_encoder_engine, vae_decoder_engine = self.engines
        time_emb_engine_bin = time_emb_engine.irb
        time_emb_engine.opt_engine = None
        gc.collect()
        torch.cuda.empty_cache()

        prompt_embeds = self.get_t5_prompt_embeds(text_emb_engine, prefill_engine, self.positive_prompt)
        negative_prompt_embeds = self.get_t5_prompt_embeds(text_emb_engine, prefill_engine, self.negative_prompt)

        # offload text models
        text_emb_engine.opt_engine = None
        prefill_engine.opt_engine = None
        gc.collect()
        torch.cuda.empty_cache()

        self.scheduler.set_timesteps(self.step)
        timesteps = self.scheduler.timesteps

        latents_outputs = self.prepare_latents(vae_encoder_engine, self.img, 48, height, width, num_frames, generator)
        latents, condition, first_frame_mask = latents_outputs

        contextlen = width // 32 * height // 32 * (num_frames // self.vae_scale_factor_temporal + 1)
        temb_block_size = contextlen * 3072 * 2

        vae_encoder_engine.opt_engine = None
        gc.collect()
        torch.cuda.empty_cache()

        for i, t in tqdm(enumerate(timesteps)):
            latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents

            with open(time_emb_engine_bin, "rb") as f:
                f.seek(temb_block_size * i)
                time_emb = f.read(temb_block_size)

            time_emb = torch.frombuffer(time_emb, dtype=self.dtype).to(self.device)
            time_emb = time_emb.reshape(1, contextlen, 3072)

            noise_cond = transformer_engine.forward([latent_model_input, time_emb, prompt_embeds], tensor_type="pt")[0]
            noise_uncond = transformer_engine.forward([latent_model_input, time_emb, negative_prompt_embeds], tensor_type="pt")[0]
            noise_pred = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
        latents = latents / self.latents_std + self.latents_mean

        transformer_engine.opt_engine = None
        gc.collect()
        torch.cuda.empty_cache()

        ts = latents.shape[2]

        frames = []
        for i in tqdm(range(3, ts)):
            frame = vae_decoder_engine.forward([latents[:,:,i-3:i]], tensor_type="pt")[0]
            b,c,t,h,w = frame.shape
            c = c // 4
            frame = frame.reshape(b,c,2,2,t,h,w)
            frame = frame.permute(0,1,4,5,3,6,2).contiguous()
            frame = frame.reshape(b,c,t,h*2,w*2)
            frame = frame.clamp(0,1)
            if i == 3:
                frames.append(frame)
            else:
                frames.append(frame[:,:,-4:])
        frames = torch.cat(frames, dim=2)
        torch.save(frames.cpu(), "video_frame.pt")
