import os
import torch
import numpy as np
from datetime import datetime
from AIPUBuilder.core import Dtype
from AIPUBuilder.logger import INFO, ERROR, WARN, DEBUG
from AIPUBuilder.model.app.llmchat import LLMChat
from AIPUBuilder.model.register import register_app, ENGINES


@register_app()
class VLChat(LLMChat):
    """VLChat is used for multi-modal chat and chat scripts uses python."""

    def __init__(self,
                 prompt: str = '',
                 sys_prompt: str = '',
                 sampling: str = "NONE",
                 sampling_params: str = "",
                 image: str = '',
                 video: str = '',
                 post_process: str = 'false',
                 **kwargs):
        """VLChat constructor.

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

        image: str
            the image path

        video: str
            the video path
        """

        super().__init__()
        self.prompts = self._gen_prompt(prompt)
        self.sys_prompt = sys_prompt
        self.sampling = sampling
        self.sampling_params = sampling_params
        self.images = image
        self.videos = video
        self.post_process = post_process.lower() != 'false'
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

    def get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
    ):
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.configs[-1].vision_config.get('spatial_merge_size', 2)
        image_token_id = self.configs[-1].image_token_id
        video_token_id = self.configs[-1].video_token_id
        vision_start_token_id = self.configs[-1].vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * \
                        self.configs[-1].vision_config.get(
                            'tokens_per_second', 2)

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas

    def _gen_messages(self, prompts, sys_prompt='', images='', videos=''):
        def _gen_image_or_vedio(imgs):
            imgs_list = imgs.replace(' ', '').split(',')
            for i, il in enumerate(imgs_list[:]):
                if not os.path.exists(il):
                    WARN(f"image or vedio path: {il} does not exist, please check it, \
                        and pop it from image/vedio list.")
                    imgs_list.pop(i)
            return imgs_list

        messages = [
            {"role": "user", 'content': []}
        ]

        if len(images) > 0:
            imgs = _gen_image_or_vedio(images)
            for img in imgs:
                messages[0]['content'].append({'type': 'image', 'image': img})
        if len(videos) > 0:
            vdos = _gen_image_or_vedio(videos)
            for vd in vdos:
                messages[0]['content'].append({'type': 'video', 'video': vd})

        if len(prompts):
            for prompt in prompts:
                messages[0]['content'].append({'type': 'text', 'text': prompt})
        return messages

    def _vl_prepare(self):

        from transformers import AutoProcessor
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as e:
            WARN(f"please pip install qwen_vl_utils first if you want use the latest code of qwen_vl_utils")
            from AIPUBuilder.model.scripts import process_vision_info

        mp = self.configs[0].model_path
        processor = AutoProcessor.from_pretrained(mp)
        messages = self._gen_messages(self.prompts, self.sys_prompt, self.images, self.videos)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if type(text)==list: #qwen omni text is list
            text = text[0]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        msg = f"1.input text = \n{text}"
        if image_inputs is not None and len(image_inputs):
            msg += f"2.input image = {self.images}, and its shape:"
            for img in image_inputs:
                msg += f"{img.size},"
        if video_inputs is not None and len(video_inputs):
            msg += f", input videos = {self.videos}"
            # TODO: add videos information

        INFO(f"now input informations for vl chat: {msg}")

        # input.keys() = ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        for inp_key, inp_value in inputs.items():
            DEBUG(f"{inp_key}'s shape = {inp_value.shape}")
        return inputs

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

    def generate(self):
        # model order: embedding(if have), vision part, llm part
        cache_shape = self.engines[-1].input_shapes[-1]
        if self.configs[-1].get('kvcache_optimization', False):
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

        pixel_values = None
        pixel_values_videos = None
        position_ids = None
        rope_deltas = None
        second_per_grid_ts = None
        with torch.no_grad():
            iters = 0
            # first time, generate inputs with input text, input image or input vedio
            inputs = self._vl_prepare()
            init_seq_len = inputs['input_ids'].shape[-1]
            input_ids = inputs['input_ids']
            pixel_values = inputs.get('pixel_values', None)
            pixel_values_videos = inputs.get('pixel_values_videos', None)
            image_grid_thw = inputs.get('image_grid_thw', [])
            video_grid_thw = inputs.get('video_grid_thw', [])
            attention_mask = inputs.get('attention_mask', None)
            while True:
                start_idx = 0
                if init_seq_len + iters > self.max_len:
                    break
                # embedding
                if not self.has_configured_embedding:
                    inputs_embeds = (self.embedding_data.to(self.device))[input_ids]
                else:
                    inputs_embeds = self.engines[0].forward(input_ids)[0]
                    start_idx += 1

                if isinstance(inputs_embeds, np.ndarray):
                    # support input_id.dtype=bfloat16, if input_id.dtype == bfloat16, torch.from_numpy not support this dtype
                    torch_dtype = Dtype.from_np(inputs_embeds.dtype).torch
                    inputs_embeds = torch.from_numpy(inputs_embeds.astype(
                        np.float32)).to(torch_dtype).to(input_ids.device)
                inputs_embeds = inputs_embeds.float()

                if iters == 0 and pixel_values is not None:

                    # for compass the pixel_values as conv3d input should match compass conv3d input shape rule
                    if self.configs[-2].get("using_3d", True):
                        shape = pixel_values.shape
                        patch_size = self.configs[-2].patch_size
                        in_channels = self.configs[-2].in_channels
                        temporal_patch_size = self.configs[-2].temporal_patch_size
                        pixel_values = pixel_values.reshape(-1, in_channels,
                                                            temporal_patch_size, patch_size, patch_size)
                        pixel_values = pixel_values.permute(0, 2, 3, 4, 1)
                        pixel_values = pixel_values.reshape(shape)
                    vision_embeds = self.engines[start_idx].forward([pixel_values])
                    image_embeds = vision_embeds[0]
                    if self.post_process:
                        window_index = torch.tensor(vision_embeds[1], device=pixel_values.device)
                        reverse_indices = torch.argsort(window_index)
                        image_embeds = image_embeds[reverse_indices.cpu().numpy(), :]

                    INFO(f"vision part finished, the image_embeds.shape={image_embeds.shape}")
                    image_embeds = torch.tensor(image_embeds, device=pixel_values.device)
                    start_idx += 1

                    # vision postprocess
                    n_image_tokens = (
                        input_ids == self.configs[-1].image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        ERROR(f"Image features and image tokens do not match: tokens: {n_image_tokens}, \
                              features {n_image_features}")
                    mask = input_ids == self.configs[-1].image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

                if iters == 0 and pixel_values_videos is not None:
                    # TODO support video
                    pass
                if isinstance(cache_k[0], np.ndarray):
                    ckn = torch.tensor(cache_k[0]).numel()
                else:
                    ckn = cache_k[0].numel()

                past_length = cache_k[0].shape[2] if ckn != 0 else 0
                cache_position = torch.arange(
                    0, input_ids.shape[1], dtype=torch.long, device=input_ids.device).int()
                cache_position += past_length

                if position_ids is None or rope_deltas is None or iters == 0:
                    position_ids, rope_deltas = self.get_rope_index(input_ids,
                                                                    image_grid_thw,
                                                                    video_grid_thw,
                                                                    second_per_grid_ts,
                                                                    attention_mask)
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

                # INFO(f"generate position_ids finished, position_ids.shape={position_ids.shape}")

                if pixel_values is None or iters > 0:
                    # iters > 0 or no vision part, will defaulty skip vision part
                    start_idx += 1
                iters += 1
                llm_inputs = self._gen_inputs(inputs_embeds, position_ids, cache_k, cache_v, init_seq_len, iters,
                                              self.configs[-1].get('kvcache_optimization', False))

                for engine in self.engines[start_idx:]:
                    out = engine.forward(llm_inputs)
                    llm_inputs = out

                if self.configs[-1].kvcache_optimization:
                    cache_k = out[1:2]
                    cache_v = out[2:]
                else:
                    cache_k = out[1: 1 + num_hidden_layers]
                    cache_v = out[1 + num_hidden_layers:]
                logits = out[0]
                token = logits[0, -1].argmax()
                input_ids = torch.tensor(token).int().flatten().reshape([1, -1])
                yield token
                if token in self.terminators:
                    break
        return self.aipu_tokenizer.eos_token_id

    def forward(self):

        self._prepare()

        for prompt in self.prompts:
            self._write_answer(f"\n{datetime.now()}\n{self.config_basic_infos}{prompt}\n")
            answer = []
            for token in self.generate():
                answer.append(token)
                s = self.aipu_tokenizer.decode(token)
                self._write_answer(s)

            self._write_answer(f"\n")
