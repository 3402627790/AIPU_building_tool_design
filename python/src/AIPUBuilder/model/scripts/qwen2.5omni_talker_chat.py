
import soundfile as sf
import torch
import os

# ===== IR 路径配置（必须在 transformers 模型 import 之前设置） =====
# 修改成你本地 IR 目录即可。留空则不启用 IR 后端。
THINKER_IR_AUDIO_DIR = "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/ir/thinker/qwen2.5thinker_audio.txt"
TALKER_IR_PREFILL_DIR  = "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/ir/talker/qwen2.5talker_prefill.txt"
TALKER_IR_DECODE_DIR  = "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/ir/talker/qwen2.5talker_seq1.txt"
TALKER_IR_DIT_DIR  = "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/ir/talker/qwen2.5token2wave_dit.txt"
TALKER_IR_BIGVGAN_DIR  = "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/ir/talker/qwen2.5token2wave_bigvgan.txt"

os.environ["QWEN_IR_THINKER_AUDIO"] = ''#THINKER_IR_AUDIO_DIR
os.environ["QWEN_IR_TALKER_LLM_PREFILL"]  = ''#TALKER_IR_PREFILL_DIR
os.environ["QWEN_IR_TALKER_LLM_DECODE"] = ''#TALKER_IR_DECODE_DIR
os.environ["QWEN_IR_TALKER_DIT"]  = TALKER_IR_DIT_DIR
os.environ["QWEN_IR_TALKER_BIGVGAN"]  = TALKER_IR_BIGVGAN_DIR

from AIPUBuilder.model.scripts import process_mm_info
from AIPUBuilder.model.scripts.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from AIPUBuilder.model.scripts.qwen2_5_omni_modeling import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniThinkerTextModel,Qwen2_5OmniConfig

# model dir
model_path = '/project/ai/scratch01/Model_Resource/huggingface/qwen2_5Omni/'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder=model_path,
    local_files_only = True,
    enable_audio_output=True,
).to(DEVICE)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

#用户配置对话，图像或者语音
# image
# what do you see?
# audio
# what do you hear?

# auido
# {"type": "audio", "audio": "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/xxx.mp3"}
# image
# {"type": "image", "image": "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/xxx.jpg"}
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "what do you hear?"},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/project/ai/scratch03/liwxu01/Qwen2.5-Omni/glass-breaking-151256.mp3"},
        ],
    },
]
USE_AUDIO_IN_VIDEO = False
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids,audio = model.generate(**inputs)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
sample_rate = 16000  # 固定采样率
sf.write("response.wav", audio.cpu().numpy().squeeze(), sample_rate)

print(text)