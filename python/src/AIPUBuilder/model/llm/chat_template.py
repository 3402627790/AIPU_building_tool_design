from AIPUBuilder.model.register import register_chat_template
from transformers import AutoTokenizer

EOT_TOKENS = {
    'llama3': lambda tokenizer: tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    'llama2_tiny': lambda tokenizer: 1,
}


@register_chat_template(['qwen2', 'llama3'])
def qwen2_chat_template(tokenizer, prompt, sys_prompt=''):
    # if sys_prompt == '':
    #     sys_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    messages = [
        {"role": "user", "content": prompt}
    ]
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# @register_chat_template(['llama3'])
# def llama3_gp_chat_template(tokenizer, prompt, sys_prompt=''):
#     query_str = f'''\
#         <|begin_of_text|><|start_header_id|>system<|end_header_id|>

#         {sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

#         {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#         '''
#     return query_str


@register_chat_template(['qwen3'])
def qwen3_chat_template(tokenizer, prompt, sys_prompt=''):
    # if sys_prompt == '':
    #     sys_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    messages = [
        {"role": "user", "content": prompt}
    ]
    if len(sys_prompt):
        messages[0].update({'system': sys_prompt})

    # thinking = len(sys_prompt) > 0
    thinking = True

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


@register_chat_template(['chatglm3', 'chatglm'])
def chatglm_chat_template(tokenizer, prompt, sys_prompt=''):
    prompt = prompt[:-2] if prompt.endswith('\\n') else prompt
    messages = [
        {"role": "user", "content": prompt}
    ]
    if sys_prompt != "":
        messages.insert(0, {"role": "system", "content": sys_prompt})
    if hasattr(tokenizer, 'apply_chat_template'):
        after_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        if sys_prompt != '':
            sys_prompt = f"{sys_prompt}\n"
            after_template = f"[gMASK]<|system|>\n{sys_prompt}<|user|>\n{prompt}<|assistant|>"
        else:
            after_template = f"[gMASK]sop<|user|>\n{prompt}<|assistant|>"

    return after_template


@register_chat_template(['llama2_chn'])
def llama2_chn_chat_template(tokenizer, prompt, sys_prompt=''):
    if len(sys_prompt) > 0 and not sys_prompt.endswith('\\n'):
        sys_prompt = f"{sys_prompt}\n"
    if prompt.endswith('\\n'):
        prompt = f"{prompt[:-2]}\n"
    else:
        prompt = f"{prompt}\n"
    if sys_prompt == '':
        after_template = f"<s>### Instruction:{prompt}### Response:"
    else:
        after_template = f'''\
            {sys_prompt}
            <s>### Instruction:{prompt}
            ### Response:\
            '''

    return after_template


@register_chat_template(['llama2'])
def llama2_chat_template(tokenizer, prompt, sys_prompt=''):
    if prompt.endswith('\\n'):
        prompt = f"{prompt[:-2]}\n"
    if sys_prompt == "":
        after_template = f"[INST]{prompt}[/INST]"
    else:
        if sys_prompt.endswith("\\n"):
            sys_prompt = f"{sys_prompt[:-2]}\n"
        else:
            sys_prompt = f"{sys_prompt}\n"
        after_template = f"[INST]<<SYS>>\n{sys_prompt}<</SYS>>\n\n{prompt}[/INST]"

    return after_template


@register_chat_template(['llama2_tiny'])
def llama2_tiny_chat_template(tokenizer, prompt, sys_prompt=''):
    # prompt = f'<s>{prompt}'
    return prompt


@register_chat_template(['deepseek_moe'])
def deepseek_moe_chat_template(tokenizer, prompt, sys_prompt=''):
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@register_chat_template(['ernie4_5'])
def ernie4_5_chat_template(tokenizer, prompt, sys_prompt=''):
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@register_chat_template(['minicpm4'])
def minicpm4_chat_template(tokenizer, prompt, sys_prompt=''):
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

@register_chat_template(['gpt_oss'])
def gpt_oss_chat_template(tokenizer, prompt, sys_prompt=''):
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)