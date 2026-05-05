from typing import List
from transformers import PreTrainedTokenizer

# Llama-2 chat tokens
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Base-model prompt markers (used by both Llama-2 and Llama-3 base variants)
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

# Llama-3 chat tokens
L3_BOS = "<|begin_of_text|>"
L3_SH, L3_EH = "<|start_header_id|>", "<|end_header_id|>"
L3_EOT = "<|eot_id|>"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_CHAT_L3 = f"{L3_SH}assistant{L3_EH}\n\n"
ADD_FROM_POS_BASE = BASE_RESPONSE


def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
    is_llama3: bool = False,
) -> List[int]:
    if is_llama3:
        content = L3_BOS
        if system_prompt is not None:
            content += f"{L3_SH}system{L3_EH}\n\n{system_prompt}{L3_EOT}"
        content += f"{L3_SH}user{L3_EH}\n\n{user_input.strip()}{L3_EOT}"
        content += f"{L3_SH}assistant{L3_EH}\n\n"
        if model_output is not None:
            content += f" {model_output.strip()}"
        return tokenizer.encode(content, add_special_tokens=False)

    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    input_content += f"{B_INST} {user_input.strip()} {E_INST}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)
