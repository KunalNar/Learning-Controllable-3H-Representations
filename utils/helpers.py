import torch as t
import matplotlib.pyplot as plt


MODEL_NAME_ALIASES = {
    "llama-3.2-1b": ("meta-llama/Llama-3.2-1B", "1b", False),
    "meta-llama/llama-3.2-1b": ("meta-llama/Llama-3.2-1B", "1b", False),
    "llama-3.2-1b-instruct": ("meta-llama/Llama-3.2-1B-Instruct", "1b", True),
    "meta-llama/llama-3.2-1b-instruct": ("meta-llama/Llama-3.2-1B-Instruct", "1b", True),
    "llama-2-7b-hf": ("meta-llama/Llama-2-7b-hf", "7b", False),
    "meta-llama/llama-2-7b-hf": ("meta-llama/Llama-2-7b-hf", "7b", False),
    "llama-2-7b-chat-hf": ("meta-llama/Llama-2-7b-chat-hf", "7b", True),
    "meta-llama/llama-2-7b-chat-hf": ("meta-llama/Llama-2-7b-chat-hf", "7b", True),
    "llama-2-13b-hf": ("meta-llama/Llama-2-13b-hf", "13b", False),
    "meta-llama/llama-2-13b-hf": ("meta-llama/Llama-2-13b-hf", "13b", False),
    "llama-2-13b-chat-hf": ("meta-llama/Llama-2-13b-chat-hf", "13b", True),
    "meta-llama/llama-2-13b-chat-hf": ("meta-llama/Llama-2-13b-chat-hf", "13b", True),
}

def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)



def add_vector_from_position(matrix, vector, position_ids, from_pos=None):
    from_id = from_pos
    if from_id is None:
        from_id = position_ids.min().item() - 1

    mask = position_ids >= from_id
    mask = mask.unsqueeze(-1)

    matrix += mask.float() * vector
    return matrix


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'


def get_model_path(size: str, is_base: bool):
    if size == "1b":
        return "meta-llama/Llama-3.2-1B" if is_base else "meta-llama/Llama-3.2-1B-Instruct"
    if is_base:
        return f"meta-llama/Llama-2-{size}-hf"
    else:
        return f"meta-llama/Llama-2-{size}-chat-hf"


def resolve_model_name(model_name: str):
    normalized_name = model_name.strip().lower()
    if normalized_name not in MODEL_NAME_ALIASES:
        supported_names = ", ".join(sorted(MODEL_NAME_ALIASES))
        raise ValueError(
            f"Unsupported model_name '{model_name}'. Supported names: {supported_names}"
        )
    return MODEL_NAME_ALIASES[normalized_name]

def is_llama3_size(size: str) -> bool:
    return size == "1b"

def model_name_format(name: str) -> str:
    name = name.lower()
    if "llama-3.2-1b" in name:
        return "Llama 3.2 1B Instruct" if "instruct" in name else "Llama 3.2 1B"
    is_chat = "chat" in name
    is_7b = "7b" in name
    if is_chat:
        if is_7b:
            return "Llama 2 Chat 7B"
        else:
            return "Llama 2 Chat 13B"
    else:
        if is_7b:
            return "Llama 2 7B"
        else:
            return "Llama 2 13B"
