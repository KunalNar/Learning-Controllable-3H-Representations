"""
Sanity check for the Llama-3.2-1B-Instruct port.

Verifies:
1. The wrapper loads.
2. tokenize_llama_chat(..., is_llama3=True) produces a prompt whose marker boundary
   (where steering would start) lands exactly between the assistant header and the
   first response token.
3. End-to-end generation produces a non-gibberish answer.

Usage: python smoke_test_llama3.py
Requires HF_TOKEN in env (.env or shell).
"""

import os
import torch as t
from dotenv import load_dotenv

from llama_wrapper import LlamaWrapper
from utils.tokenize import tokenize_llama_chat
from utils.helpers import find_instruction_end_postion

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def main():
    print("Loading Llama-3.2-1B-Instruct wrapper...")
    w = LlamaWrapper(hf_token=HF_TOKEN, size="1b", use_chat=True)
    print(f"  device: {w.device}, is_llama3: {w.is_llama3}")
    print(f"  pad_token: {w.tokenizer.pad_token}")
    print(f"  num layers: {len(w.model.model.layers)} (expected 16)")

    print("\nTokenizing a probe prompt with model_output='(A)'...")
    tokens = tokenize_llama_chat(
        tokenizer=w.tokenizer,
        user_input="Which is bigger: A) an elephant, B) a mouse?",
        model_output="(A",
        is_llama3=True,
    )
    tokens_t = t.tensor(tokens).to(w.device)
    print(f"  total tokens: {len(tokens)}")

    from_pos = find_instruction_end_postion(tokens_t, w.END_STR)
    print(f"  from_pos (steering boundary): {from_pos}")

    before = w.tokenizer.decode(tokens[: from_pos + 1])
    after = w.tokenizer.decode(tokens[from_pos + 1 :])
    print("\n--- Prompt up to and including from_pos ---")
    print(repr(before))
    print("\n--- Prompt after from_pos (this is what gets steered) ---")
    print(repr(after))

    last_two = w.tokenizer.decode(tokens[-2:])
    second_to_last = w.tokenizer.decode([tokens[-2]])
    print(f"\nLast two tokens decoded: {repr(last_two)}")
    print(f"Token at [-2] (used for activation extraction): {repr(second_to_last)}")

    print("\nRunning a quick generation test...")
    out = w.generate_text("What is 2+2? Answer in one word.", max_new_tokens=20)
    print("--- model output ---")
    print(out)


if __name__ == "__main__":
    main()
