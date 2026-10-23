import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_base_model(adapter_dir: str, fallback: str) -> str:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return fallback
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = str(cfg.get("base_model_name_or_path", "")).strip()
        return base or fallback
    except Exception:
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Qwen LoRA adapter into a standalone model")
    parser.add_argument("--adapter_dir", type=str, default="outputs/qwen-chat-v1")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--merged_out", type=str, default="outputs/qwen-chat-v1-merged")
    args = parser.parse_args()

    if not os.path.exists(args.adapter_dir):
        raise FileNotFoundError(f"Adapter not found: {args.adapter_dir}")

    base_model_name = resolve_base_model(args.adapter_dir, args.base_model)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    merged = peft_model.merge_and_unload()

    os.makedirs(args.merged_out, exist_ok=True)
    merged.save_pretrained(args.merged_out, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_out)

    print(f"Merged model saved to: {args.merged_out}")


if __name__ == "__main__":
    main()
