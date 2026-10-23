import argparse
import json
import os
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_base_model(adapter_dir: str, fallback: str) -> str:
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


def _build_inputs(tokenizer: AutoTokenizer, messages: List[Dict[str, str]], device: str):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    return {k: v.to(device) for k, v in model_inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen LoRA 聊天推理脚本")
    parser.add_argument("--adapter_dir", type=str, default="outputs/qwen-chat-v1")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    args = parser.parse_args()

    if not os.path.exists(args.adapter_dir):
        raise FileNotFoundError(f"未找到 adapter 目录: {args.adapter_dir}")

    base_model_name = _resolve_base_model(args.adapter_dir, args.base_model)
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
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    print("已加载模型，输入 /exit 退出，输入 /reset 清空上下文。")
    messages: List[Dict[str, str]] = []

    while True:
        user_text = input("\nYou> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit"}:
            print("Bye.")
            break
        if user_text.lower() == "/reset":
            messages = []
            print("上下文已清空。")
            continue

        messages.append({"role": "user", "content": user_text})
        model_inputs = _build_inputs(tokenizer, messages, device)

        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0][model_inputs["input_ids"].shape[-1] :]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        print(f"Assistant> {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
