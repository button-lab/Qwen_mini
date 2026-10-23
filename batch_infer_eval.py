import argparse
import json
import os
import time
from typing import Dict, List, Tuple

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


def load_eval_items(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                ins = str(row.get("instruction", "")).strip()
                inp = str(row.get("input", "")).strip()
                prompt = ins if not inp else f"{ins}\n{inp}"
            if not prompt:
                continue
            items.append(
                {
                    "prompt": prompt,
                    "reference": str(row.get("reference", row.get("output", ""))).strip(),
                    "id": str(row.get("id", "")),
                }
            )
    return items


def build_inputs(tokenizer: AutoTokenizer, prompt: str, device: str) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    return {k: v.to(device) for k, v in model_inputs.items()}


def generate_one(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Tuple[str, float]:
    model_inputs = build_inputs(tokenizer, prompt, device)
    start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    seconds = time.time() - start
    gen_ids = output_ids[0][model_inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch eval for Qwen LoRA chat model")
    parser.add_argument("--adapter_dir", type=str, default="outputs/qwen-chat-v1")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--eval_file", type=str, default="eval_prompts.jsonl")
    parser.add_argument("--out_file", type=str, default="eval_results.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    args = parser.parse_args()

    if not os.path.exists(args.adapter_dir):
        raise FileNotFoundError(f"Adapter not found: {args.adapter_dir}")
    if not os.path.exists(args.eval_file):
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")

    items = load_eval_items(args.eval_file)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        raise ValueError("No valid eval items loaded from eval file")

    base_model_name = resolve_base_model(args.adapter_dir, args.base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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

    total_tokens = 0
    total_seconds = 0.0
    with open(args.out_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(items, start=1):
            answer, seconds = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=item["prompt"],
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            tokens = len(tokenizer.encode(answer, add_special_tokens=False))
            total_tokens += tokens
            total_seconds += seconds

            rec = {
                "index": i,
                "id": item["id"],
                "prompt": item["prompt"],
                "reference": item["reference"],
                "prediction": answer,
                "gen_tokens": tokens,
                "latency_sec": round(seconds, 3),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(items)}] tokens={tokens} latency={seconds:.2f}s")

    tps = total_tokens / total_seconds if total_seconds > 0 else 0.0
    print(f"\nDone. results={args.out_file}")
    print(f"Samples={len(items)} total_tokens={total_tokens} total_sec={total_seconds:.2f} tok_per_sec={tps:.2f}")


if __name__ == "__main__":
    main()
