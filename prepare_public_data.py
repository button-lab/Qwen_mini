import argparse
import json
import random
from typing import Dict, List

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError


def from_alpaca(limit: int) -> List[Dict[str, str]]:
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = []
    for ex in ds:
        instruction = str(ex.get("instruction", "")).strip()
        user_input = str(ex.get("input", "")).strip()
        output = str(ex.get("output", "")).strip()
        if instruction and output:
            rows.append({"instruction": instruction, "input": user_input, "output": output})
        if len(rows) >= limit:
            break
    return rows


def from_lima(limit: int) -> List[Dict[str, str]]:
    ds = load_dataset("GAIR/lima", split="train")
    rows = []
    for ex in ds:
        conv = ex.get("conversations", [])
        if not isinstance(conv, list):
            continue
        # conversations alternates user/assistant in most samples
        for i in range(0, len(conv) - 1, 2):
            user_text = str(conv[i]).strip()
            assistant_text = str(conv[i + 1]).strip()
            if user_text and assistant_text:
                rows.append({"instruction": user_text, "input": "", "output": assistant_text})
            if len(rows) >= limit:
                return rows
    return rows


def from_oasst1(limit: int) -> List[Dict[str, str]]:
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {}
    for ex in ds:
        by_id[ex["message_id"]] = ex

    rows = []
    for ex in ds:
        role = str(ex.get("role", "")).lower()
        parent_id = ex.get("parent_id")
        if role != "assistant" or not parent_id:
            continue
        parent = by_id.get(parent_id)
        if not parent:
            continue
        parent_role = str(parent.get("role", "")).lower()
        if parent_role != "prompter":
            continue
        instruction = str(parent.get("text", "")).strip()
        output = str(ex.get("text", "")).strip()
        if instruction and output:
            rows.append({"instruction": instruction, "input": "", "output": output})
        if len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public chat datasets and convert to train.jsonl format")
    parser.add_argument("--source", choices=["alpaca", "lima", "oasst1", "mix"], default="mix")
    parser.add_argument("--limit", type=int, default=10000, help="Total number of output samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="train_public.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    all_rows: List[Dict[str, str]] = []

    try:
        if args.source == "alpaca":
            all_rows = from_alpaca(args.limit)
        elif args.source == "lima":
            all_rows = from_lima(args.limit)
        elif args.source == "oasst1":
            all_rows = from_oasst1(args.limit)
        else:
            per_source = max(args.limit // 3, 1)
            for name, fn in [
                ("alpaca", from_alpaca),
                ("lima", from_lima),
                ("oasst1", from_oasst1),
            ]:
                try:
                    rows = fn(per_source)
                    all_rows.extend(rows)
                except DatasetNotFoundError as e:
                    print(f"Skip gated dataset '{name}': {e}")
                    continue
            random.shuffle(all_rows)
            all_rows = all_rows[: args.limit]
    except DatasetNotFoundError as e:
        if args.source == "lima":
            raise RuntimeError(
                "lima 数据集需要 Hugging Face 认证（gated）。"
                "请先 `huggingface-cli login`，或改用 --source alpaca/oasst1/mix。"
            ) from e
        raise

    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
