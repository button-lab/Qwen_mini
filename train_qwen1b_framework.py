import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    # Qwen 当前公开模型没有严格 1B，1.5B 是最接近且稳定可用的选择
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_path: str = "train.jsonl"
    output_dir: str = "outputs/qwen-daily-chat-1b-like"
    max_seq_len: int = 1024
    learning_rate: float = 2e-4
    epochs: int = 2
    batch_size: int = 1
    grad_accum_steps: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_on_prompt: bool = False
    fp16: bool = True


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Qwen 日常对话微调框架（近 1B 级别）")
    parser.add_argument("--model_name", type=str, default=TrainConfig.model_name)
    parser.add_argument("--dataset_path", type=str, default=TrainConfig.dataset_path)
    parser.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--max_seq_len", type=int, default=TrainConfig.max_seq_len)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument("--lora_r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=TrainConfig.lora_dropout)
    parser.add_argument("--train_on_prompt", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        train_on_prompt=args.train_on_prompt,
        fp16=not args.no_fp16,
    )


def load_jsonl_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到数据集文件: {path}。请提供 JSONL，每行格式例如: "
            '{"instruction":"你好","input":"","output":"你好呀，很高兴见到你。"}'
        )
    ds = load_dataset("json", data_files=path, split="train")
    return ds


def build_prompt(example: Dict, train_on_prompt: bool) -> Dict[str, str]:
    instruction = str(example.get("instruction", "")).strip()
    user_input = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    if not instruction or not output:
        raise ValueError("数据字段缺失：每条样本至少需要 instruction 和 output")

    user_text = instruction if not user_input else f"{instruction}\n{user_input}"
    prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    text = prompt + output + "<|im_end|>"
    return {
        "text": text,
        "prompt": prompt,
        "answer": output,
        "train_on_prompt": train_on_prompt,
    }


def tokenize_dataset(ds: Dataset, tokenizer: AutoTokenizer, max_seq_len: int) -> Dataset:
    def _mapper(example: Dict) -> Dict[str, List[int]]:
        built = build_prompt(example, train_on_prompt=False)
        encoded = tokenizer(
            built["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return ds.map(_mapper, remove_columns=ds.column_names)


def main() -> None:
    cfg = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_ds = load_jsonl_dataset(cfg.dataset_path)
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer, cfg.max_seq_len)

    # Frozen base weights stay fp16 to fit an 8GB GPU; LoRA params stay fp32 for AMP GradScaler.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
    )
    model.add_adapter(lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=cfg.fp16,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
