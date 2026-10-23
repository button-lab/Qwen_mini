# Qwen Mini Fine-tuning Pipeline (1B-class Chat)

This repository contains a lightweight end-to-end workflow for training and evaluating a **small daily-chat model** based on Qwen (using LoRA fine-tuning on top of `Qwen/Qwen2.5-1.5B-Instruct`).

It is designed for single-GPU local development (including ~8GB VRAM setups with optimized settings).

## Features

- Public dataset preparation and conversion to JSONL
- LoRA fine-tuning script for Qwen
- Interactive chat inference script
- Batch evaluation script
- Simple automatic evaluation scoring
- LoRA merge script (export standalone merged model)

## Repository Files

- `train_qwen1b_framework.py` - training pipeline (LoRA fine-tuning)
- `prepare_public_data.py` - public dataset loader + converter
- `infer_chat.py` - interactive multi-turn chat test
- `batch_infer_eval.py` - batch inference on eval prompts
- `score_eval.py` - heuristic scoring for eval results
- `merge_lora.py` - merge LoRA adapter into base model
- `eval_prompts.jsonl` - sample evaluation prompts

## Environment Setup

Recommended: Python 3.10 with CUDA-enabled PyTorch.

```bash
conda create -n qwen_new python=3.10 -y
conda activate qwen_new
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers datasets peft accelerate sentencepiece
```

> If your GPU architecture requires a different CUDA wheel index, switch to the proper PyTorch index URL.

## 1) Prepare Training Data

Generate mixed public data:

```bash
python prepare_public_data.py --source mix --limit 6000 --out train_public.jsonl
```

Expected JSONL schema per line:

```json
{"instruction":"...","input":"...","output":"..."}
```

## 2) Train LoRA Model

```bash
python train_qwen1b_framework.py ^
  --dataset_path train_public.jsonl ^
  --output_dir outputs/qwen-chat-v1 ^
  --batch_size 1 ^
  --grad_accum_steps 16 ^
  --epochs 2 ^
  --max_seq_len 1024
```

Windows PowerShell single-line equivalent:

```bash
python train_qwen1b_framework.py --dataset_path train_public.jsonl --output_dir outputs/qwen-chat-v1 --batch_size 1 --grad_accum_steps 16 --epochs 2 --max_seq_len 1024
```

## 3) Interactive Chat Inference

```bash
python infer_chat.py --adapter_dir outputs/qwen-chat-v1
```

Commands inside chat:

- `/reset` clear conversation history
- `/exit` exit program

## 4) Batch Evaluation

```bash
python batch_infer_eval.py --adapter_dir outputs/qwen-chat-v1 --eval_file eval_prompts.jsonl --out_file eval_results.jsonl --limit 10
```

## 5) Score Evaluation Results

```bash
python score_eval.py --in_file eval_results.jsonl --out_file eval_scores.jsonl --summary_file eval_summary.json
```

Outputs:

- `eval_scores.jsonl` - per-sample metrics
- `eval_summary.json` - aggregate summary (avg score, top/bottom samples, speed stats)

## 6) Merge LoRA (Optional)

Export as standalone merged model:

```bash
python merge_lora.py --adapter_dir outputs/qwen-chat-v1 --merged_out outputs/qwen-chat-v1-merged
```

## Suggested GitHub Upload Scope

Upload:

- all `.py` scripts
- `eval_prompts.jsonl`
- `.gitignore`
- `README.md`

Do not upload:

- large model weights in `outputs/`
- large raw datasets
- local debug logs/cache files

## Notes

- You may see HF warning about unauthenticated requests; set `HF_TOKEN` for faster downloads/rate limits.
- On Windows, warnings related to `torch.distributed` redirects are usually non-blocking for this workflow.

