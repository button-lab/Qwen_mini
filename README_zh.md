# Qwen Mini 微调流程（1B 级日常对话）

本仓库提供一套轻量级端到端流程，用于基于 Qwen 训练一个**日常交流小模型**（使用 `Qwen/Qwen2.5-1.5B-Instruct` + LoRA 微调）。

该流程适合本地单卡开发（已针对约 8GB 显存场景做过参数优化）。

## 功能概览

- 公开数据下载与 JSONL 转换
- Qwen LoRA 微调训练脚本
- 交互式多轮对话推理脚本
- 批量评测脚本
- 自动评分脚本（启发式）
- LoRA 合并脚本（导出独立模型）

## 仓库文件说明

- `train_qwen1b_framework.py`：训练主流程（LoRA 微调）
- `prepare_public_data.py`：公开数据下载与转换
- `infer_chat.py`：交互式多轮对话
- `batch_infer_eval.py`：批量推理评测
- `score_eval.py`：评测结果自动打分
- `merge_lora.py`：LoRA 合并导出
- `eval_prompts.jsonl`：示例评测题集

## 环境配置

推荐：Python 3.10 + CUDA 版本 PyTorch。

```bash
conda create -n qwen_new python=3.10 -y
conda activate qwen_new
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers datasets peft accelerate sentencepiece
```

> 如果你的显卡架构需要其他 CUDA 轮子，请切换到对应的 PyTorch index-url。

## 1) 准备训练数据

生成混合公开数据：

```bash
python prepare_public_data.py --source mix --limit 6000 --out train_public.jsonl
```

输出 JSONL 每行格式：

```json
{"instruction":"...","input":"...","output":"..."}
```

## 2) 训练 LoRA 模型

```bash
python train_qwen1b_framework.py ^
  --dataset_path train_public.jsonl ^
  --output_dir outputs/qwen-chat-v1 ^
  --batch_size 1 ^
  --grad_accum_steps 16 ^
  --epochs 2 ^
  --max_seq_len 1024
```

Windows PowerShell 单行写法：

```bash
python train_qwen1b_framework.py --dataset_path train_public.jsonl --output_dir outputs/qwen-chat-v1 --batch_size 1 --grad_accum_steps 16 --epochs 2 --max_seq_len 1024
```

## 3) 交互式对话推理

```bash
python infer_chat.py --adapter_dir outputs/qwen-chat-v1
```

脚本内命令：

- `/reset`：清空上下文
- `/exit`：退出程序

## 4) 批量评测

```bash
python batch_infer_eval.py --adapter_dir outputs/qwen-chat-v1 --eval_file eval_prompts.jsonl --out_file eval_results.jsonl --limit 10
```

## 5) 自动评分

```bash
python score_eval.py --in_file eval_results.jsonl --out_file eval_scores.jsonl --summary_file eval_summary.json
```

输出文件：

- `eval_scores.jsonl`：逐条样本评分
- `eval_summary.json`：总体统计（平均分、Top/Bottom 样本、速度指标）

## 6) 合并 LoRA（可选）

导出可独立部署的合并模型：

```bash
python merge_lora.py --adapter_dir outputs/qwen-chat-v1 --merged_out outputs/qwen-chat-v1-merged
```
