#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练入口：加载 tokenizer / 数据 / 基座模型 -> 叠加 LoRA -> DPOTrainer 训练并保存 adapter

运行方式（在 code/ 目录下）：
    python -m DPO.train [--data_path ... --model_path ... 等，见 arguments.py]
"""

import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .arguments import parse_args
from .data import load_data


def build_lora_config(args):
    """构建 LoRA 配置：作用于注意力与 MLP 的全部投影层"""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def build_dpo_config(args):
    """构建 DPO 训练超参配置"""
    return DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        beta=args.dpo_beta,
        max_length=args.max_seq_len,
    )


def main():
    args = parse_args()

    if not args.data_path or not os.path.exists(args.data_path):
        raise ValueError(f"请检查数据集路径 --data_path: {args.data_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, eval_ds = load_data(args.data_path, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    model = get_peft_model(model, build_lora_config(args))
    model.print_trainable_parameters()

    trainer = DPOTrainer(
        model=model,
        args=build_dpo_config(args),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    adapter_dir = os.path.join(args.output_dir, "dpo_lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"DPO LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
