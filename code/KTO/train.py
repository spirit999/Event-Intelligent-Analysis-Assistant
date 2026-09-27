#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练入口：加载 tokenizer / 数据 / 基座模型 -> 叠加 LoRA -> KTOTrainer 训练并保存 adapter

运行方式（在 code/ 目录下）：
    python -m KTO.train [--data_path ... --kd_lora ... 等，见 arguments.py]
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from .arguments import parse_args
from .data import load_data, make_kto_collator
from .trainer import KTOTrainer


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


def build_training_args(args):
    """构建训练超参配置"""
    return TrainingArguments(
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )


def load_model(kd_lora_path, lora_config):
    print(f"加载已合并的完整模型: {kd_lora_path}")
    print("正在 CPU 上加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        kd_lora_path,
        dtype=torch.bfloat16,
    )
    print("CPU 权重加载完成")
    model.config.use_cache = False
    print(f"正在将模型迁移到 GPU (cuda:0)...")
    model = model.to("cuda:0")
    print("GPU 迁移完成")
    print("正在配置 LoRA...")
    model = get_peft_model(model, lora_config)
    print("LoRA 配置完成")
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()

    if not args.data_path:
        raise ValueError("请指定数据集路径 --data_path")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.kd_lora)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, eval_ds = load_data(args.data_path, tokenizer, args.max_seq_len)

    model = load_model(args.kd_lora, build_lora_config(args))

    trainer = KTOTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_kto_collator(tokenizer.pad_token_id or tokenizer.eos_token_id),
        beta=args.kto_beta,
    )

    trainer.train()
    output_path = os.path.join(args.output_dir, "Qwen3-0.6B-KTO-LoRA")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"KTO LoRA adapter saved to {output_path}")


if __name__ == "__main__":
    main()
