#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KD-LoRA 训练入口：知识蒸馏 + LoRA 微调（教师 Qwen3-8B -> 学生 Qwen3-0.6B）。

执行流程（与原单文件脚本完全一致）：
  1. 教师模型对原始事件文本批量生成结构化 JSON，过滤出字段完整的监督数据并落盘
  2. 重新加载教师模型（训练时在线提供 logits），加载学生模型并挂载 LoRA
  3. 联合 SFT + KL 蒸馏损失训练，保存 LoRA adapter

用法（在 code/ 目录或任意目录均可，参数与原脚本一致）：
  python KD/train.py
  python KD/train.py --limit 10 --no_teacher_4bit
  python KD/train.py --skip_distillation   # 复用已生成的 distillation_data.jsonl
"""

import os
import json
import argparse

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    TEACHER_MODEL_PATH,
    STUDENT_MODEL_PATH,
    DATA_PATH,
    OUTPUT_DIR,
    ANALYSIS_SCHEMA,
)
from teacher import load_teacher_model, generate_teacher_outputs
from data import load_data
from trainer import KDLoRATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="KD-LoRA Training")
    parser.add_argument("--teacher_model", type=str, default=TEACHER_MODEL_PATH)
    parser.add_argument("--student_model", type=str, default=STUDENT_MODEL_PATH)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--kd_lambda", type=float, default=0.3, help="蒸馏损失权重")
    parser.add_argument("--kd_temperature", type=int, default=4, help="蒸馏温度系数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_4bit", action="store_true", default=True, help="教师模型使用4-bit量化")
    parser.add_argument("--no_teacher_4bit", action="store_true", help="教师模型不使用4-bit量化")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前N条数据（调试用）")
    parser.add_argument("--skip_distillation", action="store_true", help="跳过教师模型生成，使用已有的distillation_data.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    teacher_4bit = not args.no_teacher_4bit

    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    distillation_path = os.path.join(args.output_dir, "distillation_data.jsonl")

    if not args.skip_distillation or not os.path.exists(distillation_path):
        print("=" * 60)
        print("步骤1: 使用教师模型生成监督数据")
        print("=" * 60)

        teacher_model = load_teacher_model(args.teacher_model, tokenizer, load_4bit=teacher_4bit)

        raw_data = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    text = item.get("text", "").strip()
                    if text:
                        raw_data.append(text)
                        if args.limit and len(raw_data) >= args.limit:
                            break

        print(f"待处理数据: {len(raw_data)} 条")

        teacher_outputs = generate_teacher_outputs(teacher_model, tokenizer, raw_data)

        distillation_data = []
        for output in teacher_outputs:
            if output["analysis"] and all(k in output["analysis"] for k in ANALYSIS_SCHEMA):
                distillation_data.append({
                    "text": output["text"],
                    "analysis": output["analysis"],
                })

        print(f"有效监督数据: {len(distillation_data)} 条")

        with open(distillation_path, "w", encoding="utf-8") as f:
            for item in distillation_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"监督数据已保存到: {distillation_path}")

        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"跳过教师模型生成，使用已有的监督数据: {distillation_path}")

    print("\n" + "=" * 60)
    print("步骤2: 加载教师模型和学生模型")
    print("=" * 60)

    teacher_model_for_training = load_teacher_model(args.teacher_model, tokenizer, load_4bit=teacher_4bit)

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.config.use_cache = False
    if hasattr(student_model.config, "tie_word_embeddings") and student_model.config.tie_word_embeddings:
        student_model.enable_input_require_grads()

    lora_config = LoraConfig(
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
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    print("\n" + "=" * 60)
    print("步骤3: 加载数据并开始训练")
    print("=" * 60)

    train_ds, eval_ds = load_data(distillation_path, tokenizer, args.max_seq_len)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
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
        save_safetensors=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = KDLoRATrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        teacher_model=teacher_model_for_training,
        kd_lambda=args.kd_lambda,
        kd_temperature=args.kd_temperature,
    )

    trainer.train()

    model_save_path = os.path.join(args.output_dir, "kd_lora_adapter")
    student_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nKD-LoRA adapter saved to {model_save_path}")


if __name__ == "__main__":
    main()
