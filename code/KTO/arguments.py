#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行参数解析（默认值均取自 config.py，与原脚本保持一致）
"""

import argparse

from .config import DATA_PATH, KD_LORA_PATH, OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA KTO for Qwen3-0.6B-KD-LoRA")
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--kd_lora", type=str, default=KD_LORA_PATH)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--kto_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
