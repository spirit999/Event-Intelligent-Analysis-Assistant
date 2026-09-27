#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理：DPO 样本的 prompt/chosen/rejected 构造，以及训练/验证集切分
"""

import json

from datasets import load_dataset

from .config import ANALYSIS_SCHEMA
from .prompts import build_chat_input


def preprocess_function(example, tokenizer):
    prompt = build_chat_input(tokenizer, example["prompt"])

    def format_response(analysis_dict):
        if isinstance(analysis_dict, str):
            return analysis_dict
        payload = {k: v for k, v in analysis_dict.items() if k in ANALYSIS_SCHEMA}
        return json.dumps(payload, ensure_ascii=False)

    chosen_text = format_response(example["chosen"])
    rejected_text = format_response(example["rejected"])

    return {
        "prompt": prompt,
        "chosen": chosen_text + tokenizer.eos_token,
        "rejected": rejected_text + tokenizer.eos_token,
    }


def load_data(data_path, tokenizer):
    raw = load_dataset("json", data_files=data_path, split="train")
    print(f"原始样本数量: {len(raw)}")

    processed = raw.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=raw.column_names,
        num_proc=4,
        desc="Processing DPO data",
    )

    split_dataset = processed.train_test_split(test_size=0.04, seed=42)
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]

    print(f"训练集数量: {len(train_ds)}, 验证集数量: {len(eval_ds)}")
    return train_ds, eval_ds
