#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据模块：读取蒸馏 jsonl，切分训练/验证集并完成 tokenize。

标签策略：prompt 部分 label 全部置 -100（不参与损失），仅答案 + EOS 部分计算损失。
"""

from datasets import load_dataset

from prompts import build_chat_full


def preprocess_function(example, tokenizer, max_seq_len):
    """单条样本 -> {input_ids, labels, attention_mask}。"""
    prompt, full_text = build_chat_full(
        tokenizer, example["text"], example["analysis"]
    )

    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )["input_ids"]

    # prompt 段不计算损失；长度对齐到截断后的 full_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    labels = labels[: len(full_ids)]
    input_ids = full_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def load_data(data_path, tokenizer, max_seq_len):
    """加载蒸馏数据，按 4% 固定种子切出验证集，返回 (train_ds, eval_ds)。"""
    raw = load_dataset("json", data_files=data_path, split="train")
    print(f"原始样本数量: {len(raw)}")

    split_dataset = raw.train_test_split(test_size=0.04, seed=42)
    train_raw = split_dataset["train"]
    eval_raw = split_dataset["test"]

    print(f"训练集数量: {len(train_raw)}, 验证集数量: {len(eval_raw)}")

    def process_split(dataset):
        processed = dataset.map(
            lambda x: preprocess_function(x, tokenizer, max_seq_len),
            remove_columns=dataset.column_names,
            num_proc=4,
            desc="Tokenizing",
        )
        processed = processed.filter(
            lambda x: len(x["input_ids"]) > 10
            and sum(1 for v in x["labels"] if v != -100) > 0
        )
        return processed

    train_ds = process_split(train_raw)
    eval_ds = process_split(eval_raw)

    print(f"训练集有效数量: {len(train_ds)}, 验证集有效数量: {len(eval_ds)}")
    return train_ds, eval_ds
