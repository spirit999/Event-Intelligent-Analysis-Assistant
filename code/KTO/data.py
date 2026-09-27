#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理：KTO 样本的 prompt/completion 构造、截断策略、动态 padding collator，
以及训练/验证集切分
"""

import json

import torch
from datasets import load_dataset

from .config import ANALYSIS_SCHEMA
from .prompts import GEN_HEADER, USER_SUFFIX, build_chat_input


def make_kto_collator(pad_id: int):
    """动态 padding collator：pad 到 batch 内最长。

    labels 在 padding 位置填 -100，修复旧版 pad token（=eos）被当成
    监督目标参与 loss 的泄漏问题。
    """

    def collate(features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(list(f["input_ids"]) + [pad_id] * pad)
            attention_mask.append(list(f["attention_mask"]) + [0] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label": torch.tensor([f["label"] for f in features], dtype=torch.long),
        }

    return collate


def preprocess_function(example, tokenizer, max_seq_len):
    prompt = build_chat_input(tokenizer, example["prompt"])

    def format_response(analysis_dict):
        if isinstance(analysis_dict, str):
            return analysis_dict
        if isinstance(analysis_dict, dict):
            if analysis_dict.get("parse_error"):
                raw = analysis_dict.get("raw_response", "")
                if isinstance(raw, str) and raw.startswith("{"):
                    try:
                        parsed = json.loads(raw)
                        payload = {k: v for k, v in parsed.items() if k in ANALYSIS_SCHEMA}
                        return json.dumps(payload, ensure_ascii=False)
                    except:
                        pass
                return json.dumps({k: "" for k in ANALYSIS_SCHEMA}, ensure_ascii=False)
            payload = {k: v for k, v in analysis_dict.items() if k in ANALYSIS_SCHEMA}
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps({k: "" for k in ANALYSIS_SCHEMA}, ensure_ascii=False)

    # 关键修复：completion 末尾显式拼接 EOS（Qwen3 为 <|im_end|>），
    # 让模型学到“JSON 写完 → 输出结束符”，否则推理时该停不停、循环复读。
    completion_text = format_response(example["completion"]) + tokenizer.eos_token

    # 拆出生成头，超长时只截 prompt 正文（事件文本），绝不截掉 completion 与结尾 EOS
    body = prompt
    header_ids: list[int] = []
    if prompt.endswith(GEN_HEADER):
        body = prompt[: -len(GEN_HEADER)]
        header_ids = tokenizer(GEN_HEADER, add_special_tokens=False)["input_ids"]
    suffix_ids: list[int] = []
    if header_ids:
        suffix_ids = tokenizer(USER_SUFFIX, add_special_tokens=False)["input_ids"]

    body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    fixed_len = len(header_ids) + len(suffix_ids) + len(completion_ids)
    body_budget = max_seq_len - fixed_len
    if body_budget < 0:
        # completion 本身超长（极罕见）：截 completion 头部，保留结尾 EOS
        keep = max_seq_len - len(header_ids) - len(suffix_ids)
        completion_ids = completion_ids[len(completion_ids) - keep :]
        body_ids = []
    elif len(body_ids) > body_budget:
        body_ids = body_ids[:body_budget]

    input_ids = body_ids + suffix_ids + header_ids + completion_ids
    prompt_len = len(input_ids) - len(completion_ids)

    # 只对 completion（含结尾 EOS）计算 loss，prompt 部分置 -100
    labels = [-100] * prompt_len + list(completion_ids)

    label = example.get("label", "").lower()
    label_id = 1 if label in ["desirable", "1", "true"] else 0

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "label": label_id,
    }


def load_data(data_path, tokenizer, max_seq_len):
    raw = load_dataset("json", data_files=data_path, split="train")
    print(f"原始样本数量: {len(raw)}")

    processed = raw.map(
        lambda x: preprocess_function(x, tokenizer, max_seq_len),
        remove_columns=raw.column_names,
        num_proc=4,
        desc="Processing KTO data",
    )

    split_dataset = processed.train_test_split(test_size=0.04, seed=42)
    train_ds = split_dataset["train"]
    eval_ds = split_dataset["test"]

    desirable_count = sum(1 for x in processed if x["label"] == 1)
    undesirable_count = len(processed) - desirable_count
    print(f"可取样本: {desirable_count}, 不可取样本: {undesirable_count}")
    print(f"训练集数量: {len(train_ds)}, 验证集数量: {len(eval_ds)}")

    return train_ds, eval_ds
