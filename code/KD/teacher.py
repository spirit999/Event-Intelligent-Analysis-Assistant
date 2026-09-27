#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
教师模型模块：加载 8B 教师模型（支持 4-bit 量化）并批量生成结构化蒸馏监督数据。

职责：
- load_teacher_model:       按环境（GPU/CPU、flash-attn 是否可用）加载教师模型
- generate_teacher_outputs: 贪心解码批量生成，并从原始输出中提取合法 JSON
"""

import json
from typing import Any, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from prompts import build_chat_input


def load_teacher_model(model_path: str, tokenizer: AutoTokenizer, load_4bit: bool = True):
    """加载教师模型。

    - GPU 环境：默认 NF4 4-bit 量化 + double quant；flash-attn 可用时启用，否则用 sdpa
    - CPU 环境：float32 加载
    加载后冻结全部参数并切换到 eval 模式。
    """
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }

    if torch.cuda.is_available():
        import importlib.util
        if importlib.util.find_spec("flash_attn") is not None:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            load_kwargs["attn_implementation"] = "sdpa"

    if load_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        load_kwargs["quantization_config"] = bnb_config

    if torch.cuda.is_available():
        load_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def generate_teacher_outputs(teacher_model, tokenizer, texts: List[str], max_new_tokens: int = 1024, batch_size: int = 4):
    """对输入事件文本批量执行教师模型贪心生成，并解析输出中的 JSON。

    返回每条样本的 {"text", "analysis", "raw_output"}；
    解析失败（无 JSON / JSON 非法）时 analysis 为空 dict。
    """
    results = []
    total = len(texts)

    try:
        from tqdm import tqdm
        iterator = tqdm(range(0, total, batch_size), desc="生成教师输出", unit="batch")
    except ImportError:
        iterator = range(0, total, batch_size)

    for i in iterator:
        batch_texts = texts[i:i+batch_size]
        prompts = [build_chat_input(tokenizer, text) for text in batch_texts]

        tokenized_inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_new_tokens,
            return_tensors="pt",
        ).to(teacher_model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }

        with torch.no_grad():
            outputs = teacher_model.generate(**tokenized_inputs, **gen_kwargs)

        for j, text in enumerate(batch_texts):
            start_idx = tokenized_inputs["input_ids"].shape[1]
            new_tokens = outputs[j, start_idx:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            try:
                import re
                fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
                if fence:
                    raw_output = fence.group(1).strip()
                match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", raw_output)
                if match:
                    analysis = json.loads(match.group(0))
                else:
                    analysis = {}
            except (json.JSONDecodeError, ValueError):
                analysis = {}

            results.append({
                "text": text,
                "analysis": analysis,
                "raw_output": raw_output,
            })

    return results
