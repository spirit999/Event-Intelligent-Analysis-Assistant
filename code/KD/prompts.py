#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提示词模块：负责系统/用户提示词拼装，以及 Qwen3 chat 模板封装。

对外暴露两个 chat 模板函数；_build_user_prompt 为模块内部使用。
tokenizer 由调用方传入，模块本身不持有模型状态：
- build_chat_input:  事件文本 -> 仅含生成前缀的 prompt（教师生成时使用）
- build_chat_full:   事件文本 + 分析结果 -> (prompt, 完整答案文本)（训练拼样本时使用）
"""

import json

from config import ANALYSIS_SCHEMA, SYSTEM_PROMPT


def _build_user_prompt(text: str) -> str:
    """根据事件文本构造用户提示词，字段说明由 ANALYSIS_SCHEMA 自动生成。"""
    fields_desc = "\n".join(
        f'- "{key}": {desc}' for key, desc in ANALYSIS_SCHEMA.items()
    )
    return f"""请对以下事件文本进行多维度分析。

【事件文本】
{text}

【输出要求】 
请输出 JSON，包含且仅包含以下字段（键名必须与下方完全一致）：
{fields_desc}
所有字段均为字符串类型。"""


def build_chat_input(tokenizer, text: str):
    """构造仅含 system+user 与生成前缀的对话文本（add_generation_prompt=True）。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_chat_full(tokenizer, text: str, analysis_dict: dict):
    """构造训练样本：返回 (prompt, prompt+答案JSON+EOS)。

    答案仅保留 ANALYSIS_SCHEMA 中声明的字段，序列化为紧凑 JSON 字符串。
    """
    messages_user = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages_user,
        tokenize=False,
        add_generation_prompt=True,
    )
    assistant_payload = {k: v for k, v in analysis_dict.items() if k in ANALYSIS_SCHEMA}
    assistant_text = json.dumps(assistant_payload, ensure_ascii=False)
    full_text = prompt + assistant_text + tokenizer.eos_token
    return prompt, full_text
