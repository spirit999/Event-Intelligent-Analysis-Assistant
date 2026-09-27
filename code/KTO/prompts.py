#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt 构建：将事件文本组装为对话格式的模型输入
"""

from .config import ANALYSIS_SCHEMA, SYSTEM_PROMPT

GEN_HEADER = "<|im_start|>assistant\n"  # chat 模板的生成头
USER_SUFFIX = "<|im_end|>\n"            # user 回合收尾


def _build_user_prompt(text: str) -> str:
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


def build_chat_input(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
