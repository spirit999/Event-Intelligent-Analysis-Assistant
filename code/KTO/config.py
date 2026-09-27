#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局配置：模型 / 数据 / 输出路径，以及事件分析的 Schema 与 System Prompt
"""

KD_LORA_PATH = "/home/ps/wuzhichao/event_analysis/save_models/Qwen3-0.6B-KD-LoRA"
DATA_PATH = "/home/ps/wuzhichao/event_analysis/data/kto_data.jsonl"
OUTPUT_DIR = "/home/ps/wuzhichao/event_analysis/save_models"

ANALYSIS_SCHEMA: dict[str, str] = {
    "summary": "事件摘要（严格控制文本长度，仅保留核心要素）",
    "cause_analysis": "事件原因分析（直接原因与深层原因）",
    "impact_assessment": "影响范围评估（涉及方、规模、短期与长期影响）",
    "event_deduction": "事件推演（可能的发展趋势、连锁反应、后续关注点）",
    "related_factors": "关联因素（行业背景、政策、市场、历史类似事件等）",
    "suggestions": "应对建议（面向相关方：企业、公众等的可操作建议）",
}

SYSTEM_PROMPT = """你是一名专业的事件分析顾问，擅长从新闻类文本中提取事实并进行结构化研判。
请严格基于给定的事件文本进行分析，不要编造文中未出现的事实；信息不足时在对应字段中明确说明"信息不足"。

【输出格式要求】
你必须且只输出一个合法的 JSON 对象，不要输出任何其他文字，包括：
- 不要输出"以下是分析结果"、"我来分析一下"等前缀文字
- 不要输出 markdown 代码块标记（如 ```json 或 ```）
- 不要输出任何解释性文字
- 输出必须是完整的、可直接解析的 JSON 字符串

JSON 格式示例：
{"summary": "...", "cause_analysis": "...", "impact_assessment": "...", "event_deduction": "...", "related_factors": "...", "suggestions": "..."}"""
