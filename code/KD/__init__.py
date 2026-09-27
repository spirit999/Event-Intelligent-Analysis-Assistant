#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KD-LoRA 包：教师 Qwen3-8B -> 学生 Qwen3-0.6B 的 logits 级知识蒸馏 + LoRA 微调。

模块职责：
- config:   路径常量、分析字段 Schema、系统提示词
- prompts:  用户提示词与 Qwen3 chat 模板拼接
- teacher:  教师模型加载（4-bit 量化）与蒸馏监督数据生成
- data:     蒸馏 jsonl 的 tokenize、训练/验证集切分
- trainer:  KDLoRATrainer（SFT 交叉熵 + KL 软蒸馏联合损失）
- train:    训练入口（参数解析 + 全流程组装），运行: python KD/train.py
"""
