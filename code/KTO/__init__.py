#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KTO 偏好优化训练包（Qwen3-0.6B-KD-LoRA 基座，不依赖 trl，手写 KTO loss）

模块划分：
- config:     路径常量、ANALYSIS_SCHEMA、SYSTEM_PROMPT
- prompts:    事件文本 -> 对话格式输入的构建
- data:       KTO 样本预处理、动态 padding collator、训练/验证集切分
- arguments:  命令行参数解析
- trainer:    KTOTrainer（手写 KTO loss，disable_adapter 作 reference model）
- train:      训练入口（模型加载、LoRA、TrainingArguments、训练与保存）

运行方式（在 code/ 目录下）：
    python -m KTO.train [参数]
"""
