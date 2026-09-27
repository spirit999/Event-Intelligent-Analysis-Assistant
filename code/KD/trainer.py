#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KD-LoRA Trainer：硬标签 SFT 损失 + 教师/学生 logits 的 KL 软蒸馏损失联合优化。

总损失：(1 - kd_lambda) * sft_loss + kd_lambda * kd_loss
- sft_loss：答案 token 的交叉熵（prompt 段 label=-100 被忽略）
- kd_loss：仅在答案 token 上、经温度缩放后的 KL 散度（乘 T^2 补偿梯度尺度）
教师模型前向在 no_grad 下执行，不参与反传。
"""

import torch
import torch.nn.functional as F
from transformers import Trainer


class KDLoRATrainer(Trainer):
    def __init__(self, teacher_model=None, kd_lambda=0.3, kd_temperature=4, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.kd_lambda = kd_lambda
        self.kd_temperature = kd_temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        sft_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        kd_loss = torch.tensor(0.0, device=student_logits.device)

        if self.teacher_model is not None and self.kd_lambda > 0:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

            student_probs = F.log_softmax(shift_logits / self.kd_temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits / self.kd_temperature, dim=-1)

            mask = (shift_labels != -100).float()
            mask = mask.view(-1)

            kd_loss = F.kl_div(
                student_probs.view(-1, student_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
                reduction="none",
            ).sum(dim=-1) * (self.kd_temperature ** 2)

            if mask.sum() > 0:
                kd_loss = (kd_loss * mask).sum() / mask.sum()
            else:
                kd_loss = kd_loss.mean()

        total_loss = (1 - self.kd_lambda) * sft_loss + self.kd_lambda * kd_loss

        if return_outputs:
            return total_loss, student_outputs
        return total_loss
