#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KTOTrainer：基于 transformers.Trainer 手写 KTO loss。
Reference model 通过 peft 的 disable_adapter() 在同一 GPU 上获得，
避免跨设备通信开销。
"""

import torch
import torch.nn.functional as F
from transformers import Trainer


class KTOTrainer(Trainer):
    def __init__(self, *args, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        print("KTOTrainer 初始化完成 (使用 disable_adapter 作为 reference model)")

    def _compute_seq_log_probs(self, model, inputs, labels):
        """计算序列的平均 log 概率"""
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        sample_log_probs = -self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        sample_log_probs = sample_log_probs.view(shift_labels.size(0), shift_labels.size(1))

        mask = (shift_labels != -100).float()
        valid_log_probs = sample_log_probs * mask
        sequence_log_probs = valid_log_probs.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return sequence_log_probs, logits, valid_log_probs, mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        kto_labels = inputs.pop("label", None)

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        labels = labels.to(device)
        if kto_labels is not None:
            kto_labels = kto_labels.to(device)

        sequence_log_probs, logits, valid_log_probs, mask = self._compute_seq_log_probs(model, inputs, labels)

        if not self.model.training:
            loss = (-valid_log_probs).sum() / (mask.sum() + 1e-8)
            if return_outputs:
                return loss, {"logits": logits}
            return loss

        # Reference model: 用 disable_adapter 获取 base model 输出 (同一 GPU, 无跨设备传输)
        # 处理 DataParallel 包装: 通过 model.module 访问底层 PEFT 模型
        peft_model = model.module if hasattr(model, 'module') else model
        with torch.no_grad(), peft_model.disable_adapter():
            ref_sequence_log_probs, _, _, _ = self._compute_seq_log_probs(model, inputs, labels)

        logits_kto = sequence_log_probs - self.beta * ref_sequence_log_probs

        desirable_mask = (kto_labels == 1).float()
        undesirable_mask = (kto_labels == 0).float()

        desirable_loss = -desirable_mask * F.logsigmoid(logits_kto)
        undesirable_loss = -undesirable_mask * F.logsigmoid(-logits_kto)

        loss = (desirable_loss + undesirable_loss).mean()

        if return_outputs:
            return loss, {"logits": logits}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """评估时只计算 loss，不收集 logits (避免 OOM)"""
        return super().prediction_step(
            model, inputs, prediction_loss_only=True, ignore_keys=ignore_keys
        )
