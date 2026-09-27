![](../web.png)

# 事件智能分析助手

基于 Qwen3-0.6B 的事件分析模型训练与推理代码。整体流程：KD-LoRA → 偏好对齐（DPO/KTO）→ 权重合并 → 推理 → Web 应用。


## 文件说明（按流水线顺序）

| 阶段 | 文件 | 说明 |
|------|------|------|
| 训练 | `KD/` | KD-LoRA：logits 级软蒸馏 + 硬标签联合训练（教师 8B → 学生 0.6B），入口 `python KD/train.py` |
| 训练 | `DPO/` | LoRA + DPO 偏好对齐（基座为 KD-LoRA 合并模型），入口 `python -m DPO.train` |
| 训练 | `KTO/` | LoRA + KTO 偏好对齐，自实现 KTO 损失（不依赖 trl），入口 `python -m KTO.train` |
| 合并 | `merge.py` | 将训练好的 LoRA 权重合并回基座并保存为完整权重 |
| 推理 | `infer.py` | 批量推理脚本：加载模型对 jsonl 事件逐条结构化分析，支持断点续跑 |
| 应用 | `web.py` | Gradio Web UI，可调推理参数与选择分析维度 |

## 模型与产物

- 最终模型：`../save_models/Qwen3-0.6B-KTO`（0.6B，事件分析偏好对齐，4-bit 显存约为 8B 的 1/13）
- 中间产物：`Qwen3-0.6B-LoRA`、`Qwen3-0.6B-KD-LoRA`、`Qwen3-0.6B-KTO-LoRA`
- 数据：`../data/`，推理结果：`../result/`，训练日志：`../log/`

## Web UI 运行

`web.py` 复用 `infer.py` 的提示词模板、字段定义与 JSON 解析工具。运行环境需含 torch / transformers / bitsandbytes / gradio，且需要 GPU。

```bash
cd code

# 默认：GPU 0、4-bit 量化、端口 7860
python web.py --device 0

# 不量化 / 指定端口 / 公网分享
python web.py --device 0 --no-quant --port 7870 --share

# 指定其他模型目录
python web.py --model ../save_models/Qwen3-0.6B-KTO --device 0
```

启动后浏览器访问 `http://<host>:7860`。

界面功能：

- 左侧控制面板：推理参数（temperature、max_new_tokens、top_p、enable_thinking）、模型部署（GPU 编号、4-bit 量化开关、重新加载）、分析维度勾选（摘要 / 原因分析 / 影响评估 / 事件推演 / 关联因素 / 应对建议）
- 右侧对话区：输入事件文本，流式返回结构化分析结果（Markdown 渲染），每条消息独立分析

若默认端口 7860 被占用，用 `--port` 指定空闲端口。

## 批量推理运行

```bash
python infer.py --data ../data/event_top10.jsonl --output ../result/event_analysis_kto.jsonl \
    --model ../save_models/Qwen3-0.6B-KTO --device 0
```

## 训练运行

训练脚本需在安装了 requirements.txt 依赖的环境中运行（需 GPU）。

```bash
cd code

python KD/train.py               
python -m DPO.train             
python -m KTO.train              
python merge.py                  
```

各训练脚本均支持 `--data_path` 等参数覆盖默认路径，详见各包内 `arguments.py` 或脚本参数定义。
