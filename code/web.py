#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
事件分析 Web UI（Gradio，对话式问答页面）

- 模型：Qwen3-0.6B-KTO（事件分析偏好对齐模型）
- 页面：对话式问答页面，流式输出
- 可调推理参数：温度、最大生成长度、top_p、思考模式、GPU、4-bit 量化
- 可选分析维度：事件摘要 / 原因分析 / 影响评估 / 事件推演 / 关联因素 / 应对建议
- 复用 infer.py 中的提示词模板、字段定义与 JSON 解析工具函数

运行（需 GPU，请在沙箱外执行）：
    python web.py --device 0
    python web.py --device 0 --no-quant --share
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable

import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from infer import (
    ANALYSIS_SCHEMA,
    SYSTEM_PROMPT,
    _build_user_prompt,
    _extract_json,
    _flatten_analysis,
)

MODEL_PATH = "/home/ps/wuzhichao/event_analysis/save_models/Qwen3-0.6B-KTO"

FIELD_LABELS: dict[str, str] = {
    "summary": "事件摘要",
    "cause_analysis": "原因分析",
    "impact_assessment": "影响评估",
    "event_deduction": "事件推演",
    "related_factors": "关联因素",
    "suggestions": "应对建议",
}

EXAMPLE_EVENTS: list[str] = [
    "6月7日报道，IBM将裁员超过1000人。IBM周四确认，将裁减一千多人。据知情人士称，"
    "此次裁员将影响到约1700名员工，约占IBM全球逾34万员工中的0.5%。IBM股价今年累计上涨16%，"
    "但该公司4月发布的财报显示，一季度营收下降5%，低于市场预期。",
    "有多名魅族员工表示，从6月份开始，魅族开始了新一轮裁员，重点裁员区域是营销和线下。"
    "裁员占比超过30%，剩余员工将不过千余人，魅族的知名工程师，爱讲真话的洪汉生已经从钉钉里退出了，"
    "外界传言说他去了OPPO。",
    "7月14日上午，记者从曲靖市麒麟区人民政府新闻办公室获悉，7月13日晚上20点20分左右，"
    "位于麒麟区交通路美佳华商场一楼屈臣氏商店后部约100平方米楼层发生坍塌事故。"
    "14日凌晨，救援人员成功抢救出3人送往医院救治，伤者无致命伤，病情稳定。",
]

WELCOME = (
    "你好！我是**事件智能分析助手**。\n\n"
    "请输入或粘贴一段**事件文本**，我将从以下维度进行结构化研判：\n"
    "- 事件摘要　·　原因分析　·　影响评估\n"
    "- 事件推演　·　关联因素　·　应对建议\n\n"
    "左侧控制面板可调整推理参数与勾选需要的分析维度。"
)

MODEL: Any = None
TOKENIZER: Any = None


def load_model(
    model_path: str | Path,
    device_id: int = 0,
    load_in_4bit: bool = True,
):
    """加载模型到单张 GPU（不使用 torch.compile，保证流式生成的稳定性）。"""
    path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }

    if torch.cuda.is_available():
        load_kwargs["attn_implementation"] = "sdpa"
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )
        load_kwargs["device_map"] = {"": device_id}
    else:
        if load_in_4bit:
            print("警告：4-bit 量化需要 CUDA，已在 CPU 模式下禁用")

    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    model.eval()
    return model, tokenizer


def model_info_text(device_id: int, load_in_4bit: bool) -> str:
    if not torch.cuda.is_available():
        return " 未检测到 CUDA，使用 CPU 推理（速度较慢）"
    return f" 模型已加载 | GPU cuda:{device_id} | 4-bit: {'启用' if load_in_4bit else '禁用'}"


# user prompt 直接复用 infer._build_user_prompt：始终请求全部 6 个字段，
# 与 KTO 模型训练分布一致，避免子集 prompt 导致输出退化（字段缺失/全部"信息不足"）。
# 前端勾选的维度仅影响最终展示（见 format_analysis）。
def format_analysis(analysis: dict[str, Any], selected_fields: list[str]) -> str:
    """将解析后的分析结果渲染为纯 HTML（不依赖 Markdown），避免 prose/strong 的样式漂移。

    每个字段包成独立 div，标题和内容均使用统一字重、字号（与"应对建议"小字一致），
    不再用 **{label}** 粗体，所有字段视觉风格完全一致。
    """
    blocks: list[str] = []
    for key in selected_fields:
        label = FIELD_LABELS.get(key, key)
        value = analysis.get(key, "信息不足")
        if isinstance(value, dict):
            value = "；".join(f"{k}：{v}" for k, v in value.items())
        elif isinstance(value, list):
            value = "；".join(str(item) for item in value)
        blocks.append(
            '<div class="ea-field">'
            f'<div class="ea-field__title">{label}</div>'
            f'<div class="ea-field__value">{_html_escape(str(value))}</div>'
            "</div>"
        )
    return "\n".join(blocks)


def _html_escape(s: str) -> str:
    import html as _html

    return _html.escape(s, quote=False)


def _json_is_complete(text: str) -> bool:
    """检查 JSON 字符串是否完整闭合，用于流式生成时的提前截断。

    从最后一个 '{' 开始检查，忽略前面可能存在的思考内容。
    """
    last_brace = text.rfind("{")
    if last_brace == -1:
        return False
    candidate = text[last_brace:]
    if not candidate.strip().endswith("}"):
        return False
    depth = 0
    in_str = False
    escape = False
    for ch in candidate:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return depth == 0 and not in_str


def _repair_json(raw: str) -> str:
    """对常见的不完整 JSON 做容错修复：补全未闭合的字符串与花括号。

    适配该 0.6B 模型两类高频缺陷：漏掉最后一个 `}`、或在某个字段值中途提前结束。
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return text
    text = text[start:]

    depth = 0
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    suffix = ""
    if in_str:  # 字符串未闭合
        suffix += '"'
    if depth > 0:  # 对象未闭合
        suffix += "}" * depth
    return text + suffix


def _extract_json_last(raw: str) -> dict[str, Any]:
    """优先从最后一个 '{' 开始提取 JSON，避免被思考内容中的花括号干扰。"""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1).strip()
    matches = list(re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text))
    if matches:
        for match in reversed(matches):
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return _extract_json(raw)


def _try_parse(raw: str) -> tuple[dict | None, list[str] | str]:
    """先直接解析，失败则容错修复后重试。

    返回 (analysis|None, missing)：missing 为 [] 表示字段完整；
    为列表表示解析成功但缺字段；为 "parse_fail" 表示完全无法解析。
    """
    for text in (raw, _repair_json(raw)):
        try:
            analysis = _extract_json_last(text)
        except (json.JSONDecodeError, ValueError):
            continue
        missing = [k for k in ANALYSIS_SCHEMA if k not in analysis]
        return _flatten_analysis(analysis), missing
    return None, "parse_fail"


def analyze_stream(
    text: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    enable_thinking: bool,
) -> Iterable[str]:
    """流式生成原始文本，逐段产出（供前端逐字渲染）。

    始终请求全部 6 个分析字段（与模型训练分布一致）；前端按勾选维度仅展示对应内容。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text)},
    ]
    prompt = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    pad_token_id = TOKENIZER.pad_token_id or TOKENIZER.eos_token_id

    streamer = TextIteratorStreamer(
        TOKENIZER, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0,
        "pad_token_id": pad_token_id,
        "use_cache": True,
        "repetition_penalty": 1.05 if temperature > 0 else 1.0,  # 采样时轻微惩罚重复
        "no_repeat_ngram_size": 256 if temperature > 0 else 0, # 防止短 n-gram 循环（JSON key+value）
        "streamer": streamer,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    def _generate() -> None:
        with torch.no_grad():
            MODEL.generate(**inputs, **gen_kwargs)

    thread = threading.Thread(target=_generate)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        yield partial
        # 动态截断：如果已经生成了完整的 JSON（首尾括号匹配，且字符串已闭合），提前结束流式输出
        if _json_is_complete(partial):
            break
    thread.join()


def respond(
    message: str,
    history: list[dict],
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    enable_thinking: bool,
    fields: list[str],
):
    """Chatbot 回调：每条消息作为独立事件进行分析。

    始终请求全部 6 个字段（与训练分布一致），再按勾选维度渲染展示。
    针对 0.6B 模型偶发的“漏闭合花括号 / 提前结束”问题，采用：
      1) 容错修复（补全未闭合的字符串与花括号）后解析；
      2) 解析失败或不完整时按原参数重新采样重试（贪心模式不重试，因结果确定）；
      3) 多次仍不完整则展示已生成的部分结果，而非直接报错。
    """
    message = (message or "").strip()
    if not message:
        yield history, ""
        return

    selected = list(fields) if fields else list(ANALYSIS_SCHEMA.keys())
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, ""

    cur_max = int(max_new_tokens)
    # 贪心（temperature=0）结果确定，重试无意义；采样模式最多重试 2 次
    max_attempts = 3 if temperature > 0 else 1

    best_analysis: dict | None = None
    best_missing: list[str] | str = "parse_fail"
    partial = ""

    for attempt in range(max_attempts):
        partial = ""
        try:
            for partial in analyze_stream(
                message, temperature, cur_max, top_p, enable_thinking
            ):
                history[-1]["content"] = partial
                yield history, ""
        except Exception as e:  # noqa: BLE001
            history[-1]["content"] = f" 生成失败：{e}"
            yield history, ""
            return

        analysis, missing = _try_parse(partial)
        if analysis is not None and not missing:  # 字段完整
            history[-1]["content"] = format_analysis(analysis, selected)
            yield history, ""
            return
        if analysis is not None:  # 解析成功但缺字段，记为当前最佳
            best_analysis, best_missing = analysis, missing
        if attempt < max_attempts - 1:
            history[-1]["content"] = "⏳ 输出不完整，正在重新采样重试…"
            yield history, ""

    if best_analysis is not None:
        note = f" 模型输出不完整（缺少 {best_missing}），以下为已生成部分：\n\n"
        history[-1]["content"] = note + format_analysis(best_analysis, selected)
        yield history, ""
    else:
        history[-1]["content"] = (
            f" 结果解析失败。\n\n**原始输出：**\n```\n{partial}\n```"
        )
        yield history, ""


def reload_model(device_id: int | str, load_in_4bit: bool):
    """重新加载模型（切换 GPU 或量化时使用）。"""
    global MODEL, TOKENIZER
    try:
        device_id = int(device_id)
        MODEL, TOKENIZER = load_model(MODEL_PATH, device_id, load_in_4bit)
        return model_info_text(device_id, load_in_4bit)
    except Exception as e:
        return f" 模型加载失败：{e}"


def toggle_all_fields(check: bool) -> list[str]:
    return list(ANALYSIS_SCHEMA.keys()) if check else []


CSS = """
#top_header { display:flex; align-items:center; gap:10px; padding:6px 4px 10px; }
#top_header h1 { margin:0; font-size:22px; font-weight:700; }#top_header .logo { font-size:26px; }
#top_header .sub { color:#8a8f99; font-size:13px; margin-left:auto; }
.sidebar-card { background:#f7f8fa !important; border:1px solid #eceef1 !important; border-radius:14px !important; }
footer { visibility:hidden; }
/* 事件分析字段样式：统一字号、字重，所有字段（含标题、内容）完全一致
   !important 抵消 gradio prose/.prose-invert 对 p/h4/strong 等标签的覆盖（
   旧版本最后一个字段因没有后续 <hr> 而被 prose 遗漏，导致字号异常偏小）*/
.prose .ea-field, .prose-invert .ea-field, .chat-container .ea-field {
  font-size: 14px !important;
  line-height: 1.6 !important;
  color: #1a1a1a !important;
  font-weight: 400 !important;
  margin: 4px 0 12px 0 !important;
}
.prose .ea-field__title, .prose-invert .ea-field__title,
.prose .ea-field__value, .prose-invert .ea-field__value,
.chat-container .ea-field__title, .chat-container .ea-field__value {
  font-size: 14px !important;
  line-height: 1.6 !important;
  color: #1a1a1a !important;
  font-weight: 400 !important;
  margin: 0 !important;
  padding: 0 !important;
  display: block !important;
  background: none !important;
  border: none !important;
}
.prose .ea-field__title, .prose-invert .ea-field__title {
  color: #3c424d !important;
  margin-bottom: 2px !important;
}
.prose hr, .prose-invert hr { display: none !important; }
.prose strong, .prose-invert strong { font-weight: 400 !important; color: inherit !important; }
.prose p, .prose-invert p { margin: 2px 0 !important; font-size: 14px !important; }
"""

FIELD_CHOICES = [(desc, key) for key, desc in ANALYSIS_SCHEMA.items()]


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
        radius_size="lg",
        font=[gr.themes.GoogleFont("Noto Sans SC"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(theme=theme, css=CSS, title="事件智能分析助手") as demo:
        gr.Group(
            gr.HTML(
                '<div id="top_header">'
                '<h1>事件智能分析助手</h1>'
                "</div>"
            )
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300, elem_classes="sidebar-card"):
                status_md = gr.Markdown("正在加载模型…")

                with gr.Accordion("推理参数", open=True):
                    temperature = gr.Slider(
                        0.0, 1.0, value=0.3, step=0.05,
                        label="温度 temperature", info=">采样多样性",
                    )
                    max_new_tokens = gr.Slider(
                        512, 4096, value=2048, step=128,
                        label="最大生成长度",
                    )
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
                    enable_thinking = gr.Checkbox(
                        value=True, label="启用思考模式 enable_thinking",
                    )

                with gr.Accordion("模型部署", open=False):
                    device = gr.Dropdown(
                        choices=[str(i) for i in range(8)],
                        value="0", label="GPU 编号",
                    )
                    load_in_4bit = gr.Checkbox(value=True, label="4-bit 量化")
                    reload_btn = gr.Button("重新加载模型", variant="secondary")
                    reload_btn.click(
                        reload_model,
                        inputs=[device, load_in_4bit],
                        outputs=status_md,
                    )

                with gr.Accordion("分析维度（选择需要推理的内容）", open=True):
                    select_all = gr.Checkbox(value=True, label="全选 / 取消全选")
                    fields = gr.CheckboxGroup(
                        choices=FIELD_CHOICES,
                        value=list(ANALYSIS_SCHEMA.keys()),
                        label="输出维度",
                        info="仅勾选的维度会出现在结果中",
                    )
                    select_all.change(
                        toggle_all_fields, inputs=select_all, outputs=fields
                    )

            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[{"role": "assistant", "content": WELCOME}],
                    type="messages",
                    height="70vh",
                    show_label=False,
                    avatar_images=(None, None),
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入或粘贴事件文本，回车发送…",
                        scale=8,
                        show_label=False,
                        lines=3,
                        max_lines=8,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", scale=1)

                gr.Examples(
                    examples=[[e] for e in EXAMPLE_EVENTS],
                    inputs=msg_input,
                    label="示例事件（点击填充）",
                )

        inputs = [
            msg_input, chatbot,
            temperature, max_new_tokens, top_p, enable_thinking, fields,
        ]
        outputs = [chatbot, msg_input]
        msg_input.submit(respond, inputs, outputs)
        send_btn.click(respond, inputs, outputs)
        clear_btn.click(lambda: ([{"role": "assistant", "content": WELCOME}], ""), None, [chatbot, msg_input])

    return demo


def main() -> None:
    global MODEL, TOKENIZER

    parser = argparse.ArgumentParser(description="事件分析 Web UI")
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH), help="模型目录")
    parser.add_argument("--device", type=int, default=0, help="GPU 编号")
    parser.add_argument("--no-quant", action="store_true", help="不使用 4-bit 量化")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    model_path = str(args.model)
    load_in_4bit = not args.no_quant

    if not torch.cuda.is_available():
        print(" 未检测到 CUDA，使用 CPU 推理（速度较慢）")
    else:
        print(f"使用 GPU: cuda:{args.device}")

    print(f"4-bit 量化: {'启用' if load_in_4bit else '禁用'}")
    print("正在加载模型…")
    MODEL, TOKENIZER = load_model(model_path, args.device, load_in_4bit)
    print(model_info_text(args.device, load_in_4bit))

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=False,
    )


if __name__ == "__main__":
    main()
