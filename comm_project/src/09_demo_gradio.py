# -*- coding: utf-8 -*-
"""
comm_project/src/09_demo_gradio.py

Gradio Demo for Communication Standard Q&A (Day9).

Features:
- Model selection: base / sft / dpo
- Input: question + evidence (supports multiple chunks)
- Output: raw JSON, parsed fields, citation highlighting
- Logging: data/logs/demo_calls.jsonl
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import torch

# Local imports
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from utils.prompts import SYSTEM_PROMPT, format_user_prompt, validate_json_output  # noqa: E402
from utils.retrieval import format_chunks_as_evidence_text, retrieve_topk  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # comm_project/
DEFAULT_BASE_MODEL = "qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "outputs" / "sft_adapter"
DEFAULT_DPO_ADAPTER = PROJECT_ROOT / "outputs" / "dpo_adapter"
LOG_DIR = PROJECT_ROOT / "data" / "logs"
LOG_FILE = LOG_DIR / "demo_calls.jsonl"

# Global model bundle cache
_model_cache = {}


def extract_json_robust(text: str) -> str:
    """Extract first balanced JSON object from text."""
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return ""


def load_model_bundle(model_name: str, adapter_path: Optional[Path], use_4bit: bool = False):
    """Load model and tokenizer, with caching."""
    cache_key = f"{model_name}_{adapter_path}_{use_4bit}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if use_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if adapter_path is not None and adapter_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    bundle = {"model": model, "tokenizer": tok}
    _model_cache[cache_key] = bundle
    return bundle


def build_inputs(bundle, system_text: str, user_text: str):
    """Build model inputs from system and user text."""
    tok = bundle["tokenizer"]
    messages = [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text.strip()},
    ]

    if hasattr(tok, "apply_chat_template"):
        try:
            input_ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(input_ids, torch.Tensor):
                attn = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attn}
        except Exception:
            pass

    # Fallback
    full_prompt = system_text.strip() + "\n" + user_text.strip() + "\n"
    return tok(full_prompt, return_tensors="pt")


def generate_json(bundle, system_text: str, user_text: str, max_new_tokens: int = 512):
    """Generate JSON output from model."""
    tok = bundle["tokenizer"]
    model = bundle["model"]

    inputs = build_inputs(bundle, system_text, user_text)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][input_len:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    return extract_json_robust(gen_text)


def highlight_quotes_in_evidence(evidence_text: str, quotes: list) -> str:
    """Highlight citation quotes in evidence text using HTML."""
    if not evidence_text or not quotes:
        return evidence_text

    # Sort quotes by position (longest first to avoid partial matches)
    sorted_quotes = sorted([q for q in quotes if q], key=len, reverse=True)
    
    highlighted = evidence_text
    for quote in sorted_quotes:
        if not quote or len(quote.strip()) < 3:
            continue
        # Escape special regex characters
        escaped_quote = re.escape(quote)
        # Replace with highlighted version (avoid nested highlights)
        pattern = f"({escaped_quote})"
        highlighted = re.sub(
            pattern,
            r'<mark style="background-color: #ffeb3b; padding: 2px 0;">\1</mark>',
            highlighted,
            flags=re.IGNORECASE,
        )

    return highlighted


def format_parsed_output(parsed_json: dict) -> str:
    """Format parsed JSON into readable text."""
    lines = []
    lines.append("## 解析结果\n")
    
    # Answer
    answer = parsed_json.get("answer", "")
    lines.append(f"**答案**: {answer}\n")
    
    # Citations
    citations = parsed_json.get("citations", [])
    if citations:
        lines.append(f"\n**引用** ({len(citations)} 条):\n")
        for i, cit in enumerate(citations, 1):
            doc_id = cit.get("doc_id", "N/A")
            section = cit.get("section", "N/A")
            chunk_id = cit.get("chunk_id", "N/A")
            quote = cit.get("quote", "")
            lines.append(f"{i}. [{doc_id} - {section}] (chunk: {chunk_id})")
            lines.append(f"   > {quote[:200]}{'...' if len(quote) > 200 else ''}\n")
    else:
        lines.append("\n**引用**: 无\n")
    
    # Confidence
    confidence = parsed_json.get("confidence", "N/A")
    lines.append(f"\n**置信度**: {confidence}\n")
    
    # Cannot answer reason
    reason = parsed_json.get("cannot_answer_reason")
    if reason:
        lines.append(f"\n**拒答原因**: {reason}\n")
    
    # Validation status
    is_valid, msg = validate_json_output(parsed_json)
    status_icon = "✅" if is_valid else "❌"
    lines.append(f"\n**JSON校验**: {status_icon} {msg}\n")
    
    return "\n".join(lines)


def log_call(question: str, evidence: str, model_type: str, raw_output: str, parsed_output: Optional[dict]):
    """Log demo call to JSONL file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "question": question,
        "evidence_length": len(evidence),
        "raw_output": raw_output,
        "parsed_output": parsed_output,
        "json_valid": parsed_output is not None,
    }
    
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def process_query(model_type: str, question: str, evidence: str, use_4bit: bool):
    """Process a query and return formatted outputs."""
    if not question.strip():
        return "请输入问题。", "", "", ""
    
    if not evidence.strip():
        return "请输入证据。", "", "", ""
    
    # Load model
    try:
        adapter_path = None
        if model_type == "sft":
            adapter_path = DEFAULT_SFT_ADAPTER
        elif model_type == "dpo":
            adapter_path = DEFAULT_DPO_ADAPTER
        
        bundle = load_model_bundle(DEFAULT_BASE_MODEL, adapter_path, use_4bit=use_4bit)
    except Exception as e:
        error_msg = f"模型加载失败: {str(e)}"
        return error_msg, "", "", ""
    
    # Format prompt
    user_prompt = format_user_prompt(question, evidence_text=evidence)
    
    # Generate
    try:
        raw_json = generate_json(bundle, SYSTEM_PROMPT, user_prompt, max_new_tokens=512)
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        return error_msg, "", "", ""
    
    # Parse JSON
    parsed = None
    try:
        parsed = json.loads(raw_json) if raw_json else None
    except Exception:
        pass
    
    # Format outputs
    raw_output_display = raw_json if raw_json else "(未生成有效JSON)"
    parsed_output_display = format_parsed_output(parsed) if parsed else "**JSON解析失败**\n\n原始输出:\n```\n" + raw_json + "\n```"
    
    # Highlight quotes in evidence
    quotes = []
    if parsed and isinstance(parsed.get("citations"), list):
        quotes = [c.get("quote", "") for c in parsed["citations"] if isinstance(c, dict) and c.get("quote")]
    highlighted_evidence = highlight_quotes_in_evidence(evidence, quotes)
    
    # Log call
    log_call(question, evidence, model_type, raw_json, parsed)
    
    return raw_output_display, parsed_output_display, highlighted_evidence, "✅ 处理完成，已记录日志"


def auto_retrieve_and_answer(model_type: str, question: str, top_k: int, use_4bit: bool):
    """User only inputs question; system retrieves evidence automatically and answers."""
    if not question or not question.strip():
        return "", "请输入问题。", "", "", "", "❌ 问题为空"

    # Retrieve
    try:
        chunks, info = retrieve_topk(question.strip(), top_k=int(top_k))
        evidence_text = format_chunks_as_evidence_text(chunks)
    except Exception as e:
        # Friendly guidance (especially when corpus is missing)
        msg = str(e)
        if "缺少证据库文件" in msg:
            hint = (
                msg
                + "\n\n建议按顺序执行：\n"
                + "  python comm_project/src/01_parse_docx.py\n"
                + "  python comm_project/src/02_build_corpus.py\n"
                + "  python comm_project/src/11_build_index.py\n"
            )
            return "", hint, "", "", "", "❌ 缺少证据库"
        return "", f"检索失败: {msg}", "", "", "", "❌ 检索失败"

    # Answer using retrieved evidence
    raw_json, parsed_md, highlighted_evidence, status = process_query(
        model_type=model_type, question=question, evidence=evidence_text, use_4bit=use_4bit
    )
    return evidence_text, info, raw_json, parsed_md, highlighted_evidence, status


def create_demo():
    """Create Gradio interface."""
    with gr.Blocks(title="通信标准问答助手", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 通信标准问答助手 Demo
            
            基于 Qwen2.5-1.5B-Instruct 的通信标准问答系统。
            
            **使用说明**:
            1. 选择模型版本（base / sft / dpo）
            2. 输入问题和证据（支持多chunk格式）
            3. 查看原始JSON、解析结果和引用高亮
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                model_type = gr.Dropdown(
                    choices=["base", "sft", "dpo"],
                    value="sft",
                    label="模型版本",
                    info="选择要使用的模型版本"
                )
                use_4bit = gr.Checkbox(
                    value=False,
                    label="使用4bit量化",
                    info="降低显存占用（可能影响精度）"
                )
                
                question = gr.Textbox(
                    label="问题",
                    placeholder="例如：什么是PDU Session？",
                    lines=3,
                )
                
                gr.Markdown("### 自动检索（用户只输入问题也可用）")
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="自动检索Top-K",
                    info="从本地证据库中检索最相关的 K 个 chunk 作为证据",
                )
                auto_btn = gr.Button("自动检索并回答", variant="secondary")
                retrieve_info = gr.Textbox(label="检索信息", interactive=False, lines=6)

                evidence = gr.Textbox(
                    label="证据",
                    placeholder="""支持多chunk格式，例如：
[chunk_id=TS23501-5.6.1-0001 doc_id=TS 23.501 section=5.6.1]
A PDU Session is a logical connection between the UE and a Data Network.

[chunk_id=TS23501-5.6.2-0001 doc_id=TS 23.501 section=5.6.2]
The PDU Session establishment procedure is initiated by the UE.
""",
                    lines=10,
                )
                
                submit_btn = gr.Button("提交查询", variant="primary", size="lg")
                status = gr.Textbox(label="状态", interactive=False)
            
            with gr.Column(scale=1):
                raw_json = gr.Code(
                    label="原始JSON输出",
                    language="json",
                    lines=15,
                )
                
                parsed_output = gr.Markdown(
                    label="解析结果",
                )
                
                highlighted_evidence = gr.HTML(
                    label="证据（引用高亮）",
                )
        
        submit_btn.click(
            fn=process_query,
            inputs=[model_type, question, evidence, use_4bit],
            outputs=[raw_json, parsed_output, highlighted_evidence, status],
        )

        auto_btn.click(
            fn=auto_retrieve_and_answer,
            inputs=[model_type, question, top_k, use_4bit],
            outputs=[evidence, retrieve_info, raw_json, parsed_output, highlighted_evidence, status],
        )
        
        gr.Markdown(
            """
            ---
            **注意**: 
            - 所有查询都会记录到 `data/logs/demo_calls.jsonl`
            - 引用高亮使用黄色背景标记
            - 自动检索依赖本地证据库：`comm_project/data/corpus/evidence_corpus.jsonl`
              - 若缺失，请先运行：`01_parse_docx.py` ➜ `02_build_corpus.py`（可选再跑 `11_build_index.py` 预建索引）
            - 如果模型未训练完成，相应选项可能无法使用（例如 dpo adapter 未生成）
            """
        )
    
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "sft", "dpo"], default="sft", help="默认模型版本")
    ap.add_argument("--host", default="127.0.0.1", help="服务器地址")
    ap.add_argument("--port", type=int, default=7860, help="服务器端口")
    ap.add_argument("--share", action="store_true", help="创建公共链接")
    args = ap.parse_args()
    
    demo = create_demo()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

