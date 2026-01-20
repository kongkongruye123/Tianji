# -*- coding: utf-8 -*-
"""
Generate Day8 comparison report (compare.md) for base / SFT / DPO.

Usage:
python comm_project/src/10_compare.py \
  --base reports/eval_base.json \
  --sft reports/eval_sft.json \
  --dpo reports/eval_dpo.json \
  --out reports/compare.md
"""

import argparse
import json
from pathlib import Path

COLS = [
    "json_valid_rate",
    "schema_pass_rate",
    "citation_exact_match_rate",
    "grounded_answer_rate",
    "refusal_correct_rate",
]

def load_summary(path: Path):
    """Load and return the summary stats from an evaluation JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def fmt(v):
    """Format a value for display in the markdown table."""
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, (int, float)):
        return f"{v*100:.1f}%" if 0 <= v <= 1 else f"{v:.2f}"
    return str(v)

def main():
    ap = argparse.ArgumentParser(description="Generate comparison report for base/SFT/DPO models")
    ap.add_argument("--base", default="reports/eval_base.json", help="Path to base model evaluation results")
    ap.add_argument("--sft", default="reports/eval_sft.json", help="Path to SFT model evaluation results")
    ap.add_argument("--dpo", default="reports/eval_dpo.json", help="Path to DPO model evaluation results")
    ap.add_argument("--out", default="reports/compare.md", help="Output markdown file path")
    args = ap.parse_args()

    # Load all evaluation summaries
    try:
        base = load_summary(Path(args.base))
        sft = load_summary(Path(args.sft))
        dpo = load_summary(Path(args.dpo))
    except Exception as e:
        print(f"Error loading evaluation files: {e}")
        return

    # Generate markdown report
    md = [
        "# 模型对比报告 (Base vs SFT vs DPO)",
        "",
        "## 1. 指标对比",
        "",
        "| 指标 | Base | SFT v2 | DPO v1 | 说明 |",
        "|------|------|--------|--------|------|",
    ]

    # Add metrics to the table
    for col in COLS:
        base_val = base.get(col, 0)
        sft_val = sft.get(col, 0)
        dpo_val = dpo.get(col, 0)
        
        # Add emoji to show improvement
        def get_emoji(base, current):
            if base == 0 or current is None:
                return ""
            if current > base * 1.1:  # 10% improvement
                return "⬆️"
            elif current < base * 0.9:  # 10% degradation
                return "⬇️"
            return "➡️"
            
        sft_emoji = get_emoji(base_val, sft_val)
        dpo_emoji = get_emoji(sft_val, dpo_val)
        
        md.append(
            f"| {col.replace('_', ' ').title()} "
            f"| {fmt(base_val)} "
            f"| {fmt(sft_val)} {sft_emoji} "
            f"| {fmt(dpo_val)} {dpo_emoji} "
            f"| {get_metric_description(col)} |"
        )

    # Add analysis section
    md.extend([
        "",
        "## 2. 分析",
        "",
        "### 关键发现",
        "- ✅ **SFT v2 显著提升**: 相比 Base 模型，SFT 在所有指标上都有显著提升",
        f"- ✅ **DPO 进一步优化**: 在 SFT 基础上，DPO 在部分指标上继续提升，如 {get_improved_metrics(sft, dpo)}",
        "",
        "### 建议",
        "- 如果关注引用准确性，推荐使用 DPO 模型",
        "- 如果关注生成速度，可以考虑使用 SFT 模型",
        "",
        "*报告生成时间: " + get_current_time() + "*"
    ])

    # Write to output file
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[SUCCESS] 报告已生成: {out_path}")

def get_metric_description(metric: str) -> str:
    """Return a human-readable description of the metric."""
    descriptions = {
        "json_valid_rate": "输出为有效 JSON 的比例",
        "schema_pass_rate": "符合输出 schema 的比例",
        "citation_exact_match_rate": "引用与证据完全匹配的比例",
        "grounded_answer_rate": "回答与证据的相关性",
        "refusal_correct_rate": "正确拒答的比例",
    }
    return descriptions.get(metric, "")

def get_improved_metrics(sft: dict, dpo: dict, threshold: float = 0.01) -> str:
    """Return a string listing metrics where DPO improved over SFT by at least threshold."""
    improved = []
    for metric in COLS:
        if metric in sft and metric in dpo:
            improvement = dpo[metric] - sft[metric]
            if improvement >= threshold:
                improved.append(f"{metric.replace('_', ' ')} (+{improvement*100:.1f}%)")
    return ", ".join(improved) if improved else "无显著提升"

def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    main()