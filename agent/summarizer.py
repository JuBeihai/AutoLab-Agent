"""实验总结生成器。

生成结构化 summary（创新点 + delta vs baseline + verdict + 中文叙述），
写入 ExperimentRecord.summary 字段并生成 artifacts/<exp_id>/SUMMARY.md。

设计原则：
- 数字全部由模板计算，LLM 只负责自然语言润色
- LLM 缺失/失败时降级为 template-only，不阻塞主流程
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_CITY_NAMES = {
    "100001": "示例区域A",
    "100002": "示例区域B",
    "100003": "示例区域C",
    "ALL": "全量",
}

_METRIC_KEYS = ("recall_ratio", "over_ratio", "under_ratio", "bin_mae", "severe_under_ratio")
_METRIC_ALIAS = {
    "recall_ratio": "acc",
    "over_ratio": "over",
    "under_ratio": "under",
    "bin_mae": "mae",
    "severe_under_ratio": "severe_under",
}


def _r4(x) -> float:
    """round 到 4 位小数，None 保持 None。"""
    if x is None:
        return None
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None


def compute_delta(challenger_metrics: dict, baseline_metrics: dict,
                   model_tag: str = "xgb_44") -> Dict[str, Dict[str, float]]:
    """按城市计算 challenger - baseline 的五项 delta。

    Returns
    -------
    {"ALL": {"acc": -0.005, "under": +0.003, ...}, "100001": {...}, ...}
    仅包含双方都有的城市键。
    """
    out = {}
    ch = (challenger_metrics or {}).get(model_tag, {})
    bs = (baseline_metrics or {}).get(model_tag, {})
    for city_key in list(ch.keys()):
        ch_city = ch.get(city_key)
        bs_city = bs.get(city_key)
        if not ch_city or not bs_city:
            continue
        row = {}
        for k in _METRIC_KEYS:
            a = ch_city.get(k)
            b = bs_city.get(k)
            if a is None or b is None:
                continue
            row[_METRIC_ALIAS[k]] = _r4(a - b)
        if row:
            out[city_key] = row
    return out


def compute_innovation(config_diff: dict, prev_train_cfg: dict) -> Dict[str, Any]:
    """从 config_diff 生成结构化创新点描述。

    prev_train_cfg 用于查 xgb_params 的原始值（做 from -> to 对比）。
    """
    if not config_diff:
        return {
            "title": "baseline（复刻 champion 配置）",
            "features_added": [],
            "features_removed": [],
            "xgb_params_changed": {},
            "training_filters_changed": {},
        }

    added = list(config_diff.get("add_features") or [])
    removed = list(config_diff.get("remove_features") or [])

    xgb_orig = (prev_train_cfg or {}).get("xgboost", {})
    xgb_changed = {}
    for k, v in (config_diff.get("xgb_params") or {}).items():
        xgb_changed[k] = {"from": xgb_orig.get(k), "to": v}

    filters_changed = dict(config_diff.get("training_filters") or {})

    parts = []
    if added:
        parts.append("加入特征 " + ", ".join(added))
    if removed:
        parts.append("移除特征 " + ", ".join(removed))
    if xgb_changed:
        pieces = [f"{k}={ch['from']}→{ch['to']}" for k, ch in xgb_changed.items()]
        parts.append("调参 " + ", ".join(pieces))
    if filters_changed:
        parts.append("调训练过滤 " + json.dumps(filters_changed, ensure_ascii=False))
    title = "；".join(parts) if parts else "无有效改动"

    return {
        "title": title,
        "features_added": added,
        "features_removed": removed,
        "xgb_params_changed": xgb_changed,
        "training_filters_changed": filters_changed,
    }


def classify_verdict(delta: Dict[str, Dict[str, float]],
                      gate_pass: Optional[bool] = None,
                      is_baseline: bool = False) -> str:
    """判定本次实验是 improvement / regression / mixed / baseline。

    baseline: proposer == baseline
    regression: gate_pass=False，或任一城 acc_delta < -0.01
    improvement: 三城 acc_delta 全 >= -0.005 且 BJ under_delta <= -0.005
    其余: mixed
    """
    if is_baseline:
        return "baseline"
    if gate_pass is False:
        return "regression"

    per_city = {k: v for k, v in delta.items() if k != "ALL"}
    if not per_city:
        return "mixed"

    for city, row in per_city.items():
        if row.get("acc") is not None and row["acc"] < -0.01:
            return "regression"

    bj = delta.get("100001")
    all_acc_ok = all((row.get("acc") is None or row["acc"] >= -0.005)
                     for row in per_city.values())
    bj_under_good = bj and bj.get("under") is not None and bj["under"] <= -0.005
    if all_acc_ok and bj_under_good:
        return "improvement"
    return "mixed"


def _fmt_pp(x: Optional[float]) -> str:
    """把 0.0032 这样的小数转成 "+0.3pp" / "-0.5pp"，None 用 "—"。"""
    if x is None:
        return "—"
    pp = x * 100
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.1f}pp"


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.3f}"


def render_narrative_template(innovation: dict,
                               delta: Dict[str, Dict[str, float]],
                               verdict: str,
                               baseline_ref: dict) -> str:
    """纯模板生成 4-6 行中文总结（无需 LLM）。"""
    lines = []

    if verdict == "baseline":
        lines.append(f"本次实验为 baseline 复刻，无对比 delta。")
    else:
        lines.append(f"本次改动：{innovation.get('title', '（无）')}。")

    if baseline_ref.get("experiment_id"):
        lines.append(
            f"对比对象：{baseline_ref.get('role', 'champion')} "
            f"（{baseline_ref['experiment_id']}）。"
        )

    if verdict != "baseline":
        parts = []
        for city_key in ("100001", "100002", "100003", "ALL"):
            row = delta.get(city_key)
            if not row:
                continue
            city_name = _CITY_NAMES.get(city_key, city_key)
            acc_s = _fmt_pp(row.get("acc"))
            under_s = _fmt_pp(row.get("under"))
            parts.append(f"{city_name} acc {acc_s} / under {under_s}")
        if parts:
            lines.append("关键 delta：" + "；".join(parts) + "。")

    verdict_zh = {
        "improvement": "本轮判定为提升，可作为新 champion 候选。",
        "regression": "本轮判定为退化，未通过门禁。",
        "mixed": "本轮判定为混合结果，部分指标进步、部分退化。",
        "baseline": "作为后续对比基准。",
    }.get(verdict, "结果待定。")
    lines.append(verdict_zh)

    return " ".join(lines)


def _load_summarizer_prompt(root: Path) -> str:
    p = root / "agent/prompts/summarizer.txt"
    if not p.exists():
        return "你是 等待时间 实验总结助手，用 4-6 句中文说明创新点、delta、判定与下一步建议。"
    return p.read_text(encoding="utf-8")


def render_narrative_with_llm(innovation: dict,
                               delta: Dict[str, Dict[str, float]],
                               verdict: str,
                               baseline_ref: dict,
                               template_text: str,
                               llm_cfg: dict,
                               root: Path) -> str:
    """让 LLM 基于给定的数字/创新点写一段自然语言总结。
    失败或空 key 时返回 template_text。"""
    try:
        from skills.llm_client import LLMClient, LLMError
    except Exception:
        return template_text

    try:
        client = LLMClient(llm_cfg)
    except Exception as e:
        print(f"[summarizer] LLM 初始化失败，使用模板：{e}")
        return template_text

    system_prompt = _load_summarizer_prompt(root)

    # 只把结构化数据丢给 LLM，明确说不能改数字
    user_prompt = (
        "## 创新点\n"
        + json.dumps(innovation, ensure_ascii=False, indent=2)
        + "\n\n## Delta vs baseline (小数，正值表示比 baseline 高)\n"
        + json.dumps(delta, ensure_ascii=False, indent=2)
        + "\n\n## Verdict\n"
        + verdict
        + "\n\n## 参考模板（数字必须与此保持一致）\n"
        + template_text
        + "\n\n请输出 150-300 字的纯中文段落，遵循 system prompt 要求。"
    )

    try:
        _buf = {"reasoning": [], "content": []}
        _FLUSH_CHARS = 40

        def _flush(kind: str, tail: bool = False):
            s = "".join(_buf[kind])
            if not s:
                return
            if tail or len(s) >= _FLUSH_CHARS or "\n" in s:
                prefix = "[sum-think]" if kind == "reasoning" else "[sum-out]  "
                for line in s.split("\n"):
                    if line.strip():
                        print(f"{prefix} {line}", flush=True)
                _buf[kind] = []

        def _on_delta(kind: str, piece: str):
            _buf[kind].append(piece)
            _flush(kind, tail=False)

        resp = client.chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            max_tokens=3000,
            temperature=0.4,
            stream=True,
            on_delta=_on_delta,
        )
        _flush("reasoning", tail=True)
        _flush("content", tail=True)
        text = (resp.text or "").strip()
        return text if text else template_text
    except Exception as e:
        print(f"[summarizer] LLM 中文润色失败（不影响数据准确性，退回模板摘要）：{e}", flush=True)
        return template_text


def build_summary(record_dict: dict,
                   baseline_record: Optional[dict],
                   prev_train_cfg: dict,
                   llm_cfg: Optional[dict],
                   root: Path,
                   use_llm: bool = True) -> Dict[str, Any]:
    """一次性组装完整 summary dict。

    record_dict: 已完成的 ExperimentRecord.asdict()
    baseline_record: 用来比较的记录（通常是 get_champion 返回值），None 表示无对比
    """
    is_baseline = record_dict.get("proposer") == "baseline"
    innovation = compute_innovation(
        record_dict.get("config_diff") or {},
        prev_train_cfg or {},
    )

    if baseline_record and not is_baseline:
        delta = compute_delta(
            record_dict.get("metrics_by_city") or {},
            baseline_record.get("metrics_by_city") or {},
            model_tag="xgb_44",
        )
        baseline_ref = {
            "experiment_id": baseline_record.get("experiment_id"),
            "role": "champion",
        }
    else:
        delta = {}
        baseline_ref = {"experiment_id": None, "role": "self"}

    verdict = classify_verdict(
        delta,
        gate_pass=record_dict.get("gate_pass"),
        is_baseline=is_baseline,
    )

    template_text = render_narrative_template(
        innovation, delta, verdict, baseline_ref,
    )

    if use_llm and llm_cfg and not is_baseline:
        narrative = render_narrative_with_llm(
            innovation, delta, verdict, baseline_ref,
            template_text, llm_cfg, root,
        )
        source = "template+llm" if narrative != template_text else "template-only"
    else:
        narrative = template_text
        source = "template-only"

    return {
        "innovation": innovation,
        "baseline_ref": baseline_ref,
        "delta_vs_baseline": {"xgb_44": delta},
        "verdict": verdict,
        "narrative_zh": narrative,
        "narrative_source": source,
    }


def write_summary_md(summary: dict, record_dict: dict, artifact_dir: str) -> str:
    """把 summary 写成人可读 Markdown 到 artifact_dir/SUMMARY.md。"""
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "SUMMARY.md")

    inn = summary.get("innovation", {})
    delta = (summary.get("delta_vs_baseline") or {}).get("xgb_44", {})
    ref = summary.get("baseline_ref") or {}
    verdict = summary.get("verdict", "")
    narrative = summary.get("narrative_zh", "")

    lines = [
        f"# 实验总结 · {record_dict.get('experiment_id')}",
        "",
        f"- **轮次**: {record_dict.get('round_id')} · **提议方**: {record_dict.get('proposer')}",
        f"- **时间**: {record_dict.get('timestamp')}",
        f"- **假设**: {record_dict.get('hypothesis')}",
        f"- **对比对象**: {ref.get('experiment_id') or '（无）'} ({ref.get('role')})",
        f"- **门禁**: gate_pass={record_dict.get('gate_pass')} · 原因: {record_dict.get('gate_reasons') or '—'}",
        f"- **判定**: **{verdict}**",
        "",
        "## 创新点",
        f"- 加入特征: `{', '.join(inn.get('features_added') or []) or '—'}`",
        f"- 移除特征: `{', '.join(inn.get('features_removed') or []) or '—'}`",
        f"- XGB 参数: `{json.dumps(inn.get('xgb_params_changed') or {}, ensure_ascii=False)}`",
        f"- 训练过滤: `{json.dumps(inn.get('training_filters_changed') or {}, ensure_ascii=False)}`",
        "",
        "## Delta vs baseline (xgb_44)",
        "",
        "| 城市 | acc | under | over | mae | severe_under |",
        "|---|---|---|---|---|---|",
    ]
    for city_key in ("100001", "100002", "100003", "ALL"):
        row = delta.get(city_key)
        if not row:
            continue
        city_name = _CITY_NAMES.get(city_key, city_key)
        lines.append(
            f"| {city_name} | {_fmt_pp(row.get('acc'))} | {_fmt_pp(row.get('under'))} | "
            f"{_fmt_pp(row.get('over'))} | {_fmt_num(row.get('mae'))} | "
            f"{_fmt_pp(row.get('severe_under'))} |"
        )
    lines += [
        "",
        "## 中文总结",
        "",
        narrative or "（无）",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
