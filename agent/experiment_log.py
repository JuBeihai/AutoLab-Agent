"""结构化实验日志：每次 Agent 提出/执行一个 challenger，
在 experiments/ 下追加一行 JSON Lines 记录。

Schema:
    experiment_id, round_id, timestamp, proposer, hypothesis,
    config_diff, data_window, metrics_by_city, gate_pass, artifact_dir
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.identifiers import new_id


@dataclass
class ExperimentRecord:
    experiment_id: str
    round_id: int
    timestamp: str
    proposer: str                    # 'baseline' / 'agent' / 'human'
    hypothesis: str                  # 一句话说明这轮为什么改
    config_diff: Dict[str, Any] = field(default_factory=dict)
    data_window: Dict[str, str] = field(default_factory=dict)
    metrics_by_city: Dict[str, Any] = field(default_factory=dict)
    gate_pass: Optional[bool] = None
    gate_reasons: list = field(default_factory=list)
    artifact_dir: str = ""
    notes: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    evaluation_views: Dict[str, Any] = field(default_factory=dict)


def new_experiment_id(prefix: str = "exp", root=None) -> str:
    kind = 'S' if prefix in {'mining_mock', 'mining_real', 'session'} else 'E'
    return new_id(kind, root=Path(root) if root is not None else Path(__file__).resolve().parents[1])


def append_record(log_dir: str, record: ExperimentRecord) -> str:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "experiments.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return log_path


def update_record(log_dir: str, experiment_id: str, patch: Dict[str, Any]) -> bool:
    """按 experiment_id 就地更新一条记录（用于 gate_check 结果回填）。

    Returns
    -------
    True 命中并更新；False 未找到对应 experiment_id。
    """
    log_path = os.path.join(log_dir, "experiments.jsonl")
    if not os.path.exists(log_path):
        return False
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    hit = False
    new_lines = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("experiment_id") == experiment_id:
            rec.update(patch)
            hit = True
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    if hit:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
    return hit


def load_records(log_dir: str) -> List[dict]:
    log_path = os.path.join(log_dir, "experiments.jsonl")
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_champion(log_dir: str, model_tag: str = "xgb_44",
                 city_key: str = "ALL",
                 metric: str = "recall_ratio",
                 require_gate_pass: bool = True) -> Optional[dict]:
    """从历史实验里找 champion（默认按全量 xgb_44 准确桶最高）。

    require_gate_pass:
        True  —— 只从 baseline（gate_pass=None）与 gate_pass=True 的记录中挑，
                 避免长循环里 challenger 一次意外高分就顶替 champion。
        False —— 无条件按 metric 选。
    """
    records = load_records(log_dir)
    best = None
    best_val = -1
    for r in records:
        try:
            if require_gate_pass:
                gp = r.get("gate_pass")
                # baseline 没有 gate_check，gate_pass 为 None；显式 True 也允许
                if gp is False:
                    continue
            m = r["metrics_by_city"].get(model_tag, {})
            city_m = m.get(city_key) or m.get(str(city_key))
            if not city_m:
                continue
            val = city_m.get(metric, -1)
            if val is not None and val > best_val:
                best_val = val
                best = r
        except (KeyError, TypeError):
            continue
    return best


def _pick_city_metrics(record: dict, model_tag: str, city_key: str) -> Optional[dict]:
    m = record.get("metrics_by_city", {}).get(model_tag, {})
    return m.get(city_key) or m.get(str(city_key))


def gate_check_records(
    champion: dict,
    challenger: dict,
    model_tag: str = "xgb_44",
    thresholds: Optional[dict] = None,
) -> Dict[str, Any]:
    """从两条实验记录中按 Agent调研.md 7.2 节门禁做比较。

    Returns
    -------
    {
      "pass": bool,
      "reasons": [str, ...],
      "per_city": {"100001": {champion_acc, challenger_acc, acc_delta, ...}, ...}
    }
    """
    if thresholds is None:
        thresholds = {
            "min_acc_delta_per_city": -0.01,      # 任一城市 acc 不下降超 1pp
            "bj_under_reduction_min": 0.02,       # 示例区域A低估率至少降 2pp
        }

    per_city = {}
    reasons = []
    passed = True

    for city_key in ("100001", "100002", "100003"):
        champ = _pick_city_metrics(champion, model_tag, city_key)
        chall = _pick_city_metrics(challenger, model_tag, city_key)
        if not champ or not chall:
            continue
        acc_delta = chall["recall_ratio"] - champ["recall_ratio"]
        under_delta = chall["under_ratio"] - champ["under_ratio"]
        per_city[city_key] = {
            "champion_acc": champ["recall_ratio"],
            "challenger_acc": chall["recall_ratio"],
            "acc_delta": acc_delta,
            "champion_under": champ["under_ratio"],
            "challenger_under": chall["under_ratio"],
            "under_delta": under_delta,
        }
        if acc_delta < thresholds["min_acc_delta_per_city"]:
            passed = False
            reasons.append(f"city {city_key} acc 下降 {acc_delta:.1%}"
                           f"（超过 {thresholds['min_acc_delta_per_city']:.1%} 阈值）")

    bj = per_city.get("100001")
    if bj:
        reduction = -bj["under_delta"]
        if reduction < thresholds["bj_under_reduction_min"]:
            passed = False
            reasons.append(f"示例区域A低估率仅降 {reduction:.1%}"
                           f"（未达 {thresholds['bj_under_reduction_min']:.1%}）")

    return {"pass": passed, "reasons": reasons, "per_city": per_city}
