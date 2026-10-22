"""评估 Skill：分城市 / 分切片准确桶 + 高估 + 低估。

对齐 example Notebook 的 generate_prediction_matrix + evaluate_by_city。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SliceMetrics:
    name: str
    total: int
    recall: int           # 准确桶数
    over: int             # 高估数
    under: int            # 低估数
    recall_ratio: float
    over_ratio: float
    under_ratio: float
    bin_mae: float        # 档位平均绝对误差
    severe_under: int     # 低估 >= 2 档
    severe_under_ratio: float

    def as_dict(self) -> dict:
        return asdict(self)


def _slice_metrics(df: pd.DataFrame, pred_col: str, true_col: str,
                   name: str) -> SliceMetrics:
    if len(df) == 0:
        return SliceMetrics(name, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
    y_true = df[true_col].values.astype(int)
    y_pred = df[pred_col].values.astype(int)
    total = len(df)
    correct = int((y_pred == y_true).sum())
    over = int((y_pred > y_true).sum())
    under = int((y_pred < y_true).sum())
    diffs = y_pred - y_true
    bin_mae = float(np.abs(diffs).mean())
    severe_under = int((diffs <= -2).sum())
    return SliceMetrics(
        name=name, total=total, recall=correct, over=over, under=under,
        recall_ratio=correct / total, over_ratio=over / total, under_ratio=under / total,
        bin_mae=bin_mae, severe_under=severe_under,
        severe_under_ratio=severe_under / total,
    )


def confusion_matrix(df: pd.DataFrame, true_col: str = "true_bin",
                     pred_col: str = "predict_bin") -> pd.DataFrame:
    """按实际桶编号生成混淆矩阵（含 true_all / pred_all 汇总行列）。"""
    temp = (df.groupby([true_col, pred_col])
              .size()
              .rename("count")
              .reset_index())
    m = pd.pivot_table(temp, index=true_col, columns=pred_col,
                       values="count").fillna(0)
    bins = sorted(set(m.index) | set(m.columns))
    m = m.reindex(index=bins, columns=bins, fill_value=0).astype(int)
    m["true_all"] = m.sum(axis=1)
    m.loc["pred_all"] = m.sum(axis=0)
    return m


def evaluate_by_city(
    df: pd.DataFrame,
    pred_col: str = "predict_bin",
    true_col: str = "true_bin",
    city_col: str = "city_code",
    city_names: Optional[Dict[int, str]] = None,
    label: str = "",
    verbose: bool = True,
) -> Dict[str, SliceMetrics]:
    """分城市评估。返回 {city_code: SliceMetrics}。"""
    if city_names is None:
        city_names = {100001: "示例区域A", 100002: "示例区域B", 100003: "示例区域C"}

    results = {}
    if verbose:
        print(f"\n==================== [{label}] {pred_col} 分城市 ====================")
    for city_code in sorted(df[city_col].unique()):
        city_df = df[df[city_col] == city_code]
        name = city_names.get(int(city_code), str(city_code))
        m = _slice_metrics(city_df, pred_col, true_col,
                           name=f"{city_code}_{name}")
        results[int(city_code)] = m
        if verbose:
            print(
                f"  {city_code}（{name}） n={m.total} "
                f"acc={m.recall_ratio:.1%} over={m.over_ratio:.1%} "
                f"under={m.under_ratio:.1%} MAE={m.bin_mae:.2f} "
                f"severe_under={m.severe_under_ratio:.1%}"
            )
    # 全量
    m_all = _slice_metrics(df, pred_col, true_col, name="ALL")
    results["ALL"] = m_all
    if verbose:
        print(
            f"  ALL     n={m_all.total} acc={m_all.recall_ratio:.1%} "
            f"over={m_all.over_ratio:.1%} under={m_all.under_ratio:.1%} "
            f"MAE={m_all.bin_mae:.2f}"
        )
    return results


def error_slices(
    df: pd.DataFrame,
    pred_col: str = "predict_bin",
    true_col: str = "true_bin",
    slice_cols: List[str] = ("city_code", "hour", "is_weekday",
                              "base_pred_bin", "start_h3_index"),
    min_count: int = 30,
) -> pd.DataFrame:
    """按多个维度切片给出准确率/高低估率，供 Agent 决定下一步方向。

    只保留样本量 >= min_count 的切片。
    """
    rows = []
    for col in slice_cols:
        if col not in df.columns:
            continue
        for value, sub in df.groupby(col):
            if len(sub) < min_count:
                continue
            m = _slice_metrics(sub, pred_col, true_col, name=f"{col}={value}")
            rows.append({
                "slice_col": col,
                "slice_value": value,
                "n": m.total,
                "acc": m.recall_ratio,
                "over": m.over_ratio,
                "under": m.under_ratio,
                "bin_mae": m.bin_mae,
                "severe_under_ratio": m.severe_under_ratio,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("under", ascending=False).reset_index(drop=True)
    return out


def compare_champion_challenger(
    df: pd.DataFrame,
    champion_col: str,
    challenger_col: str,
    true_col: str = "true_bin",
    city_col: str = "city_code",
    thresholds: Optional[dict] = None,
) -> dict:
    """按 Agent调研.md 7.2 节的门禁比较 champion / challenger。

    返回 {"pass": bool, "reasons": [...], "metrics": {city -> {champion, challenger, delta}}}
    """
    if thresholds is None:
        thresholds = {
            "min_acc_delta_per_city": -0.01,         # 任一城市 acc 不下降超 1pp
            "bj_under_reduction_min": 0.02,          # 示例区域A低估至少降 2pp
        }

    metrics = {}
    reasons = []
    passed = True

    for city_code, sub in df.groupby(city_col):
        champ = _slice_metrics(sub, champion_col, true_col, name="champ")
        chall = _slice_metrics(sub, challenger_col, true_col, name="chall")
        metrics[int(city_code)] = {
            "n": champ.total,
            "champion_acc": champ.recall_ratio,
            "challenger_acc": chall.recall_ratio,
            "acc_delta": chall.recall_ratio - champ.recall_ratio,
            "champion_under": champ.under_ratio,
            "challenger_under": chall.under_ratio,
            "under_delta": chall.under_ratio - champ.under_ratio,
        }
        # 任一城市 acc 不下降超阈值
        if chall.recall_ratio - champ.recall_ratio < thresholds["min_acc_delta_per_city"]:
            passed = False
            reasons.append(f"城市 {city_code} acc 下降 "
                           f"{(chall.recall_ratio - champ.recall_ratio):.1%} "
                           f"超过阈值 {thresholds['min_acc_delta_per_city']:.1%}")

    # 示例区域A低估率
    if 100001 in metrics:
        region_a = metrics[100001]
        under_reduction = region_a["champion_under"] - region_a["challenger_under"]
        if under_reduction < thresholds["bj_under_reduction_min"]:
            passed = False
            reasons.append(f"示例区域A低估率仅下降 {under_reduction:.1%}，未达 "
                           f"{thresholds['bj_under_reduction_min']:.1%} 门槛")

    return {"pass": passed, "reasons": reasons, "metrics": metrics}
