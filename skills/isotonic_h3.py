"""H3 空间保序后处理 Skill。

对齐 example Notebook 的 ordering_with_decay_city + remaining_to_bin。
分城市、分日期、分 H3 网格；组内按 create_time 顺序，对每单向前找仍在队列中
的历史订单 k，取其 bin 右边界减去已消耗时间，作为下限 floor。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# 分箱右边界（分钟），对齐 feature_builder._BIN_EDGES_MIN 前 10 个 + inf
from skills.evaluation_policy import BIN_EDGES
_RIGHT_BOUND_MIN = dict(enumerate((*BIN_EDGES, float('inf'))))


def _remaining_to_bin(remaining_minutes: float) -> int:
    bins = BIN_EDGES
    for i, b in enumerate(bins):
        if remaining_minutes <= b:
            return i
    return len(bins)


def apply_ordering(
    df: pd.DataFrame,
    city_col: str = "city_code",
    date_col: str = "dt",
    spatial_col: str = "start_h3_index",
    pred_col: str = "predict_bin",
) -> np.ndarray:
    """分组衰减保序；返回与 df 行序对齐的 ndarray。

    组内约束：对订单 x，向前找仍在等待的订单 k (end_time_k > create_time_x)，
    用 k 修正后的 bin 右边界减去已消耗时间得剩余分钟，映射回 bin 作为 x 的下限。
    """
    df = df.copy()
    df["_idx"] = np.arange(len(df))

    create_dt = pd.to_datetime(df["create_time"])
    match_dt = pd.to_datetime(df["match_time"])
    cancel_dt = pd.to_datetime(df["cancel_time"])
    end_dt = match_dt.fillna(cancel_dt).fillna(pd.Timestamp.max)

    df["_create_time_dt"] = create_dt
    df["_end_time_dt"] = end_dt

    out = np.zeros(len(df), dtype=int)

    for _, group in df.groupby([city_col, date_col, spatial_col]):
        group = group.sort_values(["_create_time_dt", "_end_time_dt"]).reset_index(drop=True)
        n = len(group)
        ordered = np.zeros(n, dtype=int)

        c_times = group["_create_time_dt"].values
        e_times = group["_end_time_dt"].values
        preds = group[pred_col].values
        orig_indices = group["_idx"].values

        ordered[0] = preds[0]

        for x in range(1, n):
            floor = None
            for k in range(x - 1, -1, -1):
                if e_times[k] > c_times[x]:
                    diff_min = (c_times[x] - c_times[k]) / np.timedelta64(1, "m")
                    right = _RIGHT_BOUND_MIN[int(ordered[k])]
                    remaining = right - diff_min
                    if remaining > 0:
                        floor = _remaining_to_bin(remaining)
                        break
            ordered[x] = max(int(preds[x]), floor) if floor is not None else int(preds[x])

        out[orig_indices] = ordered
    return out


def apply_ordering_multi_res(
    df: pd.DataFrame,
    pred_col: str = "predict_bin",
    resolutions: Iterable[int] = (6, 7, 8),
    city_col: str = "city_code",
    date_col: str = "dt",
) -> pd.DataFrame:
    """对同一预测列跑多个 H3 分辨率保序，把结果作为新列写回。

    产出列：ordered_{pred_col}_h3_res6/7/8
    df 需要已经通过 feature_builder.add_all_h3_layers 加过 h3_res6/7/8 列。
    """
    df = df.copy()
    for res in resolutions:
        grid_col = f"h3_res{res}"
        out_col = f"ordered_{pred_col}_{grid_col}"
        df[out_col] = apply_ordering(
            df, city_col=city_col, date_col=date_col,
            spatial_col=grid_col, pred_col=pred_col,
        )
    return df
