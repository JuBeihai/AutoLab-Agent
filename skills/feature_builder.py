"""特征/标签构造 Skill

对齐 example Notebook 的 ets_func / week_peak / convert_coords_to_h3 / ets_bin。
所有函数纯 pandas / numpy，无 Spark 依赖。
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_true_ets(df: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    """计算真实 等待时间 秒数 + is_answer 标记。

    优先使用 match_time；缺失时用 cancel_time，与 example 一致。
    注意：这个口径把"取消"当成了"到 cancel_time 时才会应答"，
    交接文档已标注为竞争风险问题，Agent 后续可用 survival 模型替代。
    """
    df = df.copy() if copy else df
    create_dt = pd.to_datetime(df["create_time"])
    match_dt = pd.to_datetime(df["match_time"])
    cancel_dt = pd.to_datetime(df["cancel_time"])
    target_dt = match_dt.combine_first(cancel_dt)
    df["true_ets"] = (target_dt - create_dt).dt.total_seconds()
    df["is_answer"] = np.where(df["match_time"].isnull(), 0, 1)
    return df


def add_week_peak(df: pd.DataFrame) -> pd.DataFrame:
    """从 create_time 派生 hour / weekday / is_weekday / peak。

    与 example Notebook 的 week_peak() 逻辑完全一致：
    - 工作日：07-09 morningPeak, 18-23 eveningPeak, 其他 offPeak
    - 非工作日：10-22 peak, 其他 offPeak
    """
    df = df.copy()
    create_time_dt = pd.to_datetime(df["create_time"])
    df["hour"] = create_time_dt.dt.hour
    df["weekday"] = create_time_dt.dt.dayofweek
    df["is_weekday"] = (df["weekday"] <= 4).astype(int)

    def _peak(is_wk, hr):
        if is_wk == 1:
            if 7 <= hr < 10:
                return "morningPeak"
            elif hr >= 18:
                return "eveningPeak"
            else:
                return "offPeak"
        else:
            if 10 <= hr <= 22:
                return "peak"
            else:
                return "offPeak"

    df["peak"] = [_peak(w, h) for w, h in zip(df["is_weekday"], df["hour"])]
    return df


def add_h3_index(df: pd.DataFrame, lat_col: str = "start_lat",
                 lng_col: str = "start_lng", resolution: int = 7,
                 out_col: str = "start_h3_index", *, copy: bool = True) -> pd.DataFrame:
    """加 H3 网格列（h3 v4 API）。"""
    import h3
    df = df.copy() if copy else df
    df[out_col] = [
        h3.latlng_to_cell(float(lat), float(lon), resolution)
        if pd.notna(lat) and pd.notna(lon)
        else None
        for lat, lon in zip(df[lat_col], df[lng_col])
    ]
    return df


def add_all_h3_layers(df: pd.DataFrame, resolutions: List[int] = (6, 7, 8),
                      lat_col: str = "start_lat",
                      lng_col: str = "start_lng", *, copy: bool = True) -> pd.DataFrame:
    """一次性加多个 H3 分辨率，供保序 Skill 使用。"""
    import h3
    df = df.copy() if copy else df
    for res in resolutions:
        col = f"h3_res{res}"
        df[col] = [
            h3.latlng_to_cell(float(lat), float(lon), res)
            if pd.notna(lat) and pd.notna(lon)
            else None
            for lat, lon in zip(df[lat_col], df[lng_col])
        ]
    return df


# 分箱阈值（分钟）；与指定 XGB43 / 6桶 Notebook 对齐
from skills.evaluation_policy import BIN_EDGES
_BIN_EDGES_MIN = list(BIN_EDGES)


def ets_bin(seconds) -> int:
    """把 等待时间 秒数映射到 0-5 档。"""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return -1
    minutes = float(seconds) / 60.0
    for i, edge in enumerate(_BIN_EDGES_MIN):
        if minutes <= edge:
            return i
    return len(_BIN_EDGES_MIN)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    """加 true_bin / base_pred_bin / base_sorted_pred_bin 三列。

    - true_bin: 真实 等待时间 分箱
    - base_pred_bin: 规则模型 origin_ert_result 分箱（诊断；新流程按 has_queue 路由）
    - base_sorted_pred_bin: 规则模型 ert_result（含保序）分箱
    """
    df = df.copy()
    df["true_bin"] = [ets_bin(x) for x in df["true_ets"]]
    df["base_pred_bin"] = [
        ets_bin(float(x)) if pd.notna(x) else -1 for x in df["origin_ert_result"]
    ]
    df["base_sorted_pred_bin"] = [
        ets_bin(float(x)) if pd.notna(x) else -1 for x in df["ert_result"]
    ]
    return df


def build_full_features(df: pd.DataFrame,
                         h3_resolutions=(6, 7, 8)) -> pd.DataFrame:
    """流水线：等待时间 + 峰期 + H3 多分辨率 + 分箱。

    输入应该是 data_loader.merge_all() 的输出。
    如果 hour/is_weekday 等时间/峰期字段来自 SDS 表（真实数据/mock 都有），
    就跳过 add_week_peak；否则用 create_time 现算，保证下游特征齐全。
    """
    df = compute_true_ets(df)
    required_peak_cols = {"hour", "is_weekday"}
    if not required_peak_cols.issubset(df.columns):
        df = add_week_peak(df)
    df = add_h3_index(df, resolution=7, out_col="start_h3_index")
    df = add_all_h3_layers(df, resolutions=h3_resolutions)
    df = add_bins(df)
    return df


# ================================================================
# Agent 提议的候选特征实现
#
# 已实现（可立即入模）:
#   - hour_sin, hour_cos   周期性时间编码
#   - queue_x_grid_rank    排队队列长度 × 网格排位
#   - hour_x_peak          小时 × 峰期标识（合并三个峰期为 0/1/2/3）
#
# 未实现（需要时间/空间历史聚合，先返回 NotImplementedError 让 orchestrator 跳过）:
#   - minutes_since_creation, neighbor_grid_rank_mean,
#     supply_trend_5min, demand_trend_5min, supply_demand_ratio_ratio
# ================================================================


class ProposedFeatureNotImplemented(NotImplementedError):
    pass


def _feat_hour_sin(df: pd.DataFrame) -> pd.Series:
    return np.sin(2 * np.pi * df["hour"].astype(float) / 24.0)


def _feat_hour_cos(df: pd.DataFrame) -> pd.Series:
    return np.cos(2 * np.pi * df["hour"].astype(float) / 24.0)


def _feat_queue_x_grid_rank(df: pd.DataFrame) -> pd.Series:
    # sp1+sp2 队列总长度 × grid_rank；缺失填 0
    q = df.get("sp1_queue_count", 0).fillna(0) + df.get("sp2_queue_count", 0).fillna(0)
    return q * df["grid_rank"].fillna(0)


def _feat_hour_x_peak(df: pd.DataFrame) -> pd.Series:
    # 峰期标识编码：night_peak=3, morning_peak=2, peak=1, flat=0
    peak_code = (
        df.get("is_night_peak", 0).fillna(0) * 3
        + df.get("is_morning_peak", 0).fillna(0) * 2
        + df.get("is_peak", 0).fillna(0) * 1
    )
    return df["hour"].astype(float) * peak_code


_PROPOSED_FEATURE_IMPL = {
    "hour_sin": _feat_hour_sin,
    "hour_cos": _feat_hour_cos,
    "queue_x_grid_rank": _feat_queue_x_grid_rank,
    "hour_x_peak": _feat_hour_x_peak,
    # h3_res6 / h3_res8 已经由 build_full_features 生成，但它们是字符串不能直接入 XGB
    # 若 Agent 提议 h3_res6/8 需要额外 hash encoding，暂不实现
}


def add_proposed_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """按名字加候选特征。

    Raises
    ------
    ProposedFeatureNotImplemented: 特征未实现时抛出（orchestrator 会 catch 并跳过 proposal）
    """
    df = df.copy()
    unimplemented = []
    for name in feature_names:
        if name in df.columns:
            continue  # 已存在，跳过
        impl = _PROPOSED_FEATURE_IMPL.get(name)
        if impl is None:
            unimplemented.append(name)
            continue
        df[name] = impl(df).astype(np.float32)
    if unimplemented:
        raise ProposedFeatureNotImplemented(
            f"以下候选特征暂未实现（需要历史聚合或空间邻域，请扩展 "
            f"feature_builder._PROPOSED_FEATURE_IMPL）: {unimplemented}"
        )
    return df
