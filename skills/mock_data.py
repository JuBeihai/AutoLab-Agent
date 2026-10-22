"""Mock 数据发生器 —— 用于在没有 Hive 权限时演示 Agent 完整循环。

生成的 DataFrame schema 与 skills.data_loader.merge_all() 输出完全一致，
下游 skills（feature_builder / trainer / evaluator / isotonic_h3）无需改动。

设计思路：
- 让 hour_x_peak 和 supply-demand 特征真的对 label 有信号（否则 Agent 永远拿不到 gain）
- 3 个虚构区域均匀采样，与业务数据无关
- 44 特征分布近似正态/对数正态 + 部分类别独热
- 随机种子固定，多次跑同一份 mock 数据可重复
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd


def _sample_hour(rng: np.random.Generator, n: int) -> np.ndarray:
    """按人工设定的示例小时分布采样：早高峰 8-9，晚高峰 18-20 权重更高。"""
    # 权重覆盖 5-23 点
    hours = np.arange(5, 24)
    weights = np.array([1, 1.5, 3, 5, 4, 2.5, 2, 2, 2.5, 3, 3.5, 4,
                        4.5, 6, 7, 5, 4, 3, 2])
    weights = weights / weights.sum()
    return rng.choice(hours, size=n, p=weights)


def _sample_city(rng: np.random.Generator, n: int) -> np.ndarray:
    """三个虚构区域等概率采样。"""
    return rng.choice([100001, 100002, 100003], size=n, p=[1/3, 1/3, 1/3])


def _sample_start_latlng(city_codes: np.ndarray, rng: np.random.Generator):
    """按城市中心 ± 抖动生成经纬度。"""
    centers = {
        100001: (0.0, 0.0),  # 示例区域A
        100002: (10.0, 10.0),  # 示例区域B
        100003: (-10.0, -10.0),  # 示例区域C
    }
    lats = np.zeros(len(city_codes))
    lngs = np.zeros(len(city_codes))
    for city, (lat0, lng0) in centers.items():
        mask = city_codes == city
        lats[mask] = lat0 + rng.normal(0, 0.08, mask.sum())
        lngs[mask] = lng0 + rng.normal(0, 0.10, mask.sum())
    return lats, lngs


def _generate_true_ets_seconds(
    hour: np.ndarray,
    city: np.ndarray,
    supply_demand: np.ndarray,
    sp1_queue_count: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """让 hour × 峰期 + queue + supply/demand 真的对 label 有影响，
    这样 Agent 提议的 hour_x_peak / queue_x_grid_rank 才能带来真实增益。"""
    n = len(hour)
    base_log = 3.5  # log(exp(3.5)) ≈ 33 秒

    # 峰期加成（工作日 morning/evening 加倍）
    peak_boost = np.where((hour >= 7) & (hour < 10), 0.8, 0.0)
    peak_boost += np.where(hour >= 18, 1.0, 0.0)

    # 城市差异：示例区域A最慢，示例区域B最快
    city_boost = np.where(city == 100001, 0.6,
                np.where(city == 100003, 0.3, 0.0))

    # 排队影响
    queue_boost = np.log1p(sp1_queue_count) * 0.15

    # 供需失衡影响（demand/supply 越高等待越久）
    sd_boost = np.log1p(supply_demand) * 0.3

    # 高斯噪声
    noise = rng.normal(0, 0.6, n)

    log_ets = base_log + peak_boost + city_boost + queue_boost + sd_boost + noise
    ets_sec = np.exp(log_ets).clip(1, 7200)  # 1s - 2h
    # Synthetic long-wait tail exercises all six classes in smoke training.
    long_wait = rng.random(n) < .06
    ets_sec[long_wait] = rng.uniform(1201, 3600, long_wait.sum())
    return ets_sec


def generate_mock_merged(
    start_date: str = "2024-08-05",
    end_date: str = "2024-08-19",
    n_per_day: int = 2800,
    seed: int = 42,
) -> pd.DataFrame:
    """生成 merge_all() 输出格式的 DataFrame。

    Returns
    -------
    DataFrame with columns matching real data_loader.merge_all() output:
        订单主表列: order_id, city_code, city_name, create_time, dt,
                    match_time, cancel_time, tick_service_product_code,
                    start_lat, start_lng, code_hash
        SDS 特征列: 44 项（feature_group, has_queue, plan_*, valid_*,
                    city_rank, vehicle_num, grid_rank, sp*_queue_*,
                    demand_supply, cancel_rate_*, median_*,
                    hour, is_weekday, is_*_peak, weekday_0..6）
        排队日志列: ert_result, origin_ert_result
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")

    frames = []
    for d in dates:
        n = n_per_day + rng.integers(-200, 200)
        dt_str = d.strftime("%Y-%m-%d")

        city = _sample_city(rng, n)
        hour = _sample_hour(rng, n)
        minute = rng.integers(0, 60, n)
        second = rng.integers(0, 60, n)
        start_lat, start_lng = _sample_start_latlng(city, rng)

        # 时间字段
        create_times = [
            d + timedelta(hours=int(h), minutes=int(m), seconds=int(s))
            for h, m, s in zip(hour, minute, second)
        ]

        # 44 SDS 特征：以有意义的分布生成，避免全零/全 NaN
        code_hash = rng.choice([0, 1, 2], size=n, p=[0.3, 0.5, 0.2])
        weekday = np.array([t.weekday() for t in create_times])
        is_weekday = (weekday <= 4).astype(float)

        is_morning_peak = ((hour >= 7) & (hour < 10)).astype(float) * is_weekday
        is_night_peak = (hour >= 18).astype(float) * is_weekday
        is_peak = ((hour >= 10) & (hour <= 22)).astype(float) * (1 - is_weekday)
        is_flat = 1 - is_morning_peak - is_night_peak - is_peak

        # 供给/排队特征
        supply_demand = rng.lognormal(0.0, 0.5, n)  # 中位 1.0
        sp1_queue_count = rng.poisson(3, n).astype(float)
        sp2_queue_count = rng.poisson(2, n).astype(float)
        grid_rank = rng.integers(1, 20, n).astype(float)

        # 生成真实 等待时间（依赖 hour/city/queue/supply-demand）
        true_ets_sec = _generate_true_ets_seconds(
            hour, city, supply_demand, sp1_queue_count, rng)

        # 应答/取消：真实 等待时间 越长越可能取消
        cancel_prob = np.clip((true_ets_sec / 60 - 5) / 20, 0.05, 0.4)
        is_cancel = rng.random(n) < cancel_prob
        match_times = [
            (ct + timedelta(seconds=float(e))) if not c else None
            for ct, e, c in zip(create_times, true_ets_sec, is_cancel)
        ]
        cancel_times = [
            (ct + timedelta(seconds=float(e))) if c else None
            for ct, e, c in zip(create_times, true_ets_sec, is_cancel)
        ]

        # 规则模型（base_pred_bin）：真实 等待时间 加噪声 + 系统性高估
        base_pred_sec = true_ets_sec * rng.lognormal(0.1, 0.4, n)  # 中位偏高一点
        # origin_ert_result 是"规则先验" 的秒数（未保序）
        origin_ert_sec = base_pred_sec
        ert_sec = base_pred_sec * rng.lognormal(0, 0.15, n)  # 保序后微调

        # 星期独热
        weekday_oh = np.zeros((n, 7))
        weekday_oh[np.arange(n), weekday] = 1

        df = pd.DataFrame({
            # ---- 订单主表 ----
            "order_id": [f"{d.strftime('%Y%m%d')}_{i:06d}" for i in range(n)],
            "city_code": city.astype(int),
            "city_name": np.where(city == 100001, "示例区域A市",
                         np.where(city == 100002, "示例区域B市", "示例区域C市")),
            "create_time": create_times,
            "dt": dt_str,
            "match_time": match_times,
            "cancel_time": cancel_times,
            "tick_service_product_code": np.where(code_hash == 0, "PRODUCT_A,PRODUCT_B",
                                          np.where(code_hash == 1, "PRODUCT_B", "PRODUCT_A")),
            "start_lat": start_lat,
            "start_lng": start_lng,
            "code_hash": code_hash.astype(float),

            # ---- SDS 44 特征 ----
            "feature_group": rng.integers(0, 3, n),
            "has_queue": rng.random(n) < 0.6,
            "plan_duration": rng.lognormal(6.8, 0.5, n),  # 秒
            "plan_distance": rng.lognormal(8.2, 0.6, n),  # 米
            "plan_duration_rank": rng.random(n),
            "plan_duration_rand": rng.random(n),
            "plan_distance_rank": rng.random(n),
            "plan_distance_rand": rng.random(n),
            "valid_idle_vehicles_nums": rng.poisson(15, n).astype(float),
            "valid_vehiclenum": rng.poisson(25, n).astype(float),
            "valid_work_vehicles_nums": rng.poisson(10, n).astype(float),
            "valid_remainingservicetime_mean": rng.lognormal(6, 0.4, n),
            "city_rank": rng.integers(1, 100, n).astype(float),
            "vehicle_num": rng.poisson(20, n).astype(float),
            "grid_rank": grid_rank,
            "sp1_queue_position": rng.integers(0, 10, n).astype(float),
            "sp1_queue_count": sp1_queue_count,
            "sp2_queue_position": rng.integers(0, 8, n).astype(float),
            "sp2_queue_count": sp2_queue_count,
            "demand_supply": supply_demand,
            "sp1_demand_supply": supply_demand * rng.lognormal(0, 0.2, n),
            "sp2_demand_supply": supply_demand * rng.lognormal(0, 0.2, n),
            "cancel_rate_before_match": rng.beta(2, 8, n),
            "proportion_of_cancel_after_match_and_call": rng.beta(1, 15, n),
            "median_of_match_wait": rng.lognormal(2.5, 0.8, n),
            "median_of_wait_time_of_cancel_before_match": rng.lognormal(4.2, 0.7, n),
            "median_of_all_wait_time_of_cancel_after_match": rng.lognormal(4.5, 0.6, n),
            "median_of_wait_time_of_cancel_after_match": rng.lognormal(3.5, 0.6, n),
            "median_of_pick_up_duration": rng.lognormal(6.0, 0.4, n),
            "median_of_trip_duration": rng.lognormal(6.5, 0.5, n),
            "median_of_pick_up_distance": rng.lognormal(7.8, 0.5, n),
            "median_of_trip_distance": rng.lognormal(8.5, 0.6, n),
            "hour": hour.astype(float),
            "is_weekday": is_weekday,
            "is_night_peak": is_night_peak,
            "is_morning_peak": is_morning_peak,
            "is_flat": is_flat,
            "is_peak": is_peak,
            **{f"weekday_{i}": weekday_oh[:, i] for i in range(7)},

            # ---- 排队日志 ----
            "ert_result": ert_sec,
            "origin_ert_result": origin_ert_sec,
        })
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    print(f"[mock] 生成 {len(all_df)} 行，跨 {len(dates)} 天，"
          f"城市分布: {all_df['city_code'].value_counts().to_dict()}")
    return all_df
