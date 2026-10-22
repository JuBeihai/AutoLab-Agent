"""Synthetic example schema. No production database adapter is distributed."""
import pandas as pd


def build_spark_session(*args, **kwargs):
    raise RuntimeError("演示版未提供数据库连接，请使用 --mock 合成数据模式")


def stop_spark_session(spark):
    if spark is not None:
        spark.stop()


def load_hive_tables(*args, **kwargs):
    raise RuntimeError("演示版未提供生产数据读取接口，请使用 --mock")


_SDS_FEATURE_FIELDS = [
    ("feature_group", "INT"),
    ("has_queue", "BOOLEAN"),
    ("plan_duration", "DOUBLE"),
    ("plan_distance", "DOUBLE"),
    ("plan_duration_rank", "DOUBLE"),
    ("plan_duration_rand", "DOUBLE"),
    ("plan_distance_rank", "DOUBLE"),
    ("plan_distance_rand", "DOUBLE"),
    ("valid_idle_vehicles_nums", "DOUBLE"),
    ("valid_vehiclenum", "DOUBLE"),
    ("valid_work_vehicles_nums", "DOUBLE"),
    ("valid_remainingservicetime_mean", "DOUBLE"),
    ("city_rank", "DOUBLE"),
    ("vehicle_num", "DOUBLE"),
    ("grid_rank", "DOUBLE"),
    ("sp1_queue_position", "DOUBLE"),
    ("sp1_queue_count", "DOUBLE"),
    ("sp2_queue_position", "DOUBLE"),
    ("sp2_queue_count", "DOUBLE"),
    ("demand_supply", "DOUBLE"),
    ("sp1_demand_supply", "DOUBLE"),
    ("sp2_demand_supply", "DOUBLE"),
    ("cancel_rate_before_match", "DOUBLE"),
    ("proportion_of_cancel_after_match_and_call", "DOUBLE"),
    ("median_of_match_wait", "DOUBLE"),
    ("median_of_wait_time_of_cancel_before_match", "DOUBLE"),
    ("median_of_all_wait_time_of_cancel_after_match", "DOUBLE"),
    ("median_of_wait_time_of_cancel_after_match", "DOUBLE"),
    ("median_of_pick_up_duration", "DOUBLE"),
    ("median_of_trip_duration", "DOUBLE"),
    ("median_of_pick_up_distance", "DOUBLE"),
    ("median_of_trip_distance", "DOUBLE"),
    ("hour", "DOUBLE"),
    ("is_weekday", "DOUBLE"),
    ("is_night_peak", "DOUBLE"),
    ("is_morning_peak", "DOUBLE"),
    ("is_flat", "DOUBLE"),
    ("is_peak", "DOUBLE"),
    ("weekday_0", "DOUBLE"),
    ("weekday_1", "DOUBLE"),
    ("weekday_2", "DOUBLE"),
    ("weekday_3", "DOUBLE"),
    ("weekday_4", "DOUBLE"),
    ("weekday_5", "DOUBLE"),
    ("weekday_6", "DOUBLE"),
]


def merge_all(order_df: pd.DataFrame, feature_df: pd.DataFrame,
              queue_df: pd.DataFrame) -> pd.DataFrame:
    """对齐 example Notebook 的三次 left join。"""
    # 避免特征表里的 plan_duration/plan_distance 覆盖订单表字段
    drop_cols = ["plan_duration", "plan_distance"]
    order_slim = order_df.drop(columns=[c for c in drop_cols if c in order_df.columns],
                               errors="ignore")

    merged = order_slim.merge(feature_df, how="left", on=["dt", "order_id", "city_code"])
    merged = merged.merge(queue_df, how="left", on=["dt", "order_id"])
    return merged
