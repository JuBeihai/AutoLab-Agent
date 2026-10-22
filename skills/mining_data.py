"""Read-only discovery for the synthetic waiting-time example."""
from __future__ import annotations

import gc
import json
import re

import numpy as np
import pandas as pd

from skills import data_loader, feature_builder
from skills.evaluation_policy import BIN_EDGES, CONTRACT, queue_mask

# Never expose identifiers, outcome fields or payload strings to the proposer.
DENY = re.compile(r"(^|_)(id|vid|uid|pin|phone|token|key|label|target|score|result|amount|payment|pay|actual|cancel|match_time|wait_duration|true|predict|answer)(_|$)", re.I)
TABLE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$")


def sql_literal(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_,.-]+', value):
        raise ValueError('非法 SQL 筛选值')
    return "'" + value + "'"


def sds_query(*args, **kwargs):
    raise RuntimeError("Production SQL is not distributed in this demo")


def snake(name):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def numeric_payload(raw):
    """Flatten numeric JSON leaves and numeric array summaries, without ID values."""
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {}
    if not isinstance(obj, dict):
        return {}
    out = {}

    def visit(value, path, depth):
        if depth > 4 or len(out) >= 300:
            return
        if path and (not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,65}", path) or DENY.search(snake(path))):
            return
        if isinstance(value, dict):
            for k, v in sorted(value.items()):
                if isinstance(k, str): visit(v, f"{path}_{k}" if path else k, depth + 1)
        elif isinstance(value, list):
            # E.g. vehicles[].remainingServiceTime; aggregate numeric quantities only.
            buckets = {}
            for item in value[:500]:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if isinstance(k, str) and not DENY.search(snake(k)) and type(v) in (int, float) and np.isfinite(v):
                            buckets.setdefault(k, []).append(float(v))
            for k, vals in buckets.items():
                for stat, v in (("mean", np.mean(vals)), ("min", np.min(vals)), ("max", np.max(vals)), ("count", len(vals))):
                    visit(float(v), f"{path}_{k}_{stat}", depth + 1)
        elif type(value) in (int, float, bool) or isinstance(value, str):
            try:
                v = float(value)
                if np.isfinite(v): out[path] = v
            except (ValueError, TypeError):
                pass
    visit(obj, "", 0)
    return out


def timestamp(values):
    """Support timestamp strings and epoch seconds/milliseconds in Hive logs."""
    if pd.api.types.is_datetime64_any_dtype(values): return pd.to_datetime(values)
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.to_datetime(values.where(numeric.isna()), errors="coerce", format="mixed")
    sec = numeric.notna() & (numeric.abs() < 1e11)
    ms = numeric.notna() & ~sec
    out.loc[sec] = pd.to_datetime(numeric[sec], unit="s", errors="coerce")
    out.loc[ms] = pd.to_datetime(numeric[ms], unit="ms", errors="coerce")
    return out


def merge_notebook(orders, sds, queue, cfg, *, copy=True):
    """Reproduce the notebook's date-keyed left joins, including duplicate rows."""
    if copy:
        orders, sds, queue = orders.copy(), sds.copy(), queue.copy()
    for frame in (orders, sds, queue):
        frame['dt'] = frame['dt'].astype(str)
        frame['order_id'] = frame['order_id'].astype(str)
    for frame in (orders, sds):
        frame['city_code'] = frame['city_code'].astype(int)
    orders.drop(columns=['hour', 'weekday', 'is_weekday', 'peak', 'plan_duration', 'plan_distance'], errors='ignore', inplace=True)
    queue.rename(columns={'event_time': 'queue_event_time'}, inplace=True)
    out = orders
    for right, keys in ((sds, ['dt', 'order_id', 'city_code']), (queue, ['dt', 'order_id'])):
        left_counts = out.groupby(keys, dropna=False).size().rename('left_count').reset_index()
        right_counts = right.groupby(keys, dropna=False).size().rename('right_count').reset_index()
        counts = left_counts.merge(right_counts, on=keys, how='left')
        expected = int((counts['left_count'] * counts['right_count'].fillna(1)).sum())
        if expected > cfg['max_rows']:
            raise ValueError(f'Notebook 左关联预计产生 {expected} 行，超过内存预算；请缩短日期窗口，不会去重或截断')
        del left_counts, right_counts, counts
        out = out.merge(right, on=keys, how='left')
    out['prediction_time'] = timestamp(out['event_time'])
    out.reset_index(drop=True, inplace=True)
    return out


def describe_sources(spark, cfg):
    sources = []
    for table in cfg["reference_tables"]:
        if not TABLE.fullmatch(table): raise ValueError("非法数据表名")
        try:
            rows = spark.sql(f"DESCRIBE {table}").collect()
            fields = [{"name": r[0], "type": r[1], "comment": r[2] or ""}
                      for r in rows if r[0] and not r[0].startswith("#")]
            sources.append({"table": table, "status": "schema_read", "fields": fields,
                            "materialization": "active" if table in cfg["reference_tables"][:3] else "schema_only_time_join_unverified"})
        except Exception as exc:
            sources.append({"table": table, "status": "unavailable", "error_type": type(exc).__name__})
    return sources


def city_date_counts(frame):
    counts = {}
    for (city, date), count in frame.groupby(['city_code', 'dt']).size().items():
        counts.setdefault(str(city), {})[str(date)] = int(count)
    return counts


def load_live(data_cfg, cfg, train_end, sample_rows=0, on_progress=None):
    raise RuntimeError("演示版不包含在线数据接口。请加 --mock 使用合成数据")


def prepare_raw(frames, data_cfg, cfg, train_end, audit, report):
    """Both transports use exactly the same joins and training-only discovery."""
    audit['stage_rows_by_city_date'] = {name: city_date_counts(frames[name]) for name in ('sds', 'orders')}
    report('正在按 Notebook 条件关联数据')
    merged = merge_notebook(frames['orders'], frames['sds'], frames['queue'], cfg, copy=False)
    frames.clear()
    gc.collect()
    audit['merged_rows'] = len(merged)
    audit['stage_rows_by_city_date']['notebook_left_join'] = city_date_counts(merged)
    audit['missing_sds_rows'] = int(merged['feature_map'].isna().sum())
    audit['missing_queue_rows'] = int(merged['origin_ert_result'].isna().sum())
    if cfg.get('training_city'):
        from skills.city_training import select_city
        merged = select_city(merged, cfg['training_city'])
    full, lineage = decode_features(merged, data_cfg, cfg, train_end, copy=False, on_progress=report)
    del merged
    gc.collect()
    audit['usable_rows'] = len(full)
    audit['rows_by_date'] = {str(date): int(count) for date, count in full.groupby('dt').size().items()}
    audit['rows_by_city'] = {str(city): int(count) for city, count in full.groupby('city_code').size().items()}
    audit['rows_by_queue'] = {'queue': int(queue_mask(full).sum()), 'no_queue': int((~queue_mask(full)).sum())}
    audit['stage_rows_by_city_date']['usable'] = city_date_counts(full)
    if full.empty:
        raise ValueError('Notebook 业务条件及日期窗口下无订单样本')
    return full, lineage, audit


def load_local(data_cfg, cfg, train_end, snapshot_id, *, root, sample_rows=0, on_progress=None):
    """Load only validated raw files; never initialize Spark or request a DP key."""
    from skills import local_snapshot
    from skills.runtime_memory import require_available_memory
    import pyarrow.parquet as pq
    if sample_rows:
        raise ValueError('Notebook 一致模式不允许抽样')
    report = on_progress or (lambda message: None)
    require_available_memory()
    report(f'正在校验本地 Parquet 快照：{snapshot_id}')
    folder, manifest = local_snapshot.validate(root, snapshot_id, data_cfg, verify_hashes=True)
    frames = {}
    for name in local_snapshot.NAMES:
        path = folder / (name + '.parquet')
        metadata = pq.ParquetFile(path).metadata
        cap = int(cfg['max_rows']) * (10 if name == 'queue' else 1)
        if metadata.num_rows != manifest['files'][name]['rows'] or not 0 < metadata.num_rows <= cap:
            raise ValueError(f'本地 {name} 行数不符或超过内存预算；不会抽样或截断')
        report(f'正在读取本地 {name}：{metadata.num_rows:,} 行')
        frames[name] = pd.read_parquet(path)
    audit = {'sds_rows': len(frames['sds']), 'order_rows': len(frames['orders']),
             'queue_rows': len(frames['queue']), 'data_contract': CONTRACT,
             'sample_rows_requested': 0, 'historical_population': 'notebook left-joined rows; duplicates retained',
             'point_in_time_guaranteed': False, 'sources': [], 'mode': 'real', 'data_source': 'offline',
             'snapshot_id': snapshot_id, 'snapshot_dir': str(folder),
             'snapshot_sha256': {name: manifest['files'][name]['sha256'] for name in local_snapshot.NAMES},
             'data_window': data_cfg['window'], 'feature_event_id': data_cfg['feature_filters']['event_id']}
    return prepare_raw(frames, data_cfg, cfg, train_end, audit, report)


def payload_columns(values, fields):
    columns = {name: [] for name in fields}
    if not columns:
        return columns
    for raw in values:
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            payload = {}
        for name, key in fields.items():
            columns[name].append(payload.get(key, np.nan))
    return columns


def decode_features(merged, data_cfg, cfg, train_end, *, copy=True, on_progress=None):
    out = merged.copy() if copy else merged
    report = on_progress or (lambda message: None)
    fields = {field: 'group' if field == 'feature_group' else field
              for field, _ in data_loader._SDS_FEATURE_FIELDS if field not in out}
    for name, values in payload_columns(out['feature_map'], fields).items():
        series = pd.Series(values, index=out.index)
        out[name] = (series.map(lambda value: str(value).lower() in ('true', '1'))
                     if name == 'has_queue' else pd.to_numeric(series, errors='coerce'))
    fields = {name: key for key, name in (("originErtResult", "origin_ert_result"), ("ertResult", "ert_result"))
              if name not in out}
    for name, values in payload_columns(out['json_param'], fields).items():
        out[name] = values
    lineage = {field: {"source": "sds.feature_map", "path": 'group' if field == 'feature_group' else field,
                       "availability": "notebook_date_join_time_unverified"}
               for field, _ in data_loader._SDS_FEATURE_FIELDS}
    discovery_mask = out["dt"].astype(str) <= train_end
    for source, column in (("sds", "feature_map"), ("queue", "json_param")):
        values = out[column]
        report(f'正在发现 {source} 训练期数值字段')
        seen = sorted({key for raw in values[discovery_mask].head(cfg['profile_rows']) for key in numeric_payload(raw)})
        keys = [key for key in seen[:cfg['max_discovered_fields']]
                if not (source == 'sds' and key in out) and len(f'{source}__{key}') <= 80]
        matrix = np.full((len(out), len(keys)), np.nan, dtype=float, order='F')
        if keys:
            for position, raw in enumerate(values):
                decoded = numeric_payload(raw)
                matrix[position] = [decoded.get(key, np.nan) for key in keys]
                if (position + 1) % 10000 == 0:
                    report(f'正在解析 {source} 特征：{position + 1:,}/{len(out):,} 行')
        for index, key in enumerate(keys):
            name = f'{source}__{key}'
            out[name] = matrix[:, index]
            lineage[name] = {"source": source, "path": key, "availability": "notebook_date_join_time_unverified"}
        del values, matrix
        out.drop(columns=[column], inplace=True)
        report(f'{source} 特征解析完成：{len(out):,} 行，{len(keys)} 个发现字段')
    out["code_hash"] = out["tick_service_product_code"].map(data_cfg["product_code_hash"])
    out = feature_builder.compute_true_ets(out, copy=False)
    out = feature_builder.add_h3_index(out, resolution=7, out_col='start_h3_index', copy=False)
    out = feature_builder.add_all_h3_layers(out, copy=False)
    out['true_bin'] = [notebook_bin(value) for value in out['true_ets']]
    out['base_pred_bin'] = [notebook_bin(float(value)) for value in out['origin_ert_result']]
    out['base_sorted_pred_bin'] = [notebook_bin(float(value)) for value in out['ert_result']]
    return out, lineage


def notebook_bin(seconds):
    minutes = seconds / 60
    for bucket, edge in enumerate(BIN_EDGES):
        if minutes <= edge:
            return bucket
    return len(BIN_EDGES)


def profile_frame(full, train_start, train_end, lineage, cfg):
    """Only training data enters the LLM profile; no row-level values or identifiers."""
    train = full[full["dt"].astype(str).between(train_start, train_end)]
    if len(train) > cfg["profile_rows"]:
        train = train.sample(cfg["profile_rows"], random_state=123)
    catalog = {}
    groups = {"city_code", "hour", "is_weekday", "grid_rank", "start_h3_index", "h3_res6", "h3_res7", "h3_res8", "code_hash"}
    base = {name for name, _ in data_loader._SDS_FEATURE_FIELDS} | {"code_hash"}
    for name in sorted(set(lineage) | groups | base):
        if name not in train: continue
        s = train[name]
        info = {**lineage.get(name, {"source": "derived_from_order_or_sds"}), "groupable": name in groups}
        info.update({"missing_rate": float(s.isna().mean()), "unique": int(s.nunique())})
        if pd.api.types.is_numeric_dtype(s):
            n = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            if n.nunique() < 2 or n.isna().mean() > cfg["max_missing_rate"]: continue
            info["role"] = "numeric"
            info["quantiles"] = {str(k): float(v) for k, v in n.quantile([0.05, 0.5, 0.95]).items()}
            corr = n.corr(train["true_bin"], method="spearman")
            info["train_label_spearman"] = float(corr) if np.isfinite(corr) else None
        elif name in groups:
            info["role"] = "group"
        else: continue
        if info['groupable']:
            counts = s.dropna().value_counts().head(24)
            info['training_categories'] = [{'value': value.item() if isinstance(value, np.generic) else value,
                                             'count': int(count)} for value, count in counts.items()
                                            if not isinstance(value, (float, np.floating)) or np.isfinite(value)]
            info['categories_truncated'] = int(s.nunique()) > len(counts)
        catalog[name] = info
    return {"training_rows_profiled": len(train), "train_start": train_start, "train_end": train_end,
            "catalog": catalog, "historical_target": "true_ets: training-only completed prior-day orders; never current-row input"}
