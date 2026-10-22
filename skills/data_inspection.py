"""Read-only, allowlisted aggregate inspection; never alters experiment rows."""
from __future__ import annotations

import re
import threading
import uuid

import numpy as np
import pandas as pd

from skills import data_loader

IDENTIFIER = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
TABLE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$')
SENSITIVE = re.compile(r'(^|_)(id|uid|pin|phone|mobile|name|address|token|secret|password|key|lat|lng|lon|latitude|longitude|user|account|email|device|imei|idfa|ip|vin)(_|$)', re.I)
OUTCOME = re.compile(r'(label|target|true_|match_time|cancel_time|finish_time|actual|order_id|driver|passenger)', re.I)


def inspectable(field):
    name = field.get('name', '')
    dtype = field.get('type', '').lower()
    normalized = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    return bool(IDENTIFIER.fullmatch(name) and not SENSITIVE.search(normalized) and not OUTCOME.search(normalized)
                and re.fullmatch(r'(tinyint|smallint|int|bigint|float|double|boolean|decimal\(\d+,\d+\))', dtype))


def capabilities(audit, cfg):
    if audit.get('data_source') == 'offline':
        return {'operations': ['fields'], 'max_calls': max(0, min(cfg.get('max_inspection_calls', 4), 8)),
                'max_columns': 8, 'row_limit': cfg.get('inspection_rows', 20000), 'tables': [],
                'policy': '离线数据模式仅探查已加载的训练期字段；不访问 Spark 业务表'}
    allowed = set(cfg['reference_tables'])
    return {'operations': ['fields', 'table'], 'max_calls': max(0, min(cfg.get('max_inspection_calls', 4), 8)),
            'max_columns': 8, 'row_limit': cfg.get('inspection_rows', 20000),
            'tables': [{'table': source['table'],
                        'numeric_columns': [field['name'] for field in source.get('fields', []) if inspectable(field)]}
                       for source in audit.get('sources', [])
                       if source.get('table') in allowed and source.get('status') == 'schema_read'
                       and any(field['name'] == 'dt' for field in source.get('fields', []))],
            'policy': '训练日期内只读聚合；禁止SQL/明细/敏感字段；探查不等于允许关联进模型'}


def validate(request, catalog, audit, cfg):
    if not isinstance(request, dict) or request.get('kind') not in {'fields', 'table'}:
        raise ValueError('inspect.kind 只能是 fields 或 table')
    if request['kind'] == 'table' and audit.get('data_source') == 'offline':
        raise ValueError('离线数据模式禁止访问 Spark 业务表；可探查已加载字段')
    expected = {'kind', 'columns'} | ({'table'} if request['kind'] == 'table' else set())
    if set(request) != expected:
        raise ValueError('探查只接受结构化字段，不接受 SQL、筛选条件或自选日期')
    columns = request['columns']
    if (not isinstance(columns, list) or not 1 <= len(columns) <= 8
            or any(not isinstance(name, str) or not IDENTIFIER.fullmatch(name) for name in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError('探查需要 1–8 个不同的合法字段')
    if request['kind'] == 'fields':
        if any(name not in catalog for name in columns):
            raise ValueError('只允许探查已批准的输入字段，不允许标签或订单标识')
        return
    table = request['table']
    if not isinstance(table, str) or not TABLE.fullmatch(table):
        raise ValueError('非法表名')
    permitted = {entry['table']: entry['numeric_columns'] for entry in capabilities(audit, cfg)['tables']}
    if table not in permitted or set(columns) - set(permitted[table]):
        raise ValueError('表或字段未在只读探查白名单中')
    source = next(source for source in audit['sources'] if source['table'] == table)
    if not any(field['name'] == 'dt' for field in source.get('fields', [])):
        raise ValueError('表没有 dt 分区，不能执行有界日期探查')


def summarize(frame, columns):
    fields = {}
    numeric = {}
    for name in columns:
        values = frame[name]
        entry = {'dtype': str(values.dtype), 'rows': len(values), 'unique': int(values.nunique()),
                 'missing_rate': float(values.isna().mean()) if len(values) else None}
        if pd.api.types.is_numeric_dtype(values):
            values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
            entry['missing_rate'] = float(values.isna().mean()) if len(values) else None
            entry['unique'] = int(values.nunique())
            entry['quantiles'] = {str(quantile): float(value) if pd.notna(value) else None
                                  for quantile, value in values.quantile([0, .05, .5, .95, 1]).items()}
            numeric[name] = values
        fields[name] = entry
    correlations = []
    if numeric:
        correlation = pd.DataFrame(numeric).corr(method='spearman')
        for position, first in enumerate(correlation.columns):
            for second in correlation.columns[position + 1:]:
                value = correlation.loc[first, second]
                if np.isfinite(value):
                    correlations.append({'first': first, 'second': second, 'spearman': float(value)})
    return {'fields': fields, 'correlations': correlations}


def execute(request, frame, catalog, audit, cfg, window, mock=False):
    validate(request, catalog, audit, cfg)
    if request['kind'] == 'fields':
        training = frame.loc[frame['dt'].between(window['train_start'], window['train_end'])]
        return {'kind': 'fields', 'scope': 'joined_training_rows', 'window': window,
                **summarize(training, request['columns'])}
    if mock:
        raise ValueError('mock 会话禁止访问真实数据库')
    start = pd.Timestamp(window['train_start']).strftime('%Y-%m-%d')
    end = pd.Timestamp(window['train_end']).strftime('%Y-%m-%d')
    limit = max(1, min(int(cfg.get('inspection_rows', 20000)), 20000))
    timeout = max(1, min(int(cfg.get('inspection_timeout_seconds', 60)), 120))
    spark = data_loader.build_spark_session('ets_readonly_inspection', max_executors=2, resource_limited=True)
    timer = None
    expired = threading.Event()
    try:
        group = 'ets_inspect_' + uuid.uuid4().hex
        spark.sparkContext.setJobGroup(group, '训练期白名单表聚合探查', interruptOnCancel=True)
        def cancel():
            expired.set()
            spark.sparkContext.cancelJobGroup(group)
        timer = threading.Timer(timeout, cancel)
        timer.daemon = True
        timer.start()
        columns = ', '.join(f'`{name}`' for name in request['columns'])
        snapshot = spark.sql(f"SELECT {columns} FROM {request['table']} "
                             f"WHERE dt >= '{start}' AND dt <= '{end}' LIMIT {limit}").toPandas().copy()
        for name in request['columns']:
            snapshot[name] = pd.to_numeric(snapshot[name], errors='coerce').astype(float)
        if expired.is_set():
            raise TimeoutError('表探查超时')
        return {'kind': 'table', 'table': request['table'], 'train_start': start, 'train_end': end,
                'scope': 'bounded_prefix_not_full_population', 'row_limit': limit,
                'rows_read': len(snapshot), 'truncated_possible': len(snapshot) == limit,
                'join_status': 'not_materialized; availability and join require human review',
                **summarize(snapshot, request['columns'])}
    except Exception as error:
        raise RuntimeError(f'只读表探查未完成：{type(error).__name__}') from None
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        data_loader.stop_spark_session(spark)
