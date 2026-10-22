"""Bounded preprocessing recipes fitted without evaluation labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

KINDS = {'missing_indicator', 'impute', 'one_hot', 'target_encoding'}


def validate(recipe, catalog):
    kind = recipe['kind']
    common = {'name', 'kind'}
    if kind in {'missing_indicator', 'impute'}:
        expected = common | {'column'} | ({'strategy', 'value'} if kind == 'impute' else set())
        if set(recipe) != expected or catalog.get(recipe.get('column'), {}).get('role') != 'numeric':
            raise ValueError('缺失值配方只能读取已批准的数值字段')
        if kind == 'impute':
            if recipe['strategy'] not in {'median', 'constant'}:
                raise ValueError('填补策略只能是 median 或 constant')
            value = recipe['value']
            if recipe['strategy'] == 'median' and value is not None:
                raise ValueError('median 的 value 必须为 null，由实际训练行拟合')
            if recipe['strategy'] == 'median' and catalog[recipe['column']].get('target_derived'):
                raise ValueError('标签衍生特征不能用跨日期训练中位数回填；请保留NaN或用常数，避免未来标签间接泄漏')
            if recipe['strategy'] == 'constant' and (
                    type(value) not in (int, float) or not np.isfinite(value) or abs(value) > 1e6):
                raise ValueError('填补常数必须有限且绝对值不超过 1e6')
        return
    expected = common | {'keys'} | ({'values'} if kind == 'one_hot' else {'smoothing', 'target'})
    if set(recipe) != expected:
        raise ValueError('类别编码字段不符合协议')
    keys = recipe['keys']
    if (not isinstance(keys, list) or not 1 <= len(keys) <= 3
            or any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys)
            or any(not catalog.get(key, {}).get('groupable') for key in keys)):
        raise ValueError('编码只允许 1–3 个已批准分组字段，可组合业务交叉类别')
    if kind == 'one_hot':
        values = recipe['values']
        if (not isinstance(values, list) or len(values) != len(keys)
                or any(type(value) not in (str, int, float, bool) for value in values)
                or any(isinstance(value, str) and len(value) > 100 for value in values)
                or any(type(value) in (int, float) and not np.isfinite(value) for value in values)):
            raise ValueError('one_hot values 必须与 keys 一一对应且为有限标量')
    else:
        smoothing, target = recipe['smoothing'], recipe['target']
        if type(smoothing) not in (int, float) or not np.isfinite(smoothing) or not 1 <= smoothing <= 1000:
            raise ValueError('目标编码平滑强度必须在 1–1000')
        if target != 'mean_bin' and (type(target) is not int or not 0 <= target <= 5):
            raise ValueError('target 只能是 mean_bin 或 0–5 的档位概率')


def transform(recipe, frame, fitting_index, label_cutoff, state):
    if fitting_index is None:
        raise ValueError('预处理必须显式指定实际拟合行，禁止在全量/测试行上拟合')
    fitting = frame.loc[fitting_index]
    if fitting.empty:
        raise ValueError('预处理的实际拟合行为空')
    kind = recipe['kind']
    entry = {'kind': kind, 'fit_rows': len(fitting)}
    state[recipe['name']] = entry
    if kind in {'missing_indicator', 'impute'}:
        entry['column'] = recipe['column']
        values = pd.to_numeric(frame[recipe['column']], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if kind == 'missing_indicator':
            return values.isna().astype(float)
        observed = values.loc[fitting_index].dropna()
        fill = recipe['value'] if recipe['strategy'] == 'constant' else observed.median() if len(observed) else np.nan
        if not np.isfinite(fill):
            raise ValueError('实际拟合行中没有可用中位数，不能从测试集填补')
        entry['fill_value'] = float(fill)
        return values.fillna(fill)
    keys = recipe['keys']
    if kind == 'one_hot':
        mask = pd.Series(True, index=frame.index)
        for key, value in zip(keys, recipe['values']):
            mask &= frame[key].eq(value).fillna(False)
        if not mask.loc[fitting_index].any():
            raise ValueError('独热类别组合未在实际拟合行中出现')
        entry.update(keys=keys, values=recipe['values'], unknown_policy='all_zero')
        return mask.astype(float)
    days = pd.to_datetime(frame['prediction_time']).dt.normalize()
    ready = pd.to_datetime(fitting['match_time']).combine_first(pd.to_datetime(fitting['cancel_time']))
    cutoff = pd.Timestamp(label_cutoff).normalize() + pd.Timedelta(days=1)
    labels = pd.to_numeric(fitting['true_bin'], errors='coerce')
    target = labels if recipe['target'] == 'mean_bin' else labels.eq(recipe['target']).where(labels.notna()).astype(float)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    daily = []
    mapping = []
    final_prior = None
    for day in sorted(days.dropna().unique()):
        day = pd.Timestamp(day)
        eligible = (days.loc[fitting_index] < day) & (days.loc[fitting_index] < cutoff) & (ready < min(day, cutoff))
        pool = fitting.loc[eligible, keys].copy()
        pool['_encoding_target'] = target.loc[eligible]
        pool = pool.dropna(subset=['_encoding_target'])
        if pool.empty:
            continue
        prior = float(pool['_encoding_target'].mean())
        stats = pool.groupby(keys, dropna=True)['_encoding_target'].agg(['sum', 'count'])
        stats['_encoded'] = (stats['sum'] + recipe['smoothing'] * prior) / (stats['count'] + recipe['smoothing'])
        selected = frame.index[days == day]
        joined = frame.loc[selected, keys].merge(stats[['_encoded']].reset_index(), on=keys,
                                                how='left', sort=False, validate='many_to_one')
        output.loc[selected] = joined['_encoded'].fillna(prior).to_numpy()
        daily.append({'date': str(day.date()), 'eligible_rows': len(pool), 'prior': prior})
        mapping = stats[['_encoded']].reset_index().to_dict('records')
        final_prior = prior
    entry.update(keys=keys, target=recipe['target'], smoothing=recipe['smoothing'],
                 label_cutoff=label_cutoff, policy='strict_prior_day_mature_fitting_labels_only',
                 daily_pools=daily, mapping=mapping, prior=final_prior,
                 inference_available_from=daily[-1]['date'] if daily else None,
                 cold_start='NaN; unseen group uses eligible past prior only')
    return output
