"""Aggregate-only diagnostics; never used to alter training rows or features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skills import evaluator, mining_train
from skills.evaluation_policy import queue_mask

def fitting_rows(frame, is_queue, config):
    if config.get('training_method') == 'notebook_random':
        return mining_train.notebook_split(frame, is_queue, config)[0]
    return frame.loc[mining_train.fitting_mask(frame, is_queue, config)]


def field_stats(series):
    if pd.api.types.is_numeric_dtype(series):
        series = series.replace([np.inf, -np.inf], np.nan)
    return {'rows': len(series), 'missing_rate': float(series.isna().mean()) if len(series) else None,
            'unique': int(series.nunique())}


def fitting_profiles(frame, catalog, config):
    profiles = {}
    for tag, is_queue in ((('unified', True),) if config.get('non_queue_strategy') == 'unified' else (('queue', True), ('no_queue', False))):
        if not is_queue and config.get('non_queue_strategy') == 'zero':
            continue
        fitting = fitting_rows(frame, is_queue, config)
        profiles[tag] = {'rows': len(fitting), 'fields': {
            name: field_stats(fitting[name]) for name in catalog if name in fitting}}
    return profiles


def feature_coverage(training, testing, names, config):
    fitting = fitting_profiles(training, names, config)
    result = {}
    for name in names:
        test_stats = field_stats(testing[name])
        by_date = {str(day): field_stats(group[name]) for day, group in testing.groupby('dt')}
        by_city = {str(city): field_stats(group[name]) for city, group in testing.groupby('city_code')}
        by_city_date = [{'city': str(city), 'date': str(day), **field_stats(group[name])}
                        for (city, day), group in testing.groupby(['city_code', 'dt'])]
        fit_stats = {tag: profile['fields'][name] for tag, profile in fitting.items()}
        warnings = []
        if any(stats['unique'] < 2 for stats in fit_stats.values()):
            warnings.append('实际拟合人群中全缺失或恒定')
        if any((stats['missing_rate'] or 0) > .95 for stats in fit_stats.values()):
            warnings.append('实际拟合人群缺失率超过95%')
        if any(stats['missing_rate'] == 1 for stats in by_date.values()):
            warnings.append('测试期存在整日缺失')
        result[name] = {'training': field_stats(training[name]), 'fitting': fit_stats,
                        'test': test_stats, 'test_by_date': by_date, 'test_by_city': by_city,
                        'test_by_city_date': by_city_date, 'warnings': warnings}
    return result


def model_usage(models, names):
    result = {}
    for tag in (('unified',) if 'unified' in models else ('queue', 'no_queue')):
        model = models.get(tag)
        if model is None:
            result[tag] = {'available': False, 'reason': '未训练该模型或无诊断接口'}
            continue
        if not hasattr(model, 'get_score'):
            result[tag] = {'available': False, 'reason': '模型未提供分裂统计接口'}
            continue
        weight, gain = model.get_score(importance_type='weight'), model.get_score(importance_type='gain')
        result[tag] = {'available': True, 'features': {
            name: {'split_count': int(weight.get(name, weight.get(f'f{position}', 0))),
                   'gain': float(gain.get(name, gain.get(f'f{position}', 0)))}
            for position, name in enumerate(names)}}
    return result

def feature_selection_report(training, models, columns, config, budget):
    report = {'scope': 'actual_fitting_rows_only',
              'note': '训练内重要性不是独立泛化证据；相关特征可互相替代。低重要性只建议消融，不自动删除。',
              'models': {}}
    for tag, is_queue in ((('unified', True),) if config.get('non_queue_strategy') == 'unified' else (('queue', True), ('no_queue', False))):
        model = models.get(tag)
        if model is None or not hasattr(model, 'get_score'):
            continue
        fitting = fitting_rows(training, is_queue, config)
        if fitting.empty:
            continue
        gains = model.get_score(importance_type='total_gain')
        weights = model.get_score(importance_type='weight')
        ranking = []
        for position, name in enumerate(columns):
            stats = field_stats(fitting[name])
            ranking.append({'name': name, **stats,
                            'total_gain': float(gains.get(name, gains.get(f'f{position}', 0))),
                            'split_count': int(weights.get(name, weights.get(f'f{position}', 0)))})
        total = sum(entry['total_gain'] for entry in ranking)
        for entry in ranking:
            entry['gain_share'] = entry['total_gain'] / total if total else 0.
        ranking.sort(key=lambda entry: (-entry['total_gain'], entry['name']))
        sampled = fitting.sample(n=min(len(fitting), 5000), random_state=123)
        correlation = sampled[columns].corr(method='spearman')
        redundant = []
        for position, first in enumerate(columns):
            for second in columns[position + 1:]:
                value = correlation.loc[first, second]
                if np.isfinite(value) and abs(value) >= .95:
                    redundant.append({'first': first, 'second': second, 'spearman': float(value)})
        redundant.sort(key=lambda entry: -abs(entry['spearman']))
        count = max(0, min(int(budget.get('importance_features', 6)), 12))
        importance = []
        if count and hasattr(model, 'predict'):
            import xgboost as xgb
            sample = fitting.sample(n=min(len(fitting), max(1, min(int(budget.get('importance_rows', 512)), 2000))),
                                    random_state=123)
            matrix = sample[columns].copy()
            labels = sample['true_bin'].to_numpy()
            def accuracy(values):
                prediction = model.predict(xgb.DMatrix(values, nthread=4))
                if prediction.ndim == 2:
                    prediction = prediction.argmax(axis=1)
                return float(np.mean(prediction == labels))
            baseline = accuracy(matrix)
            selected = list(dict.fromkeys([entry['name'] for entry in ranking[:2] + list(reversed(ranking))]))[:count]
            random = np.random.RandomState(123)
            for name in selected:
                original = matrix[name].copy()
                drops = []
                for repeat in range(2):
                    matrix[name] = random.permutation(original.to_numpy())
                    drops.append((baseline - accuracy(matrix)) * 100)
                matrix[name] = original
                importance.append({'name': name, 'mean_acc_drop_pp': float(np.mean(drops)),
                                   'min_acc_drop_pp': min(drops), 'max_acc_drop_pp': max(drops)})
        report['models'][tag] = {'fitting_rows': len(fitting), 'ranking': ranking,
                                'ablation_candidates': [entry['name'] for entry in reversed(ranking)][:8],
                                'high_correlation_pairs': redundant[:20],
                                'correlation_sample_rows': len(sampled),
                                'permutation': importance, 'permutation_scope': 'in_sample_not_holdout',
                                'permutation_rows': len(sample) if importance else 0, 'permutation_repeats': 2}
    return report


def accuracy_slices(predictions):
    frame = predictions.assign(queue_status=np.where(queue_mask(predictions), 'queue', 'no_queue'))
    columns = [name for name in ('city_code', 'hour', 'is_weekday', 'base_pred_bin', 'is_answer',
                                'queue_status', 'dt') if name in frame]
    slices = evaluator.error_slices(frame, pred_col='predict_bin_44', slice_cols=columns, min_count=1)
    errors = int((frame['true_bin'] != frame['predict_bin_44']).sum())
    if len(slices):
        slices['errors'] = (slices['n'] * (1 - slices['acc'])).round().astype(int)
        slices['error_share'] = slices['errors'] / errors if errors else 0.
        slices['error_contribution_pp'] = slices['errors'] / len(frame) * 100
        slices = slices.sort_values(['errors', 'n'], ascending=False, kind='stable').reset_index(drop=True)
    return slices


def accuracy_feedback(predictions):
    errors = predictions['true_bin'] != predictions['predict_bin_44']
    queue = queue_mask(predictions)
    transitions = (predictions.loc[errors].groupby(['true_bin', 'predict_bin_44']).size()
                   .rename('errors').reset_index().sort_values('errors', ascending=False).head(20))
    return {'total': len(predictions), 'errors': int(errors.sum()),
            'queue_rows': int(queue.sum()), 'queue_errors': int((queue & errors).sum()),
            'non_queue_errors': int((~queue & errors).sum()),
            'slices': accuracy_slices(predictions).head(15).to_dict('records'),
            'queue_slices': accuracy_slices(predictions.loc[queue]).head(15).to_dict('records'),
            'confusions': transitions.to_dict('records'),
            'note': '切片互相重叠，错误占比不可跨维度相加；非排队固定为0时其保序前错误不受新增XGB特征影响。'}


def prediction_changes(reference, candidate):
    if not reference.index.equals(candidate.index) or len(reference) != len(candidate):
        raise ValueError('预测对照必须保持相同样本顺序')
    for name in ('order_id', 'dt', 'true_bin'):
        if name in reference and (name not in candidate or not reference[name].equals(candidate[name])):
            raise ValueError('预测对照样本或标签不一致')
    old_correct = reference['predict_bin_44'].to_numpy() == reference['true_bin'].to_numpy()
    new_correct = candidate['predict_bin_44'].to_numpy() == candidate['true_bin'].to_numpy()
    corrected, introduced = ~old_correct & new_correct, old_correct & ~new_correct
    details = candidate.assign(_corrected=corrected, _introduced=introduced,
                               queue_status=np.where(queue_mask(candidate), 'queue', 'no_queue'))
    slices = []
    for name in ('city_code', 'dt', 'queue_status', 'is_answer'):
        if name not in details:
            continue
        for value, group in details.groupby(name):
            fixed, broken = int(group['_corrected'].sum()), int(group['_introduced'].sum())
            slices.append({'dimension': name, 'value': str(value), 'rows': len(group),
                           'corrected': fixed, 'introduced': broken, 'net_corrected': fixed - broken})
    return {'corrected': int(corrected.sum()), 'introduced': int(introduced.sum()),
            'net_corrected': int(corrected.sum() - introduced.sum()),
            'delta_pp': float((corrected.sum() - introduced.sum()) / len(candidate) * 100) if len(candidate) else 0.,
            'by_slice': slices}


def compact_features(coverage, usage, names):
    summary = {}
    for name in names:
        entry = coverage[name]
        summary[name] = {'fitting': entry['fitting'], 'test': entry['test'],
                         'warnings': entry['warnings'],
                         'fully_missing_dates': [day for day, stats in entry['test_by_date'].items()
                                                 if stats['missing_rate'] == 1],
                         'model_usage': {tag: info.get('features', {}).get(name)
                                         for tag, info in usage.items()}}
    return summary
