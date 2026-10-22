"""Population-first evaluation matching the XGB44 Notebook ordering workflow."""

import pandas as pd

from skills import evaluator, isotonic_h3


from skills.evaluation_policy import SELECTION_KEY, SELECTION_LABEL, queue_mask


def selection_metrics(views):
    view = views.get('populations', {}).get('answer_cancel', {}).get('h3_7', {})
    if 'metrics' not in view:
        raise ValueError('门禁所需 H3-7 保序指标不可用：' + view.get('unavailable_reason', '未保存指标'))
    return view['metrics']


def record_selection(record):
    """Respect the recorded policy; never relabel historical raw gates."""
    record = record or {}
    key = (record.get('config_diff') or {}).get('selection_metric', 'xgb_44')
    return (record.get('metrics_by_city') or {}).get(key, {})


def city_metrics(frame, pred_col):
    return {str(city): value.as_dict() for city, value in
            evaluator.evaluate_by_city(frame, pred_col=pred_col, verbose=False).items()}


def build_views(predictions, on_progress=None, non_queue_after_ordering='zero'):
    if non_queue_after_ordering not in ('zero', 'model'):
        raise ValueError('未知非排队保序策略')
    report = on_progress or (lambda message: None)
    frame = predictions.copy()
    queue_mask(frame)  # Fail closed if the required routing flag is absent.
    for column in ('true_bin', 'predict_bin_44'):
        if not frame[column].isin(range(6)).all():
            raise ValueError('6桶评估不接受其他分桶标签或缺失标签')
    populations = {'all': frame}
    if 'is_answer' in frame:
        populations['answer_only'] = frame[frame['is_answer'] == 1]
        populations['answer_cancel'] = frame[(frame['is_answer'] == 1) |
            ((frame['is_answer'] == 0) & (frame['true_bin'] >= frame['predict_bin_44']))]
    views = {}
    for population, subset in populations.items():
        versions = {'raw': {'metrics': city_metrics(subset, 'predict_bin_44')}}
        for resolution in (6, 7, 8):
            version = f'h3_{resolution}'
            grid = f'h3_res{resolution}'
            required = ['city_code', 'dt', grid, 'create_time', 'match_time', 'cancel_time']
            missing = [column for column in required if column not in subset]
            if missing:
                versions[version] = {'unavailable_reason': '缺少保序输入字段：' + ', '.join(missing)}
                continue
            if subset[['city_code', 'dt', grid, 'create_time']].isna().any().any():
                versions[version] = {'unavailable_reason': '城市、日期、格子或创建时间缺失，未计算保序结果'}
                continue
            report(f'正在计算 {population} 人群 H3-{resolution} 保序：{len(subset):,} 行')
            ordered = subset.copy()
            prediction = ordered['predict_bin_44'].to_numpy(dtype=int, copy=True)
            grouped = ordered.groupby(['city_code', 'dt', grid])['predict_bin_44'].transform('size').to_numpy() > 1
            if grouped.any():
                prediction[grouped] = isotonic_h3.apply_ordering(
                    ordered.loc[grouped], spatial_col=grid, pred_col='predict_bin_44')
            if non_queue_after_ordering == 'zero':
                prediction[(~queue_mask(ordered)).to_numpy()] = 0
            ordered['ordered_prediction'] = prediction
            versions[version] = {'metrics': city_metrics(ordered, 'ordered_prediction')}
        views[population] = versions
    return {'protocol': 'notebook_xgb43_6_unified_v1' if non_queue_after_ordering == 'model' else 'notebook_xgb43_6_zero_after_ordering_v1',
            'num_bins': 6, 'non_queue_after_ordering': non_queue_after_ordering, 'populations': views}
