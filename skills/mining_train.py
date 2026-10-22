"""XGBClassifier training and persisted predictions, with legacy Notebook support."""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd

from skills.trainer import _build_sample_weights
from skills import evaluator, model_io
from skills.evaluation_views import SELECTION_KEY
from skills.evaluation_policy import CONTRACT, UNIFIED_CONTRACT, queue_mask


def notebook_split(df, is_queue, train_cfg):
    tag = 'queue' if is_queue else 'no_queue'
    mask = fitting_mask(df, is_queue, train_cfg)
    data = df.loc[mask]
    if train_cfg.get('training_method') == 'temporal':
        start = pd.Timestamp(train_cfg['temporal_early_stop_start'])
        end = pd.Timestamp(train_cfg['temporal_validation_start'])
        days = pd.to_datetime(data['dt'])
        ready = pd.to_datetime(data['match_time']).combine_first(pd.to_datetime(data['cancel_time']))
        fitting = data.loc[(days < start) & (ready < start)]
        stopping = data.loc[(days >= start) & (days < end) & (ready < end)]
        if fitting.empty or stopping.empty:
            raise ValueError('时间拆分的拟合/早停人群为空')
        if set(fitting['order_id'].astype(str)) & set(stopping['order_id'].astype(str)):
            raise ValueError('同一订单跨越拟合与早停分区')
        return fitting, stopping
    random = np.random.RandomState(train_cfg['split']['seed'])
    ratio = train_cfg['split'][f'{tag}_eval_ratio']
    split_mask = random.random_sample(len(data)) < (1 - ratio)
    fitting, stopping = data.loc[split_mask], data.loc[~split_mask]
    if fitting.empty or stopping.empty:
        raise ValueError(f'{tag} Notebook 随机拆分后训练集或早停集为空，请扩大日期窗口')
    return fitting, stopping


def early_stop_rows(df, train_cfg):
    partitions = [notebook_split(df, True, train_cfg)[1]]
    if train_cfg.get('non_queue_strategy', 'model') == 'model':
        partitions.append(notebook_split(df, False, train_cfg)[1])
    return pd.concat(partitions)


def fitting_mask(df, is_queue, train_cfg):
    if train_cfg.get('non_queue_strategy') == 'unified':
        return df['is_answer'].eq(1)
    tag = 'queue' if is_queue else 'no_queue'
    mask = queue_mask(df, train_cfg.get('queue_rule', 'has_queue'))
    if not is_queue:
        mask = ~mask
    if train_cfg['training_filters'].get(tag, {}).get('require_is_answer', False):
        mask = mask & (df['is_answer'] == 1)
    return mask


def train_pair(df, columns, train_cfg, num_class, output_dir, run_meta, on_progress=None):
    from skills import model_registry
    family = model_registry.validate(train_cfg.get('model_family', 'xgb'))
    import xgboost as xgb
    if train_cfg.get('training_city'):
        from skills.city_training import assert_city
        assert_city(df, train_cfg['training_city'])
    pairs = {'_queue_rule': train_cfg.get('queue_rule', 'has_queue'), '_training_city': train_cfg.get('training_city')}
    report = on_progress or (lambda message: None)
    strategy = train_cfg.get('non_queue_strategy', 'model')
    method = train_cfg.get('training_method', 'full_train')
    if method not in {'full_train', 'notebook_random', 'temporal'}:
        raise ValueError('未知训练拆分方式')
    if strategy not in {'model', 'zero', 'unified'}:
        raise ValueError('non_queue_strategy 必须是 model 或 zero')
    routes = ((True, 'unified'),) if strategy == 'unified' else ((True, 'queue'), (False, 'no_queue'))
    for is_queue, tag in routes:
        if not is_queue and strategy == 'zero':
            pairs[tag] = None
            policy = {**run_meta, 'condition': 'has_queue is not true', 'prediction': 0,
                      'prediction_meaning': 'bucket_0_le_1_minute',
                      'after_ordering': train_cfg.get('non_queue_after_ordering', 'zero'), 'model_artifact': None}
            (output_dir / 'non_queue_policy.json').write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding='utf-8')
            print('[mining] no_queue: 按 Notebook 策略固定预测 0，不训练非排队模型', flush=True)
            continue
        if method in {'notebook_random', 'temporal'}:
            fitting, stopping = notebook_split(df, is_queue, train_cfg)
        else:
            mask = fitting_mask(df, is_queue, train_cfg)
            fitting, stopping = df.loc[mask], df.iloc[:0]
            if fitting.empty:
                raise ValueError(f'{tag} 训练窗口筛选后无可用样本')
        p = dict(train_cfg["xgboost"])
        if 'model_seed' in train_cfg:
            p['random_state'] = train_cfg['model_seed']
        if method == 'full_train':
            p.pop('early_stopping_rounds', None)
        labels = fitting['true_bin'].astype(int).to_numpy()
        if not np.array_equal(np.unique(labels), np.arange(num_class)):
            raise ValueError(f'Invalid classes: 训练样本必须覆盖配置的全部 {num_class} 桶')
        p['num_class'] = num_class
        wc = train_cfg["class_weight"]
        weights, _ = _build_sample_weights(pd.Series(labels), wc["scheme"], wc["penalty_multipliers"])
        if family == 'mlp':
            from skills.neural_model import fit
            model, metadata = fit(fitting, stopping, columns, labels, weights,
                train_cfg.get('mlp', {}), num_class, train_cfg.get('model_seed', train_cfg['split']['seed']), report)
            model.save(output_dir / model_registry.filename(family, tag),
                       {**run_meta, **metadata, 'fit_rows': len(fitting), 'early_stop_rows': len(stopping)})
            pairs[tag] = model
            report(f'{tag} MLP 训练完成并保存：最佳第 {metadata["best_epoch"]} 轮')
            continue
        fit_options = {}
        if method in {'notebook_random', 'temporal'}:
            fit_options['eval_set'] = [(fitting[columns], labels),
                                       (stopping[columns], stopping['true_bin'].astype(int).to_numpy())]
        report(f'{tag} 模型训练中：{len(fitting):,} 行、{len(columns)} 个特征，配置 {p.get("n_estimators", 100)} 轮 boosting')
        model = xgb.XGBClassifier(**p).fit(
            X=fitting[columns].astype(np.float32), y=labels, verbose=False, sample_weight=weights, **fit_options)
        best = int(model.best_iteration) + 1 if method in {'notebook_random', 'temporal'} else model.get_booster().num_boosted_rounds()
        # Persist exactly the trees used during evaluation, so loading doesn't change predictions.
        booster = model.get_booster()[:best]
        path = output_dir / model_registry.filename(family, tag)
        model_io.save_model(booster, str(path), columns,
                            experiment_id=run_meta["experiment_id"], data_window=run_meta["window"],
                            extra_meta={**run_meta, "best_num_boost_round": best,
                                        'split_method': method, 'fit_rows': len(fitting),
                                        'early_stop_rows': len(stopping)},
                            **train_cfg.get("save", {}))
        pairs[tag] = booster
        report(f'{tag} 模型训练完成并保存：{best} 轮 boosting')
        print(f"[mining] {tag}: fit={len(fitting)} early_stop={len(stopping)} features={len(columns)} trees={best}", flush=True)
    return pairs


def predict(df, pairs, columns):
    import xgboost as xgb
    if pairs.get('_training_city'):
        from skills.city_training import assert_city
        assert_city(df, pairs['_training_city'])
    out = df.copy()
    pred = np.full(len(out), -1, dtype=int)
    if 'unified' in pairs:
        model = pairs['unified']
        raw = model.predict(out[columns]) if getattr(model, 'family', None) == 'mlp' else model.predict(xgb.DMatrix(out[columns].astype(np.float32)))
        out['predict_bin_44'] = np.argmax(raw, axis=1) if raw.ndim == 2 else raw.astype(int)
        return out
    queued = queue_mask(out, pairs.get('_queue_rule', 'has_queue'))
    for mask, tag in ((queued, "queue"), (~queued, "no_queue")):
        if mask.any():
            if tag == 'no_queue' and pairs[tag] is None:
                pred[mask.to_numpy()] = 0
                continue
            raw = (pairs[tag].predict(out.loc[mask, columns]) if getattr(pairs[tag], 'family', None) == 'mlp'
                   else pairs[tag].predict(xgb.DMatrix(out.loc[mask, columns].astype(np.float32))))
            pred[mask.to_numpy()] = np.argmax(raw, axis=1) if raw.ndim == 2 else raw.astype(int)
    if (pred < 0).any(): raise ValueError("存在未被模型覆盖的样本")
    out["predict_bin_44"] = pred
    return out


def metrics(pred):
    result = {str(k): v.as_dict() for k, v in evaluator.evaluate_by_city(
        pred, pred_col="predict_bin_44", verbose=False).items()}
    long_wait = pred[pred["true_ets"] >= 600]
    result["LONG_WAIT"] = evaluator._slice_metrics(long_wait, "predict_bin_44", "true_bin", "LONG_WAIT").as_dict()
    answered = pred[pred['is_answer'] == 1] if 'is_answer' in pred else pred.iloc[:0]
    filtered = pred[(pred['is_answer'] == 1) | ((pred['is_answer'] == 0) & (pred['true_bin'] >= pred['predict_bin_44']))] if 'is_answer' in pred else pred.iloc[:0]
    return {"xgb_44": result,
            "notebook_answer_only": {str(city): value.as_dict() for city, value in evaluator.evaluate_by_city(
                answered, pred_col='predict_bin_44', verbose=False).items()},
            "notebook_answer_cancel": {str(city): value.as_dict() for city, value in evaluator.evaluate_by_city(
                filtered, pred_col='predict_bin_44', verbose=False).items()}}


def gate(base, trial):
    a, b = base.get(SELECTION_KEY, {}), trial.get(SELECTION_KEY, {})
    if not a or not b:
        return {'pass': False, 'reasons': ['门禁所需 H3-7 保序指标不可用']}
    reasons = []
    scope = trial.get('_evaluation_contract', {}).get('training_city')
    if scope is not None:
        from skills.city_training import CITIES
        if scope not in CITIES:
            return {'pass': False, 'reasons': ['未知训练城市']}
    for city in ((scope, 'ALL') if scope else ("100001", "100002", "100003", "ALL")):
        if city not in a or city not in b or a[city]['total'] <= 0 or b[city]['total'] <= 0:
            reasons.append(f"{city} 缺失或评估样本为空")
    # Notebook cancellation filtering depends on each model's raw prediction.
    # Compare the same source test rows, not the resulting filtered row counts.
    left, right = base.get('_evaluation_contract'), trial.get('_evaluation_contract')
    if not left or not right or left != right or left.get('schema') not in (CONTRACT, UNIFIED_CONTRACT):
        reasons.append('测试源样本或6桶评估协议不一致／缺失')
    if reasons: return {"pass": False, "reasons": reasons}
    metric_key = scope or 'ALL'
    accuracies = [value[metric_key].get('recall_ratio') for value in (a, b)]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
           for value in accuracies):
        reasons.append('6桶 H3-7 保序后整体召回无效')
    elif accuracies[1] < accuracies[0]:
        reasons.append('6桶 H3-7 保序后整体召回低于当前对照')
    if not reasons and not right.get('mock', False):
        reference = right.get('reference_baseline', {}).get('metrics', {}).get(metric_key, {}).get('recall_ratio')
        if type(reference) not in (int, float) or not math.isfinite(reference) or not 0 <= reference <= 1:
            reasons.append('示例阈值 缺失或无效')
        elif accuracies[1] < reference:
            reasons.append(f'低于示例阈值 {reference:.4%}')
    if scope:
        reasons = [reason.replace("整体召回", "当前城市召回") for reason in reasons]
    return {"pass": not reasons, "reasons": reasons}
