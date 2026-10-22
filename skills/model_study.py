"""Autonomous, bounded model experiments on chronological development folds.

The controller chooses subsequent configurations from measured development
results. Historical test rows are used only after the selection is locked.
No LLM, external writes, production promotion, or modification of train.json.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from agent.experiment_log import ExperimentRecord, append_record, new_experiment_id
from agent import experiment_spec
from skills import feature_program, mining_train, model_registry
from skills.evaluation_policy import queue_mask
from skills.evaluation_views import build_views, selection_metrics, SELECTION_KEY
from skills.neural_model import DEFAULTS

STUDY_KEY = 'answer_only_h3_7_6bin'
SEEDS = (123, 456, 789)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def folds(window):
    end = pd.Timestamp(window['train_end'])
    start = pd.Timestamp(window['train_start'])
    result = []
    for offset in (0, 7):
        last = end - pd.Timedelta(days=offset)
        validation = last - pd.Timedelta(days=6)
        stopping = validation - pd.Timedelta(days=7)
        if (stopping - start).days < 7:
            raise ValueError('自主模型实验需要至少28天训练数据，以保留两个时间验证窗口')
        result.append({'train_start': str(start.date()),
                       'early_stop_start': str(stopping.date()),
                       'validation_start': str(validation.date()),
                       'validation_end': str(last.date())})
    return result


def partition(frame, fold):
    """Preserve joined rows, but reject order overlap instead of leaking it."""
    masks = [frame.dt.between(fold['train_start'], str((pd.Timestamp(fold['early_stop_start']) - pd.Timedelta(days=1)).date())),
             frame.dt.between(fold['early_stop_start'], str((pd.Timestamp(fold['validation_start']) - pd.Timedelta(days=1)).date())),
             frame.dt.between(fold['validation_start'], fold['validation_end'])]
    groups = [set(frame.loc[mask, 'order_id'].astype(str)) for mask in masks]
    if any(not group for group in groups):
        raise ValueError('拟合、早停或验证时间窗口没有订单')
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('同一订单跨越时间分区，拒绝存在分区泄漏的实验')
    return masks[0] | masks[1], masks[2]


def probabilities(frame, pairs, columns):
    import xgboost as xgb
    result = np.zeros((len(frame), 6), dtype=float)
    routes = [(np.ones(len(frame), dtype=bool), 'unified')] if 'unified' in pairs else [
        (queue_mask(frame).to_numpy(), 'queue'), (~queue_mask(frame).to_numpy(), 'no_queue')]
    for mask, tag in routes:
        if not mask.any():
            continue
        model = pairs[tag]
        if model is None:
            result[mask, 0] = 1
        elif getattr(model, 'family', None) == 'mlp':
            result[mask] = model.predict(frame.loc[mask, columns])
        else:
            margin = model.predict(xgb.DMatrix(frame.loc[mask, columns].astype(np.float32)), output_margin=True)
            if margin.shape != (int(mask.sum()), 6):
                raise ValueError('XGB 未返回六桶 margin，禁止将类别编号当成概率融合')
            scores = np.exp(margin - margin.max(axis=1, keepdims=True))
            result[mask] = scores / scores.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=1e-5):
        raise ValueError('模型概率无效')
    return result


def blend(left, right, mlp_weight):
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 6 or not 0 <= mlp_weight <= 1:
        raise ValueError('融合需要相同订单顺序的六桶概率与0–1权重')
    return mlp_weight * left + (1 - mlp_weight) * right


def measure(frame, probs, strategy):
    predictions = frame.copy()
    predictions['predict_bin_44'] = probs.argmax(axis=1)
    views = build_views(predictions, non_queue_after_ordering=strategy)
    fixed = views['populations']['answer_only']['h3_7']['metrics']
    tail = predictions.loc[predictions.is_answer.eq(1) & predictions.true_ets.ge(600)]
    from skills.evaluator import _slice_metrics
    tail_metric = _slice_metrics(tail, 'predict_bin_44', 'true_bin', 'answered_long_wait').as_dict()
    if fixed['ALL']['total'] == 0 or tail_metric['total'] == 0:
        raise ValueError('验证窗口缺少已应答或已应答长等待样本，不能据空切片选择模型')
    metrics = mining_train.metrics(predictions)
    metrics[SELECTION_KEY] = selection_metrics(views)
    metrics[STUDY_KEY] = fixed
    metrics['answered_long_wait'] = {'ALL': tail_metric}
    return metrics, views


def scores(metrics):
    answered = metrics[STUDY_KEY]['ALL']
    tail = metrics['answered_long_wait']['ALL']
    return {'accuracy': answered['recall_ratio'], 'severe_under': answered['severe_under_ratio'],
            'long_wait_accuracy': tail['recall_ratio'], 'long_wait_severe_under': tail['severe_under_ratio'],
            'legacy_accuracy': metrics[SELECTION_KEY]['ALL']['recall_ratio']}


def aggregate(results):
    return {key: {'mean': float(np.mean([r[key] for r in results])),
                  'std': float(np.std([r[key] for r in results]))}
            for key in results[0]}


def acceptable(value, reference):
    return (value['severe_under'] <= reference['severe_under'] + 1e-12
            and value['long_wait_severe_under'] <= reference['long_wait_severe_under'] + 1e-12)


class BudgetEnded(Exception):
    pass


class Study:
    def __init__(self, root, full, config, manifest, args, progress, emit):
        self.root, self.full, self.config, self.manifest, self.args = Path(root), full, config, manifest, args
        self.progress, self.emit = progress, emit
        self.directory, self.logs = Path(manifest['artifact_dir']), Path(manifest['log_dir'])
        self.started = time.monotonic()
        self.limit = args.rounds * args.max_proposals_per_round
        self.folds = folds(manifest['window'])
        self.specs, self.screen, self.cache, self.events = {}, {}, {}, []
        self.trial_count = 0
        self.base = None
        self.report = {'protocol': 'temporal_model_study_v1', 'folds': self.folds, 'status': 'running',
                       'selection_metric': STUDY_KEY, 'seeds': list(SEEDS),
                       'candidate_budget': self.limit, 'screening': {}, 'confirmation': {},
                       'decisions': self.events,
                       'policy': '先最大化固定已应答人群H3-7准确率；推荐方案的严重低估和长等待严重低估不得高于本次XGB基线；另报纯准确率冠军',
                       'test_warning': '历史测试窗口曾参与旧会话搜索，最终回测不构成独立泛化证明；需要新日期验收'}
        self.manifest.update(model_study=True, research_skills=True, evaluation_split='validation', rows=len(full),
            selection_metric=STUDY_KEY, training_method='temporal',
            title=manifest['title'] + ' · 自主模型实验',
            selection_warning=self.report['test_warning'],
            selection_population='固定已应答时间验证人群；取消过滤指标另行展示',
            champion_policy='fixed_answer_accuracy_with_xgb_underprediction_guardrails_v1',
            study_budget={'candidates': self.limit, 'confirmation_max_fits': 24, 'final_refit_max_fits': 3},
            status='running', production_approved=False)
        self.manifest['fixed_train_config'] = {**deepcopy(config['train']), 'training_method': 'temporal',
                                               'study_folds': self.folds}
        self.checkpoint()

    def checkpoint(self):
        self.report['screening'] = {key: {'scores': value['scores'], 'experiment_id': value['id'],
                                         'spec': self.specs[key]} for key, value in self.screen.items()}
        write_json(self.directory / 'model_study.json', self.report)
        write_json(self.directory / 'session.json', self.manifest)
        write_json(self.root / 'experiments/feature_mining/current.json', self.manifest)
        self.emit('session_updated', self.manifest['session_id'], status=self.manifest['status'])

    def check_time(self):
        if time.monotonic() - self.started >= self.args.search_timeout_minutes * 60:
            raise BudgetEnded('时间预算已用尽，保留已完成结果，未完成复验的模型不作为最终推荐')

    def decide(self, reason):
        self.events.append({'step': len(self.events) + 1, 'reason': reason})
        self.progress('planning', reason)
        self.checkpoint()

    def fit(self, spec, fold, seed, out):
        train, validation = partition(self.full, fold)
        # Exclude the retrospective test completely from development materialization.
        frame = self.full.loc[self.full.dt.between(fold['train_start'], fold['validation_end'])].copy()
        train = train.loc[frame.index]
        cfg = deepcopy(spec['train_config'])
        cfg.update(training_method='temporal', temporal_early_stop_start=fold['early_stop_start'],
                   temporal_validation_start=fold['validation_start'], model_seed=seed)
        fitting = mining_train.notebook_split(frame.loc[train], True, cfg)[0]
        catalog = {column: {'role': 'numeric', 'groupable': True} for column in frame.columns}
        state = {}
        expanded = feature_program.materialize(frame, spec['recipes'], catalog, self.config['mining'],
            str((pd.Timestamp(fold['validation_start']) - pd.Timedelta(days=1)).date()),
            fitting_index=fitting.index, preprocessing_state=state) if spec['recipes'] else frame
        columns = experiment_spec.feature_names(spec)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / 'feature_manifest.json', {'experiment_spec': spec, 'preprocessing_state': state,
            'effective_train_config': cfg, 'feature_columns': columns, 'fold': fold, 'seed': seed,
            'evaluation_split': 'validation'})
        from skills import reproduction
        target = expanded.loc[validation.loc[expanded.index]]
        reproduction.capture(self.root, out, expanded.loc[train], target, columns, cfg,
                             protocol='model_study', window=fold,
                             snapshot_id=self.manifest.get('snapshot_id'))
        pairs = mining_train.train_pair(expanded.loc[train], columns, cfg, 6, out,
            {'model_family': cfg['model_family'], 'training_population': cfg['training_population'],
             'experiment_id': out.name, 'window': fold,
             'fold': fold, 'model_seed': seed, 'mock': self.args.mock, 'model_study': True},
            on_progress=lambda message: self.progress('training', f'{out.name} · {message}'))
        target = expanded.loc[validation.loc[expanded.index]]
        return target, probabilities(target, pairs, columns), cfg

    def evaluate(self, name, spec, fold_index=0, seed=123, stage='screen'):
        key = (name, fold_index, seed)
        if key in self.cache:
            return self.cache[key]
        self.check_time()
        exp_id = new_experiment_id(root=self.root)
        out = self.directory / exp_id
        fold = self.folds[fold_index]
        self.progress('training', f'{stage} · {name} · 时间窗口{fold_index + 1} · seed={seed}', experiment_id=exp_id)
        target, probs, cfg = self.fit(spec, fold, seed, out)
        met, views = measure(target, probs, cfg['non_queue_after_ordering'])
        score = scores(met)
        self.trial_count += 1
        result = {'id': exp_id, 'scores': score, 'probabilities': probs, 'index': target.index.to_numpy(),
                  'metrics': met}
        self.cache[key] = result
        history = []
        for path in out.glob('*.meta.json'):
            meta = json.loads(path.read_text())
            if 'mlp_config' in meta:
                history.append({k: meta[k] for k in ('best_epoch', 'epochs_run', 'early_stop_loss', 'mlp_config')})
        result['training'] = history
        is_reference = name == self.base or self.base is None
        reference = self.cache.get((self.base, fold_index, seed))
        record = ExperimentRecord(exp_id, 0 if self.base is None else self.trial_count - 1, datetime.now().isoformat(timespec='seconds'),
            'baseline' if is_reference else 'agent', name, config_diff={'session_id': self.manifest['session_id'], 'experiment_spec': spec,
                'model_family': cfg['model_family'], 'model_study': True, 'evaluation_split': 'validation',
                'parent_experiment_id': reference['id'] if reference and not is_reference else None,
                'selection_metric': STUDY_KEY, 'stage': stage, 'model_seed': seed, 'fold_index': fold_index,
                'training_population': cfg['training_population'], 'training_method': 'temporal'},
            data_window=fold, metrics_by_city=met, evaluation_views=views, artifact_dir=str(out),
            notes='时间验证实验；不与历史Notebook测试分数混比，不自动更新生产模型。',
            summary={'innovation': {'title': name}, 'narrative_zh': name,
                     'verdict': 'baseline' if is_reference else 'validation',
                     'selection_metric': STUDY_KEY, 'study_scores': score, 'training': history,
                     'selection_note': '已评估；最终按跨时间窗口和多种子复验选择'})
        append_record(str(self.logs), record)
        write_json(out / 'selection_result.json', asdict(record))
        from skills.reproduction import finalize
        finalize(out, asdict(record))
        self.emit('experiment_completed', self.manifest['session_id'], experiment_id=exp_id)
        self.progress('evaluated', f'{name}：固定已应答H3-7准确率 {score["accuracy"]:.2%}；严重低估 {score["severe_under"]:.2%}')
        self.manifest['study_trials_completed'] = self.trial_count
        if stage == 'screen':
            self.screen[name] = result
            self.specs[name] = deepcopy(spec)
        self.checkpoint()
        return result

    def candidate(self, name, spec):
        signature = experiment_spec.signature(spec)
        for previous, previous_spec in self.specs.items():
            if experiment_spec.signature(previous_spec) == signature:
                return previous
        if len(self.screen) >= self.limit:
            self.decide(f'候选预算 {self.limit} 已用完，转入已完成候选的复验')
            return None
        try:
            self.evaluate(name, spec)
        except (ValueError, FloatingPointError) as error:
            # Failed configurations are recorded, never interpreted as evidence.
            self.report.setdefault('failures', []).append({'candidate': name, 'error': str(error)})
            self.checkpoint()
            raise
        return name

    def best(self, family, guarded=False):
        eligible = [name for name, spec in self.specs.items() if spec['train_config']['model_family'] == family
                    and (not guarded or acceptable(self.screen[name]['scores'], self.screen[self.base]['scores']))]
        if not eligible:
            return self.best(family)
        return max(eligible, key=lambda name: (self.screen[name]['scores']['accuracy'],
                                              -self.screen[name]['scores']['severe_under']))

    def run(self, source_spec=None):
        columns = self.config['features']['feature_set_27'] + self.config['features']['feature_set_44_extra']
        plain = experiment_spec.baseline_spec(columns, [], self.config['train'])
        plain['train_config'].pop('model_seed', None)
        plain['train_config']['mlp'] = {**DEFAULTS, **plain['train_config'].get('mlp', {})}
        self.decide('先以相同特征、相同时间分区，对比XGB/MLP以及balanced_sqrt/none，区分权重收益和网络收益')
        for family, scheme in [('xgb', 'balanced_sqrt'), ('mlp', 'balanced_sqrt'), ('xgb', 'none'), ('mlp', 'none')]:
            spec = deepcopy(plain)
            spec['train_config']['model_family'] = family
            spec['train_config']['class_weight'] = {'scheme': scheme, 'penalty_multipliers': {str(i): 1. for i in range(6)}}
            name = self.candidate(f'原始43特征 · {family} · {scheme}', spec)
            if self.base is None:
                self.base = name
                self.manifest['baseline_id'] = self.screen[name]['id']
        if source_spec is not None:
            for family in ('mlp', 'xgb'):
                spec = deepcopy(source_spec)
                spec['train_config']['model_family'] = family
                spec['train_config']['mlp'] = {**DEFAULTS, **spec['train_config'].get('mlp', {})}
                self.candidate(f'继承冠军特征与权重 · {family}', spec)
        self.decide('固定当前最佳MLP的特征与网络，仅比较长等待桶的温和类别倍率')
        parent = deepcopy(self.specs[self.best('mlp')])
        for multipliers in [(1.1, 1.2, 1.3), (1.2, 1.4, 1.6)]:
            spec = deepcopy(parent)
            spec['train_config']['class_weight'] = {'scheme': 'none',
                'penalty_multipliers': {str(i): value for i, value in enumerate((1., 1., 1., *multipliers))}}
            self.candidate(f'长等待权重 {multipliers}', spec)
        # Stagewise coordinate search: every parameter group starts from the
        # best measured MLP. Parent is fixed within a group for attribution.
        for parameter in ('learning_rate', 'alpha', 'hidden_layers', 'batch_size'):
            parent_name = self.best('mlp', guarded=True)
            parent = deepcopy(self.specs[parent_name])
            self.decide(f'依据验证结果选择「{parent_name}」作父模型；本阶段只调整 {parameter}')
            for value in experiment_spec.MLP_SEARCH[parameter]:
                if value == parent['train_config']['mlp'][parameter]:
                    continue
                spec = deepcopy(parent)
                spec['train_config']['mlp'][parameter] = value
                self.candidate(f'{parameter}={value} / 父={parent_name}', spec)
        leader = self.best('mlp', guarded=True)
        parent = deepcopy(self.specs[leader])
        if parent['train_config']['mlp']['learning_rate'] < .001:
            spec = deepcopy(parent)
            spec['train_config']['mlp']['patience'] = 20
            self.decide('最佳方案使用低学习率，检验patience=20是否允许后期继续改善')
            self.candidate('低学习率 · patience=20', spec)
        leader = self.best('mlp', guarded=True)
        training = self.screen[leader]['training']
        if training and any(m['mlp_config']['max_epochs'] < 160 and
                            m['best_epoch'] >= .9 * m['mlp_config']['max_epochs'] for m in training):
            spec = deepcopy(self.specs[leader])
            spec['train_config']['mlp']['max_epochs'] = min(160, spec['train_config']['mlp']['max_epochs'] * 2)
            self.decide('最佳epoch接近轮数上限，验证增加训练轮数；否则不盲目延长训练')
            self.candidate('接近训练上限 · 增加epochs', spec)
        spec = deepcopy(self.specs[self.best('mlp', guarded=True)])
        redundant = {r['name'] for r in spec['recipes'] if r['kind'] == 'missing_indicator'
                     and r['column'] in spec['base_features']}
        if redundant:
            spec['recipes'] = [r for r in spec['recipes'] if r['name'] not in redundant]
            self.decide('MLP已自动追加缺失指示；单独消融重复指示，并在复验中检查稳定性')
            self.candidate('消融重复缺失指示', spec)
        self.confirm()

    def confirm(self):
        names = list(dict.fromkeys([self.base, self.best('xgb'), self.best('mlp'), self.best('mlp', guarded=True)]))
        self.decide('停止新增候选；对XGB基线、最佳XGB及MLP入围方案做两个时间窗口×三个独立初始化种子复验，拆分保持固定')
        confirmed = {}
        for name in names:
            runs = [self.evaluate(name, self.specs[name], fold, seed, 'confirmation')
                    for fold in range(2) for seed in SEEDS]
            confirmed[name] = aggregate([run['scores'] for run in runs])
            self.report['confirmation'][name] = {'aggregate': confirmed[name],
                'experiment_ids': [run['id'] for run in runs]}
            self.checkpoint()
        mlp_names = [name for name in names if self.specs[name]['train_config']['model_family'] == 'mlp']
        mlp_name = max(mlp_names, key=lambda n: confirmed[n]['accuracy']['mean'])
        xgb_names = [name for name in names if self.specs[name]['train_config']['model_family'] == 'xgb']
        xgb_name = max(xgb_names, key=lambda n: confirmed[n]['accuracy']['mean'])
        self.decide('复用已训练模型的六桶概率，验证25%、50%、75% MLP占比；每次融合后重新执行人群过滤与H3保序')
        blends = {}
        for weight in (.25, .5, .75):
            self.check_time()
            name = f'概率融合 MLP={weight:.0%}'
            values = []
            for fold in range(2):
                for seed in SEEDS:
                    left, right = self.cache[(mlp_name, fold, seed)], self.cache[(xgb_name, fold, seed)]
                    if not np.array_equal(left['index'], right['index']):
                        raise ValueError('融合模型预测订单不一致')
                    probs = blend(left['probabilities'], right['probabilities'], weight)
                    met, _ = measure(self.full.loc[left['index']], probs, self.config['train']['non_queue_after_ordering'])
                    values.append(scores(met))
            confirmed[name] = aggregate(values)
            blends[name] = {'mlp': mlp_name, 'xgb': xgb_name, 'mlp_weight': weight}
            self.report['confirmation'][name] = {'aggregate': confirmed[name], 'ensemble': blends[name]}
            self.checkpoint()
        average = {name: {key: value['mean'] for key, value in stats.items()} for name, stats in confirmed.items()}
        reference = average[self.base]
        accuracy_leader = max(average, key=lambda name: average[name]['accuracy'])
        eligible = [name for name in average if acceptable(average[name], reference)]
        winner = max(eligible, key=lambda name: (average[name]['accuracy'], -average[name]['severe_under']))
        selected = blends.get(winner, {'model': winner})
        self.report.update(status='selection_locked', accuracy_leader=accuracy_leader, recommended=winner,
                           selected=selected, guardrail_reference=self.base)
        self.checkpoint()  # persisted before any retrospective test labels are read
        self.decide(f'验证选择已锁定：推荐「{winner}」；纯准确率冠军「{accuracy_leader}」。现在才进行历史测试回测')
        self.finalize(winner, selected)

    def finalize(self, winner, selected):
        self.check_time()
        win = self.manifest['window']
        fold = {'train_start': win['train_start'],
                'early_stop_start': str((pd.Timestamp(win['train_end']) - pd.Timedelta(days=6)).date()),
                'validation_start': win['test_start'], 'validation_end': win['test_end']}
        out = self.directory / 'final_model'
        names = [selected['model']] if 'model' in selected else [selected['mlp'], selected['xgb']]
        predictions, components = {}, []
        # Same fixed refit policy for selected models and reference.
        for name in dict.fromkeys([*names, self.base]):
            self.check_time()
            subdir = out / f'component_{len(predictions)}'
            target, probs, _ = self.fit(self.specs[name], fold, 123, subdir)
            predictions[name] = (target, probs)
            if name in names:
                components.append({'directory': subdir.name, 'model_family': self.specs[name]['train_config']['model_family'],
                    'weight': 1. if 'model' in selected else selected['mlp_weight'] if name == selected['mlp'] else 1 - selected['mlp_weight']})
        frame = predictions[names[0]][0]
        if any(not np.array_equal(frame.index, predictions[name][0].index) for name in predictions):
            raise ValueError('最终回测订单不一致')
        probs = sum(component['weight'] * predictions[name][1] for component, name in zip(components, names))
        met, views = measure(frame, probs, self.config['train']['non_queue_after_ordering'])
        base_met, base_views = measure(frame, predictions[self.base][1], self.config['train']['non_queue_after_ordering'])
        bundle = {'format': 'ets_model_study_bundle_v1', 'components': components,
                  'training_population': self.config['train']['training_population'],
                  'queue_rule': self.config['train']['queue_rule'],
                  'non_queue_after_ordering': self.config['train']['non_queue_after_ordering'],
                  'selection_locked_before_retrospective': True, 'production_approved': False}
        write_json(out / 'model_bundle.json', bundle)
        write_json(self.directory / 'holdout_result.json', {'baseline': base_met, 'selected': met,
            'baseline_views': base_views, 'selected_views': views, 'selection_metric': STUDY_KEY,
            'used_for_search': False, 'historically_exposed': True, 'independent_holdout': False,
            'warning': self.report['test_warning'], 'evaluation_split': 'retrospective_test', 'production_approved': False,
            'gate': {'pass': False, 'reasons': ['仅历史回测；需要未参与搜索的新日期验收，不自动替换历史冠军']}})
        # Reload the entire bundle, including fitted feature mappings, and
        # verify saved inference, not merely the in-memory training result.
        loaded = predict_bundle(out / 'model_bundle.json', self.full, self.config['mining'])
        np.testing.assert_allclose(loaded.loc[frame.index].to_numpy(), probs, rtol=1e-5, atol=1e-6)
        self.report.update(status='complete', retrospective={'selected': scores(met), 'baseline': scores(base_met)},
                           bundle=str(out / 'model_bundle.json'), reload_verified=True)
        self.manifest.update(status='complete', champion_id=self.screen[names[0]]['id'] if len(names) == 1 else None,
            champion_spec=self.specs[names[0]] if len(names) == 1 else None,
            study_selected_components=selected,
            study_recommended=winner, study_accuracy_leader=self.report['accuracy_leader'],
            model_bundle=str(out / 'model_bundle.json'), accepted_recipes=[],
            production_approved=False)
        self.checkpoint()


def predict_bundle(path, frame, mining_config):
    """Reproduce final artifact predictions, using saved target mappings."""
    path = Path(path)
    bundle = json.loads(path.read_text())
    if bundle.get('format') != 'ets_model_study_bundle_v1':
        raise ValueError('未知模型组合格式')
    combined = np.zeros((len(frame), 6))
    if not np.isclose(sum(c['weight'] for c in bundle['components']), 1):
        raise ValueError('模型组合权重无效')
    for component in bundle['components']:
        directory = (path.parent / component['directory']).resolve()
        if directory.parent != path.parent.resolve():
            raise ValueError('模型组件路径越界')
        doc = json.loads((directory / 'feature_manifest.json').read_text())
        spec, state = doc['experiment_spec'], doc['preprocessing_state']
        target = frame.copy()
        for recipe in spec['recipes']:
            kind, name = recipe['kind'], recipe['name']
            if kind == 'target_encoding':
                entry = state[name]
                mapping = pd.DataFrame(entry['mapping'])
                mapped = target[recipe['keys']].merge(mapping, on=recipe['keys'], how='left', sort=False, validate='many_to_one')
                target[name] = mapped['_encoded'].fillna(entry['prior']).to_numpy()
            elif kind == 'missing_indicator':
                target[name] = pd.to_numeric(target[recipe['column']], errors='coerce').replace([np.inf, -np.inf], np.nan).isna().astype(float)
            elif kind == 'impute':
                target[name] = pd.to_numeric(target[recipe['column']], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(state[name]['fill_value'])
            elif kind == 'one_hot':
                target[name] = np.logical_and.reduce([target[k].eq(v).fillna(False) for k, v in zip(recipe['keys'], recipe['values'])]).astype(float)
            elif kind == 'expression':
                target[name] = feature_program.expression(recipe['expression'], target)
            else:
                raise ValueError(f'自主模型实验暂不支持保存回读此配方：{kind}')
            target[name] = target[name].clip(-1e30, 1e30).astype(np.float32)
        pairs = {'_queue_rule': bundle['queue_rule']}
        routes = ['unified'] if bundle['training_population'] == 'unified' else ['queue']
        for route in routes:
            pairs[route] = model_registry.load(directory / model_registry.filename(component['model_family'], route), component['model_family'])
        if routes == ['queue']:
            pairs['no_queue'] = None
        combined += component['weight'] * probabilities(target, pairs, doc['feature_columns'])
    return pd.DataFrame(combined, index=frame.index, columns=[f'p{i}' for i in range(6)])


def run(root, full, config, manifest, args, progress, emit):
    source = None
    if args.resume_recipes:
        from skills.city_training import latest_path
        path = latest_path(root, args.city_code, 'mlp', args.training_population)
        if path.is_file():
            previous = json.loads(path.read_text())
            if (previous.get('model_family') == 'mlp' and previous.get('mock') == args.mock
                    and previous.get('training_population') == args.training_population
                    and previous.get('training_city') == args.city_code and previous.get('status') == 'complete'):
                source = previous.get('champion_spec')
                if source and (source.get('extensions') or any(r['kind'] not in {
                        'target_encoding', 'missing_indicator', 'impute', 'one_hot', 'expression'} for r in source['recipes'])):
                    raise ValueError('继承模型包含自主模型实验尚不支持保存回读的扩展/历史配方')
                manifest['study_source_session'] = previous['session_id']
    study = Study(root, full, config, manifest, args, progress, emit)
    try:
        study.run(source)
    except BudgetEnded as error:
        study.report.update(status='budget_exhausted', stop_reason=str(error))
        manifest.update(status='complete', study_incomplete=True, study_stop_reason=str(error), production_approved=False)
        study.checkpoint()
