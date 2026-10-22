"""Training model catalog and safe artifact dispatch."""
from copy import deepcopy
from pathlib import Path
import json

CATALOG = {
    'xgb': {'id': 'xgb', 'name': 'XGBoost', 'description': '梯度提升树；现有表格特征基线，支持有界树参数搜索。'},
    'mlp': {'id': 'mlp', 'name': '简单神经网络（MLP）',
            'description': '128 → 64 隐藏层，ReLU、Adam；训练内缺失值填补与标准化；6桶分类。'},
}


def validate(family):
    if not isinstance(family, str) or family not in CATALOG:
        raise ValueError('训练模型必须是 xgb 或 mlp')
    return family


def filename(family='xgb', tag='queue'):
    validate(family)
    if tag not in ('queue', 'no_queue', 'unified'):
        raise ValueError('未知模型路由')
    return f'ets_pred_model_{tag}_44.json' if family == 'xgb' else f'ets_mlp_model_{tag}.npz'


def catalog():
    return deepcopy(list(CATALOG.values()))


def load(path, family):
    validate(family)
    if family == 'mlp':
        from skills.neural_model import NeuralModel
        return NeuralModel.load(path)
    from skills.model_io import load_model
    return load_model(str(path))


def saved_models(root, limit=100):
    from skills.evaluation_views import record_selection
    root = Path(root)
    sessions = sorted((root / 'artifacts/feature_mining').glob('*/session.json'),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for path in sessions:
        try:
            session = json.loads(path.read_text())
            if session.get('model_study') and session.get('model_bundle'):
                bundle = Path(session['model_bundle'])
                if bundle.is_file():
                    components = json.loads(bundle.read_text())['components']
                    rows.append({'session_id': session['session_id'],
                        'experiment_id': 'final_model', 'model_family': components[0]['model_family'] if len(components) == 1 else 'ensemble',
                        'model_name': '自主实验推荐 · ' + session.get('study_recommended', ''),
                        'training_population': session.get('training_population'), 'training_city': session.get('training_city'),
                        'city_name': session.get('city_name'), 'mock': session.get('mock', False),
                        'timestamp': session.get('created_at'), 'selected': True, 'gate_pass': False,
                        'recall_ratio': None, 'model_path': str(bundle)})
                    if len(rows) >= limit:
                        return rows
            journal = root / 'experiments/feature_mining' / session['session_id'] / 'experiments.jsonl'
            if not journal.is_file():
                continue
            for line in reversed(journal.read_text().splitlines()):
                record = json.loads(line)
                family = record.get('config_diff', {}).get('model_family', session.get('model_family', 'xgb'))
                model = path.parent / record['experiment_id'] / filename(family, 'unified' if session.get('training_population') == 'unified' else 'queue')
                if not model.is_file():
                    continue
                rows.append({'session_id': session['session_id'], 'experiment_id': record['experiment_id'],
                    'model_family': family, 'model_name': CATALOG[family]['name'],
                    'training_population': session.get('training_population', 'queue_only'), 'training_city': session.get('training_city'), 'city_name': session.get('city_name'),
                    'mock': session.get('mock', False), 'timestamp': record.get('timestamp'),
                    'selected': not session.get('model_study') and record['experiment_id'] == session.get('champion_id'),
                    'recall_ratio': record_selection(record).get('ALL', {}).get('recall_ratio'),
                    'gate_pass': record.get('gate_pass'), 'model_path': str(model)})
                if len(rows) >= limit:
                    return rows
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return rows
