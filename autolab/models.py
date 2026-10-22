"""A bounded model registry: adding a model does not change the experiment loop."""
from dataclasses import dataclass
import math

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelAdapter:
    label: str
    classifier: object
    regressor: object
    defaults: dict
    space: dict


REGISTRY = {
    'linear': ModelAdapter('Logistic / Ridge', LogisticRegression, Ridge, {}, {}),
    'random_forest': ModelAdapter('Random Forest', RandomForestClassifier, RandomForestRegressor,
        {'n_estimators': 80, 'max_depth': 8, 'min_samples_leaf': 2},
        {'n_estimators': [40, 80, 160], 'max_depth': [4, 8, 16], 'min_samples_leaf': [1, 2, 5]}),
    'mlp': ModelAdapter('MLP', MLPClassifier, MLPRegressor,
        {'hidden_layer_sizes': [64, 32], 'alpha': .001, 'learning_rate_init': .001, 'max_iter': 100},
        {'hidden_layer_sizes': [[32], [64, 32], [128, 64]], 'alpha': [.0001, .001, .01],
         'learning_rate_init': [.0003, .001, .003], 'max_iter': [80, 100, 160]}),
}


def registry():
    if 'xgb' not in REGISTRY:
        from xgboost import XGBClassifier, XGBRegressor
        REGISTRY['xgb'] = ModelAdapter('XGBoost', XGBClassifier, XGBRegressor,
            {'n_estimators': 80, 'max_depth': 4, 'learning_rate': .05, 'subsample': .9},
            {'n_estimators': [40, 80, 160], 'max_depth': [2, 4, 6],
             'learning_rate': [.02, .05, .1], 'subsample': [.7, .9, 1.]})
    return REGISTRY


def catalog(kind):
    result = {}
    for name, adapter in registry().items():
        defaults, space = dict(adapter.defaults), dict(adapter.space)
        if name == 'linear':
            key = 'C' if kind == 'classification' else 'alpha'
            defaults, space = {key: 1.}, {key: [.1, 1., 10.]}
        result[name] = {'label': adapter.label, 'defaults': defaults, 'space': space}
    return result


def validate_candidate(value, kind, allowed_models, feature_ids):
    if not isinstance(value, dict) or set(value) != {'model', 'params', 'features', 'hypothesis'}:
        raise ValueError('候选仅允许 model、params、features、hypothesis')
    if value['model'] not in allowed_models or value['model'] not in catalog(kind):
        raise ValueError('候选模型不在允许范围内')
    description = catalog(kind)[value['model']]
    params = value['params']
    if not isinstance(params, dict) or set(params) - set(description['space']):
        raise ValueError('候选含未允许的参数')
    for key, val in params.items():
        if type(val) is bool or val not in description['space'][key]:
            raise ValueError(f'参数 {key} 超出可执行搜索范围')
    features = value['features']
    if not isinstance(features, list) or not features or any(type(x) is not str for x in features):
        raise ValueError('候选需要非空输入特征列表')
    if len(set(features)) != len(features) or set(features) - set(feature_ids):
        raise ValueError('候选包含重复或未知特征')
    if not isinstance(value['hypothesis'], str) or not 1 <= len(value['hypothesis']) <= 1200:
        raise ValueError('候选需要有长度限制的实验假设')
    return {**value, 'params': {**description['defaults'], **params}, 'features': sorted(features)}


def build(candidate, kind, frame, seed):
    name = candidate['model']
    adapter = registry()[name]
    params = dict(candidate['params'])
    if name == 'mlp':
        params.update(hidden_layer_sizes=tuple(params['hidden_layer_sizes']), random_state=seed,
                      early_stopping=True, n_iter_no_change=10)
    elif name == 'linear':
        if kind == 'classification':
            params.update(max_iter=500, random_state=seed)
    elif name == 'xgb':
        params.update(n_jobs=2, random_state=seed, tree_method='hist')
        if kind == 'classification':
            params['eval_metric'] = 'mlogloss'
    else:
        params.update(n_jobs=2, random_state=seed)
    model = (adapter.classifier if kind == 'classification' else adapter.regressor)(**params)
    columns = candidate['features']
    numeric = [c for c in columns if frame[c].dtype.kind in 'biufc']
    category = [c for c in columns if c not in numeric]
    transforms = []
    if numeric:
        transforms.append(('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                                ('scale', StandardScaler())]), numeric))
    if category:
        transforms.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                                        max_categories=32), category))
    return Pipeline([('preprocess', ColumnTransformer(transforms)), ('model', model)])


def measure(kind, truth, prediction):
    if kind == 'classification':
        return {'accuracy': float(accuracy_score(truth, prediction)),
                'f1_macro': float(f1_score(truth, prediction, average='macro', zero_division=0))}
    # r2 is diagnostic only; selection uses finite MAE/RMSE.
    r2 = float(r2_score(truth, prediction))
    return {'rmse': float(math.sqrt(mean_squared_error(truth, prediction))),
            'mae': float(mean_absolute_error(truth, prediction)), 'r2': r2 if math.isfinite(r2) else None}


def utility(metrics, metric):
    value = metrics[metric]
    if not math.isfinite(value):
        raise ValueError('评估指标不是有限值')
    return -value if metric in ('rmse', 'mae') else value
