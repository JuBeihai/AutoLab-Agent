"""Complete candidate specifications; immutable baseline and bounded mutations."""
from copy import deepcopy
import math

from skills import feature_program as program
from skills import feature_preprocessing
from agent.search_memory import recipe_signature


PARAMETERS = {
    'n_estimators': (50, 1200, int), 'max_depth': (2, 10, int),
    'learning_rate': (0.005, 0.2, float), 'subsample': (0.5, 1.0, float),
    'colsample_bytree': (0.5, 1.0, float), 'min_child_weight': (1, 30, float),
    'reg_alpha': (0, 20, float), 'reg_lambda': (0, 30, float), 'gamma': (0, 10, float),
}
WEIGHT_SCHEMES = ('none', 'balanced', 'balanced_sqrt')
MLP_SEARCH = {
    'hidden_layers': [[64, 32], [128, 64], [256, 128]],
    'learning_rate': [0.0003, 0.0005, 0.001],
    'alpha': [0.0001, 0.001, 0.01],
    'batch_size': [128, 256, 512],
    'patience': [10, 20],
    'max_epochs': [80, 160],
}
REQUIRED = {'id', 'hypothesis', 'evidence', 'feature_recipes'}
OPTIONAL = {'remove_features', 'training', 'parent_experiment_id', 'extensions', 'requests', 'inspect'}


def capabilities(train_config=None):
    neural = (train_config or {}).get('model_family') == 'mlp'
    return {'feature_edit': ['add', 'remove', 'replace', 'ablate', 'branch'],
            'preprocessing': ['missing_indicator', 'impute', 'one_hot', 'target_encoding'],
            'model_family': 'mlp' if neural else 'xgb',
            'feature_diagnostics': ['redundancy_correlations'] if neural else ['all_input_total_gain', 'training_permutation', 'redundancy_correlations'],
            'training_parameters': {name: {'choices': values} for name, values in MLP_SEARCH.items()} if neural else {name: {'min': low, 'max': high, 'integer': kind is int}
                                    for name, (low, high, kind) in PARAMETERS.items()},
            'training_parameter_field': 'mlp_params' if neural else 'xgb_params',
            'penalty_multipliers': {'classes': list(range(6)), 'min': 0.5, 'max': 2.0},
            'weight_schemes': list(WEIGHT_SCHEMES),
            'protected': ['data_filters', 'joins', 'test_population', 'labels', 'gate',
                          'non_queue_policy', 'seed', 'early_stop_split', 'xgb_early_stopping_rounds', 'baseline'],
            'external_changes': 'Only registered, human-approved extensions; requests never auto-approve'}


def baseline_spec(columns, recipes, train_config):
    return {'base_features': list(columns), 'recipes': deepcopy(recipes),
            'train_config': deepcopy(train_config), 'extensions': []}


def feature_names(spec):
    return spec['base_features'] + [recipe['name'] for recipe in spec['recipes']] + [
        name for extension in spec.get('extensions', []) for name in extension['outputs']]


def signature(spec):
    train_config = deepcopy(spec['train_config'])
    if train_config.get('model_family') == 'mlp':
        from skills.neural_model import validate_config
        train_config['mlp'] = validate_config(train_config.get('mlp', {}))
    return program.fingerprint({'base_features': sorted(spec['base_features']),
                                'recipes': recipe_signature(spec['recipes']),
                                'train_config': train_config,
                                'extensions': spec.get('extensions', [])})


def validate_header(proposal):
    if not isinstance(proposal, dict) or not REQUIRED <= set(proposal) or set(proposal) - REQUIRED - OPTIONAL:
        raise ValueError('候选包含未知字段，不能修改取数、标签或门禁')
    for name in ('id', 'hypothesis', 'evidence'):
        if not isinstance(proposal[name], str) or not 1 <= len(proposal[name].strip()) <= 1500:
            raise ValueError(f'{name} 不能为空或过长')
    if not isinstance(proposal['feature_recipes'], list):
        raise ValueError('feature_recipes 必须是数组')


def derive(proposal, parent, catalog, cfg, approved_extensions=()):
    validate_header(proposal)
    if proposal.get('inspect') is not None or proposal.get('requests'):
        raise ValueError('只读探查/审批申请不能作为模型训练候选')
    candidate = deepcopy(parent)
    removed = proposal.get('remove_features', [])
    if (not isinstance(removed, list) or any(not isinstance(name, str) for name in removed)
            or len(set(removed)) != len(removed) or set(removed) - set(feature_names(parent))):
        raise ValueError('只能删除父实验已存在且不重复的输入特征')
    candidate['base_features'] = [name for name in parent['base_features'] if name not in removed]
    candidate['recipes'] = [recipe for recipe in parent['recipes'] if recipe['name'] not in removed]
    candidate['extensions'] = []
    for extension in parent.get('extensions', []):
        intersection = set(extension['outputs']) & set(removed)
        if intersection and intersection != set(extension['outputs']):
            raise ValueError('扩展的输出需作为一组删除，不能只删其中一列')
        if not intersection:
            candidate['extensions'].append(deepcopy(extension))
    requested = proposal.get('extensions', [])
    if not isinstance(requested, list) or any(not isinstance(name, str) for name in requested) or len(requested) > 4:
        raise ValueError('extensions 必须是最多4个已审批扩展ID')
    registered = {entry['id']: entry for entry in approved_extensions}
    existing = {entry['id'] for entry in candidate['extensions']}
    for extension_id in requested:
        if extension_id not in registered or extension_id in existing:
            raise ValueError('扩展未审批或重复')
        candidate['extensions'].append(deepcopy(registered[extension_id]))
        existing.add(extension_id)
    recipes = proposal['feature_recipes']
    if len(recipes) > cfg['max_features_per_proposal']:
        raise ValueError('单次新增配方超出预算')
    candidate['recipes'].extend(deepcopy(recipes))
    known = dict(catalog)
    extension_outputs = []
    for extension in candidate['extensions']:
        if any(catalog.get(name, {}).get('role') != 'numeric' for name in extension.get('inputs', [])):
            raise ValueError('扩展输入在当前数据画像中不可用，不能跨数据口径直接套用')
        for name in extension['outputs']:
            if name in known:
                raise ValueError('扩展输出不能覆盖现有输入')
            known[name] = {'role': 'numeric'}
            extension_outputs.append(name)
    for recipe in candidate['recipes']:
        program.validate_recipes([recipe], known, {**cfg, 'max_features_per_proposal': 1})
        known[recipe['name']] = program.recipe_info(recipe, known)
    if len(candidate['recipes']) + len(extension_outputs) > cfg['max_total_features']:
        raise ValueError('候选新增特征总数超过预算')
    if not feature_names(candidate):
        raise ValueError('不能删除全部模型输入')
    if len(feature_names(candidate)) != len(set(feature_names(candidate))):
        raise ValueError('模型输入特征重复')
    changes = proposal.get('training', {})
    if not isinstance(changes, dict) or set(changes) - {'xgb_params', 'mlp_params', 'class_weight_scheme', 'penalty_multipliers'}:
        raise ValueError('training 只允许当前模型的有界参数与类别权重')
    params = changes.get('xgb_params', {})
    if candidate['train_config'].get('model_family') == 'mlp' and params:
        raise ValueError('MLP 不支持 XGB 参数，请使用 mlp_params')
    neural_params = changes.get('mlp_params', {})
    if not isinstance(neural_params, dict) or set(neural_params) - set(MLP_SEARCH):
        raise ValueError('未知 MLP 搜索参数')
    if neural_params:
        if candidate['train_config'].get('model_family') != 'mlp':
            raise ValueError('XGB 不支持 mlp_params')
        from skills.neural_model import validate_config
        validate_config(neural_params)
        for key, value in neural_params.items():
            if value not in MLP_SEARCH[key]:
                raise ValueError(f'{key} 超出 MLP 自动搜索范围')
        candidate['train_config'].setdefault('mlp', {}).update(neural_params)
    if not isinstance(params, dict) or set(params) - set(PARAMETERS):
        raise ValueError('未知或受保护的训练参数')
    for name, value in params.items():
        low, high, kind = PARAMETERS[name]
        if (type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high
                or (kind is int and type(value) is not int)):
            raise ValueError(f'{name} 超出允许范围 {low}–{high}')
    candidate['train_config']['xgboost'].update(params)
    if 'class_weight_scheme' in changes:
        if changes['class_weight_scheme'] not in WEIGHT_SCHEMES:
            raise ValueError('类别权重方案未批准')
        candidate['train_config']['class_weight']['scheme'] = changes['class_weight_scheme']
    if 'penalty_multipliers' in changes:
        multipliers = changes['penalty_multipliers']
        if (not isinstance(multipliers, dict) or not multipliers
                or set(multipliers) - {str(i) for i in range(6)}
                or any(type(v) not in (int, float) or not math.isfinite(v) or not .5 <= v <= 2
                       for v in multipliers.values())):
            raise ValueError('类别倍率需为0–5桶的0.5–2.0有限数值')
        candidate['train_config']['class_weight'].setdefault('penalty_multipliers', {}).update(multipliers)
    if candidate['train_config']['training_method'] != parent['train_config']['training_method']:
        raise ValueError('候选不能修改 baseline 的训练拆分方法')
    if signature(candidate) == signature(parent):
        raise ValueError('候选与父实验相同，没有可执行改动')
    return candidate


def describe_change(parent, candidate):
    old, new = set(feature_names(parent)), set(feature_names(candidate))
    params = candidate['train_config']['xgboost']
    return {'added': sorted(new - old), 'removed': sorted(old - new),
            'xgb_params': {name: value for name, value in params.items()
                           if parent['train_config']['xgboost'].get(name) != value},
            'mlp_params': {name: value for name, value in candidate['train_config'].get('mlp', {}).items()
                           if parent['train_config'].get('mlp', {}).get(name) != value},
            'class_weight_scheme': candidate['train_config']['class_weight']['scheme'],
            'penalty_multipliers': candidate['train_config']['class_weight'].get('penalty_multipliers', {}),
            'feature_count': len(new)}

def has_non_additive_change(parent, candidate):
    if (set(feature_names(parent)) - set(feature_names(candidate))
            or parent['train_config'] != candidate['train_config']
            or parent.get('extensions', []) != candidate.get('extensions', [])):
        return True
    previous = {recipe['name']: recipe for recipe in parent['recipes']}
    return any(recipe != previous.get(recipe['name']) and (
        recipe['name'] in previous or recipe['kind'] in feature_preprocessing.KINDS)
        for recipe in candidate['recipes'])
