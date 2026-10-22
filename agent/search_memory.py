"""Session-owned search evidence, with exact-context deduplication."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd

from skills.feature_program import fingerprint
from skills.evaluation_policy import SELECTION_KEY


def recipe_signature(recipes):
    references = {}

    def expression(node):
        if 'col' in node:
            return references.get(node['col'], {'col': node['col']})
        if 'const' in node:
            return {'const': float(node['const'])}
        arguments = [expression(argument) for argument in node['args']]
        if node['op'] in {'add', 'mul', 'min', 'max'}:
            arguments.sort(key=lambda value: json.dumps(value, sort_keys=True))
        return {'op': node['op'], 'args': arguments}

    normalized = []
    for recipe in recipes:
        if recipe['kind'] == 'expression':
            value = {'kind': 'expression', 'expression': expression(recipe['expression'])}
        else:
            value = {key: value for key, value in recipe.items() if key != 'name'}
            if recipe['kind'] == 'one_hot':
                pairs = sorted(zip(value['keys'], value['values']))
                value['keys'] = [key for key, category in pairs]
                value['values'] = [category for key, category in pairs]
            elif 'keys' in value:
                value['keys'] = sorted(value['keys'])
            if 'column' in value:
                value['column'] = references.get(value['column'], {'col': value['column']})
        normalized.append(value)
        references[recipe['name']] = {'feature': value}
    return fingerprint(normalized)


def frame_fingerprint(frame):
    digest = hashlib.sha256()
    columns = sorted(frame.columns)
    digest.update(json.dumps([(name, str(frame[name].dtype)) for name in columns]).encode())
    for start in range(0, len(frame), 10000):
        chunk = frame.iloc[start:start + 10000][columns]
        try:
            hashed = pd.util.hash_pandas_object(chunk, index=True)
        except TypeError:
            chunk = chunk.copy()
            for name in chunk.select_dtypes(include='object'):
                chunk[name] = chunk[name].map(lambda value: json.dumps(value, sort_keys=True, default=str))
            hashed = pd.util.hash_pandas_object(chunk, index=True)
        digest.update(hashed.to_numpy().tobytes())
    return digest.hexdigest()


def context_identity(frame, config, window, mock):
    source_root = Path(__file__).resolve().parents[1]
    code = {}
    for folder in ('skills', 'agent'):
        for path in sorted((source_root / folder).glob('*.py')):
            code[str(path.relative_to(source_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'version': 1, 'mock': mock, 'data_contract': config['mining']['data_contract'],
        'evaluation_split': 'test', 'training_method': config['train']['training_method'], 'window': window,
        'selection_metric': SELECTION_KEY, 'training_population': config['train'].get('training_population', 'queue_only'), 'model_family': config['train'].get('model_family', 'xgb'), 'training_city': config['train'].get('training_city'),
        'data_fingerprint': frame_fingerprint(frame),
        'config_fingerprint': fingerprint({name: config[name] for name in ('data', 'features', 'train', 'mining')}),
        'implementation_fingerprint': fingerprint(code),
        'runtime_versions': {name: version(name) for name in ('pandas', 'numpy', 'xgboost')},
    }


def parent_signature(recipes, predictions, artifact_dir):
    models = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted([*Path(artifact_dir).glob('ets_pred_model_*.json'), *Path(artifact_dir).glob('ets_mlp_model_*.npz*')])}
    return fingerprint({'recipes': recipe_signature(recipes), 'models': models,
                        'predictions': frame_fingerprint(predictions[['true_bin', 'predict_bin_44']])})


def load_memory(root, current_session, context):
    entries = []
    for folder in sorted((Path(root) / 'artifacts/feature_mining').glob('S*')):
        if folder.name == current_session:
            continue
        try:
            manifest = json.loads((folder / 'session.json').read_text())
            if any(manifest.get(key) != context[key] for key in
                   ('mock', 'data_contract', 'evaluation_split', 'training_method')):
                continue
            if manifest.get('training_population', 'queue_only') != context.get('training_population', 'queue_only'):
                continue
            if manifest.get('model_family', 'xgb') != context.get('model_family', 'xgb'):
                continue
            if manifest.get('training_city') != context.get('training_city'):
                continue
            journal = folder / 'search_memory.jsonl'
            if journal.exists():
                for line in journal.read_text().splitlines():
                    try:
                        entry = json.loads(line)
                        if isinstance(entry, dict) and entry.get('status') in {'completed', 'execution_failed'}:
                            entries.append(entry)
                    except (ValueError, TypeError):
                        continue
            else:
                feedback_path = folder / 'search_feedback.json'
                if not feedback_path.exists():
                    continue
                for feedback in json.loads(feedback_path.read_text()):
                    if not isinstance(feedback, dict):
                        continue
                    entries.append({'session_id': folder.name, 'window': manifest.get('window'),
                                    'status': 'legacy_reference', 'recipes': feedback.get('recipes', []),
                                    'metrics': feedback.get('selection_metrics', {}).get(feedback.get('selection_metric', 'xgb_44'), {}).get('ALL'),
                                    'selection_metric': feedback.get('selection_metric', 'xgb_44'),
                                    'historical_gate_pass': feedback.get('gate_pass'),
                                    'reasons': feedback.get('reasons', []), 'error': feedback.get('rejected'),
                                    'reference_note': '旧门禁结果仅供参考；缺少完整上下文指纹，不禁止重试'})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return entries


def duplicate_entry(entries, context, parent, recipes, spec_signature=None):
    context_hash = fingerprint(context)
    signature = recipe_signature(recipes)
    return next((entry for entry in reversed(entries)
                 if entry.get('status') == 'completed'
                 and entry.get('context_hash') == context_hash
                 and entry.get('parent_signature') == parent
                 and entry.get('recipe_signature') == signature
                 and entry.get('spec_signature') == spec_signature), None)


def memory_for_prompt(entries, context, parent, limit=16, max_chars=32000):
    context_hash = fingerprint(context)
    ranked = sorted(enumerate(entries), key=lambda pair: (
        pair[1].get('context_hash') == context_hash and pair[1].get('parent_signature') == parent,
        pair[1].get('window') == context['window'], pair[0]), reverse=True)
    selected = []
    used_chars = 0
    for _, entry in ranked[:limit]:
        value = {key: entry[key] for key in ('session_id', 'experiment_id', 'parent_experiment_id',
                 'window', 'status', 'outcome', 'recipes', 'metrics', 'delta_pp', 'reasons', 'error',
                 'reference_note', 'diagnostics', 'historical_gate_pass', 'research_plan', 'candidate_changes',
                 'spec_signature', 'experiment_spec') if key in entry}
        value['exact_context'] = (entry.get('context_hash') == context_hash
                                  and entry.get('parent_signature') == parent)
        value['retry_allowed'] = not (value['exact_context'] and entry.get('status') == 'completed')
        value['selection_metric'] = entry.get('selection_metric', 'xgb_44')
        size = len(json.dumps(value, ensure_ascii=False))
        if used_chars + size > max_chars:
            continue
        used_chars += size
        selected.append(value)
    return selected


def append_memory(directory, entry):
    with (Path(directory) / 'search_memory.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def prefer_candidate(current_acc, current_features, candidate_acc, candidate_features):
    return (candidate_acc > current_acc or
            (candidate_acc == current_acc and candidate_features < current_features))
