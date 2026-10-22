"""Read-only, source-labelled retrieval of active experiment sessions."""
from __future__ import annotations

import json
from pathlib import Path
import re
from skills.evaluation_views import record_selection


def query_experiments(artifacts, experiments, session, query='', limit=12, max_chars=24000):
    requested = set(re.findall(r'(?<![0-9A-Z_])[SE][0-9A-Z]{7}(?![0-9A-Z_])', query.upper()))
    terms = set(re.findall(r'[A-Za-z_][A-Za-z_0-9]{2,}|[\u4e00-\u9fff]{2,}', query.lower()))
    matches = []
    for path in Path(artifacts).glob('feature_mining/*/session.json'):
        if path.parent.is_symlink():
            continue
        try:
            manifest = json.loads(path.read_text())
            session_id = path.parent.name
            if manifest.get('session_id') != session_id or manifest.get('mode') == 'chat':
                continue
            if not requested and session_id == session.get('session_id'):
                continue
            if not requested and session.get('mock') is not None and manifest.get('mock') != session['mock']:
                continue
            records_path = Path(experiments) / 'feature_mining' / session_id / 'experiments.jsonl'
            if records_path.is_symlink() or not records_path.is_file():
                continue
            with records_path.open() as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                        experiment_id = record.get('experiment_id', '')
                        if not isinstance(experiment_id, str) or not experiment_id:
                            continue
                        if requested and session_id.upper() not in requested and experiment_id.upper() not in requested:
                            continue
                        change = record.get('config_diff') or {}
                        metrics = record_selection(record)
                        summary = record.get('summary') or {}
                        result = {'session_id': session_id, 'experiment_id': experiment_id,
                                  'mock': manifest.get('mock'), 'timestamp': record.get('timestamp'),
                                  'training_mode': manifest.get('training_mode', 'pooled'),
                                  'training_population': manifest.get('training_population', 'queue_only'), 'model_family': manifest.get('model_family', 'xgb'), 'training_city': manifest.get('training_city'), 'city_name': manifest.get('city_name'),
                                  'window': manifest.get('window', record.get('data_window')),
                                  'evaluation_split': manifest.get('evaluation_split') or change.get('evaluation_split'),
                                  'training_method': manifest.get('training_method') or change.get('training_method'),
                                  'data_contract': manifest.get('data_contract'),
                                  'selection_metric': change.get('selection_metric', 'xgb_44'),
                                  'hypothesis': record.get('hypothesis', ''),
                                  'recipes': change.get('feature_recipes', []),
                                  'features': change.get('add_features', []),
                                  'parent_experiment_id': change.get('parent_experiment_id'),
                                  'metrics': {key: metrics[key] for key in ('ALL', '100001', '100002', '100003') if key in metrics},
                                  'historical_gate_pass': record.get('gate_pass'),
                                  'historical_gate_reasons': record.get('gate_reasons', []),
                                  'outcome': summary.get('verdict'),
                                  'research_plan': summary.get('research_plan'),
                                  'candidate_changes': change.get('candidate_changes'),
                                  'experiment_spec': change.get('experiment_spec'),
                                  'scope_note': '历史参考，不是当前会话结果；旧门禁与不同数据窗口不能直接混比'}
                        text = json.dumps(result, ensure_ascii=False).lower()
                        score = sum(term in text for term in terms)
                        timestamp = record.get('timestamp')
                        matches.append((score, timestamp if isinstance(timestamp, str) else '', experiment_id, result))
                    except (ValueError, TypeError, AttributeError):
                        continue
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    records, size = [], 0
    for _, _, _, result in sorted(matches, key=lambda item: item[:3], reverse=True):
        cost = len(json.dumps(result, ensure_ascii=False))
        if size + cost > max_chars:
            continue
        records.append(result)
        size += cost
        if len(records) >= limit:
            break
    return {'records': records, 'matched_records': len(matches), 'requested_ids': sorted(requested),
            'truncated': len(records) < len(matches),
            'note': '只查询未归档的实验记录，不读取其他对话的聊天内容；返回有限摘要，不代表全部历史。'}
