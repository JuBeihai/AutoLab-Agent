"""Independent city sessions sharing one user-requested training job."""
from skills.model_registry import filename as model_filename
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import signal
import uuid

CITIES = {'100001': '示例区域A', '100002': '示例区域B', '100003': '示例区域C'}


def valid_group_id(value):
    return isinstance(value, str) and re.fullmatch(r'city_[0-9a-f]{32}', value) is not None


def select_city(frame, city):
    if city not in CITIES:
        raise ValueError('未知训练城市')
    if 'city_code' not in frame:
        raise ValueError('分城市训练要求 city_code 字段')
    selected = frame.loc[frame['city_code'].astype(str) == city].copy()
    if selected.empty:
        raise ValueError(f'{CITIES[city]}没有可用样本，不能回退为合并训练')
    return selected.reset_index(drop=True)


def assert_city(frame, city):
    if city not in CITIES or frame.empty or not frame['city_code'].astype(str).eq(city).all():
        raise ValueError('分城市模型只能使用对应城市的非空数据')


def scoped_reference(reference, city):
    result = deepcopy(reference)
    if city is None:
        return result
    if city not in CITIES:
        raise ValueError('未知训练城市')
    metric = deepcopy(result['metrics'][city])
    if metric.get('total', 0) <= 0:
        raise ValueError('城市参考基线没有有效样本')
    metric['recall_ratio'] = metric['recall'] / metric['total']
    # ALL is the aggregate of THIS session, which contains only this city.
    result['metrics'] = {city: metric, 'ALL': deepcopy(metric)}
    result['training_city'] = city
    result['aggregate_scope'] = CITIES[city]
    result['name'] += f' · {CITIES[city]}示例区域参考'
    return result


def latest_path(root, city=None, model_family='xgb', training_population='queue_only'):
    if city is not None and city not in CITIES:
        raise ValueError('未知训练城市')
    from skills.model_registry import validate
    validate(model_family)
    if training_population not in ('queue_only', 'unified'):
        raise ValueError('未知训练人群')
    suffix = ('' if model_family == 'xgb' else '_mlp') + ('_unified' if training_population == 'unified' else '')
    return Path(root) / 'experiments/feature_mining' / (f'latest_city_{city}{suffix}.json' if city else f'latest{suffix}.json')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def run_group(root, argv, run_city):
    """Sequential runs keep each city's recipes, baseline, gate and champion isolated.

    Reuse the enclosing activity lock. A city failure does not discard successful
    siblings; a user stop terminates the batch rather than starting the next city.
    """
    root = Path(root)
    group_id = 'city_' + uuid.uuid4().hex
    path = root / 'artifacts/city_training' / (group_id + '.json')
    group = {'group_id': group_id, 'status': 'running',
             'created_at': datetime.now().isoformat(timespec='seconds'),
             'cities': {city: {'city_name': name, 'status': 'pending'} for city, name in CITIES.items()}}
    save(path, group)
    old_handler = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise KeyboardInterrupt('分城市训练已停止')
    signal.signal(signal.SIGTERM, stop)
    try:
        for city, name in CITIES.items():
            entry = group['cities'][city]
            entry['status'] = 'running'
            save(path, group)
            print(f'[city-training] {name}：开始独立训练与门禁；批次 {group_id}', flush=True)
            try:
                run_city([*argv, '--city-code', city, '--city-group-id', group_id])
            except SystemExit as error:
                if error.code not in (None, 0):
                    entry.update(status='failed', error='城市训练失败；其他城市继续，详见该城市会话日志')
            except Exception as error:
                entry.update(status='failed', error=f'城市训练失败：{type(error).__name__}')
            finally:
                current = root / 'experiments/feature_mining/current.json'
                if current.exists():
                    manifest = json.loads(current.read_text())
                    if manifest.get('city_group_id') == group_id and manifest.get('training_city') == city:
                        entry['session_id'] = manifest['session_id']
                        if entry['status'] != 'failed':
                            entry['status'] = manifest['status']
                save(path, group)
        group['status'] = 'complete' if all(e['status'] in ('complete', 'profile_complete') for e in group['cities'].values()) else 'partial_failure'
    except KeyboardInterrupt:
        group['status'] = 'cancelled'
        for entry in group['cities'].values():
            if entry['status'] not in ('complete', 'profile_complete', 'failed'):
                entry['status'] = 'cancelled'
        raise
    finally:
        save(path, group)
        signal.signal(signal.SIGTERM, old_handler)
    print(f'[city-training] 三城训练批次结束：{group["status"]}；{path}', flush=True)
    if group['status'] != 'complete':
        raise SystemExit(1)
    return group


def group_summary(root, group_id):
    root = Path(root)
    if not valid_group_id(group_id):
        raise ValueError('分城市训练批次编号无效')
    group = json.loads((root / 'artifacts/city_training' / (group_id + '.json')).read_text())
    for city, entry in group['cities'].items():
        session_id = entry.get('session_id')
        if not session_id or not re.fullmatch(r'S[0-9A-Z]{7}', session_id):
            continue
        folder = root / 'artifacts/feature_mining' / session_id
        if not (folder / 'session.json').is_file():
            entry['status'] = 'archived'
            continue
        session = json.loads((folder / 'session.json').read_text())
        if session.get('training_city') != city or session.get('city_group_id') != group_id:
            raise ValueError('城市会话与批次不匹配')
        entry.update(status=session['status'], baseline_id=session.get('baseline_id'),
                     champion_id=session.get('champion_id'))
        result_path = folder / 'holdout_result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            key = result['selection_metric']
            entry.update(baseline=result['baseline'][key].get(city),
                         selected=result['selected'][key].get(city), gate=result['gate'],
                         reference=result['reference_baseline']['metrics'][city],
                         model_path=str(folder / result['selected_id'] / model_filename(session.get('model_family', 'xgb'), 'unified' if session.get('training_population') == 'unified' else 'queue')),
                         updated=result['selected_id'] != session.get('baseline_id'))
    return group


def register_session(root, group_id, city, session_id):
    if not valid_group_id(group_id) or city not in CITIES:
        raise ValueError('城市批次无效')
    path = Path(root) / 'artifacts/city_training' / (group_id + '.json')
    group = json.loads(path.read_text())
    group['cities'][city].update(session_id=session_id, status='running')
    save(path, group)
