"""Portable, fail-closed training/evaluation replay from frozen feature inputs.

No Spark, LLM, production promotion or mutation of historical models. Historical
backfills explicitly record that source/input capture happened after the run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import uuid

PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'pyarrow', 'h3', 'threadpoolctl')
META_COLUMNS = ('order_id', 'dt', 'city_code', 'has_queue', 'base_pred_bin', 'is_answer',
                'true_bin', 'true_ets', 'create_time', 'match_time', 'cancel_time',
                'h3_res6', 'h3_res7', 'h3_res8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def environment():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(),
            'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}


def freeze_code(root, shared):
    sources = sorted(p for folder in ('agent', 'skills') for p in (Path(root) / folder).rglob('*.py')
                     if '__pycache__' not in p.parts)
    hashes = {str(p.relative_to(root)): digest(p) for p in sources}
    key = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = shared / 'code' / key
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='capture-', dir=destination.parent))
        try:
            for source in sources:
                target = temporary / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if digest(target) != hashes[str(source.relative_to(root))]:
                    raise ValueError('代码在捕获过程中发生变化，请重试')
            try:
                temporary.rename(destination)
            except FileExistsError:
                pass
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination, hashes


def freeze_frame(frame, shared):
    folder = shared / 'inputs'
    folder.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(suffix='.parquet', dir=folder)
    os.close(handle)
    temporary = Path(temporary)
    try:
        frame.to_parquet(temporary, index=True)
        sha = digest(temporary)
        destination = folder / (sha + '.parquet')
        if not destination.exists():
            temporary.replace(destination)
        return destination, sha
    finally:
        temporary.unlink(missing_ok=True)


LAUNCHER = '''#!/usr/bin/env python3
# Portable entry point: only the saved source tree is imported.
import hashlib, json, sys
from pathlib import Path
base = Path(__file__).resolve().parent
manifest = json.loads((base / 'reproduction.json').read_text())
source = (base / manifest['source']['path']).resolve()
for name, expected in manifest['source']['sha256'].items():
    if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Frozen source integrity check failed: ' + name)
sys.path.insert(0, str(source))
from skills.reproduction import replay_main
raise SystemExit(replay_main(base, sys.argv[1:]))
'''
SHELL = '''#!/usr/bin/env bash
set -euo pipefail
REPLAY_DIR="$(cd "$(dirname "$0")" && pwd)"
REPLAY_PYTHON="${ETS_PYTHON:-}"
if [ -z "$REPLAY_PYTHON" ]; then
  REPLAY_PROJECT="$(cd "$REPLAY_DIR/../../../.." && pwd)"
  for candidate in "${VIRTUAL_ENV:-/nonexistent}/bin/python" "$REPLAY_PROJECT/.venv/bin/python" "$REPLAY_PROJECT/../.venv/bin/python"; do
    if [ -x "$candidate" ]; then REPLAY_PYTHON="$candidate"; break; fi
  done
  REPLAY_PYTHON="${REPLAY_PYTHON:-python3}"
fi
if [ "$(uname -s)" = "Darwin" ]; then
  REPLAY_OMP_DIR="$("$REPLAY_PYTHON" -c 'import importlib.util; from pathlib import Path; s=importlib.util.find_spec("sklearn"); print(Path(s.origin).parent / ".dylibs" if s else "")')"
  if [ -d "$REPLAY_OMP_DIR" ]; then
    export DYLD_LIBRARY_PATH="$REPLAY_OMP_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  fi
fi
exec "$REPLAY_PYTHON" "$REPLAY_DIR/reproduce.py" "$@"
'''


def capture(root, out, training, evaluation, columns, config, *, protocol, window,
            snapshot_id=None, historical=False):
    """Capture actual post-feature inputs BEFORE fitting, with original row order."""
    root, out = Path(root).resolve(), Path(out).resolve()
    if (out / 'reproduction.json').exists():
        raise ValueError('复现档案已存在，拒绝覆盖')
    shared = out.parent / '_reproduction'
    code, hashes = freeze_code(Path(__file__).resolve().parents[1], shared)
    keep = list(dict.fromkeys(columns + [c for c in META_COLUMNS if c in training or c in evaluation]))
    inputs = {}
    for name, frame in [('train', training), ('evaluation', evaluation)]:
        frozen = frame[keep].copy()
        path, sha = freeze_frame(frozen, shared)
        inputs[name] = {'path': os.path.relpath(path, out), 'sha256': sha, 'rows': len(frame)}
    cfg = json.loads(json.dumps(config))
    if cfg.get('model_family') == 'mlp':
        from skills.neural_model import validate_config
        cfg['mlp'] = validate_config(cfg.get('mlp', {}))
    doc = {'format': 'ets_reproduction_v1', 'experiment_id': out.name,
           'captured_at': datetime.now(timezone.utc).isoformat(),
           'capture_origin': 'historical_backfill' if historical else 'before_training',
           'scope': '冻结特征输入 → 从头训练（含早停） → 预测 → 全部人群/H3评估；不重新查询Hive或运行LLM',
           'protocol': protocol, 'window': window, 'snapshot_id': snapshot_id,
           'feature_columns': columns, 'effective_train_config': cfg,
           'environment': environment(), 'inputs': inputs,
           'source': {'path': os.path.relpath(code, out), 'sha256': hashes},
           'status': 'captured', 'models': []}
    write(out / 'reproduction.json', doc)
    (out / 'reproduce.py').write_text(LAUNCHER)
    (out / 'reproduce.sh').write_text(SHELL)
    (out / 'reproduce.sh').chmod(0o755)
    (out / 'requirements-reproduce.txt').write_text('\n'.join(
        f'{k}=={v}' for k, v in doc['environment']['packages'].items()) + '\n')
    return doc


def finalize(out, record):
    out = Path(out)
    doc = read(out / 'reproduction.json')
    from skills.model_registry import filename
    cfg = doc['effective_train_config']
    routes = ['unified'] if cfg['non_queue_strategy'] == 'unified' else ['queue', 'no_queue']
    files = []
    for route in routes:
        if route == 'no_queue' and cfg['non_queue_strategy'] == 'zero':
            continue
        model = out / filename(cfg.get('model_family', 'xgb'), route)
        for path in [model, Path(str(model) + '.meta.json')]:
            files.append({'path': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size})
    doc.update(status='ready', models=files, selection_metric=record['config_diff']['selection_metric'],
               expected_metrics=record['metrics_by_city'], expected_views=record.get('evaluation_views', {}))
    write(out / 'reproduction.json', doc)


def differences(expected, actual, path=''):
    """Compare all saved metrics, not just rounded ALL accuracy."""
    import math
    result = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [path + ': type mismatch']
        for key, value in expected.items():
            if key == '_evaluation_contract':
                continue
            if key not in actual:
                result.append(path + '/' + str(key) + ': missing')
            else:
                result.extend(differences(value, actual[key], path + '/' + str(key)))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12):
            result.append(f'{path}: expected={expected}, actual={actual}')
    elif expected != actual:
        result.append(f'{path}: value mismatch')
    return result


def replay_main(out, argv=None):
    parser = argparse.ArgumentParser(description='离线复现单次实验；默认重新训练，原模型不被覆盖')
    parser.add_argument('--mode', choices=['retrain', 'evaluate'], default='retrain')
    parser.add_argument('--allow-environment-drift', action='store_true', help='允许依赖版本变化，报告会注明；仍严格比较结果')
    args = parser.parse_args(argv)
    out = Path(out).resolve()
    run = out / 'reproduction_runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    report = {'mode': args.mode, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat()}
    write(run / 'result.json', report)
    write(out / ('reproduction_' + args.mode + '_result.json'), report)
    try:
        doc = read(out / 'reproduction.json')
        if doc['format'] != 'ets_reproduction_v1' or doc['status'] != 'ready':
            raise ValueError('复现档案未就绪')
        actual_env = environment()
        drift = differences(doc['environment']['packages'], actual_env['packages'], 'packages')
        if doc['environment']['python'].split('.')[:2] != actual_env['python'].split('.')[:2]:
            drift.append('Python major/minor version mismatch')
        report.update(environment=actual_env, environment_differences=drift,
                      platform_changed=actual_env['platform'] != doc['environment']['platform'])
        if drift and not args.allow_environment_drift:
            raise ValueError('环境不一致：' + '; '.join(drift) + '。请使用 requirements-reproduce.txt 和记录的 Python 版本')
        for item in list(doc['inputs'].values()) + doc['models']:
            if digest(out / item['path']) != item['sha256']:
                raise ValueError('文件校验失败：' + item['path'])
        import numpy as np
        import pandas as pd
        from skills import mining_train, model_registry, model_study
        from skills.evaluation_views import build_views, selection_metrics, SELECTION_KEY
        training = pd.read_parquet(out / doc['inputs']['train']['path'])
        target = pd.read_parquet(out / doc['inputs']['evaluation']['path'])
        cfg, columns = doc['effective_train_config'], doc['feature_columns']
        pairs = {'_queue_rule': cfg.get('queue_rule', 'has_queue'), '_training_city': cfg.get('training_city')}
        routes = ['unified'] if cfg['non_queue_strategy'] == 'unified' else ['queue', 'no_queue']
        for route in routes:
            pairs[route] = None if route == 'no_queue' and cfg['non_queue_strategy'] == 'zero' else model_registry.load(
                out / model_registry.filename(cfg.get('model_family', 'xgb'), route), cfg.get('model_family', 'xgb'))
        reference = mining_train.predict(target, pairs, columns)['predict_bin_44'].to_numpy()
        if args.mode == 'retrain':
            pairs = mining_train.train_pair(training, columns, cfg, 6, run,
                {'experiment_id': doc['experiment_id'], 'window': doc['window'], 'reproduction': True},
                on_progress=lambda message: print(message, flush=True))
        prediction = mining_train.predict(target, pairs, columns)
        if doc['protocol'] == 'model_study':
            metrics, views = model_study.measure(target, model_study.probabilities(target, pairs, columns), cfg['non_queue_after_ordering'])
        else:
            metrics = mining_train.metrics(prediction)
            views = build_views(prediction, non_queue_after_ordering=cfg['non_queue_after_ordering'])
            metrics[SELECTION_KEY] = selection_metrics(views)
        mismatches = differences(doc['expected_metrics'], metrics, 'metrics') + differences(doc['expected_views'], views, 'views')
        changed = int(np.count_nonzero(reference != prediction['predict_bin_44'].to_numpy()))
        write(run / 'metrics.json', {'metrics_by_city': metrics, 'evaluation_views': views})
        report.update(status='matched' if not mismatches and changed == 0 else 'mismatch',
                      prediction_mismatches=changed, metric_mismatches=mismatches,
                      selection_metric=doc['selection_metric'],
                      accuracy=metrics[doc['selection_metric']]['ALL']['recall_ratio'])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['result_path'] = str(run / 'result.json')
    write(run / 'result.json', report)
    write(out / ('reproduction_' + args.mode + '_result.json'), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'matched' else 1


def detail(root, record):
    """Read-only UI information, rebased to the server's current project root."""
    root = Path(root).resolve()
    session = record.get('config_diff', {}).get('session_id')
    exp = record.get('experiment_id')
    import re
    if not all(isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in (session, exp)):
        return {'status': 'unavailable', 'reason': '历史记录缺少安全的会话或实验ID'}
    out = root / 'artifacts/feature_mining' / session / exp
    if not (out / 'reproduction.json').is_file():
        return {'status': 'unavailable', 'reason': '该历史实验尚未生成复现档案，不能宣称一键可复现'}
    doc = read(out / 'reproduction.json')
    import shlex
    entry = str(out / 'reproduce.sh')
    training_details = []
    for item in doc['models']:
        if item['path'].endswith('.meta.json'):
            meta = read(out / item['path'])
            training_details.append({key: meta[key] for key in ('model_family', 'fit_rows', 'early_stop_rows',
                'best_epoch', 'epochs_run', 'early_stop_loss', 'best_num_boost_round', 'preprocessing') if key in meta})
    return {**{key: doc[key] for key in ('status', 'capture_origin', 'captured_at', 'scope', 'effective_train_config',
                                      'environment', 'feature_columns', 'window', 'snapshot_id', 'selection_metric') if key in doc},
            'artifact_directory': str(out), 'entrypoint': entry,
            'command': 'bash ' + shlex.quote(entry),
            'portable_command': 'bash ' + shlex.quote(str(Path('artifacts/feature_mining') / session / exp / 'reproduce.sh')),
            'source_directory': str((out / doc['source']['path']).resolve()),
            'source_files': {label: str((out / doc['source']['path'] / filename).resolve()) for label, filename in {
                '复现与校验实现': 'skills/reproduction.py', '模型训练': 'skills/mining_train.py',
                'MLP实现': 'skills/neural_model.py', '人群与保序评估': 'skills/evaluation_views.py',
                '特征计算': 'skills/feature_program.py'}.items()},
            'manifest_path': str(out / 'reproduction.json'),
            'requirements_path': str(out / 'requirements-reproduce.txt'),
            'feature_manifest_path': str(out / 'feature_manifest.json'),
            'training_details': training_details,
            'inputs': {k: {**v, 'path': str((out / v['path']).resolve())} for k, v in doc['inputs'].items()},
            'models': [{**v, 'path': str(out / v['path'])} for v in doc['models']],
            'verification': {mode: read(out / f'reproduction_{mode}_result.json')
                             for mode in ('evaluate', 'retrain') if (out / f'reproduction_{mode}_result.json').exists()}}
