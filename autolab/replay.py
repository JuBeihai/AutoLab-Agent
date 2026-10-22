"""Retrain a completed run's selected candidate from local frozen inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from autolab.models import build, measure
from autolab.tasks import TaskSpec


def replay(directory):
    directory = Path(directory).resolve()
    root = Path(__file__).resolve().parents[1]
    state = json.loads((directory/'run.json').read_text())
    if state['status'] != 'complete' or not state.get('selection_locked'):
        raise ValueError('只有完成且已锁定最优方案的会话可复现')
    provenance = json.loads((directory/'provenance.json').read_text())
    for name, digest in provenance['source_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('执行代码已变化，请使用实验对应的 Git 版本复现')
    for name, version in provenance['packages'].items():
        if importlib.metadata.version(name) != version: raise ValueError('依赖版本已变化：'+name)
    for name, digest in provenance['input_hashes'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError('冻结输入校验失败：'+name)
    splits = json.loads((directory/'splits.json').read_text())
    data = pd.read_parquet(directory/'data.parquet')
    X, y = data.drop(columns='__target__'), data['__target__'].to_numpy()
    task = TaskSpec.parse(json.loads((directory/'config.json').read_text())['task'])
    locked = json.loads((directory/'selection.json').read_text())
    if locked['id'] != state['champion_id']: raise ValueError('最优方案与锁定记录不一致')
    candidate = locked['candidate']; columns = candidate['features']
    model = build(candidate, task.kind, X, task.seed)
    with threadpool_limits(limits=2):
        model.fit(X.iloc[splits['train']][columns], y[splits['train']])
        predictions = model.predict(X.iloc[splits['test']][columns])
    expected = np.load(directory/'test_predictions.npy', allow_pickle=False)
    np.testing.assert_allclose(predictions, expected, rtol=1e-6, atol=1e-8)
    metrics = measure(task.kind, y[splits['test']], predictions)
    for name, value in metrics.items():
        if value is not None: np.testing.assert_allclose(value, state['test_metrics'][name], rtol=1e-6, atol=1e-8)
    return {'status': 'matched', 'run_id': state['id'], 'test_metrics': metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory')
    print(json.dumps(replay(parser.parse_args().directory), ensure_ascii=False))
