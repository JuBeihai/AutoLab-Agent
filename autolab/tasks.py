"""Task contracts and local-only data preparation, independent of any business schema."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class TaskSpec:
    kind: str = 'classification'
    dataset: str = 'synthetic'
    target: str = 'target'
    file: str = ''
    metric: str = 'accuracy'
    split: str = 'random'
    time_column: str = ''
    seed: int = 42

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('任务配置包含未知字段')
        config = dict(value)
        config.setdefault('metric', 'rmse' if config.get('kind') == 'regression' else 'accuracy')
        task = cls(**config)
        if task.kind not in ('classification', 'regression') or task.dataset not in ('synthetic', 'local_csv'):
            raise ValueError('支持表格分类、回归；数据来源为 synthetic 或 local_csv')
        allowed = ('accuracy', 'f1_macro') if task.kind == 'classification' else ('rmse', 'mae')
        if task.metric not in allowed or task.split not in ('random', 'time'):
            raise ValueError('任务指标或划分方式无效')
        if type(task.seed) is not int or not 0 <= task.seed <= 2**31-1:
            raise ValueError('seed 必须为非负整数')
        for name in ('target', 'file', 'time_column'):
            if not isinstance(getattr(task, name), str) or len(getattr(task, name)) > 160:
                raise ValueError('数据字段须为不超过160字符的文本')
        if not task.target or (task.split == 'time' and not task.time_column):
            raise ValueError('需要目标列；时间划分还需提供时间列')
        return task

    def public(self):
        return asdict(self)


def local_path(root, name):
    directory = (Path(root) / 'local_data').resolve()
    if not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('CSV 必须位于 local_data/，请填写相对文件名')
    path = directory / name
    if any(p.is_symlink() for p in [path, *path.parents] if p != directory.parent):
        raise ValueError('数据路径不能包含符号链接')
    if not path.resolve().is_relative_to(directory) or path.suffix.lower() != '.csv' or not path.is_file():
        raise ValueError('local_data/ 中没有该 CSV 文件')
    if path.stat().st_size > 50 * 1024**2:
        raise ValueError('演示执行器单个 CSV 上限为50MB')
    return path


def load_task(root, task):
    if task.dataset == 'synthetic':
        if task.split != 'random':
            raise ValueError('合成示例使用固定随机划分；时间划分请接入带时间列的本地 CSV')
        if task.kind == 'classification':
            values, y = make_classification(n_samples=900, n_features=12, n_informative=6,
                                           n_redundant=2, n_classes=3, random_state=task.seed)
        else:
            values, y = make_regression(n_samples=900, n_features=12, n_informative=6,
                                       noise=15, random_state=task.seed)
        frame = pd.DataFrame(values, columns=[f'feature_{i+1}' for i in range(values.shape[1])])
        frame[task.target] = y
    else:
        frame = pd.read_csv(local_path(root, task.file), nrows=50001)
    if not 60 <= len(frame) <= 50000 or not 2 <= len(frame.columns) <= 129:
        raise ValueError('数据须为60–50000行、1–128个输入特征')
    if task.target not in frame or frame[task.target].isna().any():
        raise ValueError('目标列缺失或包含空值')
    if task.split == 'time':
        if task.time_column == task.target or task.time_column not in frame:
            raise ValueError('时间列必须存在且不能是目标列')
        times = pd.to_datetime(frame[task.time_column], errors='raise', utc=True)
        if times.isna().any():
            raise ValueError('时间列不能有空值')
        frame = frame.assign(_autolab_time=times).sort_values('_autolab_time', kind='stable').reset_index(drop=True)
    y = frame[task.target]
    labels = None
    if task.kind == 'classification':
        if not 2 <= y.nunique() <= 50 or y.value_counts().min() < 5:
            raise ValueError('分类任务需要2–50类，每类至少5条记录')
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y.astype(str)))
        labels = [str(x) for x in encoder.classes_]
    else:
        y = pd.to_numeric(y, errors='raise')
        if not np.isfinite(y).all() or y.nunique() < 2:
            raise ValueError('回归目标须为有限且非常量的数值')
    excluded = {task.target, task.time_column, '_autolab_time'}
    features = frame.drop(columns=[x for x in excluded if x in frame]).copy()
    if features.empty:
        raise ValueError('没有可用于训练的输入列')
    # Names are kept locally; external planners receive only these anonymous IDs.
    mapping = {f'f{i:03d}': str(name) for i, name in enumerate(features.columns)}
    features.columns = list(mapping)
    for name in features:
        if pd.api.types.is_numeric_dtype(features[name]):
            features[name] = pd.to_numeric(features[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
        else:
            features[name] = features[name].fillna('__missing__').astype(str)
    estimated_width = sum(1 if features[c].dtype.kind in 'biufc' else 32 for c in features)
    if len(features) * estimated_width > 8_000_000:
        raise ValueError('编码后的预计矩阵超过本地演示内存预算，请减少行数或类别特征')
    indices = np.arange(len(frame))
    if task.split == 'time':
        # Equal timestamps never cross a split boundary.
        unique_times = frame['_autolab_time'].drop_duplicates().tolist()
        if len(unique_times) < 10:
            raise ValueError('时间划分至少需要10个不同时间点')
        a, b = unique_times[int(len(unique_times)*.6)], unique_times[int(len(unique_times)*.8)]
        times = frame['_autolab_time']
        train, valid, test = indices[times < a], indices[(times >= a) & (times < b)], indices[times >= b]
    else:
        stratify = y if task.kind == 'classification' else None
        train, rest = train_test_split(indices, test_size=.4, random_state=task.seed, stratify=stratify)
        valid, test = train_test_split(rest, test_size=.5, random_state=task.seed,
                                      stratify=y.iloc[rest] if stratify is not None else None)
    if min(len(train), len(valid), len(test)) < 10:
        raise ValueError('训练、验证、测试分区均须至少10条记录')
    if task.kind == 'classification' and set(y.iloc[train]) != set(y):
        raise ValueError('训练分区未覆盖所有类别；请调整时间窗口或数据')
    data = features.copy()
    data['__target__'] = np.asarray(y)
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes()).hexdigest()
    return {'X': features, 'y': np.asarray(y), 'labels': labels, 'feature_names': mapping,
            'splits': {k: v.tolist() for k, v in [('train', train), ('validation', valid), ('test', test)]},
            'fingerprint': fingerprint, 'rows': len(data)}
