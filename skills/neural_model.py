"""Small numeric MLP with train-only preprocessing and portable non-pickle artifacts."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import warnings

import numpy as np

DEFAULTS = {'hidden_layers': [128, 64], 'max_epochs': 80, 'patience': 10,
            'batch_size': 256, 'learning_rate': .001, 'alpha': .0001}


BOUNDS = {'max_epochs': (1, 1000), 'patience': (1, 100), 'batch_size': (1, 8192),
          'learning_rate': (0.000001, 0.1), 'alpha': (0, 10)}


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) - set(DEFAULTS):
        raise ValueError('MLP 配置包含未知参数')
    params = {**DEFAULTS, **cfg}
    layers = params['hidden_layers']
    if (not isinstance(layers, list) or not 1 <= len(layers) <= 4
            or any(type(v) is not int or not 1 <= v <= 1024 for v in layers)):
        raise ValueError('隐藏层需为 1–4 个整数，每层 1–1024 个神经元')
    for key, (lo, hi) in BOUNDS.items():
        value = params[key]
        if (type(value) not in (int, float) or not np.isfinite(value) or not lo <= value <= hi
                or (key in ('max_epochs', 'patience', 'batch_size') and type(value) is not int)):
            raise ValueError(f'{key} 必须在 [{lo}, {hi}] 内，轮数和批大小必须为整数')
    return params


class NeuralModel:
    family = 'mlp'

    def __init__(self, columns, median, mean, scale, coefs, intercepts):
        self.columns = list(columns)
        self.median, self.mean, self.scale = median, mean, scale
        self.coefs, self.intercepts = coefs, intercepts

    def transform(self, frame):
        values = frame[self.columns].to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        missing = np.isnan(values)
        values = np.where(missing, self.median, values)
        return np.concatenate([(values - self.mean) / self.scale, missing.astype(float)], axis=1).astype(np.float32)

    def predict(self, frame):
        values = self.transform(frame)
        for index, (coef, bias) in enumerate(zip(self.coefs, self.intercepts)):
            values = values @ coef + bias
            if index < len(self.coefs) - 1:
                values = np.maximum(values, 0)
        values = values - values.max(axis=1, keepdims=True)
        probabilities = np.exp(values)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def save(self, path, metadata):
        path = Path(path)
        arrays = {'median': self.median, 'mean': self.mean, 'scale': self.scale}
        for i, (coef, bias) in enumerate(zip(self.coefs, self.intercepts)):
            arrays[f'coef_{i}'], arrays[f'bias_{i}'] = coef, bias
        np.savez_compressed(path, **arrays)
        meta = {**metadata, 'model_family': 'mlp', 'format': 'ets_mlp_npz_v1',
                'feature_columns': self.columns, 'layers': len(self.coefs),
                'preprocessing': 'fit-only median imputation, standard scaling, missing indicators',
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        Path(str(path) + '.meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False))

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads(Path(str(path) + '.meta.json').read_text())
        if meta.get('format') != 'ets_mlp_npz_v1' or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('MLP 模型格式或校验和不符')
        with np.load(path, allow_pickle=False) as data:
            return cls(meta['feature_columns'], data['median'], data['mean'], data['scale'],
                       [data[f'coef_{i}'] for i in range(meta['layers'])],
                       [data[f'bias_{i}'] for i in range(meta['layers'])])


def fit(fitting, stopping, columns, labels, weights, cfg, num_class, seed, report):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import log_loss
    from threadpoolctl import threadpool_limits
    if 'sample_weight' not in inspect.signature(MLPClassifier.partial_fit).parameters:
        raise ValueError('MLP 加权训练需要 scikit-learn >= 1.7，请更新本地依赖')
    params = validate_config(cfg)
    values = fitting[columns].to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    filled = np.where(np.isnan(values), median, values)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    adapter = NeuralModel(columns, median, mean, scale, [], [])
    x = adapter.transform(fitting)
    validation = adapter.transform(stopping) if len(stopping) else None
    model = MLPClassifier(hidden_layer_sizes=tuple(params['hidden_layers']), activation='relu',
        solver='adam', batch_size=min(params['batch_size'], len(fitting)),
        learning_rate_init=params['learning_rate'], alpha=params['alpha'],
        random_state=seed, shuffle=True, early_stopping=False)
    best_loss, best_epoch, stale, best = float('inf'), 0, 0, None
    history = []
    with threadpool_limits(limits=4):
        for epoch in range(1, params['max_epochs'] + 1):
            model.partial_fit(x, labels, classes=np.arange(num_class), sample_weight=weights)
            loss = log_loss(stopping['true_bin'].astype(int), model.predict_proba(validation),
                            labels=np.arange(num_class)) if validation is not None else float(model.loss_)
            if not np.isfinite(loss):
                raise ValueError('MLP 训练损失无效，请检查输入数值范围')
            history.append({'epoch': epoch, 'train_loss': float(model.loss_), 'validation_log_loss': float(loss)})
            if loss < best_loss - 1e-6 or validation is None:
                best_loss, best_epoch, stale = loss, epoch, 0
                best = deepcopy((model.coefs_, model.intercepts_))
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0:
                report(f'MLP 第 {epoch}/{params["max_epochs"]} 轮；{"早停集" if validation is not None else "训练集"}交叉熵 {loss:.5f}')
            if validation is not None and stale >= params['patience']:
                break
    adapter.coefs, adapter.intercepts = best
    return adapter, {'best_epoch': best_epoch, 'epochs_run': epoch, 'early_stop_loss': best_loss,
                     'early_stop_metric': 'log_loss' if validation is not None else None,
                     'mlp_config': params, 'model_seed': seed, 'learning_curve': history}
