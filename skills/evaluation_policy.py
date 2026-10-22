"""Versioned six-bucket Notebook contract (historical artifacts stay immutable)."""
import json
from pathlib import Path

BIN_EDGES = (1, 5, 10, 20, 30)
NUM_BINS = 6
CONTRACT = 'notebook_xgb43_6_queue_flag_zero_after_ordering_v1'
SELECTION_KEY = 'answer_cancel_h3_7_6bin'
SELECTION_LABEL = '6桶·应答＋取消过滤·H3-7保序后'


UNIFIED_CONTRACT = 'notebook_xgb43_6_unified_v1'


def apply_training_population(config, population):
    if population not in ('unified', 'queue_only'):
        raise ValueError('训练人群必须为 unified 或 queue_only')
    config['training_population'] = population
    config['non_queue_strategy'] = 'unified' if population == 'unified' else 'zero'
    config['non_queue_after_ordering'] = 'model' if population == 'unified' else 'zero'
    return config


def reference_baseline(population='queue_only'):
    path = Path(__file__).resolve().parents[1] / ('configs/baseline_reference_unified.json' if population == 'unified' else 'configs/baseline_reference.json')
    return json.loads(path.read_text())


def queue_mask(frame, rule='has_queue'):
    if rule == 'base_pred_bin':  # Explicit legacy reproduction only.
        return frame['base_pred_bin'] > 0
    if rule != 'has_queue':
        raise ValueError('未知排队判断规则')
    if 'has_queue' not in frame:
        raise ValueError('6桶 Notebook 口径要求 has_queue，不能以 base_pred_bin 替代')
    return frame['has_queue'].fillna(False).astype(bool)
