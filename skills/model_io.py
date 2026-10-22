"""模型 IO Skill：JSON + MD5 + Meta 三件套保存与加载。

对齐 example Notebook 里的 save_xgb_model / ensure_xgb_version，
并修复"save_path 缺失导致 TypeError"的已知 bug。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import List, Optional


class XgbVersionMismatchError(RuntimeError):
    pass


def ensure_xgb_version(required_version: str = "2.1.4",
                       strict: bool = True) -> str:
    """检查 XGBoost 版本。

    - strict=True 时版本不一致会抛异常，用于产线保存/加载
    - 环境变量 ETS_XGB_STRICT=0 时把 strict 强制降级为 False，
      方便本地不同 XGB 版本调试（不改代码即可放宽）
    """
    import xgboost as xgb
    if os.environ.get("ETS_XGB_STRICT", "").strip() == "0":
        strict = False
    actual = xgb.__version__
    if actual != required_version:
        msg = (f"XGBoost 版本不匹配：当前={actual}，要求={required_version}。"
               f"请 pip install xgboost=={required_version} "
               f"或 export ETS_XGB_STRICT=0 临时放宽。")
        if strict:
            raise XgbVersionMismatchError(msg)
        print(f"WARNING: {msg}")
    return actual


def _md5(path: str, chunk: int = 1 << 20) -> str:
    m = hashlib.md5()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            m.update(buf)
    return m.hexdigest()


def save_model(
    model,
    save_path: str,
    feature_columns: List[str],
    experiment_id: str = "",
    data_window: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
    required_xgb_version: str = "2.1.4",
    write_md5: bool = True,
    write_meta: bool = True,
    strict_version: bool = True,
) -> dict:
    """保存 XGB 模型为 JSON，并生成 <path>.md5 与 <path>.meta.json。

    与 example 版本的差异：
    - save_path 必填（原版本因为该参数缺失会 TypeError）
    - experiment_id / data_window 强制写入 meta，便于 Agent 后续追溯
    """
    if not save_path:
        raise ValueError("save_path 必填，不允许留空。")
    if not save_path.endswith(".json"):
        base, _ = os.path.splitext(save_path)
        new_path = base + ".json"
        print(f"WARNING: 强制改为 JSON 格式: {save_path} -> {new_path}")
        save_path = new_path

    xgb_version = ensure_xgb_version(required_xgb_version, strict=strict_version)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save_model(save_path)

    md5_hex = _md5(save_path)
    size = os.path.getsize(save_path)

    if write_md5:
        md5_path = save_path + ".md5"
        with open(md5_path, "w", encoding="utf-8") as f:
            f.write(f"{md5_hex}  {os.path.basename(save_path)}\n")

    if write_meta:
        meta = {
            "model_file": os.path.basename(save_path),
            "experiment_id": experiment_id,
            "xgb_version": xgb_version,
            "required_xgb_version": required_xgb_version,
            "format": "json",
            "md5": md5_hex,
            "size_bytes": size,
            "train_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_features": len(feature_columns),
            "feature_columns": list(feature_columns),
            "data_window": data_window,
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path = save_path + ".meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[model_io] saved {save_path}  xgb={xgb_version}  md5={md5_hex}  size={size}B")
    return {"path": save_path, "md5": md5_hex, "size_bytes": size,
            "xgb_version": xgb_version}


def load_model(save_path: str, verify_md5: bool = True):
    """加载已保存的 JSON 模型，可选校验 MD5。"""
    import xgboost as xgb

    if not os.path.exists(save_path):
        raise FileNotFoundError(save_path)

    if verify_md5:
        md5_path = save_path + ".md5"
        if os.path.exists(md5_path):
            with open(md5_path) as f:
                expected = f.read().strip().split()[0]
            actual = _md5(save_path)
            if expected != actual:
                raise RuntimeError(
                    f"MD5 校验失败：期望 {expected}，实际 {actual}"
                )

    booster = xgb.Booster()
    booster.load_model(save_path)
    return booster


def save_all_four(
    results: dict,             # {"queue_27": TrainResult, ...}
    artifact_dir: str,
    experiment_id: str,
    data_window: Optional[dict] = None,
) -> dict:
    """一次性保存四个模型 + 通用 meta 头。"""
    os.makedirs(artifact_dir, exist_ok=True)
    summary = {}
    for label, r in results.items():
        path = os.path.join(artifact_dir, f"ets_pred_model_{label}.json")
        info = save_model(
            r.model, save_path=path, feature_columns=r.feature_columns,
            experiment_id=experiment_id, data_window=data_window,
            extra_meta={
                "label": label,
                "train_shape": list(r.train_shape),
                "eval_shape": list(r.eval_shape),
                "best_iteration": r.best_iteration,
                "train_auc_last": r.train_auc_last,
                "eval_auc_last": r.eval_auc_last,
                "class_weights": r.class_weights,
            },
        )
        summary[label] = info
    return summary
