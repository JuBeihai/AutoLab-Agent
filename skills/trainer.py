"""训练 Skill：排队 / 非排队 双 XGBoost 模型。

对齐 example Notebook 的训练流程，改造点：
1. 训练样本口径（is_answer==1）来自 configs/train.yaml，不再靠注释切换
2. 训练产物带 experiment_id + 数据窗口 + 随机种子元数据
3. 支持 27/44 两套特征列一次训完
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TrainResult:
    """单个模型训练产物。"""
    model: object                          # xgb.XGBClassifier
    feature_columns: List[str]
    train_shape: tuple
    eval_shape: tuple
    best_iteration: Optional[int] = None
    train_auc_last: Optional[float] = None
    eval_auc_last: Optional[float] = None
    label: str = ""                        # 'queue_27' / 'queue_44' / 'no_queue_27' / 'no_queue_44'
    class_weights: Dict[int, float] = field(default_factory=dict)


def _filter_training_data(df: pd.DataFrame, is_queue: bool,
                          filters_cfg: dict) -> pd.DataFrame:
    """按 base_pred_bin 拆分排队/非排队，再按 configs/train.yaml 里的
    require_is_answer 决定是否过滤 is_answer==1。"""
    if is_queue:
        out = df[df["base_pred_bin"] > 0].copy()
        rule = filters_cfg.get("queue", {})
    else:
        out = df[df["base_pred_bin"] == 0].copy()
        rule = filters_cfg.get("no_queue", {})

    if rule.get("require_is_answer", False):
        out = out[out["is_answer"] == 1]
    return out.reset_index(drop=True)


def _build_sample_weights(y: pd.Series, scheme: str = "balanced_sqrt",
                          penalty_multipliers: Optional[dict] = None):
    """样本权重：balanced -> sqrt 平滑 -> 惩罚倍率。"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    if scheme == "balanced_sqrt":
        weights = np.power(weights, 0.5)
    elif scheme == "balanced":
        pass
    elif scheme == 'none':
        weights = np.ones_like(weights)
    else:
        raise ValueError(f"不支持的 class_weight scheme: {scheme}")

    if penalty_multipliers is None:
        penalty_multipliers = {}
    adjusted = {}
    for idx, cl in enumerate(classes):
        mult = float(penalty_multipliers.get(str(int(cl)), penalty_multipliers.get(int(cl), 1.0)))
        adjusted[int(cl)] = float(weights[idx]) * mult
    sample_weight = np.array([adjusted[int(cl)] for cl in y])
    return sample_weight, adjusted


def train_bucket_model(
    train_df: pd.DataFrame,
    feature_columns: List[str],
    label_col: str = "true_bin",
    xgb_params: Optional[dict] = None,
    eval_ratio: float = 0.10,
    class_weight_scheme: str = "balanced_sqrt",
    penalty_multipliers: Optional[dict] = None,
    seed: int = 123,
    label_name: str = "",
    num_class: Optional[int] = None,
) -> TrainResult:
    """训练单个 XGB 多分类模型。

    对齐 example Notebook 的排队/非排队 27/44 训练 cell，抽出通用函数。
    """
    import xgboost as xgb

    np.random.seed(seed)
    mask = np.random.sample(train_df.shape[0]) < (1.0 - eval_ratio)
    X_train = train_df[feature_columns][mask].astype(np.float32)
    X_eval = train_df[feature_columns][~mask].astype(np.float32)
    y_train = train_df[label_col][mask]
    y_eval = train_df[label_col][~mask]

    sample_weight, adjusted = _build_sample_weights(
        y_train, scheme=class_weight_scheme,
        penalty_multipliers=penalty_multipliers,
    )

    params = {
        "n_estimators": 600, "learning_rate": 0.01, "max_depth": 6,
        "subsample": 0.8, "objective": "multi:softmax",
        "n_jobs": 4, "reg_alpha": 4, "reg_lambda": 2, "random_state": seed,
        "early_stopping_rounds": 40, "eval_metric": "auc",
        "tree_method": "hist", "max_bin": 32,
    }
    if xgb_params:
        params.update(xgb_params)
    # 多分类需要显式声明 num_class：优先用配置传入的桶数（全局），
    # 否则退回到 max(label)+1，避免用 np.unique().size 在训练集缺桶时出错。
    if num_class is not None:
        params["num_class"] = int(num_class)
    else:
        params["num_class"] = int(max(int(y_train.max()), int(y_eval.max())) + 1)

    # eval_set 使用与训练集同分布的权重；y_eval 走 balanced 权重（不做 sqrt/penalty），
    # 只是为了让 XGBoost 的 eval 指标不被类不平衡拉偏。
    from sklearn.utils.class_weight import compute_class_weight
    try:
        eval_classes = np.unique(y_eval)
        eval_w = compute_class_weight(class_weight="balanced", classes=eval_classes, y=y_eval)
        eval_w_map = {int(c): float(w) for c, w in zip(eval_classes, eval_w)}
        eval_sample_weight = np.array([eval_w_map[int(c)] for c in y_eval])
    except ValueError:
        eval_sample_weight = None

    model = xgb.XGBClassifier(**params)
    model.fit(
        X=X_train, y=y_train, verbose=20,
        eval_set=[(X_train, y_train), (X_eval, y_eval)],
        sample_weight=sample_weight,
        sample_weight_eval_set=(
            [sample_weight, eval_sample_weight] if eval_sample_weight is not None else None
        ),
    )

    # 提取最后一次评估指标（多分类 AUC 在 eval 集可能是 nan，已知问题）
    evals_result = getattr(model, "evals_result_", None) or {}
    train_auc = None
    eval_auc = None
    if evals_result:
        try:
            train_auc = float(evals_result["validation_0"]["auc"][-1])
            eval_auc = float(evals_result["validation_1"]["auc"][-1])
        except (KeyError, IndexError, TypeError):
            pass

    return TrainResult(
        model=model,
        feature_columns=list(feature_columns),
        train_shape=X_train.shape,
        eval_shape=X_eval.shape,
        best_iteration=getattr(model, "best_iteration", None),
        train_auc_last=train_auc,
        eval_auc_last=eval_auc,
        label=label_name,
        class_weights=adjusted,
    )


def train_all_four(
    train_df_all: pd.DataFrame,
    feature_columns_27: List[str],
    feature_columns_44: List[str],
    train_cfg: dict,
    num_class: Optional[int] = None,
) -> Dict[str, TrainResult]:
    """一次训完 排队/非排队 × 27/44 特征 四个模型，与 example Notebook 结构一致。

    输入 train_df_all 是全量训练窗口订单（已拼接特征、含 base_pred_bin/is_answer/true_bin）。
    num_class 显式指定桶数（推荐从 features.num_bins 传入），避免非连续标签导致 XGB 报错。
    """
    filters_cfg = train_cfg.get("training_filters", {})
    train_queue = _filter_training_data(train_df_all, is_queue=True, filters_cfg=filters_cfg)
    train_no_queue = _filter_training_data(train_df_all, is_queue=False, filters_cfg=filters_cfg)

    xgb_params = dict(train_cfg["xgboost"])
    class_weight_cfg = train_cfg["class_weight"]
    split_cfg = train_cfg["split"]

    common = dict(
        xgb_params=xgb_params,
        class_weight_scheme=class_weight_cfg["scheme"],
        penalty_multipliers=class_weight_cfg["penalty_multipliers"],
        seed=split_cfg["seed"],
        num_class=num_class,
    )

    print(f"[trainer] 排队训练集 {train_queue.shape} / 非排队训练集 {train_no_queue.shape}")

    results = {}
    for is_queue, df, eval_ratio, tag in [
        (True, train_queue, split_cfg["queue_eval_ratio"], "queue"),
        (False, train_no_queue, split_cfg["no_queue_eval_ratio"], "no_queue"),
    ]:
        for feat_cols, feat_tag in [
            (feature_columns_27, "27"),
            (feature_columns_44, "44"),
        ]:
            key = f"{tag}_{feat_tag}"
            print(f"[trainer] === training {key} ===")
            results[key] = train_bucket_model(
                train_df=df,
                feature_columns=feat_cols,
                eval_ratio=eval_ratio,
                label_name=key,
                **common,
            )
    return results


def predict_test_set(
    test_df: pd.DataFrame,
    results: Dict[str, TrainResult],
    inplace: bool = False,
) -> pd.DataFrame:
    """把 4 个模型的预测结果加到 test_df 上。

    产出列：predict_queue_27, predict_queue_44, predict_no_queue_27, predict_no_queue_44
    以及最终组合列 predict_bin_27, predict_bin_44（按 base_pred_bin 拆分选模型）
    """
    df = test_df if inplace else test_df.copy()

    for feat_tag in ["27", "44"]:
        col_name = f"predict_bin_{feat_tag}"
        preds = np.full(len(df), -1, dtype=int)

        queue_mask = df["base_pred_bin"] > 0
        no_queue_mask = df["base_pred_bin"] == 0

        r_q = results[f"queue_{feat_tag}"]
        r_nq = results[f"no_queue_{feat_tag}"]

        if queue_mask.any():
            X = df.loc[queue_mask, r_q.feature_columns].astype(np.float32)
            preds[queue_mask.values] = r_q.model.predict(X)
        if no_queue_mask.any():
            X = df.loc[no_queue_mask, r_nq.feature_columns].astype(np.float32)
            preds[no_queue_mask.values] = r_nq.model.predict(X)

        df[col_name] = preds

    # 与 example 兼容：predict_bin 默认指向 44 特征（性能更好），
    # 若上层需要 27 版本可直接读 predict_bin_27。
    df["predict_bin"] = df["predict_bin_44"]
    return df
