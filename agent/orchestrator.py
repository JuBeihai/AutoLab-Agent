"""Agent Orchestrator - 主循环

Round 0: baseline（复刻 example Notebook）
Round N (N≥1):
    1. 加载上一轮 champion metrics + error slices
    2. 通过 propose_challengers() 调 LLM 网关生成候选（受白名单约束）
    3. 每个候选走 run_one_experiment（训练 + 评估 + 保存 artifact）
    4. 用 gate_check_records 与 champion 对比，写 experiments.jsonl

使用：
    export AUTOLAB_DP_KEY='<...>'
    export AUTOLAB_LLM_API_KEY='<...>'
    python agent/orchestrator.py --rounds 2 --max-proposals-per-round 3

暂时不接入 LLM 时（无 AUTOLAB_LLM_API_KEY），可用 --no-agent 只跑 baseline。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from agent.experiment_log import (
    ExperimentRecord, append_record, gate_check_records,
    get_champion, load_records, new_experiment_id, update_record,
)
from agent.identifiers import activity_lock


# ================================================================
# 配置加载
# ================================================================

def load_all_configs(root: Path) -> dict:
    """加载 configs/*.json。JSON 而不是 YAML —— 避免额外 YAML 依赖。"""
    def _load(name):
        with open(root / f"configs/{name}", "r", encoding="utf-8") as f:
            return json.load(f)
    data_cfg = _load("data.json")
    data_cfg["cities"] = {int(k): v for k, v in data_cfg["cities"].items()}
    return {
        "data": data_cfg,
        "features": _load("features.json"),
        "train": _load("train.json"),
        "llm": _load("llm.json"),
    }


# ================================================================
# LLM 提议 + 白名单校验
# ================================================================

_XGB_PARAM_WHITELIST = {
    "max_depth": (3, 12),
    "learning_rate": (0.005, 0.1),
    "n_estimators": (100, 1500),
    "subsample": (0.5, 1.0),
    "reg_alpha": (0.0, 20.0),
    "reg_lambda": (0.0, 20.0),
    "max_bin": (16, 256),
}

_XGB_STR_WHITELIST = {
    "objective": {"multi:softmax", "multi:softprob"},
    "eval_metric": {"auc", "mlogloss", "merror"},
    "tree_method": {"hist", "approx"},
}


def _normalize_proposal(prop: dict) -> dict:
    """接受两种 schema:
       (a) {"changes": {"add_features": [...], "remove_features": [...], "xgb_params": {...}}}
       (b) 顶层直接 {"features_to_add": [...], "xgb_params": {...}, ...}
    统一成 (a) 形式后返回新 dict。"""
    if isinstance(prop.get("changes"), dict) and prop["changes"]:
        return prop
    changes = dict(prop.get("changes") or {})
    # 顶层 -> changes 的键映射
    alias = {
        "features_to_add": "add_features",
        "add_features": "add_features",
        "features_to_remove": "remove_features",
        "remove_features": "remove_features",
        "xgb_params": "xgb_params",
        "params": "xgb_params",
        "training_filters": "training_filters",
    }
    for src, dst in alias.items():
        if src in prop and prop[src] is not None:
            changes.setdefault(dst, prop[src])
    out = dict(prop)
    out["changes"] = changes
    return out


def _validate_proposal(prop: dict, feat_cfg: dict) -> Optional[str]:
    """校验一个 proposal 是否满足白名单；不通过返回原因，通过返回 None。
    调用前应先经过 _normalize_proposal。"""
    if not isinstance(prop, dict):
        return f"proposal 不是 dict: {type(prop)}"

    changes = prop.get("changes", {})
    if not isinstance(changes, dict):
        return "changes 不是 dict"

    # 必须至少有一个改动
    if not any(changes.get(k) for k in
               ("add_features", "remove_features", "xgb_params", "training_filters")):
        return "proposal 无任何有效改动（changes 全空）"

    # add_features 必须在 proposable_features_pool 白名单里
    pool = set()
    for key, group_feats in feat_cfg.get("proposable_features_pool", {}).items():
        if key.startswith("_") or not isinstance(group_feats, list):
            continue  # 跳过 _note_* 说明字段
        pool.update(group_feats)
    for f in changes.get("add_features", []) or []:
        if f not in pool:
            return f"add_features 中的 {f!r} 不在 proposable_features_pool 白名单"

    # remove_features 必须在当前 44 特征里
    current_44 = set(feat_cfg["feature_set_27"]) | set(feat_cfg["feature_set_44_extra"])
    for f in changes.get("remove_features", []) or []:
        if f not in current_44:
            return f"remove_features 中的 {f!r} 不在当前 44 特征"

    # xgb_params 必须在白名单里，且值在允许区间
    xgb_params = changes.get("xgb_params", {}) or {}
    if not isinstance(xgb_params, dict):
        return "xgb_params 不是 dict"
    for k, v in xgb_params.items():
        if k in _XGB_PARAM_WHITELIST:
            lo, hi = _XGB_PARAM_WHITELIST[k]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return f"xgb_params[{k}] 不是数字: {v!r}"
            if not (lo <= fv <= hi):
                return f"xgb_params[{k}]={v} 超出白名单区间 [{lo}, {hi}]"
        elif k in _XGB_STR_WHITELIST:
            if v not in _XGB_STR_WHITELIST[k]:
                return f"xgb_params[{k}]={v!r} 不在白名单 {_XGB_STR_WHITELIST[k]}"
        else:
            return f"xgb_params[{k}] 不在允许调整的 XGB 参数白名单里"

    return None


def _format_prev_summary(prev_records: List[dict], model_tag: str = "xgb_44") -> str:
    """把最近一次实验的指标浓缩成 LLM 能吃的短描述。"""
    if not prev_records:
        return "无历史实验记录"
    last = prev_records[-1]
    m = last.get("metrics_by_city", {}).get(model_tag, {})
    lines = [f"最近实验 id={last['experiment_id']} ({last['proposer']})",
             f"假设: {last.get('hypothesis','')}"]
    for city_key in ("100001", "100002", "100003", "ALL"):
        city_m = m.get(city_key) or m.get(str(city_key))
        if not city_m:
            continue
        lines.append(
            f"  city {city_key}: n={city_m['total']} "
            f"acc={city_m['recall_ratio']:.1%} "
            f"over={city_m['over_ratio']:.1%} "
            f"under={city_m['under_ratio']:.1%} "
            f"MAE={city_m['bin_mae']:.2f} "
            f"severe_under={city_m['severe_under_ratio']:.1%}"
        )
    return "\n".join(lines)


def _format_error_slices(slices_csv_path: Path, top_n: int = 15) -> str:
    """读上一轮误差切片 CSV 的前 N 行给 LLM 参考。"""
    if not slices_csv_path.exists():
        return "无误差切片数据"
    import pandas as pd
    df = pd.read_csv(slices_csv_path).head(top_n)
    lines = ["低估率最高的切片 (top {}):".format(len(df))]
    for _, r in df.iterrows():
        lines.append(
            f"  {r['slice_col']}={r['slice_value']} n={int(r['n'])} "
            f"acc={r['acc']:.1%} under={r['under']:.1%} "
            f"severe_under={r['severe_under_ratio']:.1%}"
        )
    return "\n".join(lines)


def _load_system_prompt(root: Path) -> str:
    p = root / "agent/prompts/feature_proposer.txt"
    return p.read_text(encoding="utf-8")


def propose_challengers(
    prev_records: List[dict],
    feat_cfg: dict,
    llm_cfg: dict,
    root: Path,
    max_proposals: int = 5,
    error_slices_path: Optional[Path] = None,
    user_hint: Optional[str] = None,
) -> List[dict]:
    """调用 LLM 生成 challenger 提议，返回通过白名单校验的 proposals。

    user_hint 会作为额外的"用户指导"段落拼进 user_prompt，让 Agent 定向优化。
    """
    from skills.llm_client import LLMClient, extract_json_object, LLMError

    print(f"[propose] 步骤 1/5：加载历史实验 —— 共 {len(prev_records)} 条")
    system_prompt = _load_system_prompt(root)
    prev_summary = _format_prev_summary(prev_records)
    print(f"[propose] 步骤 2/5：提取 champion 指标摘要（{len(prev_summary)} 字符）")
    slices_summary = (
        _format_error_slices(error_slices_path) if error_slices_path else "（本轮无切片输入）"
    )
    if error_slices_path and error_slices_path.exists():
        print(f"[propose] 步骤 3/5：读取 champion 误差切片 {error_slices_path.name}")
    else:
        print(f"[propose] 步骤 3/5：无误差切片输入")

    pool = feat_cfg.get("proposable_features_pool", {})
    pool_clean = {k: v for k, v in pool.items()
                   if not k.startswith("_") and isinstance(v, list)}
    whitelist_xgb = {
        **{k: list(v) for k, v in _XGB_PARAM_WHITELIST.items()},
        **{k: sorted(v) for k, v in _XGB_STR_WHITELIST.items()},
    }
    n_pool = sum(len(v) for v in pool_clean.values())
    print(f"[propose] 步骤 4/5：装配 prompt —— 特征池 {n_pool} 项 / XGB 白名单 {len(whitelist_xgb)} 项"
          + (f" / 用户指导: {user_hint[:40]}..." if user_hint else ""))

    hint_section = ""
    if user_hint and user_hint.strip():
        hint_section = f"\n## 用户额外指导（务必优先考虑）\n{user_hint.strip()}\n"

    user_prompt = f"""## 上一轮 champion 指标
{prev_summary}

## 误差切片
{slices_summary}
{hint_section}
## 可提议特征池（add_features 只能从这里选）
{json.dumps(pool_clean, ensure_ascii=False, indent=2)}

## XGBoost 参数白名单 (xgb_params 只能改这些)
{json.dumps(whitelist_xgb, ensure_ascii=False, indent=2)}

## 输出要求
- 严格 JSON，不加 markdown 围栏（不要写 ```json）
- 顶层只有一个键 "proposals"，其值是数组，最多 {max_proposals} 个元素
- 不要输出 reasoning_summary / notes / thoughts 等额外顶层字段——保持 JSON 小
- 每个 proposal 必须包含 id / hypothesis / changes / expected_impact / risk 五个字段
- hypothesis / expected_impact / risk 各限 80 中文字以内，越简短越好
- 输出必须是完整闭合的 JSON（每个 {{ 都要配对 }}）；如内容较长请减少 proposal 数量或压缩描述
- 优先目标：降低示例区域A低估率至少 2pp，同时任一城市 acc 不下降超 1pp
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    client = LLMClient(llm_cfg)
    print(f"[propose] 步骤 5/5：调用 LLM {llm_cfg['model']} @ {llm_cfg['endpoint']}（流式，首轮通常 30-90s）",
          flush=True)

    # 流式回调：思维链前缀 [llm-think]，正式内容前缀 [llm-out]，
    # 用小段落聚合（避免每个字一行日志），~40 字或遇换行 flush 一次
    _buf = {"reasoning": [], "content": []}
    _FLUSH_CHARS = 40

    def _flush(kind: str, tail: bool = False):
        s = "".join(_buf[kind])
        if not s:
            return
        if tail or len(s) >= _FLUSH_CHARS or "\n" in s:
            prefix = "[llm-think]" if kind == "reasoning" else "[llm-out]  "
            for line in s.split("\n"):
                if line.strip():
                    print(f"{prefix} {line}", flush=True)
            _buf[kind] = []

    def _on_delta(kind: str, piece: str):
        _buf[kind].append(piece)
        _flush(kind, tail=False)

    resp = client.chat(messages, stream=True, on_delta=_on_delta)
    _flush("reasoning", tail=True)
    _flush("content", tail=True)
    print(f"[propose] LLM 返回 usage={resp.usage} finish={resp.finish_reason}", flush=True)

    if resp.finish_reason == "length":
        completion = (resp.usage or {}).get("completion_tokens")
        print(f"[propose] ✗ LLM 输出被 max_tokens 截断 (completion_tokens={completion})；"
              f"请在 configs/llm.json 里调大 max_tokens，或减小 max_proposals 数量",
              flush=True)
        return []

    try:
        parsed = extract_json_object(resp.text)
    except LLMError as e:
        print(f"[propose] JSON 解析失败: {e}", flush=True)
        return []

    proposals = parsed.get("proposals", [])
    if not isinstance(proposals, list):
        print(f"[propose] proposals 不是 list: {type(proposals)}")
        return []

    valid = []
    for i, p in enumerate(proposals[:max_proposals]):
        norm = _normalize_proposal(p) if isinstance(p, dict) else p
        err = _validate_proposal(norm, feat_cfg)
        if err:
            print(f"[propose] ✗ 候选 {i+1} 被拒绝：{err}", flush=True)
            continue
        valid.append(norm)
        ch = norm.get("changes", {}) or {}
        adds = ch.get("add_features") or []
        rems = ch.get("remove_features") or []
        params = ch.get("xgb_params") or {}
        tag_bits = []
        if adds: tag_bits.append(f"加入特征 {','.join(adds)}")
        if rems: tag_bits.append(f"移除特征 {','.join(rems)}")
        if params: tag_bits.append("调参 " + ",".join(f"{k}={v}" for k, v in params.items()))
        tag = "; ".join(tag_bits) or "无 changes"
        print(f"[propose] ✓ 候选 {i+1} 创新点：{tag}", flush=True)
        print(f"[propose]   假设：{norm.get('hypothesis', '')[:100]}", flush=True)
    print(f"[propose] 共 {len(valid)}/{len(proposals)} 个候选通过白名单校验，即将逐个训练", flush=True)
    return valid


# ================================================================
# 训练 + 评估一次完整实验
# ================================================================

def run_one_experiment(
    cfg: dict,
    proposal_changes: dict,
    round_id: int,
    proposer: str,
    hypothesis: str,
    artifact_root: Path,
    log_dir: Path,
    use_mock: bool = False,
) -> ExperimentRecord:
    from skills import data_loader, feature_builder, trainer, evaluator, model_io
    from agent import summarizer

    exp_id = new_experiment_id(root=artifact_root.parent)
    artifact_dir = artifact_root / exp_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 提前读一次 champion，作为本次实验的对比 baseline（写入 summary 用）
    pre_champion = get_champion(str(log_dir), model_tag="xgb_44", city_key="ALL")

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    train_cfg = json.loads(json.dumps(cfg["train"]))  # 深拷贝

    # 应用 proposal 里的临时修改
    if proposal_changes.get("xgb_params"):
        train_cfg["xgboost"].update(proposal_changes["xgb_params"])
    if proposal_changes.get("training_filters"):
        for grp, patch in proposal_changes["training_filters"].items():
            train_cfg["training_filters"].setdefault(grp, {}).update(patch)

    feature_columns_27 = list(feat_cfg["feature_set_27"])
    feature_columns_44 = feature_columns_27 + list(feat_cfg["feature_set_44_extra"])

    # 支持 remove_features（44 特征方案里剔除）
    remove = set(proposal_changes.get("remove_features", []) or [])
    if remove:
        feature_columns_44 = [f for f in feature_columns_44 if f not in remove]
        feature_columns_27 = [f for f in feature_columns_27 if f not in remove]

    # 取数：mock 模式跳过 Spark
    if use_mock:
        from skills.mock_data import generate_mock_merged
        print(f"[orch] 阶段 1/4：生成 mock 数据 ({data_cfg['window']['start_date']} → {data_cfg['window']['end_date']})", flush=True)
        merged = generate_mock_merged(
            start_date=data_cfg["window"]["start_date"],
            end_date=data_cfg["window"]["end_date"],
            n_per_day=2800,
            seed=42,
        )
    else:
        print(f"[orch] 阶段 1/4：从 Hive 拉取数据", flush=True)
        spark = data_loader.build_spark_session(app_name=exp_id)
        try:
            tables = data_loader.load_hive_tables(data_cfg, spark=spark)
            merged = data_loader.merge_all(
                tables["order"], tables["feature"], tables["queue"])
        finally:
            try:
                spark.stop()
            except Exception:
                pass
    print(f"[orch] 阶段 1/4 完成：合并后 {len(merged)} 行", flush=True)

    print(f"[orch] 阶段 2/4：构造特征（H3 网格 + 分箱标签）", flush=True)
    full = feature_builder.build_full_features(merged, h3_resolutions=(6, 7, 8))

    # 追加 Agent 提议的候选特征（如果有）
    add_feats = list(proposal_changes.get("add_features", []) or [])
    if add_feats:
        try:
            full = feature_builder.add_proposed_features(full, add_feats)
            feature_columns_44 = feature_columns_44 + [f for f in add_feats
                                                       if f not in feature_columns_44]
            print(f"[orch] 追加 agent 提议特征: {add_feats}（当前 44 特征方案共 {len(feature_columns_44)} 列）", flush=True)
        except feature_builder.ProposedFeatureNotImplemented as e:
            raise RuntimeError(f"proposal 拒绝执行: {e}") from e

    w = data_cfg["window"]
    train_df = full[(full["dt"] >= w["train_start"]) & (full["dt"] <= w["train_end"])].copy()
    test_df = full[(full["dt"] >= w["test_start"]) & (full["dt"] <= w["test_end"])].copy()
    print(f"[orch] 阶段 3/4：训练 4 个 XGB 模型（train={len(train_df)} test={len(test_df)}）", flush=True)

    results = trainer.train_all_four(train_df, feature_columns_27,
                                      feature_columns_44, train_cfg,
                                      num_class=feat_cfg.get("num_bins"))
    test_df = trainer.predict_test_set(test_df, results)

    print(f"[orch] 阶段 4/4：评估 & 保存 artifact", flush=True)
    metrics_by_city = {}
    for pred_col, tag in [("predict_bin_27", "xgb_27"),
                           ("predict_bin_44", "xgb_44"),
                           ("base_pred_bin", "base")]:
        m = evaluator.evaluate_by_city(test_df, pred_col=pred_col, label=tag)
        metrics_by_city[tag] = {str(k): v.as_dict() for k, v in m.items()}

    slices = evaluator.error_slices(
        test_df, pred_col="predict_bin_44",
        slice_cols=("city_code", "hour", "is_weekday", "base_pred_bin"),
    )
    slices_path = artifact_dir / "error_slices_predict_bin_44.csv"
    slices.to_csv(slices_path, index=False, encoding="utf-8")

    # slices.head(1) 里可能含 numpy 标量，用 json.loads(json.dumps(..., default=str))
    # 一次性 sanitize，避免 append_record 时 json.dumps 报错
    top_slice = []
    if len(slices):
        top_slice = json.loads(json.dumps(
            slices.head(1).to_dict("records"), default=str, ensure_ascii=False,
        ))

    save_summary = model_io.save_all_four(
        results, artifact_dir=str(artifact_dir), experiment_id=exp_id,
        data_window=w,
    )

    record = ExperimentRecord(
        experiment_id=exp_id, round_id=round_id,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        proposer=proposer, hypothesis=hypothesis,
        config_diff={
            "xgb_params": proposal_changes.get("xgb_params", {}),
            "add_features": proposal_changes.get("add_features", []),
            "remove_features": list(remove),
            "training_filters": proposal_changes.get("training_filters", {}),
        },
        data_window=w, metrics_by_city=metrics_by_city,
        artifact_dir=str(artifact_dir),
        notes=f"artifacts: {list(save_summary.keys())}, slices_top_under: {top_slice}",
    )

    # 生成实验总结（创新点 + delta vs baseline + 中文叙述）
    try:
        rec_dict = json.loads(json.dumps({
            "experiment_id": record.experiment_id,
            "round_id": record.round_id,
            "timestamp": record.timestamp,
            "proposer": record.proposer,
            "hypothesis": record.hypothesis,
            "config_diff": record.config_diff,
            "metrics_by_city": record.metrics_by_city,
            "gate_pass": record.gate_pass,
            "gate_reasons": record.gate_reasons,
        }, ensure_ascii=False, default=str))
        summary = summarizer.build_summary(
            record_dict=rec_dict,
            baseline_record=pre_champion,
            prev_train_cfg=cfg["train"],
            llm_cfg=cfg.get("llm"),
            root=_ROOT,
            use_llm=bool(os.environ.get("AUTOLAB_LLM_API_KEY")),
        )
        record.summary = summary
        summarizer.write_summary_md(summary,
                                     {"experiment_id": record.experiment_id,
                                      "round_id": record.round_id,
                                      "proposer": record.proposer,
                                      "timestamp": record.timestamp,
                                      "hypothesis": record.hypothesis,
                                      "gate_pass": record.gate_pass,
                                      "gate_reasons": record.gate_reasons},
                                     str(artifact_dir))
        print(f"[orch] 已生成实验总结: verdict={summary['verdict']} "
              f"source={summary['narrative_source']}")
    except Exception as e:
        print(f"[orch] 生成总结失败（不影响实验落盘）：{e}")
        traceback.print_exc()

    append_record(str(log_dir), record)
    return record


# ================================================================
# 主入口
# ================================================================

def _consume_user_hint(state_dir: Path) -> Optional[str]:
    """从 webui/state/user_hints.jsonl 读一条 consumed=false 的最新 hint 并标记消费。

    没有则返回 None。文件不存在也返回 None（webui 未使用时静默跳过）。
    """
    path = state_dir / "user_hints.jsonl"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(x) for x in f if x.strip()]
    except Exception as e:
        print(f"[orch] user_hints 解析失败：{e}")
        return None

    picked = None
    picked_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].get("consumed"):
            picked = lines[i]
            picked_idx = i
            break
    if picked is None:
        return None

    lines[picked_idx]["consumed"] = True
    try:
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[orch] user_hints 回写失败：{e}")

    return picked.get("hint")


def main():
    # Feature discovery is the default. Keep the old tuner explicitly opt-in.
    if "--legacy-tuning" not in sys.argv:
        from agent.feature_mining import main as mining_main
        return mining_main()
    sys.argv.remove("--legacy-tuning")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1,
                        help="Agent 挑战轮数（不含 baseline）")
    parser.add_argument("--max-proposals-per-round", type=int, default=3)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="跳过 baseline，直接从最新一条实验开始 propose")
    parser.add_argument("--no-agent", action="store_true",
                        help="只跑 baseline，不调 LLM")
    parser.add_argument("--mock", action="store_true",
                        help="用合成数据代替 Hive 取数（IP 白名单未加或本地调试时用）")
    args = parser.parse_args()

    cfg = load_all_configs(_ROOT)
    log_dir = _ROOT / "experiments"
    artifact_root = _ROOT / "artifacts"
    state_dir = _ROOT / "webui" / "state"
    log_dir.mkdir(exist_ok=True)
    artifact_root.mkdir(exist_ok=True)

    prev_records = load_records(str(log_dir))

    # Round 0: baseline
    if not args.skip_baseline and not prev_records:
        print(f"\n{'='*60}\n Round 0: baseline{' (mock 数据)' if args.mock else ''}\n{'='*60}")
        run_one_experiment(
            cfg, proposal_changes={}, round_id=0, proposer="baseline",
            hypothesis="复刻 example Notebook 的 27/44 双 XGB 结果" + (" [mock]" if args.mock else ""),
            artifact_root=artifact_root, log_dir=log_dir,
            use_mock=args.mock,
        )
        prev_records = load_records(str(log_dir))

    if args.no_agent:
        print("[orch] --no-agent 模式，不进入 Agent 循环。")
        return

    # 后续轮：Agent 循环
    start_round = 1 + max((r["round_id"] for r in prev_records), default=-1)
    for round_id in range(start_round, start_round + args.rounds):
        print(f"\n{'='*60}\n Round {round_id}: agent challenger\n{'='*60}")
        prev_records = load_records(str(log_dir))
        last = prev_records[-1]
        slices_path = Path(last["artifact_dir"]) / "error_slices_predict_bin_44.csv"

        try:
            user_hint = _consume_user_hint(state_dir)
            if user_hint:
                print(f"[orch] 本轮吸收用户指导: {user_hint[:120]}")
            proposals = propose_challengers(
                prev_records, cfg["features"], cfg["llm"], _ROOT,
                max_proposals=args.max_proposals_per_round,
                error_slices_path=slices_path,
                user_hint=user_hint,
            )
        except Exception as e:
            print(f"[orch] LLM 提议失败：{e}")
            traceback.print_exc()
            break

        if not proposals:
            print("[orch] 本轮无有效 proposal，中止。")
            break

        champion = get_champion(str(log_dir), model_tag="xgb_44", city_key="ALL")

        for i, p in enumerate(proposals):
            print(f"\n[orch] ===== 训练候选 {i+1}/{len(proposals)}：{p.get('hypothesis','')[:80]} =====",
                  flush=True)
            try:
                rec = run_one_experiment(
                    cfg, proposal_changes=p.get("changes", {}),
                    round_id=round_id, proposer="agent",
                    hypothesis=p.get("hypothesis", ""),
                    artifact_root=artifact_root, log_dir=log_dir,
                    use_mock=args.mock,
                )
                if champion:
                    gate = gate_check_records(champion, {"metrics_by_city": rec.metrics_by_city},
                                              model_tag="xgb_44")
                    verdict = "✅ 通过门禁" if gate["pass"] else "❌ 未通过门禁"
                    print(f"[orch] 候选 {i+1} 门禁：{verdict}  原因={gate['reasons']}", flush=True)
                    update_record(str(log_dir), rec.experiment_id, {
                        "gate_pass": bool(gate["pass"]),
                        "gate_reasons": list(gate["reasons"]),
                    })
            except Exception as e:
                print(f"[orch] 候选 {i+1} 执行失败: {e}", flush=True)
                traceback.print_exc()

    print("\n[orch] 完成。检查 experiments/experiments.jsonl 和 artifacts/。")


if __name__ == "__main__":
    with activity_lock(_ROOT):
        main()
