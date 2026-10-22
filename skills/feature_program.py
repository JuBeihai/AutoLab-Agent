"""Validated feature recipes. No eval, arbitrary Python, SQL or file access.

Recipes operate on an observed input catalog, not a prewritten feature-name pool.
Historical labels are restricted to completed training orders before each day.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from skills import feature_preprocessing

NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,79}$")
OPS = {"add": 2, "sub": 2, "mul": 2, "div": 2, "min": 2, "max": 2,
       "log1p": 1, "abs": 1, "sqrt": 1}
AGGS = {"mean", "median", "std", "min", "max", "count", "p25", "p75", "p90"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()

def recipe_info(recipe, catalog):
    dependencies = []
    def collect(node):
        if 'col' in node:
            dependencies.append(node['col'])
        for argument in node.get('args', []):
            collect(argument)
    if recipe['kind'] == 'expression':
        collect(recipe['expression'])
    elif 'column' in recipe:
        dependencies.append(recipe['column'])
    elif 'value' in recipe:
        dependencies.append(recipe['value'])
    supervised = recipe['kind'] == 'target_encoding' or (
        recipe['kind'] == 'history' and recipe['value'] == 'true_ets')
    supervised |= any(catalog.get(name, {}).get('target_derived', False) for name in dependencies)
    return {'role': 'numeric', 'target_derived': supervised}


def validate_recipes(recipes, catalog, cfg, existing=()):
    if not isinstance(recipes, list) or not 1 <= len(recipes) <= cfg["max_features_per_proposal"]:
        raise ValueError("feature_recipes 数量不合法")
    names = set(catalog) | set(existing)
    known = dict(catalog)

    def expr(node, depth=0):
        if depth > 5 or not isinstance(node, dict):
            raise ValueError("表达式必须为对象，嵌套不能超过 5 层")
        if set(node) == {"col"}:
            if node["col"] not in known or known[node["col"]].get("role") != "numeric":
                raise ValueError(f"不可用数值列: {node.get('col')}")
            return
        if set(node) == {"const"}:
            v = node["const"]
            if type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 1e6:
                raise ValueError("常数必须有限且绝对值不超过 1e6")
            return
        if set(node) != {"op", "args"} or node["op"] not in OPS:
            raise ValueError("未知表达式操作")
        if not isinstance(node["args"], list) or len(node["args"]) != OPS[node["op"]]:
            raise ValueError("运算参数数量不匹配")
        for a in node["args"]:
            expr(a, depth + 1)

    for r in recipes:
        if not isinstance(r, dict) or not NAME.fullmatch(str(r.get("name", ""))):
            raise ValueError("特征名称不合法")
        name = r["name"]
        if name in names:
            raise ValueError(f"重复或覆盖已有字段: {name}")
        if r.get("kind") == "expression":
            if set(r) != {"name", "kind", "expression"}:
                raise ValueError("expression 字段不符合协议")
            expr(r["expression"])
        elif r.get('kind') in feature_preprocessing.KINDS:
            feature_preprocessing.validate(r, known)
        elif r.get("kind") in {"history", "rolling"}:
            window = "days" if r["kind"] == "history" else "minutes"
            if set(r) != {"name", "kind", "keys", "value", "agg", window}:
                raise ValueError("聚合字段不符合协议")
            keys = r["keys"]
            if not isinstance(keys, list) or not 1 <= len(keys) <= 3 or len(set(keys)) != len(keys):
                raise ValueError("历史聚合需要 1-3 个不同分组键")
            if any(k not in catalog or not catalog[k].get("groupable") for k in keys):
                raise ValueError("历史分组键不在已探查的分组字段中")
            if r["value"] != "true_ets" and (r["value"] not in catalog or catalog[r["value"]].get("role") != "numeric"):
                raise ValueError("历史聚合值不可用")
            if r["kind"] == "history":
                if r["agg"] not in AGGS or type(r["days"]) is not int or r["days"] not in cfg["history_days"]:
                    raise ValueError("历史窗口或统计量不合法")
            elif (r["value"] == "true_ets" or r["agg"] not in {"mean", "sum", "count", "std", "min", "max"}
                  or type(r["minutes"]) is not int or r["minutes"] not in cfg["rolling_minutes"]):
                raise ValueError("分钟窗口只允许已观测数值输入，禁止标签")
        else:
            raise ValueError("未知特征配方类型")
        names.add(name)
        known[name] = recipe_info(r, known)
    return recipes


def expression(node, df):
    if "col" in node:
        return pd.to_numeric(df[node["col"]], errors="coerce").astype(float)
    if "const" in node:
        return pd.Series(float(node["const"]), index=df.index)
    args = [expression(a, df) for a in node["args"]]
    a = args[0]
    op = node["op"]
    with np.errstate(all="ignore"):
        if op == "add": out = a + args[1]
        elif op == "sub": out = a - args[1]
        elif op == "mul": out = a * args[1]
        elif op == "div": out = a / args[1].where(args[1].abs() > 1e-8)
        elif op == "min": out = np.minimum(a, args[1])
        elif op == "max": out = np.maximum(a, args[1])
        elif op == "abs": out = a.abs()
        elif op == "log1p": out = np.log1p(a.where(a >= 0))
        elif op == "sqrt": out = np.sqrt(a.where(a >= 0))
        else: raise ValueError(op)
    return pd.Series(out, index=df.index).replace([np.inf, -np.inf], np.nan)


def historical(recipe, targets, history, label_cutoff, min_count=5):
    """Daily [d-days,d) window; label maturity and training-only target history.

    No global/validation mean fallback. Missing groups remain NaN. Same-day rows,
    including earlier rows and equal timestamps, never enter the historical pool.
    """
    keys, value, agg = recipe["keys"], recipe["value"], recipe["agg"]
    src = history.copy()
    src_day = pd.to_datetime(src["create_time"]).dt.normalize()
    days = pd.to_datetime(targets["create_time"]).dt.normalize()
    src_value = pd.to_numeric(src[value], errors="coerce").replace([np.inf, -np.inf], np.nan)
    src = src.assign(_value=src_value)
    label = value == "true_ets"
    if label:
        # Labels are known at match_time, or cancel_time if no match occurred.
        ready = pd.to_datetime(src["match_time"]).combine_first(pd.to_datetime(src["cancel_time"]))
        allowed = (src_day <= pd.Timestamp(label_cutoff)) & (ready < pd.Timestamp(label_cutoff) + pd.Timedelta(days=1))
    else:
        ready = pd.to_datetime(src["prediction_time"])
        allowed = pd.Series(True, index=src.index)
    out = pd.Series(np.nan, index=targets.index, dtype=float)
    for day in sorted(days.dropna().unique()):
        day = pd.Timestamp(day)
        mask = allowed & (src_day >= day - pd.Timedelta(days=recipe["days"])) & (src_day < day) & (ready < day)
        pool = src.loc[mask & src["_value"].notna()]
        if pool.empty:
            continue
        grouped = pool.groupby(keys, dropna=True)["_value"]
        if agg.startswith("p"):
            stats = grouped.quantile(int(agg[1:]) / 100)
        else:
            stats = grouped.agg(agg)
        stats = stats.where(grouped.count() >= min_count).rename("_hist_value").reset_index()
        idx = targets.index[days == day]
        joined = targets.loc[idx, keys].merge(stats, on=keys, how="left", validate="many_to_one", sort=False)
        out.loc[idx] = joined["_hist_value"].to_numpy()
    return out


def materialize(df, recipes, catalog, cfg, label_cutoff, fitting_index=None, preprocessing_state=None):
    out = df.copy()
    # Permit a cumulative bundle, while preserving per-proposal limits at admission.
    validate_recipes(recipes, catalog, {**cfg, "max_features_per_proposal": cfg["max_total_features"]})
    for r in recipes:
        if r["kind"] == "expression":
            values = expression(r["expression"], out)
        elif r['kind'] in feature_preprocessing.KINDS:
            values = feature_preprocessing.transform(r, out, fitting_index, label_cutoff,
                                                     preprocessing_state if preprocessing_state is not None else {})
        elif r["kind"] == "rolling":
            values = rolling(r, out, df, cfg["history_min_count"])
        else:
            values = historical(r, out, df, label_cutoff, cfg["history_min_count"])
        out[r["name"]] = values.clip(-1e30, 1e30).astype(np.float32)
    return out


def rolling(recipe, targets, history, min_count=5):
    """Strict [prediction_time-window,prediction_time), including exclusion of ties."""
    keys, value, agg = recipe["keys"], recipe["value"], recipe["agg"]
    source = history.copy()
    source["_value"] = pd.to_numeric(source[value], errors="coerce").replace([np.inf, -np.inf], np.nan)
    source["_time"] = pd.to_datetime(source["prediction_time"])
    source = source.dropna(subset=keys + ["_time", "_value"]).sort_values("_time")
    result = pd.Series(np.nan, index=targets.index, dtype=float)
    grouped = source.groupby(keys, dropna=True)
    for key, group in targets.groupby(keys, dropna=True):
        try: src = grouped.get_group(key)
        except KeyError: continue
        times = src["_time"].to_numpy(dtype="datetime64[ns]")
        t = pd.to_datetime(group["prediction_time"]).to_numpy(dtype="datetime64[ns]")
        left = np.searchsorted(times, t - np.timedelta64(recipe["minutes"], "m"), side="left")
        right = np.searchsorted(times, t, side="left")
        values = src["_value"].to_numpy(dtype=float)
        count = right - left
        csum = np.r_[0., np.cumsum(values)]
        sums = csum[right] - csum[left]
        with np.errstate(all="ignore"):
            if agg == "count": out = count.astype(float)
            elif agg == "sum": out = sums
            elif agg == "mean": out = sums / count
            elif agg == "std":
                squares = np.r_[0., np.cumsum(values ** 2)]
                out = np.sqrt(np.maximum(0, (squares[right] - squares[left] - sums ** 2 / count) / (count - 1)))
            else:
                fn = np.min if agg == "min" else np.max
                out = np.array([fn(values[a:b]) if b > a else np.nan for a, b in zip(left, right)])
        out[(count < min_count) | np.isnat(t)] = np.nan
        result.loc[group.index] = out
    return result


def quality_report(df, names, max_missing=0.95):
    report = {}
    for name in names:
        s = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing, unique = float(s.isna().mean()), int(s.nunique())
        report[name] = {"missing_rate": missing, "unique": unique}
        if missing > max_missing or unique < 2:
            raise ValueError(f"特征质量不合格 {name}: missing={missing:.3f}, unique={unique}")
    return report
