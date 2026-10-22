"""Data-driven feature discovery, executed by the default orchestrator entry point."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.experiment_log import ExperimentRecord, append_record, new_experiment_id
from agent.identifiers import activity_lock
from agent import search_memory
from agent import experiment_spec
from agent.experiment_history import query_experiments
from agent.research_controller import ResearchController
from skills import feature_program as program
from skills import mining_data, mining_train, feature_builder, evaluator
from skills import research_extensions
from skills import data_inspection
from skills import llm_profiles
from skills.experiment_research import ExperimentResearchSession
from skills.llm_client import LLMEmptyContentError
from skills.runtime_memory import MemoryBudgetError
from skills import mining_diagnostics as diagnostics
from skills.evaluation_views import build_views, selection_metrics, SELECTION_KEY, SELECTION_LABEL
from skills.evaluation_policy import CONTRACT, UNIFIED_CONTRACT, BIN_EDGES, reference_baseline, apply_training_population
from skills import city_training


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def emit_event(event_type, session_id, **payload):
    print("[event] " + json.dumps({"type": event_type, "session_id": session_id, **payload},
                                 ensure_ascii=False, allow_nan=False), flush=True)


def load_env():
    env = Path(os.environ.get("ETS_SECRETS_FILE", str(ROOT / ".env")))
    if not env.exists(): return
    if env.stat().st_mode & 0o777 != 0o600: raise ValueError(".env 权限必须为 600")
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "): line = line[7:]
        k, sep, v = line.partition("=")
        if sep and k in {"ETS_DP_KEY", "ETS_LLM_API_KEY", "ETS_GPT_API_KEY"} and not os.environ.get(k):
            os.environ[k] = v.strip().strip("\"'")


def windows(data):
    w = dict(data["window"])
    train_start, train_end = pd.Timestamp(w['train_start']), pd.Timestamp(w['train_end'])
    if train_end < train_start or pd.Timestamp(w['test_start']) <= train_end or pd.Timestamp(w['test_end']) < pd.Timestamp(w['test_start']):
        raise ValueError('需要互不重叠且顺序正确的训练、测试窗口')
    if pd.Timestamp(w["end_date"]) < pd.Timestamp(w["test_end"]):
        raise ValueError("订单快照日期不能早于测试结束日期")
    return {"train_start": str(train_start.date()), "train_end": str(train_end.date()),
            "test_start": w["test_start"], "test_end": w["test_end"]}


def propose(profile, audit, champion, history, slices, accepted, limit, llm_cfg, hint,
            memory=None, champion_diagnostics=None, research_state=None):
    from skills.llm_client import LLMClient, extract_json_object
    system = (ROOT / "agent/prompts/feature_miner.txt").read_text()
    if research_state is not None:
        system += '\n\n' + (ROOT / 'agent/prompts/research_planner.txt').read_text()
    if profile.get('research_capabilities'):
        system += '\n\n' + (ROOT / 'agent/prompts/research_skills.txt').read_text()
    if profile.get('web_research'):
        system += '\n\n' + (ROOT / 'agent/prompts/web_research.txt').read_text()
    payload = {"profile": {key: value for key, value in profile.items() if key != 'historical_experiments'},
               'historical_experiments': profile.get('historical_experiments', {}),
               'research_state': research_state, "sources": audit.get("sources", []),
               'evaluation_split': 'test', 'selection_metric': SELECTION_KEY,
               'evaluation_note': '同一测试窗口用于所有实验评估及候选筛选，存在选择偏差，不是未参与调优的独立评估',
               "champion_selection_metrics": champion, "recent_experiments": history[-12:],
               "selection_error_slices": slices, "accepted_recipes": accepted,
               "search_memory": memory or [], "champion_diagnostics": champion_diagnostics or {},
               "user_guidance": hint, "max_proposals": limit}
    # 挖矿 prompt 的 catalog 可能有几十项 JSON，CoT 需求大；给 12000 token 预算 + 流式
    client = LLMClient({**llm_cfg, "max_tokens": max(12000, llm_cfg.get("max_tokens", 4000))})

    _buf = {"reasoning": [], "content": []}
    _FLUSH_CHARS = 40
    def _flush(kind, tail=False):
        s = "".join(_buf[kind])
        if not s: return
        if tail or len(s) >= _FLUSH_CHARS or "\n" in s:
            prefix = "[llm-think]" if kind == "reasoning" else "[llm-out]  "
            for line in s.split("\n"):
                if line.strip():
                    print(f"{prefix} {line}", flush=True)
            _buf[kind] = []
    def _on_delta(kind, piece):
        _buf[kind].append(piece); _flush(kind)

    response = client.chat([{"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}],
                           stream=True, on_delta=_on_delta)
    _flush("reasoning", tail=True); _flush("content", tail=True)
    if response.finish_reason == "length": raise ValueError("特征配方输出被截断，请减少候选数量")
    result = extract_json_object(response.text)
    if research_state is not None:
        return result
    if not isinstance(result.get("proposals"), list): raise ValueError("LLM 未返回 proposals 数组")
    return result["proposals"][:limit]


def validate_proposal(p, catalog, accepted, cfg):
    if not isinstance(p, dict) or set(p) != {"id", "hypothesis", "evidence", "feature_recipes"}:
        raise ValueError("候选只能包含 id/hypothesis/evidence/feature_recipes；禁止调参")
    for key in ("id", "hypothesis", "evidence"):
        if not isinstance(p[key], str) or not p[key].strip() or len(p[key]) > 1500:
            raise ValueError(f"{key} 不能为空或过长")
    known = {**catalog, **{r["name"]: {"role": "numeric"} for r in accepted}}
    program.validate_recipes(p["feature_recipes"], known, cfg)
    if len(accepted) + len(p["feature_recipes"]) > cfg["max_total_features"]:
        raise ValueError("累计特征数超过预算")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-proposals-per-round", type=int, default=3)
    parser.add_argument('--continuous', action='store_true', help='逐次规划与反馈；尝试预算为轮数×每轮候选数')
    parser.add_argument('--research-skills', action='store_true', help='允许特征编辑、同会话分支、有界调参及扩展申请')
    parser.add_argument('--model-study', action='store_true', help='MLP自主模型实验：时间验证、XGB公平对照、调参、多种子复验与概率融合')
    parser.add_argument('--web-research', action='store_true', help='联网检索公开论文与官方文档，作为实验规划参考')
    parser.add_argument('--llm-profile', help='选择 configs/llm.json 中的模型：chat 或 responses')
    parser.add_argument('--search-timeout-minutes', type=int, default=120, help='baseline 后搜索时间预算；不强杀已开始的实验')
    parser.add_argument('--search-patience', type=int, default=9, help='连续无冠军提升的最大尝试次数')
    parser.add_argument("--mock", action="store_true")
    parser.add_argument('--data-source', choices=('online', 'offline'), default='online')
    parser.add_argument('--snapshot-id', help='data_snapshots 下的完整快照目录名；离线模式必填')
    parser.add_argument("--no-agent", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=0, help="兼容旧参数；Notebook 一致模式只接受 0，不允许抽样")
    parser.add_argument("--proposal-file", type=Path, help="读取人工配方 JSON 验证执行链路，不调用 LLM")
    parser.add_argument("--smoke", action="store_true", help="仅 mock：缩小样本和树数，不能代表效果")
    parser.add_argument("--skip-baseline", action="store_true", help="兼容旧脚本；新实验仍重建同口径 baseline")
    parser.add_argument("--resume-recipes", action="store_true",
                        help="从上一个 session 继承 accepted_recipes 作为新 session 的起点；默认从空开始")
    parser.add_argument('--training-population', choices=('unified', 'queue_only'), default=None)
    parser.add_argument('--model-family', choices=('xgb', 'mlp'), default='xgb')
    parser.add_argument('--training-mode', choices=('pooled', 'by_city'), default='pooled')
    parser.add_argument('--city-code', choices=tuple(city_training.CITIES), help=argparse.SUPPRESS)
    parser.add_argument('--city-group-id', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.model_study and (args.model_family != 'mlp' or args.no_agent or args.profile_only or args.proposal_file or args.rounds * args.max_proposals_per_round < 4):
        parser.error('自主模型实验需要MLP、至少4个候选预算；不能同时只跑baseline/画像/人工配方')
    if args.city_code and args.training_mode != 'by_city':
        parser.error('--city-code 仅用于分城市训练')
    if args.city_group_id and (not args.city_code or not city_training.valid_group_id(args.city_group_id)):
        parser.error('分城市训练批次编号无效')
    if not args.mock and args.data_source == 'offline' and not args.snapshot_id:
        parser.error('离线模式需要 --snapshot-id')
    if args.snapshot_id and (args.mock or args.data_source != 'offline'):
        parser.error('--snapshot-id 仅用于离线真实数据模式')
    if not 1 <= args.search_timeout_minutes <= 1440 or not 1 <= args.search_patience <= 160:
        parser.error('搜索时间需为1–1440分钟，连续无提升预算需为1–160次')
    if args.continuous and args.proposal_file:
        parser.error('--continuous 不能和 --proposal-file 同时使用')
    if not 0 <= args.rounds <= 20 or not 1 <= args.max_proposals_per_round <= 8 or args.sample_rows < 0:
        parser.error("轮数、候选数或抽样数超出范围")
    if args.smoke and not args.mock: parser.error("--smoke 只允许与 --mock 一起使用")
    if args.sample_rows and not args.mock:
        parser.error('Notebook 一致模式不允许抽样；请移除 --sample-rows 或设为 0，数据过大时缩短日期窗口')
    if args.mock and args.sample_rows:
        print('[mining] mock 模式不使用真实 SDS 抽样参数', flush=True)
    if args.training_mode == 'by_city' and not args.city_code:
        return city_training.run_group(ROOT, list(sys.argv[1:] if argv is None else argv), main)
    load_env()
    config = {name: json.loads((ROOT / f"configs/{name}.json").read_text())
              for name in ("data", "features", "train", "llm", "mining")}
    try:
        config['llm'] = llm_profiles.resolve(config['llm'], args.llm_profile)
    except ValueError as error:
        parser.error(str(error))
    config['mining'] = {**config['mining'], 'research_skills': args.research_skills}
    cfg, train_cfg = config["mining"], config["train"]
    train_cfg.update(training_mode=args.training_mode, training_city=args.city_code, model_family=args.model_family)
    args.training_population = args.training_population or train_cfg.get('training_population', 'queue_only')
    apply_training_population(train_cfg, args.training_population)
    cfg['training_population'] = args.training_population
    cfg['training_city'] = args.city_code
    reference = city_training.scoped_reference(reference_baseline(args.training_population), args.city_code)
    latest_path = city_training.latest_path(ROOT, args.city_code, args.model_family, args.training_population)
    if config['features']['num_bins'] != 6 or config['features']['bin_edges_minutes'] != list(BIN_EDGES):
        parser.error('新实验必须使用 Notebook 的6桶边界')
    win = windows(config['data'])
    if not args.mock and args.data_source == 'offline':
        from skills.local_snapshot import validate
        try:
            validate(ROOT, args.snapshot_id, config['data'])
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.smoke:
        train_cfg["xgboost"].update(n_estimators=12, max_depth=3)
        train_cfg["mlp"] = {"max_epochs": 3, "patience": 2}
    # --resume-recipes 会继承上一次成功 session 的 accepted_recipes。
    # 只读 latest.json —— 它按约定只指向"最近一次真正成功"的 session（status=complete
    # 且至少有一个 recipe 通过门禁）；失败/无进展的 session 不会污染这里。
    _prev_accepted_recipes = []
    _prev_session_id = None
    if args.resume_recipes:
        _prev = latest_path
        if _prev.exists():
            try:
                _pm = json.loads(_prev.read_text())
                if _pm.get("mock") is not args.mock:
                    parser.error("不能在 mock 与真实数据之间继承特征；请取消继承或选择相同数据模式")
                if _pm.get('training_population', 'queue_only') != args.training_population:
                    parser.error('不能跨统一训练与只训练排队继承特征')
                if _pm.get('training_city') != args.city_code:
                    parser.error('不能跨城市或跨训练模式继承特征')
                if _pm.get('data_contract') != CONTRACT:
                    parser.error('旧会话取数口径与 Notebook 不一致，不能继承其配方；请关闭继承')
                if (_pm.get('evaluation_split') != 'test' or _pm.get('training_method') != train_cfg['training_method']
                        or _pm.get('selection_metric') != SELECTION_KEY):
                    parser.error('旧会话筛选口径不同，不能直接继承；请关闭继承')
                _prev_accepted_recipes = _pm.get("accepted_recipes") or []
                _prev_session_id = _pm.get("session_id")
            except Exception:
                pass
    session = new_experiment_id("session", root=ROOT)
    directory = ROOT / "artifacts/feature_mining" / session
    logs = ROOT / "experiments/feature_mining" / session
    directory.mkdir(parents=True)
    logs.mkdir(parents=True)
    manifest = {"session_id": session, "created_at": datetime.now().isoformat(timespec="seconds"),
                "mode": "feature_mining", "mock": args.mock, "smoke": args.smoke,
                'training_population': args.training_population, 'model_family': args.model_family, 'training_mode': args.training_mode, 'training_city': args.city_code,
                'city_name': city_training.CITIES.get(args.city_code), 'city_group_id': args.city_group_id,
                'title': (city_training.CITIES[args.city_code] + ' · 独立训练') if args.city_code else '三城合并训练',
                'data_source': 'mock' if args.mock else args.data_source,
                'snapshot_id': args.snapshot_id,
                'data_contract': CONTRACT,
                'evaluation_split': 'test', 'training_method': train_cfg['training_method'],
                'selection_metric': SELECTION_KEY,
                'num_bins': 6, 'bin_edges_minutes': list(BIN_EDGES), 'reference_baseline': reference,
                'selection_population': '固定测试源样本；各模型按保序前预测筛选应答＋取消订单，再做H3-7保序；过滤后人群可不同。非排队策略：' + train_cfg['non_queue_after_ordering'],
                'selection_warning': '测试集用于候选筛选和 Agent 反馈，存在选择偏差，不是未参与调优的独立评估',
                'reference_notebook': reference['source_notebook'],
                "web_research_enabled": args.web_research,
                "web_research_mode": "autonomous_tools" if args.web_research else "disabled",
                "status": "loading", "window": win, "fixed_train_config": train_cfg,
                'llm': {key: config['llm'][key] for key in ('profile_id', 'model', 'protocol')},
                "sample_rows_requested": args.sample_rows if not args.mock else 0,
                "non_queue_strategy": train_cfg.get('non_queue_strategy', 'model'),
                "config_hash": program.fingerprint(config),
                "log_dir": str(logs), "artifact_dir": str(directory), "accepted_recipes": [],
                "resume_from_session_id": _prev_session_id}
    write_json(directory / "session.json", manifest)
    if args.city_group_id:
        city_training.register_session(ROOT, args.city_group_id, args.city_code, session)
    # 每次 session 都无条件写 current.json（记录最近一次跑，无论成败）
    # latest.json 只在成功完成且至少有一次通过门禁时更新（在 finally 中判断）
    write_json(ROOT / "experiments/feature_mining/current.json", manifest)
    emit_event("session_updated", session, status=manifest["status"])
    def progress(stage, message, **details):
        emit_event('progress', session, stage=stage, message=message, **details)

    print(f'[mining] session={session}; 6桶、43特征、训练方式={train_cfg["training_method"]}；测试窗口：{win}', flush=True)
    print(f'[mining] 示例阈值={reference["metrics"]["ALL"]["recall_ratio"]:.4%}；新会话对照另行训练，非独立评估', flush=True)
    print(f"[mining] 本次研究模型：{config['llm']['model']} · {config['llm']['protocol']}", flush=True)
    if args.skip_baseline: print("[mining] 新会话会重建同样本 baseline，旧实验不会混入比较", flush=True)
    try:
        if args.mock:
            progress('loading', '正在生成 mock 数据')
            from skills.mock_data import generate_mock_merged
            start = str((pd.Timestamp(win["train_start"]) - pd.Timedelta(days=28)).date())
            full = generate_mock_merged(start, win["test_end"], n_per_day=300 if args.smoke else 1200)
            full["prediction_time"] = pd.to_datetime(full["create_time"])
            full["queue__vehicleServiceTime"] = full["valid_remainingservicetime_mean"]
            full["queue__queueSize"] = full["sp1_queue_count"] + full["sp2_queue_count"]
            full = feature_builder.build_full_features(full)
            lineage = {n: {"source": "mock"} for n in full if n.startswith("queue__")}
            audit = {"sources": [], "mode": "mock", "historical_population": "synthetic SDS sample"}
        elif args.data_source == 'offline':
            full, lineage, audit = mining_data.load_local(config['data'], cfg, win['train_end'],
                args.snapshot_id, root=ROOT, sample_rows=args.sample_rows,
                on_progress=lambda message: progress('loading', message))
        else:
            full, lineage, audit = mining_data.load_live(config["data"], cfg, win["train_end"], args.sample_rows,
                on_progress=lambda message: progress('loading', message))
        if args.city_code:
            full = city_training.select_city(full, args.city_code)
            audit.update(training_city=args.city_code, city_name=city_training.CITIES[args.city_code],
                         training_rows=len(full))
        full["dt"] = full["dt"].astype(str)
        if args.model_study:
            from skills import model_study
            model_study.run(ROOT, full, config, manifest, args, progress, emit_event)
            return
        progress('profiling', f'取数完成：{len(full):,} 行；正在生成训练期字段画像')
        profile = mining_data.profile_frame(full, win["train_start"], win["train_end"], lineage, cfg)
        profile['training_scope'] = {'mode': args.training_mode, 'city_code': args.city_code,
            'city_name': city_training.CITIES.get(args.city_code),
            'reference_baseline': reference,
            'note': '当前城市独立取样、训练、评估与更新；ALL 仅表示当前城市全部评估样本' if args.city_code else '三城合并训练'}
        catalog = profile["catalog"]
        profile['fitting_populations'] = diagnostics.fitting_profiles(
            full.loc[full['dt'].between(win['train_start'], win['train_end'])], catalog, train_cfg)
        inspection_count = 0
        inspection_seen = set()
        if args.research_skills:
            profile['data_inspection'] = data_inspection.capabilities(audit, cfg)
            profile['inspection_results'] = []
        published_profile = dict(profile)
        write_json(directory / "data_profile.json", published_profile)
        write_json(directory / "source_audit.json", audit)
        manifest.update(status="profiled", profile_columns=len(catalog), rows=len(full))
        write_json(directory / "session.json", manifest)
        emit_event("session_updated", session, status=manifest["status"])
        print(f"[mining] 已探查 {len(catalog)} 个可用字段，共 {len(full)} 行；画像只使用训练期", flush=True)
        if args.profile_only:
            manifest["status"] = "profile_complete"
            return
        base_columns = config["features"]["feature_set_27"] + config["features"]["feature_set_44_extra"]
        missing = set(base_columns) - set(full)
        if missing: raise ValueError(f"baseline 缺少字段: {sorted(missing)}")
        train_mask = full["dt"].between(win["train_start"], win["train_end"])
        test_mask = full["dt"].between(win["test_start"], win["test_end"])
        if any(not mask.any() for mask in (train_mask, test_mask)):
            raise ValueError('训练/测试至少一个窗口无可用数据')
        selection = full.loc[test_mask]
        selection_hash = program.fingerprint(selection[['order_id', 'dt']].astype(str).values.tolist())
        manifest['rows_by_split'] = {'train': int(train_mask.sum()), 'test': int(test_mask.sum())}
        manifest['selection_sample_hash'] = selection_hash
        manifest['selection_rows_by_city'] = {str(city): int(count) for city, count in selection.groupby('city_code').size().items()}
        sample_hash = program.fingerprint(full.loc[train_mask | test_mask, ['order_id', 'dt']].astype(str).values.tolist())
        accepted, feedback = [], []
        context = search_memory.context_identity(full, config, win, args.mock)
        memories = search_memory.load_memory(ROOT, session, context)
        manifest['search_context'] = context
        manifest['memory_entries_loaded'] = len(memories)
        manifest['champion_policy'] = 'answer_cancel_h3_7_6bin_reference_floor_then_fewer_features_v1'
        evaluation_contract = {'schema': UNIFIED_CONTRACT if args.training_population == 'unified' else CONTRACT, 'training_population': args.training_population, 'source_sample_hash': selection_hash, 'mock': args.mock,
                               'reference_baseline': reference, 'training_city': args.city_code, 'model_family': args.model_family}
        manifest['exploration_branches'] = []
        profile['historical_experiments'] = query_experiments(ROOT / 'artifacts', ROOT / 'experiments', manifest)
        write_json(directory / 'history_retrieval.json', profile['historical_experiments'])
        write_json(directory / 'search_context.json', context)
        print(f'[mining] 已加载 {len(memories)} 条跨会话实验记忆；仅完整上下文相同的已完成配方禁止重复训练', flush=True)

        # --resume-recipes：从上一个 session 继承 accepted_recipes 作为起点
        # 注意 _prev_accepted_recipes 是在覆盖 latest.json 之前读到的，避免自覆盖
        if args.resume_recipes:
            if _prev_accepted_recipes:
                known = dict(catalog)
                survived = []
                for r in _prev_accepted_recipes:
                    try:
                        program.validate_recipes([r], known, {**cfg, "max_features_per_proposal": 1})
                        survived.append(r)
                        known[r["name"]] = program.recipe_info(r, known)
                    except Exception as exc:
                        print(f"[mining] --resume-recipes: 丢弃 {r.get('name')}：{exc}", flush=True)
                accepted = survived
                if accepted:
                    print(f"[mining] --resume-recipes: 从 {_prev_session_id} 继承 {len(accepted)} 个配方："
                          f"{[r['name'] for r in accepted]}", flush=True)
                else:
                    print(f"[mining] --resume-recipes: 上一 session 的配方与当前 catalog 均不兼容", flush=True)
            else:
                print(f"[mining] --resume-recipes: 上一 session 没有可继承的配方", flush=True)

        inherited_recipes = list(accepted)
        manifest.update(inherited_recipes=inherited_recipes, accepted_recipes=list(accepted),
                        resume_from_session_id=_prev_session_id if inherited_recipes else None)

        baseline_spec = experiment_spec.baseline_spec(base_columns, accepted, train_cfg)
        champion_spec = baseline_spec
        experiments_by_id = {}
        predictions_by_id = {}
        registered_extensions = research_extensions.approved(ROOT, mock=args.mock) if args.research_skills else []
        if args.research_skills:
            profile['research_capabilities'] = experiment_spec.capabilities(train_cfg)
            profile['approved_extensions'] = registered_extensions
        manifest['research_skills'] = args.research_skills

        def experiment(recipes, round_id, p, candidate_spec=None, parent_record=None):
            is_baseline = round_id == 0
            spec = candidate_spec or experiment_spec.baseline_spec(base_columns, recipes, train_cfg)
            effective_train = spec['train_config']
            exp_id = new_experiment_id(root=ROOT)
            out_dir = directory / exp_id
            out_dir.mkdir()
            progress('features', f'{"baseline" if is_baseline else "候选"} {exp_id}：正在准备特征',
                     experiment_id=exp_id, round_id=round_id)
            inputs = research_extensions.materialize(ROOT, full, spec['extensions']) if spec['extensions'] else full
            effective_catalog = {**catalog, **{name: {'role': 'numeric'} for extension in spec['extensions'] for name in extension['outputs']}}
            preprocessing_state = {}
            fitting = mining_train.fitting_mask(inputs, True, effective_train)
            if effective_train.get('non_queue_strategy') != 'zero':
                fitting |= mining_train.fitting_mask(inputs, False, effective_train)
            if effective_train['training_method'] == 'notebook_random':
                fit_rows = mining_train.notebook_split(inputs.loc[train_mask], True, effective_train)[0].index
                fitting &= inputs.index.isin(fit_rows)
            expanded = program.materialize(inputs, recipes, effective_catalog, cfg, win["train_end"],
                        fitting_index=inputs.index[train_mask & fitting],
                        preprocessing_state=preprocessing_state) if recipes else inputs
            added = [r["name"] for r in recipes] + [name for extension in spec['extensions'] for name in extension['outputs']]
            quality = program.quality_report(expanded.loc[train_mask], added, cfg["max_missing_rate"])
            # Reject pure copies and duplicate columns using training data only.
            for index, name in enumerate(added):
                for old in spec['base_features'] + added[:index]:
                    if np.allclose(expanded.loc[train_mask, name], expanded.loc[train_mask, old], equal_nan=True):
                        raise ValueError(f"{name} 与 {old} 重复，无新增信号")
            columns = experiment_spec.feature_names(spec)
            recipe_doc = {"recipes": recipes, "input_catalog": catalog, "window": win,
                          'evaluation_split': 'test', 'selection_sample_hash': selection_hash,
                          "fixed_train_config": effective_train, "sample_hash": sample_hash, "quality": quality,
                          'experiment_spec': spec, 'feature_columns': columns,
                          'preprocessing_state': preprocessing_state}
            write_json(out_dir / "feature_manifest.json", recipe_doc)
            from skills import reproduction
            reproduction.capture(ROOT, out_dir, expanded.loc[train_mask], expanded.loc[test_mask],
                                 columns, effective_train, protocol='feature_mining', window=win,
                                 snapshot_id=args.snapshot_id)
            pairs = mining_train.train_pair(expanded.loc[train_mask], columns, effective_train,
                        config["features"]["num_bins"], out_dir,
                        {"experiment_id": exp_id, "window": win, "recipes": recipes,
                         'training_population': args.training_population, 'model_family': args.model_family, 'training_mode': args.training_mode, 'training_city': args.city_code,
                         'num_bins': 6, 'bin_edges_minutes': list(BIN_EDGES), 'data_contract': CONTRACT,
                         'queue_rule': effective_train['queue_rule'], 'non_queue_after_ordering': effective_train['non_queue_after_ordering'],
                         "sample_hash": sample_hash, "train_config_hash": program.fingerprint(effective_train), "mock": args.mock},
                        on_progress=lambda message: progress('training', f'{exp_id} · {message}',
                                                             experiment_id=exp_id, round_id=round_id))
            coverage = diagnostics.feature_coverage(expanded.loc[train_mask], expanded.loc[test_mask], columns, effective_train)
            usage = diagnostics.model_usage(pairs, columns)
            selection_report = {}
            if args.research_skills:
                progress('diagnostics', f'{exp_id}：评估全部输入的重要性、冗余和训练内置换敏感度',
                         experiment_id=exp_id, round_id=round_id)
                selection_report = diagnostics.feature_selection_report(
                    expanded.loc[train_mask], pairs, columns, effective_train, cfg)
            progress('evaluating', f'{exp_id}：正在评估固定测试集 {int(test_mask.sum()):,} 行',
                     experiment_id=exp_id, round_id=round_id)
            selection_pred = mining_train.predict(expanded.loc[test_mask], pairs, columns)
            met = mining_train.metrics(selection_pred)
            evaluation_views = build_views(selection_pred, non_queue_after_ordering=effective_train['non_queue_after_ordering'], on_progress=lambda message:
                progress('evaluating', f'{exp_id} · {message}', experiment_id=exp_id, round_id=round_id))
            met[SELECTION_KEY] = selection_metrics(evaluation_views)
            met['_evaluation_contract'] = evaluation_contract
            slices = diagnostics.accuracy_slices(selection_pred)
            slices.to_csv(out_dir / "error_slices_predict_bin_44.csv", index=False)
            errors = diagnostics.accuracy_feedback(selection_pred)
            changes = {} if is_baseline else {
                'parent': diagnostics.prediction_changes(predictions_by_id[parent_record.experiment_id] if parent_record else champion_frame, selection_pred),
                'baseline': diagnostics.prediction_changes(baseline_frame, selection_pred)}
            if not is_baseline and args.research_skills:
                changes['champion'] = diagnostics.prediction_changes(champion_frame, selection_pred)
            search_diagnostics = {'errors': errors, 'prediction_changes': changes,
                                  'features': diagnostics.compact_features(coverage, usage, added),
                                  'feature_selection': selection_report,
                                  'fitting_rows': {tag: entry['rows'] for tag, entry in profile['fitting_populations'].items()}}
            if args.model_family == 'mlp':
                search_diagnostics['mlp_training'] = [
                    {key: meta[key] for key in ('best_epoch', 'epochs_run', 'early_stop_loss', 'mlp_config') if key in meta}
                    for path in out_dir.glob('*.npz.meta.json')
                    for meta in [json.loads(path.read_text())]]
            write_json(out_dir / 'diagnostics.json', {
                **search_diagnostics, 'feature_coverage': coverage, 'model_usage': usage})
            verdict = {"pass": None, "reasons": []} if is_baseline else mining_train.gate(champion_metrics, met)
            selection_deltas = {} if is_baseline else {
                key: 100 * (met[SELECTION_KEY]['ALL']['recall_ratio'] - metrics[SELECTION_KEY]['ALL']['recall_ratio'])
                for key, metrics in {'parent': parent_record.metrics_by_city if parent_record else champion_metrics,
                                     'baseline': baseline_metrics, 'champion': champion_metrics}.items()}
            delta = selection_deltas.get('champion')
            outcome = 'baseline' if is_baseline else 'improvement' if delta > 0 else 'neutral' if delta == 0 else 'regression'
            eligible = not is_baseline and verdict['pass'] and search_memory.prefer_candidate(
                champion_metrics[SELECTION_KEY]['ALL']['recall_ratio'], len(experiment_spec.feature_names(champion_spec)),
                met[SELECTION_KEY]['ALL']['recall_ratio'], len(columns))
            parent_spec = parent_record.config_diff['experiment_spec'] if parent_record else champion_spec
            candidate_changes = {} if is_baseline else experiment_spec.describe_change(parent_spec, spec)
            title = '固定配置 baseline' if is_baseline else ', '.join(added)
            if args.research_skills and not is_baseline:
                title = ('新增 ' + ', '.join(candidate_changes['added']) if candidate_changes['added'] else '')
                if candidate_changes['removed']:
                    title += '；删除 ' + ', '.join(candidate_changes['removed'])
                if candidate_changes['xgb_params'] or candidate_changes.get('mlp_params') or parent_spec['train_config']['class_weight'] != effective_train['class_weight']:
                    title += '；训练策略调整'
                if not title:
                    title = '替换特征定义／分支复验'
            selection_note = ('baseline 已建立' if is_baseline else '具备冠军晋级资格，轮末统一比较' if eligible
                              else '通过门禁但未优于当前冠军，保留探索分支' if verdict['pass'] else '未通过门禁')
            rec = ExperimentRecord(experiment_id=exp_id, round_id=round_id,
                    timestamp=datetime.now().isoformat(timespec="seconds"), proposer="baseline" if is_baseline else "agent",
                    hypothesis=p["hypothesis"], config_diff={"add_features": added, "feature_recipes": recipes,
                    "xgb_params": candidate_changes.get('xgb_params', {}), "evidence": p.get("evidence", ""), "session_id": session,
                    "mlp_params": candidate_changes.get('mlp_params', {}),
                    'experiment_spec': spec, 'candidate_changes': candidate_changes,
                    "mock": args.mock, "evaluation_split": "test", "training_method": effective_train['training_method'], "sample_hash": sample_hash,
                    'data_source': manifest['data_source'], 'snapshot_id': args.snapshot_id,
                    'training_population': args.training_population, 'model_family': args.model_family, 'training_mode': args.training_mode, 'training_city': args.city_code, 'city_name': city_training.CITIES.get(args.city_code),
                    'num_bins': 6, 'reference_baseline': reference,
                    'selection_sample_hash': selection_hash, 'selection_metric': SELECTION_KEY,
                    "parent_experiment_id": None if is_baseline else (parent_record or champion_rec).experiment_id,
                    'compared_champion_id': None if is_baseline else champion_rec.experiment_id,
                    'web_research_report': (f'web_research_{round_id:02d}.json'
                        if not is_baseline and profile.get('web_research') else None)},
                    data_window=win, metrics_by_city=met, gate_pass=verdict["pass"], gate_reasons=verdict["reasons"],
                    artifact_dir=str(out_dir), evaluation_views=evaluation_views,
                    notes=(f'{city_training.CITIES[args.city_code]}独立训练；ALL仅表示该城市；' if args.city_code else '') + '6桶；应答＋取消过滤后H3-7整体召回；按本会话训练策略预测及保序；须达到Notebook参考门槛且不低于当前对照；取消过滤后人群可不同；低估率仅供观察，非独立评估',
                    summary={"verdict": outcome, 'champion_eligible': eligible,
                             'research_plan': None if is_baseline else current_plan,
                             'selection_metric': SELECTION_KEY, 'selection_delta_pp': selection_deltas,
                             'reference_baseline_delta_pp': 100 * (met[SELECTION_KEY]['ALL']['recall_ratio'] - reference['metrics']['ALL']['recall_ratio']),
                             'selection_note': selection_note, 'search_diagnostics': search_diagnostics,
                             "innovation": {"title": title.lstrip('；')},
                             "narrative_zh": p["hypothesis"] + "；" + selection_note + "；" + "; ".join(verdict["reasons"])})
            append_record(str(logs), rec)
            experiments_by_id[exp_id] = rec
            predictions_by_id[exp_id] = selection_pred[['order_id', 'dt', 'true_bin', 'predict_bin_44']].copy()
            write_json(out_dir / 'selection_result.json', asdict(rec))
            reproduction.finalize(out_dir, asdict(rec))
            print(f'[mining] {exp_id}: features={added}, test_gate={verdict}', flush=True)
            emit_event("experiment_completed", session, experiment_id=exp_id)
            progress('evaluated', f'{exp_id} 评估完成 · {SELECTION_LABEL}准确率 {met[SELECTION_KEY]["ALL"]["recall_ratio"]:.2%} · '
                     + selection_note,
                     experiment_id=exp_id, round_id=round_id)
            clean_slices = json.loads(slices.head(15).to_json(orient="records"))
            return pairs, met, clean_slices, rec, selection_pred

        champion_metrics = None
        # baseline 用当前继承的 recipes 训一次（无继承时就是纯 44 特征基线）
        # 这样 challenger 的对比对象是"继承基线"，而非"零特征基线"，公平比较增量收益
        _baseline_hypothesis = "Notebook 43特征6桶配置的本会话重训对照（非人工示例值）" + (
            f"（继承 {len(accepted)} 个 recipe: {[r['name'] for r in accepted]}）" if accepted else "")
        baseline_pairs, champion_metrics, slices, baseline_rec, champion_frame = experiment(
            list(accepted), 0, {"hypothesis": _baseline_hypothesis})
        baseline_metrics = champion_metrics
        baseline_frame = champion_frame
        champion_pairs, champion_rec = baseline_pairs, baseline_rec
        iteration_budget = args.rounds * args.max_proposals_per_round if args.continuous else args.rounds
        controller = ResearchController(iteration_budget, args.search_timeout_minutes, args.search_patience,
                                        research_skills=args.research_skills) if args.continuous else None

        def save_research():
            if controller is not None:
                manifest['research_state'] = controller.snapshot()
                write_json(directory / 'research_state.json', {**controller.snapshot(), 'steps': controller.steps})
                write_json(directory / 'session.json', manifest)

        manifest.update(status="searching", champion_id=champion_rec.experiment_id,
                        baseline_id=baseline_rec.experiment_id)
        save_research()
        write_json(directory / "session.json", manifest)
        emit_event("session_updated", session, status=manifest["status"])
        web_session = ExperimentResearchSession(ROOT, directory, config['llm']) if args.web_research else None
        active_hint = None
        for round_id in range(1, 1 + (0 if args.no_agent else iteration_budget)):
            if controller is not None and not args.research_skills and len(accepted) >= cfg['max_total_features']:
                controller.stop_reason = '冠军特征已达到累计预算，停止新增特征搜索'
            if controller is not None and controller.stopped():
                break
            current_plan = {}
            previous_champion_id = champion_rec.experiment_id
            progress('proposing', f'第 {round_id}/{iteration_budget} 次规划：Agent 正在根据最新结果生成下一项实验，请等待模型响应',
                     round_id=round_id)
            from agent.orchestrator import _consume_user_hint
            hint = _consume_user_hint(Path(os.environ.get("ETS_WEBUI_STATE_DIR", str(ROOT / "webui/state"))))
            if controller is not None:
                active_hint = hint or active_hint
                hint = active_hint
            print(f'[mining] 第 {round_id} 轮：根据训练字段画像和测试集误差生成配方', flush=True)
            parent = search_memory.parent_signature(accepted, champion_frame, champion_rec.artifact_dir)
            recalled = search_memory.memory_for_prompt(memories, context, parent)
            if web_session is not None and not args.proposal_file:
                progress('web_research', '模型正在根据实验反馈决定是否补充联网证据', round_id=round_id)
                try:
                    research_state = controller.snapshot() if controller else None
                    research_context = {
                        'task': 'XGBoost 等待时长6分类，实验前自主资料研究', 'mock': args.mock,
                        'evaluation_split': 'test', 'selection_metric': SELECTION_KEY,
                        'selection_warning': manifest['selection_warning'],
                        'champion_selection_metrics': champion_metrics[SELECTION_KEY],
                        'champion_diagnostics': champion_rec.summary['search_diagnostics'],
                        'champion_spec': champion_spec,
                        'selection_error_slices': slices, 'recent_experiments': feedback[-6:],
                        'historical_experiments': recalled[-4:], 'accepted_recipes': accepted,
                        'research_state': research_state, 'user_guidance': hint,
                        'allowed_changes': (profile.get('research_capabilities') if args.research_skills else
                            {'feature_recipes_only': True, 'training_config_fixed': True}),
                    }
                    profile['web_research'] = web_session.update(round_id, research_context,
                        progress=lambda value: progress('web_research', value.get('stage', '自主调研中'),
                                                       round_id=round_id, research_trace=value.get('trace', [])),
                        budget_seconds=min(480, research_state['remaining_seconds']) if research_state else 480)
                    manifest['web_research_calls'] = web_session.calls
                    manifest['web_research_status'] = (profile['web_research'] or {}).get('status', 'unavailable')
                    manifest['web_research_last_report'] = f'web_research_{round_id:02d}.json'
                    print('[research] ' + profile['web_research'].get('summary', '')[:6000], flush=True)
                except Exception as error:
                    # An unavailable public source must not abort existing model research.
                    profile['web_research'] = {'status': 'unavailable', 'sources': [], 'error': type(error).__name__}
                    manifest['web_research_status'] = 'unavailable'
                write_json(directory / 'session.json', manifest)
                progress('web_research', '联网资料状态：' + manifest['web_research_status'], round_id=round_id)
                if controller is not None and controller.stopped():
                    save_research()
                    break
            if args.research_skills:
                profile['champion_spec'] = champion_spec
                profile['available_parents'] = [{'experiment_id': record.experiment_id,
                    'acc': record.metrics_by_city[SELECTION_KEY]['ALL']['recall_ratio'],
                    'spec': record.config_diff['experiment_spec']} for record in experiments_by_id.values()][-20:]
                profile['extension_requests'] = research_extensions.requests(ROOT, session)
                profile['data_inspection']['remaining_calls'] = max(0, profile['data_inspection']['max_calls'] - inspection_count)
            write_json(directory / 'memory_retrieval.json', {'round_id': round_id, 'entries': recalled})
            if controller is not None:
                try:
                    planned = propose(profile, audit, champion_metrics, feedback, slices, accepted, 1,
                                      config['llm'], hint, recalled, champion_rec.summary['search_diagnostics'],
                                      research_state=controller.snapshot())
                    current_plan, proposals = controller.validate(planned)
                    write_json(directory / f'round_{round_id}_plan.json', planned)
                    if current_plan['action'] == 'stop':
                        controller.stop_reason = 'Agent 主动结束：' + current_plan['reason']
                        controller.record(current_plan, 'stopped')
                        save_research()
                        break
                    if controller.stopped():
                        save_research()
                        break
                    progress('planning', f"{current_plan['action']} / {current_plan['direction']}：{current_plan['reason']}",
                             round_id=round_id)
                except Exception as exc:
                    reason = str(exc)[:500] if isinstance(exc, (ValueError, TypeError, KeyError, LLMEmptyContentError)) else type(exc).__name__
                    feedback.append({'status': 'planning_failed', 'rejected': reason})
                    controller.record({}, 'planning_failed', detail=reason)
                    write_json(directory / 'search_feedback.json', feedback)
                    save_research()
                    progress('rejected', f'本次规划未完成，将在剩余预算内重新规划：{reason}', round_id=round_id)
                    continue
            else:
                proposals = (json.loads(args.proposal_file.read_text())["proposals"][:args.max_proposals_per_round]
                             if args.proposal_file else propose(profile, audit, champion_metrics, feedback, slices,
                                                              accepted, args.max_proposals_per_round, config["llm"], hint,
                                                              recalled, champion_rec.summary['search_diagnostics']))
            write_json(directory / f"round_{round_id}_proposals.json", {"proposals": proposals})
            if not proposals: break
            # All siblings compare to the same champion; choose after the round.
            successes = []
            for proposal_number, p in enumerate(proposals, start=1):
                progress('features', f'第 {round_id}/{iteration_budget} 次 · 候选 {proposal_number}/{len(proposals)}：准备实验',
                         round_id=round_id)
                parent_record = champion_rec
                parent = search_memory.parent_signature(accepted, champion_frame, champion_rec.artifact_dir)
                try:
                    parent_record = champion_rec
                    candidate_spec = None
                    spec_signature = None
                    if args.research_skills:
                        experiment_spec.validate_header(p)
                        if 'inspect' in p:
                            if p['feature_recipes'] or any(p.get(key) for key in (
                                    'remove_features', 'training', 'extensions', 'parent_experiment_id', 'requests')):
                                raise ValueError('只读探查与训练/审批申请必须分开提交')
                            if inspection_count >= min(cfg.get('max_inspection_calls', 4), 8):
                                raise ValueError('本会话只读探查预算已用尽，请使用已有结果设计实验')
                            data_inspection.validate(p['inspect'], catalog, audit, cfg)
                            request_signature = program.fingerprint(p['inspect'])
                            if request_signature in inspection_seen:
                                raise ValueError('相同探查已成功完成，请复用 inspection_results')
                            inspection_count += 1
                            manifest['inspection_calls'] = inspection_count
                            progress('inspection', 'Agent 正在只读探查训练期字段统计，不改变训练/测试数据', round_id=round_id)
                            result = data_inspection.execute(p['inspect'], full, catalog, audit, cfg, win, mock=args.mock)
                            inspection_seen.add(request_signature)
                            inspected = {'request': p['inspect'], 'result': result}
                            profile['inspection_results'].append(inspected)
                            profile['data_inspection']['remaining_calls'] = max(0, profile['data_inspection']['max_calls'] - inspection_count)
                            write_json(directory / 'data_profile.json', published_profile)
                            write_json(directory / f'inspection_{inspection_count}.json', inspected)
                            feedback.append({'id': p['id'], 'status': 'inspected',
                                             'inspection': inspected, 'selection_note': '只读探查完成；未训练或关联新数据'})
                            write_json(directory / 'search_feedback.json', feedback)
                            progress('inspection', '只读探查完成，结果将供下一次规划使用', round_id=round_id)
                            continue
                        pending = p.get('requests', [])
                        if not isinstance(pending, list) or len(pending) > 2:
                            raise ValueError('每次最多提交两个扩展申请')
                        if pending:
                            if p['feature_recipes'] or any(p.get(key) for key in ('remove_features', 'training', 'extensions', 'parent_experiment_id')):
                                raise ValueError('扩展申请与模型实验必须分开提交')
                            submitted = [research_extensions.stage(ROOT, session, request, catalog, mock=args.mock) for request in pending]
                            feedback.append({'id': p['id'], 'status': 'pending_review', 'requests': submitted,
                                             'selection_note': '扩展已保存待人工审批；没有执行训练或合入代码'})
                            write_json(directory / 'search_feedback.json', feedback)
                            progress('planning', '扩展申请已保存，未批准前不会执行：' + ', '.join(item['id'] for item in submitted), round_id=round_id)
                            continue
                        parent_id = p.get('parent_experiment_id', champion_rec.experiment_id)
                        if not isinstance(parent_id, str) or parent_id not in experiments_by_id:
                            raise ValueError('只能选择当前同样本会话中已完成的父实验；跨会话历史仅供参考')
                        parent_record = experiments_by_id[parent_id]
                        candidate_spec = experiment_spec.derive(p, parent_record.config_diff['experiment_spec'], catalog, cfg, registered_extensions)
                        if (controller is not None and controller.snapshot()['require_alternative_to_add_only']
                                and not experiment_spec.has_non_additive_change(parent_record.config_diff['experiment_spec'], candidate_spec)):
                            raise ValueError('小收益阶段需要真实的删改、编码或训练变化；空调参、继承的编码和原样替换不算切换方向')
                        bundle = candidate_spec['recipes']
                        spec_signature = experiment_spec.signature(candidate_spec)
                        parent = search_memory.parent_signature(parent_record.config_diff['feature_recipes'],
                                    predictions_by_id[parent_id], parent_record.artifact_dir)
                    else:
                        validate_proposal(p, catalog, accepted, cfg)
                        bundle = accepted + p["feature_recipes"]
                    duplicate = search_memory.duplicate_entry(memories, context, parent, bundle, spec_signature)
                    if duplicate:
                        reason = f'相同数据、训练配置和父模型下已完成该配方：{duplicate.get("experiment_id", "历史实验")}'
                        feedback.append({'id': p['id'], 'recipes': p['feature_recipes'], 'status': 'duplicate_skipped',
                                         'rejected': reason, 'source_experiment_id': duplicate.get('experiment_id')})
                        progress('rejected', reason, round_id=round_id)
                        write_json(directory / 'search_feedback.json', feedback)
                        continue
                    pairs, met, trial_slices, rec, expanded = experiment(bundle, round_id, p, candidate_spec, parent_record)
                    feedback.append({"id": p["id"], 'experiment_id': rec.experiment_id, 'status': 'completed',
                                     'research_plan': current_plan, "recipes": p["feature_recipes"], "selection_metrics": met,
                                     "gate_pass": rec.gate_pass, "reasons": rec.gate_reasons,
                                     'outcome': rec.summary['verdict'], 'selection_note': rec.summary['selection_note'],
                                     'selection_metric': SELECTION_KEY, 'selection_delta_pp': rec.summary['selection_delta_pp'],
                                     'candidate_changes': rec.config_diff['candidate_changes'],
                                     'diagnostics': {
                                         'features': {name: value for name, value in rec.summary['search_diagnostics']['features'].items()
                                                      if name in {recipe['name'] for recipe in p['feature_recipes']}},
                                         'prediction_changes': rec.summary['search_diagnostics']['prediction_changes']}})
                    entry = {'session_id': session, 'experiment_id': rec.experiment_id,
                             'parent_experiment_id': parent_record.experiment_id, 'window': win,
                             'context_hash': program.fingerprint(context), 'parent_signature': parent,
                             'recipe_signature': search_memory.recipe_signature(bundle), 'recipes': bundle,
                             'status': 'completed', 'outcome': rec.summary['verdict'],
                             'research_plan': current_plan,
                             'metrics': met[SELECTION_KEY]['ALL'], 'reasons': rec.gate_reasons,
                             'selection_metric': SELECTION_KEY,
                             'delta_pp': rec.summary['selection_delta_pp']['parent'],
                             'diagnostics': feedback[-1]['diagnostics']}
                    if args.research_skills:
                        entry.update(spec_signature=spec_signature, experiment_spec=candidate_spec,
                                     candidate_changes=rec.config_diff['candidate_changes'])
                    search_memory.append_memory(directory, entry)
                    memories.append(entry)
                    if rec.gate_pass:
                        successes.append((met[SELECTION_KEY]['ALL']['recall_ratio'], bundle, pairs, met, trial_slices, rec, expanded))
                except Exception as exc:
                    reason = str(exc)[:500] if isinstance(exc, (ValueError, TypeError, KeyError)) else type(exc).__name__
                    rejected_recipes = p.get('feature_recipes', []) if isinstance(p, dict) else []
                    feedback.append({"id": p.get("id") if isinstance(p, dict) else None, "rejected": reason,
                                     'recipes': rejected_recipes, 'status': 'execution_failed'})
                    failed_changes = {key: p[key] for key in ('remove_features', 'training', 'parent_experiment_id', 'extensions')
                                      if isinstance(p, dict) and key in p}
                    feedback[-1]['candidate_changes'] = failed_changes
                    entry = {'session_id': session, 'window': win, 'context_hash': program.fingerprint(context),
                             'parent_signature': parent, 'parent_experiment_id': parent_record.experiment_id,
                             'status': 'execution_failed', 'recipes': rejected_recipes, 'error': reason}
                    entry['research_plan'] = current_plan
                    entry['candidate_changes'] = failed_changes
                    search_memory.append_memory(directory, entry)
                    memories.append(entry)
                    progress('rejected', f'第 {round_id} 轮候选 {proposal_number} 未完成：{reason}', round_id=round_id)
                    print(f"[mining] 候选拒绝: {reason}", flush=True)
                write_json(directory / "search_feedback.json", feedback)
            if successes:
                best = max(successes, key=lambda candidate: (candidate[0], -len(experiment_spec.feature_names(candidate[5].config_diff['experiment_spec']))))
                if search_memory.prefer_candidate(champion_metrics[SELECTION_KEY]['ALL']['recall_ratio'],
                                                  len(experiment_spec.feature_names(champion_spec)), best[0],
                                                  len(experiment_spec.feature_names(best[5].config_diff['experiment_spec']))):
                    _, accepted, champion_pairs, champion_metrics, slices, champion_rec, champion_frame = best
                    champion_spec = champion_rec.config_diff['experiment_spec']
                for candidate in successes:
                    if candidate[5].experiment_id != champion_rec.experiment_id:
                        manifest['exploration_branches'].append({
                            'experiment_id': candidate[5].experiment_id,
                            'parent_experiment_id': candidate[5].config_diff['parent_experiment_id'],
                            'recipes': candidate[1], 'acc': candidate[0],
                            'outcome': candidate[5].summary['verdict']})
            manifest.update(accepted_recipes=accepted, champion_id=champion_rec.experiment_id, champion_spec=champion_spec,
                            extension_requests=research_extensions.requests(ROOT, session) if args.research_skills else [])
            if controller is not None:
                latest_feedback = feedback[-1]
                controller.record(current_plan, latest_feedback.get('status', 'completed'),
                                  improved=champion_rec.experiment_id != previous_champion_id,
                                  experiment_id=latest_feedback.get('experiment_id'),
                                  detail=latest_feedback.get('rejected') or latest_feedback.get('selection_note'),
                                  delta_pp=latest_feedback.get('selection_delta_pp', {}).get('champion'))
                save_research()
            write_json(directory / "session.json", manifest)
            emit_event("session_updated", session, status=manifest["status"])
        if controller is not None:
            if args.no_agent:
                controller.stop_reason = '仅运行 baseline，未执行搜索'
            controller.stopped()
            save_research()
            progress('finishing', controller.stop_reason or '搜索结束，正在保存结果')
        progress('finishing', '正在汇总测试指标并保存最终结果')
        holdout_base = baseline_metrics
        holdout_champion = champion_metrics
        has_improvement = champion_rec.experiment_id != baseline_rec.experiment_id
        final_gate = mining_train.gate(holdout_base, holdout_champion) if has_improvement else {"pass": False, "reasons": ['本轮没有候选替换冠军；持平且更复杂的候选仅保留为探索分支']}
        write_json(directory / "holdout_result.json", {"baseline": holdout_base, "selected": holdout_champion,
                    'reference_baseline': reference,
                    "baseline_views": baseline_rec.evaluation_views, "selected_views": champion_rec.evaluation_views,
                    "gate": final_gate, "selected_id": champion_rec.experiment_id, 'selection_metric': SELECTION_KEY,
                    "used_for_search": True, "evaluation_split": "test", "production_approved": False})
        manifest.update(status="complete", champion_id=champion_rec.experiment_id,
                        accepted_recipes=accepted, champion_spec=champion_spec,
                        holdout_gate=final_gate, production_approved=False)
        if controller is not None and controller.planning_failures >= 3:
            manifest.update(status='failed', error_type='planning_failed',
                            error_hint='连续三次规划失败，搜索未正常完成；已完成的 baseline 和实验结果已保留。'
                                       + str(feedback[-1].get('rejected', ''))[:300])
            raise SystemExit(1)
        print(f"[mining] 完成；冠军输入 {len(experiment_spec.feature_names(champion_spec))} 个特征，保留 {len(accepted)} 个配方。测试集筛选门禁={final_gate}；产物={directory}", flush=True)
    except KeyboardInterrupt:
        manifest.update(status='cancelled', error_hint='分城市训练批次已由用户停止')
        raise
    except Exception as exc:
        frames = [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                  for f in traceback.extract_tb(exc.__traceback__)]
        manifest.update(status="failed", error_type=type(exc).__name__, error_frames=frames)
        if "whitelist" in str(exc).lower():
            manifest["error_hint"] = "HDFS 拒绝当前服务器 IP；请完成白名单授权后重跑真实数据画像"
        elif type(exc).__name__ == "SecretMissingError":
            manifest["error_hint"] = "缺少 ETS 凭证；检查环境变量或权限为 600 的 .env"
        elif isinstance(exc, (MemoryBudgetError, LLMEmptyContentError)):
            manifest['error_hint'] = str(exc)
        # Spark exception text may contain SQL/credentials. Keep detailed vendor errors out of stdout.
        print(f"[mining] 失败: {type(exc).__name__}" + (f": {exc}" if isinstance(exc, ValueError) else "；检查环境与数据源访问"), flush=True)
        print(f"[mining] 错误位置: {frames[-5:]}", flush=True)
        if manifest.get("error_hint"): print(f"[mining] {manifest['error_hint']}", flush=True)
        raise SystemExit(1)
    finally:
        write_json(directory / "session.json", manifest)
        # current.json 无条件更新（无论成败），记录最近一次跑
        write_json(ROOT / "experiments/feature_mining/current.json", manifest)
        emit_event("session_updated", session, status=manifest["status"])
        is_success = (
            manifest.get("status") == "complete"
            and not manifest.get('model_study')
            and manifest.get('champion_id') != manifest.get('baseline_id')
        )
        if is_success:
            write_json(latest_path, manifest)
            print(f"[mining] latest.json 已更新为 {session}（冠军 {manifest['champion_id']}）",
                  flush=True)
        elif not manifest.get('model_study'):
            # 失败/无进展 session 另存到 failed/ 便于排查，但不污染 latest.json
            fail_dir = ROOT / "experiments/feature_mining/failed"
            fail_dir.mkdir(parents=True, exist_ok=True)
            write_json(fail_dir / f"{session}.json", manifest)
            print(f"[mining] latest.json 未更新（本次 status={manifest.get('status')}, "
                  f"accepted={len(manifest.get('accepted_recipes') or [])}）；失败副本已存至 failed/",
                  flush=True)


if __name__ == "__main__":
    with activity_lock(ROOT):
        main()
