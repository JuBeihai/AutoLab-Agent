"""Offline setup check: no LLM request, no Hive query, no training."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.start_webui import load_environment
from skills.llm_profiles import resolve, key_env

load_environment()
missing = [p for p in ['flask', 'numpy', 'pandas', 'scipy', 'sklearn', 'xgboost', 'h3', 'pyarrow', 'threadpoolctl', 'psutil']
           if importlib.util.find_spec(p) is None]
print('Python:', sys.version.split()[0])
print('缺失依赖:', ', '.join(missing) or '无')
prompt_names = ['feature_miner', 'feature_proposer', 'research_planner', 'research_skills', 'summarizer', 'web_research']
missing_prompts = [name for name in prompt_names if not (ROOT/'agent/prompts'/(name+'.txt')).is_file()]
print('缺失提示词:', ', '.join(missing_prompts) or '无')
cfg = resolve(json.loads((ROOT/'configs/llm.json').read_text()))
print('LLM 接口:', cfg['endpoint'])
print('LLM 模型 / 协议:', cfg['model'], '/', cfg['protocol'])
print('LLM 凭证:', '已配置（未验证授权）' if os.environ.get(key_env(cfg)) else '未配置；聊天/自动规划需填写')
weighted = True
if 'sklearn' not in missing:
    from sklearn.neural_network import MLPClassifier
    weighted = 'sample_weight' in inspect.signature(MLPClassifier.partial_fit).parameters
    print('主 Agent MLP 加权训练支持:', weighted)
print('本检查没有发送任何网络请求。')
raise SystemExit(1 if missing or missing_prompts or not weighted else 0)
