"""Configure a compatible endpoint. Never accepts or stores an API key."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def configure(path, *, endpoint, model, protocol='chat_completions', profile='custom', label=None):
    url = urlsplit(endpoint)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('endpoint 应为不含账号、密钥、查询参数的完整 HTTP(S) 接口 URL')
    if protocol not in ('chat_completions', 'responses'):
        raise ValueError('协议必须为 chat_completions 或 responses')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', profile) or not model.strip():
        raise ValueError('请提供有效 profile 和非空模型名')
    cfg = json.loads(path.read_text())
    cfg.setdefault('profiles', {})[profile] = {
        'label': label or model, 'model': model, 'protocol': protocol,
        'endpoint': endpoint, 'api_key_env': 'ETS_LLM_API_KEY',
    }
    cfg.update(default_profile=profile, endpoint=endpoint, model=model, protocol=protocol,
               api_key_env='ETS_LLM_API_KEY')
    cfg.pop('api_key_fallback_env', None)
    import sys
    sys.path.insert(0, str(ROOT))
    from skills.llm_profiles import resolve
    resolve(cfg, profile)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    return cfg['profiles'][profile]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', required=True, help='完整 /v1/chat/completions 或 /v1/responses 地址')
    p.add_argument('--model', required=True, help='提供方真实的模型 ID')
    p.add_argument('--protocol', choices=['chat_completions', 'responses'], default='chat_completions')
    p.add_argument('--profile', default='custom')
    p.add_argument('--label')
    configured = configure(ROOT/'configs/llm.json', **vars(p.parse_args()))
    print(json.dumps(configured, ensure_ascii=False, indent=2))
    print('配置已更新；密钥请填写 .env 中的 ETS_LLM_API_KEY。重启服务后在前端选择此模型。')


if __name__ == '__main__':
    main()
