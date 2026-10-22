"""Server-owned model profiles; browsers select IDs, never endpoints or keys."""
import os


def resolve(config, profile=None):
    profiles = config.get('profiles', {})
    selected = profile if profile is not None else config.get('default_profile', 'chat')
    if not isinstance(selected, str) or (profiles and selected not in profiles) or (not profiles and selected != 'chat'):
        raise ValueError('未知模型选项，只能选择服务端配置的模型')
    base = {key: value for key, value in config.items() if key not in ('profiles', 'default_profile')}
    base.update(profiles.get(selected, {}))
    base['profile_id'] = selected
    base.setdefault('protocol', 'chat_completions')
    return base


def public(config):
    identifiers = config.get('profiles') or {'chat': {}}
    return {'default_profile': config.get('default_profile', 'chat'), 'profiles': [
        {key: value for key, value in resolve(config, identifier).items()
         if key in ('profile_id', 'label', 'model', 'protocol')} for identifier in identifiers]}


def key_env(config):
    primary = config.get('api_key_env', 'ETS_LLM_API_KEY')
    fallback = config.get('api_key_fallback_env')
    return primary if os.environ.get(primary, '').strip() or not fallback else fallback
