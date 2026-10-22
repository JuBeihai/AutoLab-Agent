"""Secret access and validation for 等待时间 integrations."""

from __future__ import annotations

import base64
import binascii
import json
import os
import time


class SecretMissingError(RuntimeError):
    pass


class SecretInvalidError(RuntimeError):
    pass


def _get_env(name: str, hint: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise SecretMissingError(f"环境变量 {name} 未设置。{hint}")
    return val


def _validate_dp_key(value: str) -> str:
    raise SecretInvalidError("Production credential formats are not distributed")


def get_dp_key() -> str:
    raise SecretMissingError("演示版不支持生产数据凭证")


def get_llm_api_key(name='AUTOLAB_LLM_API_KEY') -> str:
    """Read the user-configured LLM API key."""
    return _get_env(
        name,
        "请 export key，或通过权限为 600 的项目 .env 加载。",
    )
