"""Tiny config helper.

Reads ``config.yaml`` from the repository root, or a config file selected by
``CONFIG_PATH`` / ``LEANSEARCH_CONFIG`` / ``LEANSEARCHV2_CONFIG``, then layers
``config.local.yaml`` on top if it exists. Each setting can also be overridden
by an environment variable of the same name; environment variables win over
both files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

# Matches `org/name` HF repo ids (single slash, no path separators).
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_CACHE: dict[str, Any] | None = None


def _base_config_path() -> Path:
    for env_key in ("CONFIG_PATH", "LEANSEARCH_CONFIG", "LEANSEARCHV2_CONFIG"):
        value = os.environ.get(env_key)
        if value:
            return Path(value)
    return REPO_ROOT / "config.yaml"


def _config() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        base_path = _base_config_path()
        cfg = _load_yaml(base_path)
        local_path = REPO_ROOT / "config.local.yaml"
        if local_path.exists():
            cfg = _deep_merge(cfg, _load_yaml(local_path))
        _CACHE = cfg
    return _CACHE


def _resolve(value: str) -> str:
    value = os.path.expanduser(os.path.expandvars(value))
    if _HF_REPO_ID.match(value):
        return value
    p = Path(value)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return str(p)


def get(env_key: str, *yaml_path: str, default: Any = None) -> Any:
    """Return ``$env_key`` if set, else look up ``yaml_path`` in config files, else ``default``."""
    if env_key in os.environ:
        return os.environ[env_key]
    node: Any = _config()
    for key in yaml_path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node if node is not None else default


def get_path(env_key: str, *yaml_path: str, default: str | None = None) -> str:
    value = get(env_key, *yaml_path, default=default)
    if value is None:
        raise KeyError(
            f"Missing path: set ${env_key} or config.yaml:{'.'.join(yaml_path)}"
        )
    return _resolve(value)
