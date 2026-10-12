"""Config loading: `inherit:` chain + `--set a.b.c=value` overrides."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _coerce(v: str):
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return v


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = path.parent
    if cfg.get("inherit"):
        inh = (parent / cfg["inherit"]) if not Path(cfg["inherit"]).is_file() \
            else Path(cfg["inherit"])
        base = load_config(inh)
        cfg.pop("inherit")
        deep_update(base, cfg)
        cfg = base
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {ov!r}")
        _set(cfg, key, _coerce(val))
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
