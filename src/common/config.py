"""Config loading and project paths.

Configs are YAML by preference (PyYAML), with a JSON fallback so the pipeline
still runs in a stripped-down environment. ``load_config`` also resolves any
relative paths in the config against the project root so targets work no matter
the current working directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# src/common/config.py -> project root is two parents up from this file's dir.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "PyYAML is required to read YAML configs. Run `make setup` or "
                "`pip install pyyaml`, or pass a .json config instead."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping.")
    return data


def resolve_path(p: str | Path) -> Path:
    """Resolve ``p`` against the project root unless it is already absolute."""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a config file (YAML or JSON) into a dict."""
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return _load_yaml_or_json(path)
