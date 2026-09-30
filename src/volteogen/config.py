"""Config loading: default.yaml <- preset <- explicit overrides.

The config *is* the generator contract, so this layer stays deliberately thin: a deep
merge, dotted-path access, and loud failure on unknown keys rather than a schema class
per block. Modules read the config; they never carry their own defaults.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
SEEDS_DIR = REPO_ROOT / "seeds"
PACKS_DIR = REPO_ROOT / "packs"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; scalars and lists from ``override`` win outright.

    Lists are replaced rather than concatenated — a preset that says
    ``entities: [DE01]`` means *only* DE01, not "the defaults plus DE01".
    """
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML rules (so ints/bools/lists just work)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


class _Missing:
    """Sentinel so ``get(path, None)`` can mean "default to None", not "no default"."""

    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


@dataclass
class Config:
    """Dotted-path view over the merged config tree."""

    data: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def get(self, path: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"config path not found: {path!r}")
                return default
            node = node[part]
        return node

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise KeyError(f"cannot set {path!r}: {part!r} is a leaf value")
        node[parts[-1]] = value

    # -- convenience accessors used all over the engine -------------------------
    @property
    def seed(self) -> int:
        return int(self.get("seed"))

    @property
    def entities(self) -> list[str]:
        return list(self.get("entities"))

    @property
    def months(self) -> int:
        return int(self.get("period.months"))

    @property
    def industry_pack(self) -> str:
        return str(self.get("industry_pack"))

    def defect_enabled(self, name: str) -> bool:
        return bool(self.get(f"defects.{name}", False))


def load_config(
    preset: str | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
) -> Config:
    """Load default.yaml, deep-merge the named preset, then apply dotted overrides."""
    cdir = Path(config_dir) if config_dir else CONFIG_DIR
    default_path = cdir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"missing master config: {default_path}")

    data = yaml.safe_load(default_path.read_text()) or {}
    sources = [str(default_path)]

    preset = preset or data.get("scale_preset")
    if preset:
        preset_path = cdir / "presets" / f"{preset}.yaml"
        if not preset_path.exists():
            available = sorted(p.stem for p in (cdir / "presets").glob("*.yaml"))
            raise FileNotFoundError(
                f"unknown scale preset {preset!r}; available: {available}"
            )
        data = deep_merge(data, yaml.safe_load(preset_path.read_text()) or {})
        sources.append(str(preset_path))

    cfg = Config(data=data, sources=sources)
    for path, value in (overrides or {}).items():
        cfg.set(path, _coerce(value) if isinstance(value, str) else value)
    return cfg


def load_pack(cfg: Config, packs_dir: Path | None = None) -> dict:
    """Load the industry pack: every industry-specific fact lives here, not in code.

    The engine must contain no industry knowledge, so a missing pack is a hard error —
    falling back to built-in electronics assumptions would defeat the whole layer.
    """
    pdir = Path(packs_dir) if packs_dir else PACKS_DIR
    name = cfg.industry_pack
    root = pdir / name
    if not root.is_dir():
        available = sorted(p.name for p in pdir.iterdir() if p.is_dir()) if pdir.is_dir() else []
        raise FileNotFoundError(
            f"industry pack {name!r} not found under {pdir}; available: {available}"
        )

    pack: dict[str, Any] = {"name": name, "root": root}
    for stem in (
        "product_taxonomy",
        "seasonality",
        "economics",
        "flows",
        "narrative_defaults",
    ):
        path = root / f"{stem}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"industry pack {name!r} is missing {path.name}")
        pack[stem] = yaml.safe_load(path.read_text()) or {}

    overrides = root / "applicability_overrides.csv"
    pack["applicability_overrides"] = overrides if overrides.exists() else None
    return pack
