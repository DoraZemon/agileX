"""
Configuration loading: default.yaml + site overlays + command-line overrides.

Usage:
    cfg = load_config()                                  # default + overlays
    cfg = load_config(extra=['my.yaml'], overrides=['robot.speed_percent=50'])
    cfg.get('robot.gripper.max_width')
    cfg.path('camera.extrinsics_file')                   # absolute pathlib.Path
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

_MISSING = object()


_LOG = logging.getLogger('cube_task.config')
_FIX = ('rebuild with "cd ~/vision/ws && colcon build --symlink-install --packages-select '
        'cube_task", or export CUBE_TASK_ROOT=~/vision/ws/src/cube_task')
_warned_roots: Set[str] = set()


def _find_root(module_file: Path) -> Path:
    """
    Return the package root for this module file (see package_root).

    Source tree (parents[2] has config/default.yaml), else the share
    directory of an installed copy (<prefix>/share/cube_task, also
    <prefix>/local/...), else RuntimeError with the fix.
    """
    here = Path(module_file).resolve()
    src = here.parents[2]
    if (src / 'config' / 'default.yaml').is_file():
        return src
    # Installed copy: <prefix>[/local]/lib/pythonX.Y/site-packages/cube_task/core/config.py.
    for prefix in here.parents[3:7]:
        share = prefix / 'share' / 'cube_task'
        if (share / 'config' / 'default.yaml').is_file():
            if str(share) not in _warned_roots:
                _warned_roots.add(str(share))
                _LOG.warning(
                    'cube_task runs from an installed copy (plain colcon build without '
                    '--symlink-install): reading %s read-only. config/site/*.yaml and models/ '
                    'of the source tree are NOT used, and files the tools write land in the '
                    'install tree and are lost on the next build. Fix: %s.', share, _FIX)
            return share
    raise RuntimeError('cube_task is installed as a copy (plain colcon build): %s has no '
                       'config/default.yaml and no installed share/cube_task was found. Fix: %s.'
                       % (src, _FIX))


def package_root() -> Path:
    """
    Directory containing config/ and models/.

    CUBE_TASK_ROOT overrides it. Otherwise the source tree of this package is
    used (colcon --symlink-install keeps the module in the source tree). After
    a plain colcon build (module copied into site-packages) the installed
    <prefix>/share/cube_task is used read-only with a warning; RuntimeError
    (naming the fix) when that has no config/default.yaml either.
    """
    env = os.environ.get('CUBE_TASK_ROOT')
    if env:
        return Path(env).expanduser().resolve()
    return _find_root(Path(__file__))


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Merge over into a copy of base. Dicts merge recursively; other values replace."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def _set_dotted(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split('.')
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


class Config:
    """Read-mostly nested dict with dotted access."""

    def __init__(self, data: Dict[str, Any], root: Path, sources: List[str]):
        self.data = data
        self.root = Path(root)
        self.sources = list(sources)

    def get(self, key: str, default: Any = _MISSING) -> Any:
        cur: Any = self.data
        for p in key.split('.'):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                if default is _MISSING:
                    raise KeyError('config key not found: %s' % key)
                return default
        return cur

    def has(self, key: str) -> bool:
        return self.get(key, None) is not None

    def set(self, key: str, value: Any) -> None:
        _set_dotted(self.data, key, value)

    def section(self, key: str) -> Dict[str, Any]:
        v = self.get(key, {})
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError('config key %s is not a mapping' % key)
        return v

    def resolve(self, p: Optional[str]) -> Optional[Path]:
        """Resolve a path string relative to the package root ('' / None -> None)."""
        if p is None or str(p).strip() == '':
            return None
        q = Path(str(p)).expanduser()
        if not q.is_absolute():
            q = self.root / q
        return q

    def path(self, key: str) -> Optional[Path]:
        return self.resolve(self.get(key, None))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False)

    def copy(self) -> 'Config':
        return Config(copy.deepcopy(self.data), self.root, self.sources)


def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError('%s: top level must be a mapping' % path)
    return data


def write_yaml(path: Path, data: Dict[str, Any], header: str = '') -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        if header:
            for line in header.splitlines():
                f.write('# %s\n' % line)
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


def load_config(base: Optional[str] = None, extra: Iterable[str] = (),
                overrides: Iterable[str] = (), use_overlays: bool = True,
                root: Optional[Path] = None) -> Config:
    """
    Load default.yaml, merge site overlays that exist, then extra files, then overrides.

    overrides: 'dotted.key=yaml_value' strings, e.g. 'robot.speed_percent=50'.
    """
    root = Path(root) if root is not None else package_root()
    base_path = Path(base) if base else root / 'config' / 'default.yaml'
    data = read_yaml(base_path)
    sources = [str(base_path)]
    if use_overlays:
        for rel in data.get('site_overlays', []) or []:
            p = Path(rel)
            p = p if p.is_absolute() else root / p
            if p.exists():
                data = deep_merge(data, read_yaml(p))
                sources.append(str(p))
    for e in extra:
        p = Path(e).expanduser()
        data = deep_merge(data, read_yaml(p))
        sources.append(str(p))
    for ov in overrides:
        if '=' not in ov:
            raise ValueError('override must look like key=value: %r' % ov)
        k, v = ov.split('=', 1)
        _set_dotted(data, k.strip(), _parse_value(v))
        sources.append('override:%s' % ov)
    return Config(data, root, sources)
