"""Parse place zones from the config."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .types import Zone


def parse_zone(name: str, d: Dict[str, Any]) -> Zone:
    if 'center' not in d:
        raise ValueError('zone %s: missing center [x, y, floor_z]' % name)
    c = [float(v) for v in d['center']]
    if len(c) == 2:
        c.append(0.0)
    if len(c) != 3:
        raise ValueError('zone %s: center needs 2 or 3 values' % name)
    size = d.get('size', [0.10, 0.10])
    if len(size) != 2 or min(float(size[0]), float(size[1])) <= 0:
        raise ValueError('zone %s: size must be two positive numbers' % name)
    rh = d.get('release_height', None)
    return Zone(name=name, center=np.array(c, dtype=float),
                size=(float(size[0]), float(size[1])),
                yaw=float(d.get('yaw', 0.0)),
                wall_height=float(d.get('wall_height', 0.0)),
                release_height=None if rh is None else float(rh),
                label=str(d.get('label', '')))


def parse_zones(cfg) -> Dict[str, Zone]:
    """cfg: core.config.Config or a plain dict holding a 'zones' mapping."""
    raw = cfg.get('zones', {}) if hasattr(cfg, 'get') else {}
    raw = raw or {}
    return {str(k): parse_zone(str(k), v or {}) for k, v in raw.items()}
