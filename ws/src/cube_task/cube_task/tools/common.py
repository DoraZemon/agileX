"""
Shared command-line plumbing for the tools: config loading and session setup.

Modes:
  real   - real PiPER (pyAgxArm) + real camera (ROS), config = default + site overlays.
  sim    - simulated world, arm and camera from sim/scenarios; site overlays are
           ignored unless --site is given, so taught real poses never leak in.
  replay - recorded frames (camera.backend=file) with a kinematic sim arm and no
           world, for checking perception on real captures offline.
  mujoco - the sim scenario as a MuJoCo physics scene (sim/mujoco): MuJoCo arm,
           rendered RGB-D camera, contact physics; only for the tools that pass
           add_common_args(..., mujoco=True) and only with MuJoCo installed.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Config, deep_merge, load_config

_LOG = logging.getLogger('cube_task.tools')

TASKS = ('L1', 'L2', 'L3')


def add_common_args(p: argparse.ArgumentParser, task_required: bool = False,
                    default_task: Optional[str] = 'L1', mujoco: bool = False) -> None:
    """Add the options every tool understands (mujoco: also offer --mode mujoco)."""
    p.add_argument('--task', choices=TASKS, required=task_required,
                   default=None if task_required else default_task,
                   help='competition task (selects labels, classifier, zones mapping)')
    modes = ('real', 'sim', 'replay') + (('mujoco',) if mujoco else ())
    p.add_argument('--mode', choices=modes, default='real',
                   help='real robot, simulation, or replay of recorded frames'
                   + (' (mujoco: physics simulation)' if mujoco else ''))
    p.add_argument('--config', action='append', default=[], metavar='YAML',
                   help='extra YAML merged after default + site overlays (repeatable)')
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   help='override one config key, e.g. robot.speed_percent=50 (repeatable)')
    p.add_argument('--site', action='store_true',
                   help='sim mode: also merge config/site overlays (ignored otherwise)')
    p.add_argument('--seed', type=int, default=0, help='sim scenario seed')
    p.add_argument('--mount', choices=('eye_to_hand', 'eye_in_hand'), default='eye_to_hand',
                   help='sim camera mount')
    p.add_argument('--difficulty', choices=('normal', 'hard'), default='normal',
                   help='sim scenario difficulty')
    p.add_argument('--extrinsics-error', type=float, nargs=2, default=(0.0, 0.0),
                   metavar=('M', 'DEG'), help='sim: error of the extrinsics estimate')
    p.add_argument('--replay-dir', default='', help='replay mode: directory from capture')
    p.add_argument('--log-dir', default=None, help='override logging.dir')
    p.add_argument('-v', '--verbose', action='store_true', help='debug logging')


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format='%(asctime)s %(levelname).1s %(name)s: %(message)s',
                        datefmt='%H:%M:%S')


def _scenario_task(task: Optional[str]) -> str:
    return task if task in TASKS else 'L1'


def load_cfg_for(args: argparse.Namespace, task: Optional[str] = None):
    """
    Build the Config for a tool run.

    Returns (cfg, sim_world or None, sim_info or None).
    """
    task = task or getattr(args, 'task', None)
    mode = args.mode
    world = None
    info = None
    if mode in ('sim', 'mujoco'):
        from ..sim.scenarios import make_scenario, scenario_config_overlay
        base = load_config(use_overlays=bool(getattr(args, 'site', False)))
        world, info = make_scenario(_scenario_task(task), int(args.seed), mount=args.mount,
                                    difficulty=args.difficulty)
        data = deep_merge(base.data, scenario_config_overlay(
            info, extrinsics_error=tuple(args.extrinsics_error)))
        cfg = Config(data, base.root, base.sources + ['sim:%s:%d' % (task, args.seed)])
        cfg = _apply_extra(cfg, args.config, args.set)
        if mode == 'mujoco':
            cfg, world = _mujoco_cfg_world(cfg, world, info, args)
    else:
        cfg = load_config(extra=args.config, overrides=args.set)
        if mode == 'replay':
            if not args.replay_dir:
                raise SystemExit('--mode replay needs --replay-dir DIR (written by capture)')
            cfg.set('camera.backend', 'file')
            cfg.set('camera.file.dir', str(Path(args.replay_dir).expanduser().resolve()))
            cfg.set('robot.backend', 'sim')
    if args.log_dir:
        cfg.set('logging.dir', args.log_dir)
    return cfg, world, info


def _mujoco_cfg_world(cfg: Config, sim_world, info, args: argparse.Namespace):
    """
    Return (cfg, MujocoWorld) for --mode mujoco.

    The scene is built from the scenario (with the --config / --set values,
    e.g. robot.poses.home_joints, sim.mujoco.*), then its overlay
    (MujocoWorld.config_overlay: backends, MuJoCo TCP, rendered camera,
    real zone floors) is merged and --config / --set are applied again so
    they still win.
    """
    from ..sim.mujoco.scene import build_scene
    mw = build_scene(info, cfg, world=sim_world)
    data = deep_merge(cfg.data, mw.config_overlay(
        extrinsics_error=tuple(args.extrinsics_error)))
    out = _apply_extra(Config(data, cfg.root, []), args.config, args.set)
    out.sources = list(cfg.sources) + ['mujoco:%s' % Path(mw.xml_path).name]
    return out, mw


def _apply_extra(cfg: Config, extra: List[str], overrides: List[str]) -> Config:
    """Merge extra YAML files, then key=value overrides, on top of cfg."""
    import yaml
    from ..core.config import read_yaml
    data = cfg.data
    for e in extra:
        data = deep_merge(data, read_yaml(Path(e).expanduser()))
    out = Config(data, cfg.root, cfg.sources + list(extra))
    for ov in overrides:
        if '=' not in ov:
            raise SystemExit('--set needs KEY=VALUE, got %r' % ov)
        k, v = ov.split('=', 1)
        out.set(k.strip(), yaml.safe_load(v))
    return out


class Session:
    """Everything a tool needs, built from one config. Use as a context manager."""

    def __init__(self, cfg: Config, mode: str, task: Optional[str], clock, world=None,
                 info=None):
        self.cfg = cfg
        self.mode = mode
        self.task = task
        self.clock = clock
        self.world = world
        self.info = info or {}
        self.arm = None
        self.camera = None
        self.perception = None
        self.logger = None
        self.operator = None

    def __enter__(self) -> 'Session':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Stop the camera and disconnect the arm (never disables the motors)."""
        for name, fn in (('camera', lambda: self.camera.stop()),
                         ('arm', lambda: self.arm.close()),
                         ('operator', lambda: self.operator.close())):
            obj = getattr(self, name)
            if obj is None:
                continue
            try:
                fn()
            except Exception:  # noqa: B902 - best effort cleanup
                _LOG.exception('closing %s failed', name)
            setattr(self, name, None)


def ensure_sim_knn(cfg: Config, task: Optional[str], mount: str) -> None:
    """In sim mode, train (once, cached) a KNN model from simulated ground truth."""
    from ..perception.factory import classifier_backend
    if task is None or 'knn' not in classifier_backend(cfg, task):
        return
    path = cfg.path('perception.classifier.knn.model_file')
    if path is not None and path.exists() and cfg.get('sim.use_site_knn', False):
        return
    cache = Path(os.environ.get('CUBE_TASK_CACHE', '~/.cache/cube_task')).expanduser()
    # The file name carries a hash of the sim / perception sources, so a code change
    # that alters the features or the rendering retrains instead of reusing a stale model.
    out = cache / ('knn_sim_%s_%s.npz' % (mount, _sim_knn_fingerprint()))
    if not _knn_loadable(out):
        from ..sim.teach import train_knn_from_sim
        _LOG.info('training a simulation KNN model -> %s (one time)', out)
        train_knn_from_sim(out, mount=mount)
    cfg.set('perception.classifier.knn.model_file', str(out))


def _sim_knn_fingerprint() -> str:
    """Return a short hash of the sim/ and perception/ sources (KNN cache key)."""
    import hashlib
    h = hashlib.sha1()
    pkg = Path(__file__).resolve().parents[1]
    for sub in ('sim', 'perception'):
        for f in sorted((pkg / sub).glob('*.py')):
            h.update(f.name.encode('utf-8'))
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def _knn_loadable(path: Path) -> bool:
    """Return True when path holds a KNN model that loads with the current features."""
    if not path.is_file():
        return False
    from ..perception.classify_knn import KnnClassifier
    try:
        KnnClassifier.load(path)
    except (OSError, ValueError, KeyError) as e:
        _LOG.warning('cached KNN model %s is unusable (%s); retraining', path, e)
        return False
    return True


def open_session(args: argparse.Namespace, task: Optional[str] = None, arm: bool = True,
                 camera: bool = True, perception: bool = True, logger: bool = False,
                 operator: bool = False, connect: bool = True,
                 cfg_world_info: Optional[tuple] = None) -> Session:
    """
    Load config and build the requested components.

    In real mode connect() / enable() are called on the arm only when connect is
    True. The caller owns the session and must close it (with-statement).
    """
    from ..core.clock import RealClock, SimClock
    task = task or getattr(args, 'task', None)
    cfg, world, info = cfg_world_info or load_cfg_for(args, task)
    sim_like = args.mode in ('sim', 'replay', 'mujoco')
    clock = SimClock() if args.mode in ('sim', 'mujoco') else RealClock()
    s = Session(cfg, args.mode, task, clock, world, info)
    try:
        if args.mode == 'mujoco':
            world.attach(clock)         # physics steps with every clock.sleep from now on
        if arm:
            from ..robot.factory import make_arm
            s.arm = make_arm(cfg, clock, sim_world=world)
            if connect:
                s.arm.connect()
                if not s.arm.enable(timeout=float(cfg.get('robot.enable_timeout_s', 5.0))):
                    raise SystemExit('arm did not enable; check the e-stop and the CAN link')
        if camera:
            from ..camera.factory import make_camera
            s.camera = make_camera(cfg, clock, sim_world=world, arm=s.arm)
            s.camera.start()
        if perception:
            if args.mode in ('sim', 'mujoco'):
                # The KNN trained on the kinematic renderer also classifies MuJoCo
                # renders (see sim/mujoco/run.py), so both modes share it.
                ensure_sim_knn(cfg, task, args.mount)
            from ..perception.factory import make_perception
            s.perception = make_perception(cfg, task)
        if logger:
            from ..core.logger import RunLogger
            s.logger = RunLogger(cfg.get('logging.dir', '~/vision_runs'), task or 'tool', clock,
                                 save_images=bool(cfg.get('logging.save_images', True)))
        if operator:
            from ..task.operator import Operator
            s.operator = Operator(enabled=not sim_like)
    except BaseException:
        s.close()
        raise
    return s


def summarize_cfg(cfg: Config, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    keys = keys or ['robot.backend', 'robot.speed_percent', 'robot.tcp_offset',
                    'robot.gripper.max_width', 'camera.backend', 'camera.mount',
                    'camera.extrinsics_file']
    return {k: cfg.get(k, None) for k in keys}
