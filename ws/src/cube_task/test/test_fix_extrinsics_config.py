"""Review fix in core/config.package_root: installed copies (plain colcon build)."""
from __future__ import annotations

import contextlib
import logging

import pytest

from cube_task.core import config


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _capture(name='cube_task.config'):
    lg = logging.getLogger(name)
    h = _Records()
    lg.addHandler(h)
    try:
        yield h.records
    finally:
        lg.removeHandler(h)


def _module(base, *parts):
    """Create base/<parts>/cube_task/core/config.py and return its path."""
    d = base.joinpath(*parts, 'cube_task', 'core')
    d.mkdir(parents=True)
    f = d / 'config.py'
    f.write_text('# copy\n', encoding='utf-8')
    return f


def _default_yaml(root):
    (root / 'config').mkdir(parents=True)
    (root / 'config' / 'default.yaml').write_text('robot: {speed_percent: 30}\n',
                                                  encoding='utf-8')


def test_source_tree_is_used_when_it_has_the_config(tmp_path):
    src = tmp_path / 'src' / 'cube_task'
    f = _module(src)
    _default_yaml(src)
    assert config._find_root(f) == src.resolve()


@pytest.mark.parametrize('layout', [('lib', 'python3.10', 'site-packages'),
                                    ('local', 'lib', 'python3.10', 'dist-packages')])
def test_installed_copy_falls_back_to_the_share_directory(tmp_path, layout):
    prefix = tmp_path / 'install' / 'cube_task'
    f = _module(prefix, *layout)
    share = prefix.joinpath(*layout[:-3], 'share', 'cube_task')   # <prefix>[/local]/share
    _default_yaml(share)
    config._warned_roots.discard(str(share.resolve()))
    with _capture() as records:
        assert config._find_root(f) == share.resolve()
        assert config._find_root(f) == share.resolve()
    msgs = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
    assert len(msgs) == 1                                  # warned once per process
    assert 'read-only' in msgs[0] and '--symlink-install' in msgs[0]
    assert 'CUBE_TASK_ROOT' in msgs[0]
    cfg = config.load_config(root=config._find_root(f))
    assert cfg.get('robot.speed_percent') == 30


def test_installed_copy_without_share_raises_with_the_fix(tmp_path):
    f = _module(tmp_path / 'install' / 'cube_task', 'lib', 'python3.10', 'site-packages')
    with pytest.raises(RuntimeError) as e:
        config._find_root(f)
    msg = str(e.value)
    assert 'plain colcon build' in msg
    assert 'colcon build --symlink-install --packages-select cube_task' in msg
    assert 'export CUBE_TASK_ROOT=~/vision/ws/src/cube_task' in msg


def test_package_root_env_wins_and_module_path_is_used_without_it(tmp_path, monkeypatch):
    monkeypatch.setenv('CUBE_TASK_ROOT', str(tmp_path))
    assert config.package_root() == tmp_path.resolve()
    monkeypatch.delenv('CUBE_TASK_ROOT')
    f = _module(tmp_path / 'install' / 'cube_task', 'lib', 'python3.10', 'site-packages')
    monkeypatch.setattr(config, '__file__', str(f))
    with pytest.raises(RuntimeError, match='CUBE_TASK_ROOT'):
        config.package_root()
    with pytest.raises(RuntimeError, match='symlink-install'):
        config.load_config()
