"""check_setup: ip-link parsing, CAN / ROS checks with fake commands, config-only and sim live."""
from __future__ import annotations

import importlib
import sys

from cube_task.core import config as core_config
from cube_task.tools import check_setup, common
import pytest

IP_UP = """\
3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default
    link/can  promiscuity 0 minmtu 0 maxmtu 0
    can state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 0
\t  bitrate 1000000 sample-point 0.750
\t  tq 50 prop-seg 7 phase-seg1 7 phase-seg2 5 sjw 1
\t  gs_usb: tseg1 1..16 tseg2 1..8 sjw 1..4 brp 1..1024 brp-inc 1
\t  clock 48000000 numtxqueues 1 numrxqueues 1 parentbus usb parentdev 1-2:1.0
"""
IP_DOWN = """\
3: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT group default qlen 10
    link/can  promiscuity 0 minmtu 0 maxmtu 0
    can state STOPPED restart-ms 0
\t  bitrate 1000000 sample-point 0.750
"""
IP_SLOW = IP_UP.replace('bitrate 1000000', 'bitrate 500000')
IP_BUSOFF = IP_UP.replace('ERROR-ACTIVE', 'BUS-OFF')
IP_VCAN = """\
7: vcan0: <NOARP,UP,LOWER_UP> mtu 72 qdisc noqueue state UNKNOWN mode DEFAULT group default
    link/can  promiscuity 0 minmtu 0 maxmtu 0
    vcan numtxqueues 1 numrxqueues 1 gso_max_size 65536 gso_max_segs 65535
"""


def fake_run(ip_out='', ip_rc=0, br='', topics=(), ros_rc=0):
    calls = []

    def run(cmd, timeout):
        calls.append(list(cmd))
        if cmd[:3] == ['ip', '-details', 'link']:
            return ip_rc, ip_out, '' if ip_rc == 0 else 'Device "%s" does not exist.' % cmd[-1]
        if cmd[:2] == ['ip', '-br']:
            return 0, br, ''
        if cmd[:3] == ['ros2', 'topic', 'list']:
            return ros_rc, ''.join(t + '\n' for t in topics), ''
        raise AssertionError('unexpected command %s' % cmd)

    run.calls = calls
    return run


def test_parse_ip_link_up():
    info = check_setup.parse_ip_link(IP_UP)
    assert info['exists'] and info['up'] and info['name'] == 'can0'
    assert info['bitrate'] == 1000000 and info['can_state'] == 'ERROR-ACTIVE'
    assert info['parentdev'] == '1-2:1.0' and not info['virtual']


def test_parse_ip_link_variants():
    down = check_setup.parse_ip_link(IP_DOWN)
    assert down['exists'] and not down['up'] and down['can_state'] == 'STOPPED'
    assert check_setup.parse_ip_link('')['exists'] is False
    assert check_setup.parse_ip_link('Device "can0" does not exist.')['exists'] is False
    v = check_setup.parse_ip_link(IP_VCAN)
    assert v['virtual'] and v['up'] and v['bitrate'] is None
    assert check_setup.parse_ip_link(IP_BUSOFF)['can_state'] == 'BUS-OFF'


@pytest.mark.parametrize('out,rc,level,needle', [
    (IP_UP, 0, 'PASS', 'UP at 1000000'),
    (IP_DOWN, 0, 'FAIL', 'DOWN'),
    (IP_SLOW, 0, 'FAIL', 'bitrate 500000'),
    (IP_BUSOFF, 0, 'FAIL', 'BUS-OFF'),
    (IP_VCAN, 0, 'WARN', 'virtual'),
    ('', 1, 'FAIL', 'does not exist'),
])
def test_check_can(out, rc, level, needle):
    rep = check_setup.Report()
    check_setup.check_can(rep, 'can0', run=fake_run(out, rc))
    assert [i.level for i in rep.items] == [level]
    item = rep.items[0]
    assert needle in item.text
    if level == 'FAIL':
        assert 'can_up.sh can0' in item.fix or 'CAN cable' in item.fix


def test_missing_can_lists_other_interfaces():
    rep = check_setup.Report()
    check_setup.check_can(rep, 'can0', run=fake_run('', 1, br='can1  UP  <NOARP,UP>\n'))
    assert 'can1' in rep.items[0].text and 'robot.can_channel: can1' in rep.items[0].fix


def test_check_ros_topics():
    cfg = core_config.load_config(use_overlays=False)
    topics = [cfg.get('camera.ros.color_topic'), cfg.get('camera.ros.info_topic'), '/other']
    rep = check_setup.Report()
    check_setup.check_ros(rep, cfg, run=fake_run(topics=topics))
    levels = {i.text.split()[0]: i.level for i in rep.items}
    assert levels['camera.ros.color_topic'] == 'PASS'
    assert levels['camera.ros.info_topic'] == 'PASS'
    assert levels['camera.ros.depth_topic'] == 'FAIL'
    rep = check_setup.Report()
    check_setup.check_ros(rep, cfg, run=fake_run(ros_rc=1))
    assert rep.failed and 'ros2 daemon' in rep.items[0].fix


def test_missing_required_module_fails_with_fix(monkeypatch):
    real = importlib.import_module

    def imp(name, *a):
        if name == 'pyAgxArm':
            raise ImportError('No module named pyAgxArm')
        return real(name, *a)

    monkeypatch.setattr(check_setup.importlib, 'import_module', imp)
    rep = check_setup.Report()
    check_setup.check_modules(rep, need_arm=True, need_ros=False)
    bad = [i for i in rep.items if i.level == 'FAIL']
    assert len(bad) == 1 and 'pyAgxArm' in bad[0].text and 'pip install --user' in bad[0].fix
    rep = check_setup.Report()
    check_setup.check_modules(rep, need_arm=False, need_ros=False)
    assert not rep.failed


def _no_site_overlays(monkeypatch):
    real = core_config.load_config

    def load(*a, **kw):
        kw['use_overlays'] = False
        return real(*a, **kw)

    monkeypatch.setattr(common, 'load_config', load)


def test_default_config_reports_fails_without_crashing(monkeypatch, capsys):
    _no_site_overlays(monkeypatch)
    run = fake_run('', 1, topics=[])
    monkeypatch.setattr(check_setup, 'run_cmd', run)
    monkeypatch.setattr(check_setup, 'ros_sourced', lambda: True)
    monkeypatch.setattr(sys, 'argv', ['check_setup'])
    assert check_setup.main() == 1
    out = capsys.readouterr().out
    for needle in ('no zones configured', 'no extrinsics', 'class_to_zone is empty',
                   'can0 does not exist', 'L3 (optional) is not configured'):
        assert needle in out, needle
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith('[FAIL]'):
            assert lines[i + 1].strip().startswith('fix: '), ln
    assert ['ros2', 'topic', 'list'] in run.calls
    assert 'NOT READY' in lines[-1]


def test_default_config_single_task_with_skip(monkeypatch, capsys):
    _no_site_overlays(monkeypatch)
    monkeypatch.setattr(check_setup, 'run_cmd', fake_run())
    assert check_setup.main(['--task', 'L2', '--skip', 'can,ros']) == 1
    out = capsys.readouterr().out
    assert 'knn model file missing' in out and 'teach_classes --task L2' in out
    assert 'skipped: --skip' in out


def test_unknown_skip_group():
    assert check_setup.main(['--skip', 'cam']) == 2


def test_sim_scenario_config_passes(capsys):
    assert check_setup.main(['--mode', 'sim', '--task', 'L1']) == 0
    out = capsys.readouterr().out
    assert '[FAIL]' not in out
    assert 'reachable with the tool leaned' in out        # kinematic reachability ran
    assert 'skipped: robot.backend is sim' in out


def test_observe_joints_outside_limits_fail():
    from cube_task.sim.scenarios import make_scenario, scenario_config_overlay
    base = core_config.load_config(use_overlays=False)
    _w, info = make_scenario('L1', 0)
    cfg = core_config.Config(core_config.deep_merge(base.data, scenario_config_overlay(info)),
                             base.root, base.sources)
    cfg.set('robot.poses.observe_joints', [0.0, 3.5, 0.0, 0.0, 0.0, 0.0])
    msgs = check_setup.config_messages(cfg, 'L1')
    assert any(m.startswith('[FAIL] robot.poses.observe_joints outside') for m in msgs)
    assert check_setup.config_fix('observe_joints outside the joint limits', 'L1').startswith(
        'teach robot.poses.observe_joints')


@pytest.mark.parametrize('mount', ['eye_to_hand', 'eye_in_hand'])
def test_live_on_the_simulated_rig_reads_without_moving(monkeypatch, capsys, mount):
    from cube_task.robot import sim_arm
    moves = []
    for name in ('move_tcp', 'move_joints', 'gripper', 'enable'):
        real = getattr(sim_arm.SimArm, name)
        monkeypatch.setattr(sim_arm.SimArm, name,
                            lambda self, *a, _n=name, _r=real, **k: moves.append(_n) or _r(
                                self, *a, **k))
    code = check_setup.main(['--mode', 'sim', '--task', 'L1', '--live', '--mount', mount])
    out = capsys.readouterr().out
    assert code == 0, out
    assert moves == []
    for needle in ('arm: ready', 'arm: joints', 'arm: TCP', 'gripper: width',
                   'camera: 640x480', 'camera: intrinsics', 'valid depth in the image',
                   'frames/s'):
        assert needle in out, needle
