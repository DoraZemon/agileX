"""scripts/can_up.sh and scripts/start.sh with fake ip / sudo / python commands."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAN_UP = ROOT / 'scripts' / 'can_up.sh'
START = ROOT / 'scripts' / 'start.sh'

UP = ('3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT\\n'
      '    link/can  promiscuity 0\\n'
      '    can state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 0\\n'
      '\\t  bitrate 1000000 sample-point 0.750\\n')
DOWN = ('3: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT\\n'
        '    link/can  promiscuity 0\\n'
        '    can state STOPPED restart-ms 0\\n')

FAKE_IP = r'''#!/usr/bin/env bash
# fake ip: link state comes from $FAKE_DIR/state (up / down / missing)
state="$(cat "$FAKE_DIR/state")"
if [ "$1" = "-details" ] || [ "$1" = "link" ]; then
    [ "$state" = missing ] && { echo "Device \"can0\" does not exist." >&2; exit 1; }
    [ "$state" = up ] && printf '%UP%' || printf '%DOWN%'
    exit 0
fi
if [ "$1" = "-br" ]; then exit 0; fi
exit 0
'''.replace('%UP%', UP).replace('%DOWN%', DOWN)

FAKE_SUDO = r'''#!/usr/bin/env bash
echo "$*" >> "$FAKE_DIR/sudo.log"
[ "$*" = "ip link set can0 up" ] && echo up > "$FAKE_DIR/state"
exit 0
'''

FAKE_PYTHON = r'''#!/usr/bin/env bash
echo "$*" >> "$FAKE_DIR/python.log"
if [ "$1" = "-c" ]; then echo can0; exit 0; fi
case "$*" in *check_setup*) exit "${FAKE_CHECK_RC:-0}" ;; esac
exit 0
'''


def _write(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def fake(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    _write(bin_dir / 'ip', FAKE_IP)
    _write(bin_dir / 'sudo', FAKE_SUDO)
    _write(bin_dir / 'fake_python', FAKE_PYTHON)
    env = dict(os.environ)
    env.update(PATH='%s:/usr/bin:/bin' % bin_dir, FAKE_DIR=str(tmp_path),
               PYTHON=str(bin_dir / 'fake_python'), ROS_SETUP=str(tmp_path / 'no_ros.bash'))

    def run(script, *args, state='up', **extra_env):
        (tmp_path / 'state').write_text(state + '\n')
        e = dict(env, **extra_env)
        return subprocess.run(['bash', str(script)] + list(args), env=e, timeout=20,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True)

    run.dir = tmp_path
    return run


def _log(fake, name):
    p = fake.dir / name
    return p.read_text().splitlines() if p.exists() else []


def test_scripts_parse():
    for s in (CAN_UP, START):
        assert subprocess.run(['bash', '-n', str(s)]).returncode == 0
        assert os.access(str(s), os.X_OK)


def test_can_up_check_ok(fake):
    r = fake(CAN_UP, '--check', state='up')
    assert r.returncode == 0, r.stdout
    assert 'OK: can0 is UP at 1000000' in r.stdout
    assert _log(fake, 'sudo.log') == []


def test_can_up_check_down_never_uses_sudo(fake):
    r = fake(CAN_UP, '--check', 'can0', state='down')
    assert r.returncode == 1
    assert 'NOT READY: can0 is DOWN' in r.stdout and 'can_up.sh can0 1000000' in r.stdout
    assert _log(fake, 'sudo.log') == []


def test_can_up_missing_interface_mentions_the_official_script(fake):
    r = fake(CAN_UP, state='missing')
    assert r.returncode == 1
    assert 'does not exist' in r.stdout
    assert 'can_activate.sh' in r.stdout
    assert _log(fake, 'sudo.log') == []


def test_can_up_brings_the_link_up(fake):
    r = fake(CAN_UP, 'can0', '1000000', state='down')
    assert r.returncode == 0, r.stdout
    assert _log(fake, 'sudo.log') == ['ip link set can0 down',
                                      'ip link set can0 type can bitrate 1000000',
                                      'ip link set can0 up']
    assert 'OK: can0 is UP' in r.stdout


def test_can_up_rejects_bad_names(fake):
    assert fake(CAN_UP, 'can0;reboot').returncode == 2
    assert fake(CAN_UP, 'can0', 'fast').returncode == 2
    assert _log(fake, 'sudo.log') == []


def test_start_requires_a_task(fake):
    r = fake(START)
    assert r.returncode == 2 and 'L1, L2 or L3' in r.stdout
    assert fake(START, 'L1', '--mode', 'sim').returncode == 2


def test_start_runs_checks_then_run_task(fake):
    r = fake(START, 'L1', '--targets', 'red,blue', '--set', 'robot.speed_percent=20',
             state='up')
    assert r.returncode == 0, r.stdout
    log = _log(fake, 'python.log')
    check = [ln for ln in log if 'check_setup' in ln]
    run = [ln for ln in log if 'run_task' in ln]
    assert check == ['-m cube_task check_setup --task L1 --mode real '
                     '--set robot.speed_percent=20']
    assert run == ['-m cube_task run_task --task L1 --mode real --targets red,blue '
                   '--set robot.speed_percent=20']
    assert log.index(check[0]) < log.index(run[0])


def test_start_stops_on_failed_checks_unless_forced(fake):
    r = fake(START, 'L2', state='down', FAKE_CHECK_RC='1')
    assert r.returncode == 1 and 'STOP' in r.stdout
    assert not [ln for ln in _log(fake, 'python.log') if 'run_task' in ln]
    r = fake(START, 'L2', '--force', '--dry-run', state='down', FAKE_CHECK_RC='1')
    assert r.returncode == 0, r.stdout
    run = [ln for ln in _log(fake, 'python.log') if 'run_task' in ln]
    assert run == ['-m cube_task run_task --task L2 --mode real --dry-run']
