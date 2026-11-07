"""run_task.main() in sim mode: exit codes, refusals, faults, Ctrl-C and the real-mode prompt."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys

from cube_task.core.config import Config, deep_merge, load_config
from cube_task.sim.scenarios import make_scenario, scenario_config_overlay
from cube_task.task.runner import TaskRunner
from cube_task.tools import run_task
import pytest


def _main(monkeypatch, tmp_path, *extra):
    monkeypatch.setattr(sys, 'argv', ['run_task', '--mode', 'sim', '--log-dir', str(tmp_path)]
                        + list(extra))
    return run_task.main()


def _capture_sessions(monkeypatch):
    """Wrap run_task.open_session; return the list the opened sessions are appended to."""
    sessions = []
    real_open = run_task.open_session

    def wrapped(*a, **kw):
        s = real_open(*a, **kw)
        s.test_sim_arm = s.arm.inner           # kept after Session.close() drops s.arm
        sessions.append(s)
        return s

    monkeypatch.setattr(run_task, 'open_session', wrapped)
    return sessions


def _summary(tmp_path: Path) -> dict:
    files = sorted(tmp_path.glob('*/summary.json'))
    assert files, 'no summary.json under %s' % tmp_path
    return json.loads(files[-1].read_text(encoding='utf-8'))


def test_l1_sim_complete_exits_0_and_prints_report(monkeypatch, capsys, tmp_path):
    sessions = _capture_sessions(monkeypatch)
    assert _main(monkeypatch, tmp_path, '--task', 'L1') == run_task.EXIT_OK
    out = capsys.readouterr().out
    for text in ('run_task L1 | SIM mode', 'labels:', 'red -> zone_red', 'extrinsics:',
                 'speed:        30 %', 'observe pose: taught joints', 'classifier:   color',
                 'log dir:', 'L1 COMPLETE: placed 3/3',
                 'ground truth (sim): correct 3, wrong 0, remaining 0', 'exit code 0'):
        assert text in out, text
    s = _summary(tmp_path)
    assert s['exit_code'] == 0 and s['stop_reason'] == 'done' and s['placed'] == 3
    assert s['ground_truth'] == {'correct': 3, 'wrong': 0, 'remaining': 0}
    assert (Path(s['_logger']['dir']) / 'config.yaml').is_file()
    assert sessions[0].arm is None and sessions[0].operator is None     # session closed


def test_timeout_exits_1(monkeypatch, capsys, tmp_path):
    code = _main(monkeypatch, tmp_path, '--task', 'L1', '--set', 'tasks.L1.time_limit_s=25')
    out = capsys.readouterr().out
    assert code == run_task.EXIT_INCOMPLETE
    assert 'stop reason: timeout' in out and 'INCOMPLETE' in out


def test_targets_restrict_labels(monkeypatch, capsys, tmp_path):
    code = _main(monkeypatch, tmp_path, '--task', 'L1', '--targets', 'blue，RED',
                 '--set', 'tasks.L1.required=2')
    out = capsys.readouterr().out
    assert code == run_task.EXIT_OK
    assert 'targets:      blue, red' in out
    assert 'ground truth (sim): correct 2, wrong 0, remaining 0' in out


@pytest.mark.parametrize('extra,needle', [
    (['--speed', '50'], 'above robot.speed_percent 30'),
    (['--speed', '0'], '--speed must be in 1..100'),
    (['--targets', 'purple'], "target 'purple' is not a L1 label"),
    (['--instruction', 'pick the apple'], 'are for --task L3'),
    (['--set', 'tasks.L1.class_to_zone={}'], 'class_to_zone is empty'),
])
def test_refusals_exit_2_without_motion(monkeypatch, capsys, caplog, tmp_path, extra, needle):
    sessions = _capture_sessions(monkeypatch)
    moved = []
    monkeypatch.setattr(TaskRunner, '_go_observe', lambda self: moved.append(1) or (False, ''))
    code = _main(monkeypatch, tmp_path, '--task', 'L1', *extra)
    text = capsys.readouterr().out + caplog.text
    assert code == run_task.EXIT_REFUSED
    assert needle in text
    assert moved == []
    for s in sessions:
        assert s.arm is None                                # closed again


def test_l3_without_instructions_is_refused(monkeypatch, capsys, tmp_path):
    empty = tmp_path / 'none.txt'
    empty.write_text('# only a comment\n\n', encoding='utf-8')
    code = _main(monkeypatch, tmp_path, '--task', 'L3', '--instructions-file', str(empty))
    assert code == run_task.EXIT_REFUSED
    assert 'no L3 instructions' in capsys.readouterr().out


def test_allow_faster_raises_the_speed(monkeypatch, capsys, tmp_path):
    sessions = _capture_sessions(monkeypatch)
    # 1 s limit: the runner stops before the first pick, so the dry run is incomplete.
    assert _main(monkeypatch, tmp_path, '--task', 'L1', '--speed', '60', '--allow-faster',
                 '--set', 'tasks.L1.time_limit_s=1', '--dry-run') == run_task.EXIT_INCOMPLETE
    assert 'RAISED from 30 %' in capsys.readouterr().out
    assert sessions[0].cfg.get('robot.speed_percent') == 60


def test_dry_run_visits_every_block_and_exits_0(monkeypatch, capsys, tmp_path):
    sessions = _capture_sessions(monkeypatch)
    code = _main(monkeypatch, tmp_path, '--task', 'L1', '--dry-run')
    out = capsys.readouterr().out
    assert code == run_task.EXIT_OK, out
    assert 'DRY RUN' in out
    assert 'ground truth (sim): correct 0, wrong 0, remaining 3' in out
    world = sessions[0].world
    assert world.held is None
    assert all(world.zone_of(o) is None for o in world.objects)


def test_comm_fault_exits_3(monkeypatch, capsys, tmp_path):
    real_open = run_task.open_session

    def faulty(*a, **kw):
        s = real_open(*a, **kw)
        s.arm.inner.inject('comm_loss', at_call=15)
        return s

    monkeypatch.setattr(run_task, 'open_session', faulty)
    code = _main(monkeypatch, tmp_path, '--task', 'L1')
    out = capsys.readouterr().out
    assert code == run_task.EXIT_FAULT
    assert 'stop reason: fault' in out and 'exit code 3' in out
    assert _summary(tmp_path)['stop_reason'] == 'fault'


def test_ctrl_c_stops_the_arm_saves_the_log_and_exits_130(monkeypatch, capsys, tmp_path):
    sessions = _capture_sessions(monkeypatch)
    calls = []

    def interrupted_run(self):
        s = sessions[0]
        real_stop = s.arm.stop
        s.arm.stop = lambda: calls.append('stop') or real_stop()
        s.arm.disable = lambda: calls.append('disable')
        op = s.operator
        real_close = op.close
        op.close = lambda: calls.append('operator.close') or real_close()
        raise KeyboardInterrupt

    monkeypatch.setattr(TaskRunner, 'run', interrupted_run)
    code = _main(monkeypatch, tmp_path, '--task', 'L1')
    out = capsys.readouterr().out
    assert code == run_task.EXIT_INTERRUPTED
    assert calls[0] == 'stop' and 'disable' not in calls and 'operator.close' in calls
    assert 'Ctrl-C' in out and 'log saved' in out
    s = _summary(tmp_path)
    assert s['stop_reason'] == 'interrupted' and s['exit_code'] == 130
    assert sessions[0].arm is None                          # closed (disconnected, not disabled)


def test_crash_in_the_runner_still_stops_the_arm(monkeypatch, capsys, tmp_path):
    sessions = _capture_sessions(monkeypatch)
    calls = []

    def broken_run(self):
        sessions[0].arm.stop = lambda: calls.append('stop')
        raise RuntimeError('boom')

    monkeypatch.setattr(TaskRunner, 'run', broken_run)
    assert _main(monkeypatch, tmp_path, '--task', 'L1') == run_task.EXIT_FAULT
    assert calls == ['stop']
    assert 'run_task crashed: boom' in capsys.readouterr().out


def test_real_mode_asks_before_any_motion(monkeypatch, capsys, tmp_path):
    """--mode real with a simulated rig behind it: answering 'n' moves and enables nothing."""
    base = load_config(use_overlays=False)
    world, info = make_scenario('L1', 0)
    cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root, base.sources)
    cfg.set('logging.dir', str(tmp_path))         # load_cfg_for applies --log-dir; faked here
    monkeypatch.setattr(run_task, 'load_cfg_for', lambda args, task=None: (cfg, world, info))
    sessions = _capture_sessions(monkeypatch)
    prompts = []
    monkeypatch.setattr(builtins, 'input', lambda text='': prompts.append(text) or 'n')
    monkeypatch.setattr(sys, 'argv', ['run_task', '--task', 'L1', '--mode', 'real',
                                      '--speed', '20', '--log-dir', str(tmp_path)])
    code = run_task.main()
    out = capsys.readouterr().out
    assert code == run_task.EXIT_REFUSED
    assert prompts and 'REAL robot' in prompts[0]
    assert 'About to: enable the motors, move to the observe pose' in out
    assert 'speed:        20 % (lowered from 30 %)' in out
    assert 'waits for Enter' in out
    assert 'Nothing moved' in out
    sim_arm = sessions[0].test_sim_arm
    assert sim_arm.command_log == [] and sim_arm.n_calls == 0
    assert sim_arm.health().enabled is False             # never enabled
    assert sessions[0].arm is None                       # session closed
    assert _summary(tmp_path)['stop_reason'] == 'not_confirmed'


def test_real_mode_self_check_failure_refuses_before_the_prompt(monkeypatch, capsys, tmp_path):
    base = load_config(use_overlays=False)
    world, info = make_scenario('L1', 0)
    cfg = Config(deep_merge(base.data, scenario_config_overlay(info)), base.root, base.sources)
    cfg.set('logging.dir', str(tmp_path))
    cfg.set('zones', {})                          # e.g. zones never taught
    monkeypatch.setattr(run_task, 'load_cfg_for', lambda args, task=None: (cfg, world, info))
    sessions = _capture_sessions(monkeypatch)
    prompts = []
    monkeypatch.setattr(builtins, 'input', lambda text='': prompts.append(text) or 'y')
    monkeypatch.setattr(sys, 'argv', ['run_task', '--task', 'L1', '--mode', 'real'])
    assert run_task.main() == run_task.EXIT_REFUSED
    out = capsys.readouterr().out
    assert prompts == []
    assert 'self check [FAIL] no zones configured' in out
    assert 'nothing was enabled or moved' in out
    sim_arm = sessions[0].test_sim_arm
    assert sim_arm.n_calls == 0 and sim_arm.health().enabled is False
    assert _summary(tmp_path)['stop_reason'] == 'self_check'
