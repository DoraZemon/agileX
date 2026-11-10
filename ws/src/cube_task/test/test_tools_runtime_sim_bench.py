"""sim_bench: argument parsing, fault points, a small real bench run, report highlighting."""
from __future__ import annotations

import json

from cube_task.tools import sim_bench
import pytest


def test_parse_seeds_and_lists():
    assert sim_bench.parse_seeds('0-3') == [0, 1, 2, 3]
    assert sim_bench.parse_seeds('5, 0,2-3,2') == [0, 2, 3, 5]
    for bad in ('', '3-1', '-2', 'a', '1-'):
        with pytest.raises(ValueError):
            sim_bench.parse_seeds(bad)
    assert sim_bench.parse_list('L1,L3,L1', sim_bench.TASKS, '--tasks') == ['L1', 'L3']
    with pytest.raises(ValueError, match='unknown L4'):
        sim_bench.parse_list('L4', sim_bench.TASKS, '--tasks')


def test_make_specs_expands_the_grid():
    args = sim_bench.build_parser().parse_args(
        ['--tasks', 'L1,L2', '--seeds', '0-2', '--mounts', 'eye_to_hand',
         '--difficulty', 'normal,hard', '--faults', 'none,drop'])
    specs = sim_bench.make_specs(args)
    assert len(specs) == 2 * 3 * 1 * 2 * 2
    assert {s['fault'] for s in specs} == {'none', 'drop'}


def test_fault_points_are_seeded_and_inside_the_run():
    spec = {'task': 'L1', 'seed': 3, 'mount': 'eye_in_hand', 'difficulty': 'normal'}
    for fault in sim_bench.FAULTS:
        a = sim_bench.fault_point(dict(spec, fault=fault), 3)
        b = sim_bench.fault_point(dict(spec, fault=fault), 3)
        assert a == b
        if fault == 'none':
            assert a == {}
        elif fault in sim_bench.ARM_FAULTS:
            assert 3 <= a['at_call'] < 3 + 24
        else:
            assert 1 <= a['at_call'] <= 9 and 1 <= a['count'] <= 3
    pts = {sim_bench.fault_point(dict(spec, fault='drop', seed=s), 3)['at_call']
           for s in range(10)}
    assert len(pts) > 1


def test_small_bench_run_writes_json(tmp_path, capsys):
    out = tmp_path / 'bench.json'
    code = sim_bench.main(['--tasks', 'L1', '--seeds', '0', '--mounts', 'eye_to_hand',
                           '--faults', 'none,grasp_miss,camera_none', '--json', str(out)])
    text = capsys.readouterr().out
    assert code == 0, text
    data = json.loads(out.read_text(encoding='utf-8'))
    runs = {r['fault']: r for r in data['runs']}
    assert set(runs) == {'none', 'grasp_miss', 'camera_none'}
    for r in runs.values():
        assert r['success'] and r['correct'] == 3 and r['wrong'] == 0 and not r['error']
        assert r['placed'] == r['required'] == 3 and r['stop_reason'] == 'done'
        assert 'sim_bench --tasks L1 --seeds 0' in r['reproduce']
    assert runs['grasp_miss']['fault_fired'] and runs['grasp_miss']['attempts'] == 4
    assert runs['camera_none']['fault_fired']
    agg = data['aggregate']['overall']
    assert agg['runs'] == 3 and agg['success_rate'] == 1.0 and agg['wrong_placements'] == 0
    assert 'wrong placements: none' in text


@pytest.mark.parametrize('required', [1, 3, 6])
@pytest.mark.parametrize('mount', sim_bench.MOUNTS)
def test_grasp_miss_is_armed_before_the_last_continuous_close(required, mount):
    # Preparation takes 2 commands; the first close is command 6, then every 9.
    for seed in range(20):
        spec = dict(task='L1', seed=seed, mount=mount, difficulty='normal', fault='grasp_miss')
        assert sim_bench.fault_point(spec, required)['at_call'] <= 6 + 9 * (required - 1)


def test_parallel_jobs_match_sequential(tmp_path):
    seq, par = tmp_path / 'seq.json', tmp_path / 'par.json'
    argv = ['--tasks', 'L1', '--seeds', '0-1', '--mounts', 'eye_to_hand', '--faults', 'drop']
    assert sim_bench.main(argv + ['--json', str(seq)]) == 0
    assert sim_bench.main(argv + ['--jobs', '2', '--json', str(par)]) == 0
    keys = ('seed', 'placed', 'correct', 'wrong', 'elapsed', 'attempts', 'stop_reason')
    a = [[r[k] for k in keys] for r in json.loads(seq.read_text())['runs']]
    b = [[r[k] for k in keys] for r in json.loads(par.read_text())['runs']]
    assert a == b


def _run(**kw):
    r = {'task': 'L1', 'seed': 0, 'mount': 'eye_to_hand', 'difficulty': 'normal',
         'fault': 'none', 'placed': 3, 'required': 3, 'correct': 3, 'wrong': 0,
         'remaining': 0, 'elapsed': 60.0, 'time_limit': 180.0, 'attempts': 3,
         'stop_reason': 'done', 'error': '', 'reproduce': 'python3 -m cube_task sim_bench x'}
    r.update(kw)
    r['success'] = sim_bench.is_success(r)
    return r


def test_wrong_placements_are_highlighted_and_fail_the_bench():
    runs = [_run(), _run(seed=1, correct=2, wrong=1, placed=3), _run(seed=2, elapsed=190.0)]
    assert [r['success'] for r in runs] == [True, False, False]
    agg = sim_bench.aggregate(runs)
    assert agg['overall']['wrong_placements'] == 1 and agg['overall']['runs_with_wrong'] == 1
    assert agg['overall']['success_rate'] == pytest.approx(1 / 3.0, abs=1e-3)
    text = sim_bench.format_report(runs, agg)
    assert '!! WRONG PLACEMENTS (worst outcome) in 1 run(s), 1 object(s)' in text
    assert 'reproduce: python3 -m cube_task sim_bench x' in text
    assert '!! WRONG' in sim_bench._row(runs[1])


def test_bad_arguments_exit_2(capsys):
    assert sim_bench.main(['--tasks', 'L9']) == 2
    assert sim_bench.main(['--seeds', 'x']) == 2
    assert sim_bench.main(['--jobs', '0']) == 2
