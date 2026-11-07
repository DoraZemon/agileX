"""L3 parser on realistic Chinese instructions, the LLM veto and the run_task parse preview."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
import urllib.error
import urllib.request

from cube_task.core.zones import parse_zones
from cube_task.task.instruction import COUNT_ALL, NOT_SUPPORTED, parse_instruction
from cube_task.tools import run_task
import pytest
from test_task_fakes import make_cfg

LABELS = ['apple', 'banana', 'orange', 'carrot', 'red', 'green', 'blue']


@pytest.fixture
def cfg():
    c = make_cfg('L3')
    c.set('tasks.L3.labels', list(LABELS))
    c.set('zones.fruit_bin', {'center': [0.40, 0.30, 0.0], 'label': '水果篮'})
    c.set('zones.veg_bin', {'center': [0.40, -0.30, 0.0], 'label': '蔬菜框'})
    c.set('zones.plate', {'center': [0.45, 0.0, 0.0], 'label': 'plate'})
    c.set('tasks.L3.zone_aliases', {'plate': ['盘子'], 'zone_blue': ['蓝色托盘']})
    return c


@pytest.fixture
def zones(cfg):
    return parse_zones(cfg)


ACCEPTED = [
    ('把最左边的苹果放到蓝色托盘', 'apple', 'leftmost', 'zone_blue', 1),
    ('将红色方块放到绿色区域', 'red', 'any', 'zone_green', 1),
    ('把香蕉放进水果篮', 'banana', 'any', 'fruit_bin', 1),
    ('把离机器人最近的橙子放到盘子里', 'orange', 'nearest', 'plate', 1),
    ('把所有的苹果放到蓝色托盘', 'apple', 'any', 'zone_blue', COUNT_ALL),
    ('把两个苹果都放到蓝色托盘', 'apple', 'any', 'zone_blue', 2),
    ('请把桌上的胡萝卜放入蔬菜框', 'carrot', 'any', 'veg_bin', 1),
    ('把放在左边的苹果放到蓝色托盘', 'apple', 'leftmost', 'zone_blue', 1),
    ('帮我把最右边的香蕉放到黄色托盘上', 'banana', 'rightmost', 'zone_yellow', 1),
    ('把这两个橙子都放进水果篮', 'orange', 'any', 'fruit_bin', 2),
    ('把俩苹果都放到蓝色托盘', 'apple', 'any', 'zone_blue', 2),
    ('把桌子上最大的苹果放到红色托盘', 'apple', 'largest', 'zone_red', 1),
    ('把左边第一个苹果放到蓝色托盘', 'apple', 'leftmost', 'zone_blue', 1),
    ('1. 把苹果放到蓝色托盘', 'apple', 'any', 'zone_blue', 1),
    ('2、把香蕉放到黄色托盘', 'banana', 'any', 'zone_yellow', 1),
    ('（3）把橙子放进水果篮。', 'orange', 'any', 'fruit_bin', 1),
    ('第四条：把胡萝卜放到绿色区域', 'carrot', 'any', 'zone_green', 1),
    ('把蓝色方块移动到红色区域', 'blue', 'any', 'zone_red', 1),
    ('把苹果放回蓝色托盘', 'apple', 'any', 'zone_blue', 1),
    ('把全部胡萝卜都放到绿色区域', 'carrot', 'any', 'zone_green', COUNT_ALL),
    ('把放在桌子右边的橙子放到红色托盘里', 'orange', 'rightmost', 'zone_red', 1),
    ('把放在最左边的红色方块放到绿色区域', 'red', 'leftmost', 'zone_green', 1),
    ('把中间的苹果放到蓝色托盘', 'apple', 'middle', 'zone_blue', 1),
    ('把最远的香蕉放置于黄色托盘', 'banana', 'farthest', 'zone_yellow', 1),
    ('请将离机械臂最近的那个苹果放入蓝色托盘中', 'apple', 'nearest', 'zone_blue', 1),
    ('把苹果投放到蓝色托盘', 'apple', 'any', 'zone_blue', 1),
]


@pytest.mark.parametrize('text,label,selector,zone,count', ACCEPTED)
def test_realistic_instructions_parse(cfg, zones, text, label, selector, zone, count):
    s = parse_instruction(text, cfg, LABELS, zones)
    assert s.ok, s.reason
    assert (s.label, s.selector, s.zone, s.count) == (label, selector, zone, count)
    assert s.raw == text and s.source == 'rules'


REFUSED = [
    ('把左边第二个苹果放到蓝色托盘', NOT_SUPPORTED + 'ordinal "第二"'),
    ('把从左数第二个苹果放到蓝色托盘', NOT_SUPPORTED + 'ordinal'),
    ('把第三个橙子放进水果篮', NOT_SUPPORTED + 'ordinal "第三"'),
    ('把右边倒数第一个香蕉放到黄色托盘', NOT_SUPPORTED + 'ordinal "倒数"'),
    ('把第一个苹果放到蓝色托盘', NOT_SUPPORTED + 'ordinal "第一"'),     # first from where?
    ('把2号苹果放到蓝色托盘', NOT_SUPPORTED + 'ordinal "2号"'),
    ('put the second apple into the blue tray', NOT_SUPPORTED + 'ordinal "second"'),
    ('把左边第二个苹果搁到蓝色托盘', NOT_SUPPORTED + 'ordinal'),       # no known verb
    ('把红色托盘里的苹果放到蓝色托盘', NOT_SUPPORTED + 'object described by a zone'),
    ('把放在蓝色托盘旁边的苹果放到红色托盘', NOT_SUPPORTED + 'object described by a zone'),
    ('把放在角落的苹果放到蓝色托盘', NOT_SUPPORTED + 'object described by its place'),
    ('把香蕉和橙子放进水果篮', 'several objects named (banana, orange)'),
    ('把苹果和香蕉都放到水果篮', 'several objects named'),
    ('请把桌上所有的水果放进水果篮', 'category word "水果"'),
    ('将所有蔬菜放入绿色区域', 'category word "蔬菜"'),
    ('把最左边的方块放到红色区域', 'no known object label'),
    ('把放在左边的东西放到蓝色托盘', 'no known object label in "把放在左边的东西"'),
]


@pytest.mark.parametrize('text,why', REFUSED)
def test_unsupported_instructions_are_refused_with_a_reason(cfg, zones, text, why):
    s = parse_instruction(text, cfg, LABELS, zones)
    assert not s.ok
    assert why in s.reason, s.reason


def test_ordinal_never_becomes_a_count(cfg, zones):
    for text in ('把左边第二个苹果放到蓝色托盘', '把第3个苹果放到蓝色托盘', '把第二只香蕉放到黄色托盘'):
        s = parse_instruction(text, cfg, LABELS, zones)
        assert not s.ok and s.count == 1, (text, s)


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _llm_reply(obj):
    content = obj if isinstance(obj, str) else json.dumps(obj)
    return _Resp(json.dumps({'choices': [{'message': {'content': content}}]}).encode())


def _llm(cfg, monkeypatch, reply=None, error=None):
    """Enable the LLM with a mocked endpoint; return the list of user messages it got."""
    cfg.set('tasks.L3.llm.enabled', True)
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(json.loads(req.data.decode())['messages'][1]['content'])
        if error is not None:
            raise error
        return _llm_reply(reply)
    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    return seen


def test_llm_is_not_asked_about_constructs_a_spec_cannot_express(cfg, zones, monkeypatch):
    seen = _llm(cfg, monkeypatch, {'label': 'apple', 'selector': 'leftmost',
                                   'zone': 'zone_blue', 'count': 2})
    for text in ('把左边第二个苹果放到蓝色托盘', '把红色托盘里的苹果放到蓝色托盘'):
        s = parse_instruction(text, cfg, LABELS, zones)
        assert not s.ok and s.source == 'rules' and s.reason.startswith(NOT_SUPPORTED)
    assert seen == []
    s = parse_instruction('把香蕉和橙子放进水果篮', cfg, LABELS, zones)   # LLM may still try
    assert s.source == 'llm' and len(seen) == 1


def test_llm_gets_the_text_without_the_list_number(cfg, zones, monkeypatch):
    seen = _llm(cfg, monkeypatch, {'label': 'banana', 'selector': 'any',
                                   'zone': 'zone_yellow', 'count': 1})
    s = parse_instruction('2. 把香蕉放到黄色托盘', cfg, LABELS, zones)
    assert seen == ['把香蕉放到黄色托盘']
    assert s.ok and s.source == 'llm' and s.raw == '2. 把香蕉放到黄色托盘' and s.reason == ''


# ------------------------------------------------------------- run_task preview
def test_preview_without_llm_shows_the_rule_parse(cfg, zones):
    assert run_task._instruction_notes(cfg, '把两个苹果都放到蓝色托盘', zones) == [
        '  -> apple x2 (any) -> zone_blue']
    note = run_task._instruction_notes(cfg, '把左边第二个苹果放到蓝色托盘', zones)
    assert len(note) == 1 and note[0].startswith('  -> NOT UNDERSTOOD: ' + NOT_SUPPORTED)


def test_preview_with_llm_shows_both_parses(cfg, zones, monkeypatch):
    _llm(cfg, monkeypatch, {'label': 'apple', 'selector': 'leftmost', 'zone': 'zone_blue',
                            'count': 1})
    note = run_task._instruction_notes(cfg, '把最左边的苹果放到蓝色托盘', zones)
    assert note == ['', '       rules: apple x1 (leftmost) -> zone_blue',
                    '       llm:   apple x1 (leftmost) -> zone_blue  [the run uses the LLM '
                    'parse]']


def test_preview_with_llm_flags_disagreement_failure_and_veto(cfg, zones, monkeypatch):
    _llm(cfg, monkeypatch, {'label': 'apple', 'selector': 'any', 'zone': 'zone_blue',
                            'count': 'all'})
    note = run_task._instruction_notes(cfg, '把两个苹果都放到蓝色托盘', zones)
    assert note[1] == '       rules: apple x2 (any) -> zone_blue'
    assert 'llm:   apple xall (any) -> zone_blue' in note[2] and 'DISAGREES' in note[2]
    note = run_task._instruction_notes(cfg, '把左边第二个苹果放到蓝色托盘', zones)
    assert 'NOT UNDERSTOOD: ' + NOT_SUPPORTED in note[1] and 'not asked' in note[2]
    _llm(cfg, monkeypatch, error=urllib.error.URLError('connection refused'))
    note = run_task._instruction_notes(cfg, '把香蕉放进水果篮', zones)
    assert note[1] == '       rules: banana x1 (any) -> fruit_bin'
    assert 'FAILED (' in note[2] and 'connection refused' in note[2]
    assert 'the run uses the rules parse' in note[2]


class _Arm:
    def health(self):
        return SimpleNamespace(ok=True, reason='', detail={})

    def get_joints(self):
        return None


def test_summary_lists_both_parses_under_each_instruction(cfg, monkeypatch, capsys):
    _llm(cfg, monkeypatch, {'label': 'banana', 'selector': 'any', 'zone': 'fruit_bin',
                            'count': 1})
    args = SimpleNamespace(task='L3', mode='sim', dry_run=False, seed=0, difficulty='normal',
                           extrinsics_error=(0.0, 0.0))
    opts = SimpleNamespace(instructions=['把香蕉放进水果篮', '把左边第二个苹果放到蓝色托盘'],
                           instructions_source='command line', speed=30, configured_speed=30,
                           real=False, targets=[])
    lines = run_task.summary_lines(args, SimpleNamespace(cfg=cfg, arm=_Arm(), logger=None), opts)
    i = lines.index('  1. 把香蕉放进水果篮')
    assert lines[i + 1] == '       rules: banana x1 (any) -> fruit_bin'
    assert lines[i + 2].startswith('       llm:   banana x1 (any) -> fruit_bin')
    assert lines[i + 3] == '  2. 把左边第二个苹果放到蓝色托盘'
    assert 'not asked' in lines[i + 5]
    assert 'asking the LLM to parse 2 instruction(s)' in capsys.readouterr().out
