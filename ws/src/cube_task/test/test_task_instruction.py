"""Tests for task/instruction.py (rules and the mocked LLM backend; no network)."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from cube_task.core.zones import parse_zones
from cube_task.task.instruction import COUNT_ALL, parse_instruction
import pytest
from test_task_fakes import make_cfg

LABELS = ['apple', 'banana', 'orange', 'carrot', 'tomato', 'green_pepper', 'red', 'blue']


@pytest.fixture
def cfg():
    c = make_cfg('L3')
    c.set('zones.fruit_bin', {'center': [0.40, 0.30, 0.0], 'label': '水果'})
    c.set('tasks.L3.label_aliases', {'carrot': ['红萝卜'], 'apple': ['苹果模型']})
    c.set('tasks.L3.zone_aliases', {'zone_green': ['绿盘', '左边的盘子']})
    return c


@pytest.fixture
def zones(cfg):
    return parse_zones(cfg)


OK_CASES = [
    ('把最左边的苹果放到蓝色托盘', 'apple', 'leftmost', 'zone_blue', 1),
    ('把最右边的香蕉放进红色盒子', 'banana', 'rightmost', 'zone_red', 1),
    ('将所有橙子放入黄色的框里', 'orange', 'any', 'zone_yellow', COUNT_ALL),
    ('把两个胡萝卜放到绿色区域', 'carrot', 'any', 'zone_green', 2),
    ('把最大的西红柿放到水果框', 'tomato', 'largest', 'fruit_bin', 1),
    ('把最小的番茄放在红色托盘上', 'tomato', 'smallest', 'zone_red', 1),
    ('把离机械臂最近的青椒放到绿盘', 'green_pepper', 'nearest', 'zone_green', 1),
    ('请把最远的橘子移到蓝色盘子', 'orange', 'farthest', 'zone_blue', 1),
    ('把红色方块放到蓝色区域', 'red', 'any', 'zone_blue', 1),
    ('把蓝色的积木放进红色框', 'blue', 'any', 'zone_red', 1),
    ('把最前面的苹果模型放到黄色托盘', 'apple', 'front', 'zone_yellow', 1),
    ('把最后面的红萝卜放到左边的盘子', 'carrot', 'back', 'zone_green', 1),
    ('把中间的香蕉放到红色托盘', 'banana', 'middle', 'zone_red', 1),
    ('把三个苹果都放到蓝色托盘', 'apple', 'any', 'zone_blue', 3),  # 都 stresses the number
    ('把所有的三个苹果放到蓝色托盘', None, None, None, None),      # conflicting counts
    ('Put the leftmost apple into the blue tray.', 'apple', 'leftmost', 'zone_blue', 1),
    ('move all bananas to the yellow bin', 'banana', 'any', 'zone_yellow', COUNT_ALL),
]


@pytest.mark.parametrize('text,label,selector,zone,count', OK_CASES)
def test_rule_parser(cfg, zones, text, label, selector, zone, count):
    s = parse_instruction(text, cfg, LABELS, zones)
    if label is None:
        assert not s.ok and s.reason
        return
    assert s.ok, s.reason
    assert (s.label, s.selector, s.zone, s.count) == (label, selector, zone, count)
    assert s.source == 'rules' and s.raw == text


FAIL_CASES = [
    ('把西瓜放到蓝色托盘', 'watermelon'),                 # known fruit, not a task label
    ('把苹果放到紫色托盘', 'no zone matches purple'),      # no such zone
    ('把苹果放到托盘', 'no known zone'),                  # zone not named
    ('把苹果和香蕉放到蓝色托盘', 'several objects'),       # two labels
    ('把最左边最大的苹果放到蓝色托盘', 'conflicting selectors'),
    ('最左边的苹果', 'no destination'),
    ('把东西放到蓝色托盘', 'no known object'),
    ('', 'empty'),
    ('把黄瓜放到红色托盘', 'cucumber'),                  # 黄 must not become a yellow block
    ('把苹果放到红色或蓝色托盘', 'ambiguous'),
]


@pytest.mark.parametrize('text,why', FAIL_CASES)
def test_rule_parser_failures_never_guess(cfg, zones, text, why):
    s = parse_instruction(text, cfg, LABELS, zones)
    assert not s.ok
    assert why in s.reason, s.reason


def test_zone_names_list_and_default_labels(cfg):
    s = parse_instruction('把苹果放到zone_red', cfg, None, ['zone_red', 'zone_blue'])
    assert s.ok and s.label == 'apple' and s.zone == 'zone_red'


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _llm_reply(obj):
    content = obj if isinstance(obj, str) else json.dumps(obj)
    return _Resp(json.dumps({'choices': [{'message': {'content': content}}]}).encode())


def test_llm_backend_used_and_validated(cfg, zones, monkeypatch):
    cfg.set('tasks.L3.llm.enabled', True)
    cfg.set('tasks.L3.llm.timeout_s', 3)
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen['timeout'] = timeout
        seen['body'] = json.loads(req.data.decode())
        seen['url'] = req.full_url
        return _llm_reply('```json\n{"label": "apple", "selector": "leftmost", '
                          '"zone": "zone_blue", "count": 1}\n```')
    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    s = parse_instruction('把最左边的苹果放到蓝色托盘', cfg, LABELS, zones)
    assert s.ok and s.source == 'llm' and s.zone == 'zone_blue' and s.reason == ''
    assert seen['timeout'] == 3 and seen['url'].endswith('/v1/chat/completions')
    assert 'zone_blue' in seen['body']['messages'][0]['content']


@pytest.mark.parametrize('reply', [
    {'label': 'watermelon', 'selector': 'any', 'zone': 'zone_blue', 'count': 1},
    {'label': 'apple', 'selector': 'leftmost', 'zone': 'zone_purple', 'count': 1},
    {'label': 'apple', 'selector': 'weird', 'zone': 'zone_blue', 'count': 1},
    {'label': 'apple', 'selector': 'any', 'zone': 'zone_blue', 'count': 99},
    'sorry, I cannot help',
])
def test_llm_invalid_reply_falls_back_to_rules(cfg, zones, monkeypatch, reply):
    cfg.set('tasks.L3.llm.enabled', True)
    monkeypatch.setattr(urllib.request, 'urlopen', lambda req, timeout=None: _llm_reply(reply))
    s = parse_instruction('把最左边的苹果放到蓝色托盘', cfg, LABELS, zones)
    assert s.ok and s.source == 'rules' and 'llm fallback' in s.reason
    assert (s.label, s.selector, s.zone) == ('apple', 'leftmost', 'zone_blue')


def test_llm_network_error_falls_back_and_unknown_stays_unknown(cfg, zones, monkeypatch):
    cfg.set('tasks.L3.llm.enabled', True)

    def down(req, timeout=None):
        raise urllib.error.URLError('connection refused')
    monkeypatch.setattr(urllib.request, 'urlopen', down)
    s = parse_instruction('把西瓜放到蓝色托盘', cfg, LABELS, zones)
    assert not s.ok and 'watermelon' in s.reason and 'connection refused' in s.reason


def test_llm_disagreement_is_reported(cfg, zones, monkeypatch):
    cfg.set('tasks.L3.llm.enabled', True)
    reply = {'label': 'apple', 'selector': 'rightmost', 'zone': 'zone_blue', 'count': 'all'}
    monkeypatch.setattr(urllib.request, 'urlopen', lambda req, timeout=None: _llm_reply(reply))
    s = parse_instruction('把最左边的苹果放到蓝色托盘', cfg, LABELS, zones)
    assert s.ok and s.source == 'llm' and s.count == COUNT_ALL
    assert 'rules disagree' in s.reason
