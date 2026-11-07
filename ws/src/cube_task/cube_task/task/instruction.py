"""
Parse L3 natural-language pick-and-place instructions (Chinese / English).

Example: "把最左边的苹果放到蓝色托盘" -> InstructionSpec(label='apple',
selector='leftmost', zone='zone_blue', count=1).

Rules first drop a list number ("1.", "2、", "第三条："), split the text at the
destination verb ("放到", "into", ...; with several Chinese verbs the first one
whose left part names an object, so "把放在左边的苹果放到…" works), then look for
the object label and selector / count in the first part and the zone in the
second part. A label or zone that is unknown or ambiguous gives ok=False with a
reason; the parser never guesses. One spec holds one label: several objects
("香蕉和橙子") and category words ("所有水果") are refused. Constructs a spec
cannot express are refused with a reason starting with NOT_SUPPORTED, and the
LLM is not asked about them: ordinals ("左边第二个"; "左边第一个" is the
leftmost) and objects described by a zone ("托盘里的苹果") or an unknown place
("放在角落的苹果"). With tasks.L3.llm.enabled an OpenAI-compatible chat endpoint
is asked first (strict JSON, validated against the known labels and zones); any
error falls back to the rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
import unicodedata
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

_LOG = logging.getLogger('cube_task.task.instruction')

SELECTORS = ('any', 'leftmost', 'rightmost', 'front', 'back', 'nearest', 'farthest',
             'largest', 'smallest', 'middle')
#: InstructionSpec.count value meaning "all matching objects".
COUNT_ALL = -1
MAX_COUNT = 6


@dataclass
class InstructionSpec:
    """Parsed instruction. count: number of objects, COUNT_ALL (-1) for all."""

    label: str = ''
    selector: str = 'any'
    zone: str = ''
    count: int = 1
    raw: str = ''
    source: str = 'rules'          # 'rules' | 'llm'
    ok: bool = False
    reason: str = ''

    def summary(self) -> dict:
        """Return a JSON-friendly dict."""
        return {'label': self.label, 'selector': self.selector, 'zone': self.zone,
                'count': self.count, 'raw': self.raw, 'source': self.source,
                'ok': self.ok, 'reason': self.reason}


BUILTIN_LABEL_ALIASES: Dict[str, List[str]] = {
    'apple': ['苹果'], 'banana': ['香蕉'], 'orange': ['橙子', '橘子', '桔子'],
    'lemon': ['柠檬'], 'grape': ['葡萄'], 'carrot': ['胡萝卜'], 'eggplant': ['茄子'],
    'cucumber': ['黄瓜'], 'corn': ['玉米'], 'tomato': ['西红柿', '番茄'],
    'green_pepper': ['青椒'], 'pear': ['梨'], 'peach': ['桃子'], 'strawberry': ['草莓'],
    'watermelon': ['西瓜'], 'potato': ['土豆', '马铃薯'], 'mango': ['芒果'],
    'kiwi': ['猕猴桃'], 'broccoli': ['西兰花'], 'onion': ['洋葱'], 'pumpkin': ['南瓜'],
    'garlic': ['大蒜'], 'pineapple': ['菠萝'], 'cabbage': ['白菜'],
}
#: English attribute word -> Chinese words, used for color blocks and zone matching.
ATTRIBUTE_WORDS: Dict[str, List[str]] = {
    'red': ['红'], 'green': ['绿'], 'blue': ['蓝'], 'yellow': ['黄'], 'orange': ['橙'],
    'purple': ['紫'], 'white': ['白'], 'black': ['黑'], 'fruit': ['水果'],
    'vegetable': ['蔬菜'],
}
COLOR_NAMES = ('red', 'green', 'blue', 'yellow', 'orange', 'purple', 'white', 'black')
BLOCK_WORDS_ZH = '方块|积木|立方体|木块|块'
BLOCK_WORDS_EN = 'blocks?|cubes?'

DEST_MARKERS_ZH = ['放置到', '放置在', '放置于', '投放到', '移动到', '移动至', '分拣到', '归类到',
                   '转移到', '转移至', '放回到', '放到', '放入', '放进', '放在', '放至', '放于',
                   '放回', '移到', '移至', '搬到', '搬进', '搬至', '送到', '送入', '送至', '拿到',
                   '拿至', '扔到', '扔进', '丢到', '丢进', '投到', '投入', '投进', '装进', '装入',
                   '装到', '挪到']
DEST_MARKERS_EN = [' into ', ' onto ', ' to ', ' in ', ' on ']

SELECTOR_WORDS: List[Tuple[str, List[str]]] = [
    ('leftmost', ['最左', '左边', '左侧', '左面', 'leftmost', 'left-most', 'left most',
                  'far left']),
    ('rightmost', ['最右', '右边', '右侧', '右面', 'rightmost', 'right-most', 'right most',
                   'far right']),
    ('front', ['最前', '前面', '前边', '前方', 'frontmost', 'front']),
    ('back', ['最后(?=[面边方侧的])', '最里', '后面', '后边', '后方', 'rearmost', 'backmost']),
    ('nearest', ['最近', 'nearest', 'closest']),
    ('farthest', ['最远', 'farthest', 'furthest']),
    ('largest', ['最大', 'largest', 'biggest']),
    ('smallest', ['最小', 'smallest']),
    ('middle', ['中间', '中央', 'middle', 'center', 'centre']),
    ('any', ['任意', '随便', '任一', '任何', 'any']),
]
COUNT_ALL_WORDS = ['所有', '全部', '每个', '每一个', '都', 'all', 'every', 'each']
ZH_NUMBERS = {'一': 1, '两': 2, '二': 2, '俩': 2, '三': 3, '四': 4, '五': 5, '六': 6}
EN_NUMBERS = {'one': 1, 'two': 2, 'both': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}

#: Reason prefix of instructions an InstructionSpec cannot express (the LLM is not asked).
NOT_SUPPORTED = 'not supported: '
_ZH_NUM = r'(?:\d+|[一二两三四五六七八九十]+)'
#: List number in front of an instruction line (after NFKC): "1." "2、" "(3)" "第四条：" "step 5".
_LIST_PREFIX_RE = re.compile(
    r'^\s*(?:\(\s*(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*\)'
    r'|(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*[.、:)](?!\d)'
    r'|第\s*(?:\d{1,2}|[一二三四五六七八九十]{1,3})\s*[条步项]\s*[.、:,]?'
    r'|\d{1,2}\s*(?=[把将请帮拿])'
    r'|step\s*\d{1,2}\s*[.:)]?)\s*', re.IGNORECASE)
#: "第二个" / "倒数" / "the second" / "2nd" / "2号": the N-th object. Group 1 = the Chinese N.
_ORDINAL_RE = re.compile(
    r'第\s*(%s)|倒数|编号|%s\s*号'
    r'|(?<![a-z0-9])(?:second|third|fourth|fifth|sixth|\d+(?:st|nd|rd|th))(?![a-z])'
    r'|(?<![a-z])the (?:first|last)(?![a-z])' % (_ZH_NUM, _ZH_NUM))
#: Selectors that make "第一个" unambiguous ("左边第一个" = leftmost).
_FIRST_OK = ('leftmost', 'rightmost', 'front', 'back', 'nearest', 'farthest')
#: Zone / container words: an object described by one ("托盘里的苹果") is refused.
_CONTAINER_RE = re.compile(
    r'[盘区盒框篮筐箱桶碗]|容器|(?<![a-z])(?:trays?|bins?|box(?:es)?|baskets?|plates?|zones?'
    r'|areas?|containers?|bowls?)(?![a-z])')
#: Words allowed besides selector words in a "放在…的" clause of the object part.
_CLAUSE_FILLER_RE = re.compile(r'桌子上|桌面上|台面上|桌子|桌面|桌上|台面|台上|[的最靠边面侧\s]')
_CATEGORY_RE = re.compile(
    r'水果|蔬菜|果蔬|蔬果|(?<![a-z])(?:fruits?|vegetables?|veggies?)(?![a-z])')


class _ParseError(ValueError):
    pass


class _NoLabel(_ParseError):
    """The text names no object at all (used to choose the destination verb)."""


class _Unsupported(_ParseError):
    """A construct an InstructionSpec cannot express; reason starts with NOT_SUPPORTED."""

    def __init__(self, text: str):
        super().__init__(NOT_SUPPORTED + text)


def _norm(text: str) -> str:
    s = unicodedata.normalize('NFKC', str(text)).lower().strip()
    s = re.sub(r'[。！!？?，,；;、"“”\'‘’]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _strip_list_prefix(text: str) -> str:
    """Drop a list number ("2. 把香蕉…" must not mean two bananas)."""
    s = unicodedata.normalize('NFKC', str(text)).strip()
    return _LIST_PREFIX_RE.sub('', s, count=1)


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def _pattern(word: str, regex: bool = False) -> str:
    """Regex for one alias; ASCII words need word boundaries and may be plural."""
    body = word if regex else re.escape(word)
    if _is_ascii(word):
        return r'(?<![a-z0-9_])' + body + r'(?:s|es)?(?![a-z0-9_])'
    return body


def _match_longest(text: str, table: Sequence[Tuple[str, object]],
                   regex: bool = False) -> List[Tuple[object, int, int]]:
    """Non-overlapping matches of (alias, key) pairs, longest alias first."""
    taken = [False] * len(text)
    out: List[Tuple[object, int, int]] = []
    for alias, key in sorted(table, key=lambda t: -len(t[0])):
        if not alias:
            continue
        for m in re.finditer(_pattern(alias, regex), text):
            a, b = m.start(), m.end()
            if b <= a or any(taken[a:b]):
                continue
            for i in range(a, b):
                taken[i] = True
            out.append((key, a, b))
    return sorted(out, key=lambda t: t[1])


def _split_destination(text: str, names_object=None) -> Tuple[str, str, Optional[int]]:
    """
    Split at the destination verb: (object part, destination part, clause start).

    With several Chinese verbs, the first one whose left part names an object
    (names_object(left) is true) is used, else the last one. Clause start is
    the end of the first verb skipped ("把放在左边的苹果放到…": the "左边的"
    clause), or None when the first verb is used.
    """
    hits = sorted(((mm.start(), mm.end()) for m in DEST_MARKERS_ZH
                   for mm in re.finditer(re.escape(m), text)), key=lambda h: (h[0], -h[1]))
    verbs: List[Tuple[int, int]] = []
    for a, b in hits:
        if not verbs or a >= verbs[-1][1]:      # skip "放到" inside "投放到"
            verbs.append((a, b))
    if verbs:
        pick = verbs[-1]
        for a, b in verbs:
            if names_object is None or names_object(text[:a]):
                pick = (a, b)
                break
        clause = None if pick == verbs[0] else verbs[0][1]
        return text[:pick[0]], text[pick[1]:], clause
    padded = ' %s ' % text
    for m in DEST_MARKERS_EN:
        i = padded.find(m)
        if i >= 0:
            return padded[:i].strip(), padded[i + len(m):].strip(), None
    raise _ParseError('no destination verb (e.g. 放到 / 放进 / into) found')


def _user_aliases(cfg, key: str) -> Dict[str, List[str]]:
    raw = cfg.get(key, {}) or {}
    if not isinstance(raw, dict):
        raise _ParseError('%s must be a mapping' % key)
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        vals = [v] if isinstance(v, str) else list(v or [])
        out[str(k)] = [_norm(x) for x in vals if str(x).strip()]
    return out


def _label_hits(obj: str, cfg, labels: Sequence[str]) -> List[Tuple[object, int, int]]:
    """Alias matches in obj: ((('known' | 'foreign'), label), start, end)."""
    user = _user_aliases(cfg, 'tasks.L3.label_aliases')
    table: List[Tuple[str, Tuple[str, str]]] = []
    for lab in labels:
        names = {_norm(lab), _norm(lab.replace('_', ' '))}
        names.update(BUILTIN_LABEL_ALIASES.get(lab, []))
        names.update(user.get(lab, []))
        table.extend((n, ('known', lab)) for n in names)
    for lab, names in list(BUILTIN_LABEL_ALIASES.items()) + list(user.items()):
        if lab not in labels:
            for n in set(names) | {_norm(lab.replace('_', ' '))}:
                table.append((n, ('foreign', lab)))
    return _match_longest(obj, table)


def _resolve_label(obj: str, cfg, labels: Sequence[str]) -> str:
    hits = _label_hits(obj, cfg, labels)
    known = sorted({k[1] for k, _a, _b in hits if k[0] == 'known'})
    foreign = sorted({k[1] for k, _a, _b in hits if k[0] == 'foreign'})
    if foreign:
        raise _ParseError('object %s is not a label of this task %s' % (
            '/'.join(foreign), list(labels)))
    if len(known) > 1:
        raise _ParseError('several objects named (%s): one object label per instruction is '
                          'supported' % ', '.join(known))
    if known:
        return known[0]
    m = _CATEGORY_RE.search(obj)
    if m:
        raise _ParseError('category word "%s" names no single label: name the object '
                          '(labels: %s)' % (m.group(0), list(labels)))
    return _resolve_color_label(obj, labels)


def _names_object(text: str, cfg, labels: Sequence[str]) -> bool:
    """Return True when text names an object (even a wrong or ambiguous one)."""
    try:
        _resolve_label(text, cfg, labels)
    except _NoLabel:
        return False
    except _ParseError:
        return True
    return True


def _color_to_label(color: str, labels: Sequence[str]) -> Optional[str]:
    if color in labels:
        return color
    cands = [lab for lab in labels if color in lab.lower().split('_')]
    return cands[0] if len(cands) == 1 else None


def _resolve_color_label(obj: str, labels: Sequence[str]) -> str:
    """'红色方块' / 'red block' / '红色的' -> the label of that color."""
    zh_to_en = {zh: en for en in COLOR_NAMES for zh in ATTRIBUTE_WORDS[en]}
    zh_cls = '|'.join(zh_to_en)
    colors = set()
    for m in re.finditer(r'(%s)色?的?(?:%s)' % (zh_cls, BLOCK_WORDS_ZH), obj):
        colors.add(zh_to_en[m.group(1)])
    for m in re.finditer(r'(%s)色' % zh_cls, obj):
        colors.add(zh_to_en[m.group(1)])
    for m in re.finditer(r'(?<![a-z])(%s)(?![a-z])' % '|'.join(COLOR_NAMES), obj):
        colors.add(m.group(1))
    if not colors:
        raise _NoLabel('no known object label in "%s" (labels: %s)' % (obj, list(labels)))
    if len(colors) > 1:
        raise _ParseError('several colors named: %s' % ', '.join(sorted(colors)))
    color = colors.pop()
    lab = _color_to_label(color, labels)
    if lab is None:
        raise _ParseError('no label of color %s in this task %s' % (color, list(labels)))
    return lab


def _resolve_selector(obj: str) -> str:
    table = [(w, sel) for sel, words in SELECTOR_WORDS for w in words]
    found = sorted({k for k, _a, _b in _match_longest(obj, table, regex=True)})
    if len(found) > 1:
        raise _ParseError('conflicting selectors: %s' % ', '.join(found))
    return found[0] if found else 'any'


def _check_ordinal(obj: str, numbered: bool = True) -> None:
    """Refuse "第二个" / "倒数" / "2号" (numbered=False: not 号): only extremes are picked."""
    try:
        selector = _resolve_selector(obj)
    except _ParseError:
        selector = 'any'
    for m in _ORDINAL_RE.finditer(obj):
        if m.group(1) in ('一', '1') and selector in _FIRST_OK:
            continue                        # "左边第一个" = the leftmost one
        if not numbered and m.group(0).endswith('号'):
            continue                        # whole text: may be a zone ("1号区")
        raise _Unsupported(
            'ordinal "%s" (the N-th object); only the extreme one can be picked: '
            '最左边 / 最右边 / 最前面 / 最后面 / 最近 / 最远 / 最大 / 最小 / 中间' % m.group(0))


def _check_location(obj: str, clause_at: Optional[int], cfg, labels: Sequence[str]) -> None:
    """Refuse objects described by a zone ("托盘里的") or a place no selector covers."""
    masked = obj
    for _k, a, b in _label_hits(obj, cfg, labels):     # an alias may contain 盒 / 盘 ...
        masked = masked[:a] + ' ' * (b - a) + masked[b:]
    m = _CONTAINER_RE.search(masked)
    if m:
        raise _Unsupported('object described by a zone or container ("%s" in "%s")'
                           % (m.group(0), obj.strip()))
    if clause_at is None:
        return
    tail = obj[clause_at:]
    j = tail.rfind('的')                    # "放在桌子左边的苹果": clause "桌子左边"
    rest = tail[:j] if j >= 0 else tail
    table = [(w, sel) for sel, words in SELECTOR_WORDS for w in words]
    for _k, a, b in _match_longest(rest, table, regex=True):
        rest = rest[:a] + ' ' * (b - a) + rest[b:]
    rest = _CLAUSE_FILLER_RE.sub('', rest)
    if j < 0 or rest:
        raise _Unsupported('object described by its place "%s"; only 左边 / 右边 / 前面 / '
                           '后面 / 中间 / 最近 ... are understood' % obj[clause_at:].strip())


def _resolve_count(obj: str) -> int:
    all_hits = _match_longest(obj, [(w, 'all') for w in COUNT_ALL_WORDS])
    rest = obj
    for _k, a, b in sorted(all_hits, key=lambda t: -t[1]):
        rest = rest[:a] + ' ' * (b - a) + rest[b:]
    nums = set()
    for m in re.finditer(r'(?<![第\d])([一两二俩三四五六]|\d+)\s*(?:个|只|根|块|颗|条|枚)|(俩)',
                         rest):
        g = m.group(1) or m.group(2)
        nums.add(int(g) if g.isdigit() else ZH_NUMBERS[g])
    for m in re.finditer(r'(?<![a-z0-9])(%s|\d+)(?![a-z0-9])' % '|'.join(EN_NUMBERS), rest):
        g = m.group(1)
        nums.add(int(g) if g.isdigit() else EN_NUMBERS[g])
    if all_hits and nums and len(nums) == 1 and all(obj[a:b] == '都' for _k, a, b in all_hits):
        all_hits = []                       # "把两个苹果都…": 都 only stresses the number
    if all_hits and nums:
        raise _ParseError('conflicting counts: all and %s' % sorted(nums))
    if all_hits:
        return COUNT_ALL
    if len(nums) > 1:
        raise _ParseError('conflicting counts: %s' % sorted(nums))
    n = nums.pop() if nums else 1
    if not 1 <= n <= MAX_COUNT:
        raise _ParseError('count %d out of range 1..%d' % (n, MAX_COUNT))
    return n


def _zone_text(name: str, zone) -> str:
    label = '' if zone is None else str(getattr(zone, 'label', '') or '')
    return _norm('%s %s' % (name.replace('_', ' '), label))


def _resolve_zone(dest: str, cfg, zone_map: Dict[str, object]) -> str:
    if not zone_map:
        raise _ParseError('no zones configured')
    user = _user_aliases(cfg, 'tasks.L3.zone_aliases')
    table: List[Tuple[str, str]] = []
    for name, zone in zone_map.items():
        names = {_norm(name), _norm(name.replace('_', ' '))}
        label = '' if zone is None else _norm(getattr(zone, 'label', '') or '')
        if label:
            names.add(label)
        names.update(user.get(name, []))
        table.extend((n, name) for n in names)
    hits = sorted({k for k, _a, _b in _match_longest(dest, table)})
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise _ParseError('several zones named: %s' % ', '.join(hits))
    attrs = []
    for en, zh_words in ATTRIBUTE_WORDS.items():
        if any(w in dest for w in zh_words) or re.search(_pattern(en), dest):
            attrs.append(en)
    if not attrs:
        raise _ParseError('no known zone in "%s" (zones: %s)' % (dest, sorted(zone_map)))
    colors = [a for a in attrs if a in COLOR_NAMES]
    if len(colors) > 1:
        raise _ParseError('zone is ambiguous: several colors named (%s)' % ', '.join(colors))

    def has(text: str, en: str) -> bool:
        return (re.search(_pattern(en), text) is not None
                or any(w in text for w in ATTRIBUTE_WORDS[en]))
    cands = sorted(n for n, z in zone_map.items()
                   if all(has(_zone_text(n, z), a) for a in attrs))
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise _ParseError('no zone matches %s (zones: %s)' % ('+'.join(attrs), sorted(zone_map)))
    raise _ParseError('zone %s is ambiguous: %s' % ('+'.join(attrs), ', '.join(cands)))


def _zone_map(cfg, zones) -> Dict[str, object]:
    if zones is None:
        from ..core.zones import parse_zones
        return dict(parse_zones(cfg))
    if isinstance(zones, dict):
        return dict(zones)
    out: Dict[str, object] = {}
    for z in zones:
        if isinstance(z, str):
            out[z] = None
        else:
            out[str(z.name)] = z
    return out


def parse_rules(text: str, cfg, labels: Sequence[str], zones) -> InstructionSpec:
    """Rule-based parser; see the module docstring."""
    spec = InstructionSpec(raw=str(text), source='rules')
    labels = list(labels)
    try:
        s = _norm(_strip_list_prefix(text))
        if not s:
            raise _ParseError('empty instruction')
        try:
            obj, dest, clause_at = _split_destination(
                s, lambda left: _names_object(left, cfg, labels))
        except _ParseError:
            _check_ordinal(s, numbered=False)   # no verb: never hand an ordinal to the LLM
            raise
        _check_ordinal(obj)
        _check_location(obj, clause_at, cfg, labels)
        spec.label = _resolve_label(obj, cfg, labels)
        spec.selector = _resolve_selector(obj)
        spec.count = _resolve_count(obj)
        spec.zone = _resolve_zone(dest, cfg, _zone_map(cfg, zones))
        spec.ok = True
    except _ParseError as e:
        spec.ok = False
        spec.reason = str(e)
    return spec


def _extract_json(content: str) -> dict:
    s = str(content).strip()
    s = re.sub(r'^```(?:json)?\s*|\s*```$', '', s)
    i, j = s.find('{'), s.rfind('}')
    if i < 0 or j <= i:
        raise ValueError('no JSON object in LLM reply: %r' % content[:200])
    obj = json.loads(s[i:j + 1])
    if not isinstance(obj, dict):
        raise ValueError('LLM reply is not a JSON object')
    return obj


def _validate_llm(obj: dict, raw: str, labels: Sequence[str],
                  zone_names: Sequence[str]) -> InstructionSpec:
    label = obj.get('label')
    zone = obj.get('zone')
    selector = obj.get('selector') or 'any'
    count = obj.get('count', 1)
    if label not in labels:
        raise ValueError('LLM label %r not in %s' % (label, list(labels)))
    if zone not in zone_names:
        raise ValueError('LLM zone %r not in %s' % (zone, list(zone_names)))
    if selector not in SELECTORS:
        raise ValueError('LLM selector %r not in %s' % (selector, SELECTORS))
    if isinstance(count, str) and count.strip().lower() == 'all':
        count = COUNT_ALL
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError('LLM count %r is not an integer or "all"' % (count,))
    if count != COUNT_ALL and not 1 <= count <= MAX_COUNT:
        raise ValueError('LLM count %r out of range' % (count,))
    return InstructionSpec(label=str(label), selector=str(selector), zone=str(zone),
                           count=int(count), raw=raw, source='llm', ok=True)


def parse_llm(text: str, cfg, labels: Sequence[str], zone_names: Sequence[str]) -> InstructionSpec:
    """
    Ask an OpenAI-compatible chat endpoint; raise on any error or invalid reply.

    Config: tasks.L3.llm.url, .model, .timeout_s, optional .api_key_env (name of
    the environment variable holding a bearer token).
    """
    url = str(cfg.get('tasks.L3.llm.url', '') or '')
    if not url:
        raise ValueError('tasks.L3.llm.url is empty')
    timeout = float(cfg.get('tasks.L3.llm.timeout_s', 20))
    left = cfg.get('tasks.L3.left_axis', [0.0, 1.0])
    system = (
        'You convert one robot pick-and-place instruction into JSON. Reply with exactly '
        'one JSON object and nothing else: {"label": L, "selector": S, "zone": Z, '
        '"count": N}. L must be one of %s, Z one of %s, S one of %s, N an integer 1-%d '
        'or "all". Use null for label or zone if the instruction names something that is '
        'not in the lists. Left means base direction %s. "Front" means farther from the '
        'robot base.' % (json.dumps(list(labels), ensure_ascii=False),
                         json.dumps(list(zone_names), ensure_ascii=False),
                         json.dumps(list(SELECTORS)), MAX_COUNT, list(left)))
    body = {'model': str(cfg.get('tasks.L3.llm.model', '') or ''),
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': str(text)}],
            'temperature': 0, 'max_tokens': 200, 'stream': False}
    headers = {'Content-Type': 'application/json'}
    key_env = cfg.get('tasks.L3.llm.api_key_env', '') or ''
    if key_env and os.environ.get(str(key_env)):
        headers['Authorization'] = 'Bearer ' + os.environ[str(key_env)]
    req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'),
                                 headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    content = payload['choices'][0]['message']['content']
    return _validate_llm(_extract_json(content), str(text), labels, zone_names)


def parse_instruction(text: str, cfg, labels: Optional[Sequence[str]] = None,
                      zones=None) -> InstructionSpec:
    """
    Parse one instruction.

    labels: known object labels (default tasks.L3.labels). zones: dict name ->
    core Zone, a list of Zone objects or names, or None to parse cfg zones.
    """
    labels = list(labels if labels is not None else (cfg.get('tasks.L3.labels', []) or []))
    zmap = _zone_map(cfg, zones)
    rules = parse_rules('' if text is None else text, cfg, labels, zmap)
    if not bool(cfg.get('tasks.L3.llm.enabled', False)) or not rules.raw.strip():
        return rules
    if rules.reason.startswith(NOT_SUPPORTED):
        return rules                        # a spec cannot express it: an LLM answer is wrong
    try:
        spec = parse_llm(_strip_list_prefix(rules.raw), cfg, labels, sorted(zmap))
        spec.raw = rules.raw
    except Exception as e:  # noqa: B902 - any LLM / network / JSON problem -> rules
        _LOG.warning('LLM parse failed (%s); using the rule parser', e)
        rules.reason = ('%s; ' % rules.reason if rules.reason else '') + 'llm fallback: %s' % e
        return rules
    if rules.ok and (rules.label, rules.selector, rules.zone, rules.count) != (
            spec.label, spec.selector, spec.zone, spec.count):
        _LOG.warning('LLM and rules disagree on %r: llm=%s rules=%s', rules.raw,
                     spec.summary(), rules.summary())
        spec.reason = 'rules disagree: label=%s selector=%s zone=%s count=%d' % (
            rules.label, rules.selector, rules.zone, rules.count)
    return spec
