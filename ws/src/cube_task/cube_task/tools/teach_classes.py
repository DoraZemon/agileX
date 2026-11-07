"""
Teach the KNN instance classifier from the real camera (L2; also L1 / L3).

Live loop: each view is one capture (median depth of camera.frames_to_average
frames) segmented with the classifier disabled. The valid instances are
numbered on the overlay (headless: preview image + table). Label them, move /
rotate the objects, capture the next view, repeat. The arm is never moved
(eye_in_hand: its pose is read at every view; drag-teach mode is fine).

Commands (headless: type a line; window: press the key, Enter = type a line)
    all <label>              label every valid instance of this view
    <n> <label>[; <n> ...]   label instances, e.g. '0 apple; 1,2 banana'
                             ('0 lemon; all apple': #0 lemon, the rest apple)
    c new view (move / rotate the objects first)   u undo the last labeling
    e leave-one-out evaluation   w write the model   s status   h help
    g sim only: label this view from the ground truth
    q quit (asks to write unsaved samples)

Labels are checked against tasks.<task>.labels: a label outside the list (or,
when the list is empty, every new label) needs a confirmation. Aim for at
least 8 samples per class from several positions and orientations.

--from-dir [LABEL=]DIR ... learns offline from capture sessions (one class
per session: LABEL=, else the meta extra 'session', else the directory name;
a date prefix and a trailing _N are dropped unless the full name is a task
label) with the configured extrinsics, without hardware. --mode sim first
learns from --sim-scenes simulated scenes with ground-truth labels, then runs
the live loop on the simulated rig ('c' rearranges the objects).

Output: perception.classifier.knn.model_file (sim: <logging.dir>/sim_site;
--out overrides), written atomically after a timestamped backup; --append
keeps the samples of the existing model. <model>_report.json next to it holds
samples per class, accuracy, confusion, feature names, created and source.

Example commands
    teach_classes --task L2
    teach_classes --task L2 --headless --append
    teach_classes --task L2 --from-dir ~/vision_data/20261112_apple_1 banana=/data/b1
    teach_classes --mode sim --task L2 --headless
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import difflib
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .common import add_common_args, load_cfg_for, open_session, setup_logging
from .onsite import (add_onsite_args, assign_command, backup_file, camera_pose_for_frame,
                     capture_view, compose_view, connect_arm_readonly, flange_to_tcp, fmt_mm,
                     format_table, go_observe, ground_truth_label, load_extrinsics_or_none,
                     make_ui, MotionGuard, next_command, operator_action, output_path,
                     overlay_file, read_pose, recorded_flange_pose, sim_relayout, teach_hint,
                     text_panel, use_hint, wall_stamp)
from ..core.logger import atomic_write_text
from ..perception.classify_knn import KnnClassifier
from ..perception.factory import classifier_backend, task_labels
from ..perception.features import FEATURE_NAMES, feature_vector

_LOG = logging.getLogger('cube_task.tools.teach_classes')

MODEL_KEY = 'perception.classifier.knn.model_file'
DEFAULT_MODEL = 'models/knn_model.npz'
N_FEATURES = len(FEATURE_NAMES)
MIN_SAMPLES = 8            # warn below this many samples per class
CONFUSE_RATE = 0.10        # warn when leave-one-out mixes two classes up this often
GENERIC_SESSIONS = ('', 'capture', 'view', 'teach_classes', 'session', 'unknown')
SIM_FIRST_SEED = 100       # first ground-truth scene (as sim/teach.train_knn_from_sim)
SIM_RELAYOUT_SEED = 500    # sim 'c': layout of scenario seed --seed + this + k

LABEL_HELP = "'all <label>' or '<n> <label>[; <n> <label>]' (e.g. '0 apple; 1,2 banana')"
PROMPT = "teach ['<n> <label>' | 'all <label>' | c u e w s h q]"
SIM_PROMPT = "teach ['<n> <label>' | 'all <label>' | g c u e w s h q]"
HEADLESS_HELP = ('type %s, or c new view, u undo, e evaluate, w write, s status, h help, '
                 'q quit' % LABEL_HELP)
WINDOW_HELP = ('keys in the window: c new view, u undo, e evaluate, w write, s status, h help, '
               'q quit; Enter = type a labeling command (%s) in the terminal' % LABEL_HELP)
SIM_HELP = ('sim: g labels this view from the ground truth; c lets the simulated operator '
            'rearrange the objects')

COMMANDS = {'q': 'quit', 'quit': 'quit', 'exit': 'quit', 'c': 'capture', 'capture': 'capture',
            'u': 'undo', 'undo': 'undo', 'e': 'evaluate', 'eval': 'evaluate',
            'evaluate': 'evaluate', 'w': 'write', 'write': 'write', 'save': 'write',
            's': 'status', 'status': 'status', 'h': 'help', 'help': 'help', '?': 'help',
            'g': 'ground_truth', 'gt': 'ground_truth'}


# ------------------------------------------------------------------ label commands
def command_kind(text: Optional[str]) -> str:
    """Return the command of an operator line: a COMMANDS value, 'label' or 'none' (empty)."""
    s = str(text or '').strip()
    if not s:
        return 'none'
    return COMMANDS.get(s.lower(), 'label')


def parse_label_command(text: str, n_instances: int
                        ) -> Tuple[Optional[List[Tuple[Optional[List[int]], str]]], str]:
    """
    Parse 'all <label>' / '<n>[,<m>...] <label>' parts separated by ';'.

    Returns ([(instance numbers, or None for 'all', label), ...], '') or
    (None, error message). An instance may be named only once per command
    and 'all' may appear once.
    """
    parts = [p.strip() for p in str(text or '').split(';') if p.strip()]
    if not parts:
        return None, 'empty labeling command; expected %s' % LABEL_HELP
    if n_instances <= 0:
        return None, 'no valid instances in this view (c = capture a new view)'
    out: List[Tuple[Optional[List[int]], str]] = []
    named: Set[int] = set()
    n_all = 0
    for part in parts:
        idx, label, err = assign_command(part, n_instances)
        if idx is None:
            hint = ' (a label has no spaces, e.g. green_pepper)' if len(part.split()) > 2 else ''
            return None, '%r: %s%s' % (part, err, hint)
        if part.split()[0].lower() == 'all':
            n_all += 1
            if n_all > 1:
                return None, "'all' may appear only once per command"
            out.append((None, label))
            continue
        twice = named.intersection(idx)
        if twice:
            return None, 'instance #%d is named twice' % min(twice)
        named.update(idx)
        out.append((idx, label))
    return out, ''


def plan_labeling(parts: Sequence[Tuple[Optional[List[int]], str]], already: Dict[int, str],
                  usable: Sequence[bool]) -> Tuple[Dict[int, str], List[str], str]:
    """
    Turn parsed label parts into {instance number: label} for one view.

    already: instances labeled earlier in this view; usable: instances whose
    features are complete. Explicit numbers win over 'all', which takes the
    remaining unlabeled instances. Instances with incomplete features are
    skipped with a note. Naming an instance labeled earlier is an error
    (nothing is labeled). Returns (assignment, notes, error).
    """
    assign: Dict[int, str] = {}
    notes: List[str] = []
    named: Set[int] = set()
    for idx, label in parts:
        for i in idx or []:
            if i in already:
                return {}, [], ('instance #%d is already labeled %r in this view (u = undo '
                                'the last labeling)' % (i, already[i]))
            named.add(i)
            if usable[i]:
                assign[i] = label
            else:
                notes.append('#%d skipped: incomplete features (no color pixels or geometry)'
                             % i)
    for idx, label in parts:
        if idx is not None:
            continue
        rest = [i for i in range(len(usable)) if i not in named]
        kept = [i for i in rest if i in already]
        bad = [i for i in rest if i not in already and not usable[i]]
        for i in rest:
            if i not in already and usable[i]:
                assign[i] = label
        if kept:
            notes.append("'all' kept the earlier labels %s"
                         % ', '.join('#%d %s' % (i, already[i]) for i in kept))
        if bad:
            notes.append("'all' skipped %s: incomplete features"
                         % ', '.join('#%d' % i for i in bad))
    if not assign:
        return {}, notes, 'nothing left to label in this view'
    return assign, notes, ''


def check_label(label: str, known: Sequence[str]) -> Tuple[str, str, List[str]]:
    """
    Check a typed label against the task labels.

    Returns (label in the spelling of the known list, status, close matches)
    with status 'ok' (listed, case-insensitive), 'unlisted' (not in a
    non-empty list) or 'unchecked' (the list is empty).
    """
    lab = str(label).strip()
    if not known:
        return lab, 'unchecked', []
    low = {str(k).lower(): str(k) for k in known}
    if lab.lower() in low:
        return low[lab.lower()], 'ok', []
    close = difflib.get_close_matches(lab.lower(), list(low), n=3, cutoff=0.6)
    return lab, 'unlisted', [low[c] for c in close]


def accepted_labels(book: 'SampleBook', known: Sequence[str]) -> Set[str]:
    """
    Return the labels that need no confirmation at the start of a run.

    With an empty task label list, the labels already in the samples (for
    example of an appended model) are not new. With a non-empty list, every
    unlisted label is confirmed once per run, even when the samples have it.
    """
    return set() if known else set(book.counts())


# ------------------------------------------------------------------ sample bookkeeping
def _rows(X: Any, n_features: int = N_FEATURES) -> np.ndarray:
    """Return X as an N x n_features float64 array (an empty input gives 0 rows)."""
    a = np.asarray(X, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, int(n_features)))
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != int(n_features):
        raise ValueError('expected N x %d features, got shape %s' % (n_features, a.shape))
    return a


@dataclasses.dataclass
class Batch:
    """One labeling step, undone as a whole: feature rows X with one label each."""

    text: str                      # what produced it (operator command, directory, ...)
    labels: List[str]
    X: np.ndarray
    view: int = -1                 # live view number; -1 = not from a live view
    indices: List[int] = dataclasses.field(default_factory=list)  # instances of that view
    source: str = 'live'           # live | sim_gt | from_dir


class SampleBook:
    """
    Labeled samples of one teaching run with per-command undo (pure bookkeeping).

    The base samples (an existing model kept with --append) cannot be undone.
    dirty is True when the samples changed since mark_saved().
    """

    def __init__(self, n_features: int = N_FEATURES):
        self.n_features = int(n_features)
        self.base_X = np.zeros((0, self.n_features))
        self.base_y: List[str] = []
        self.batches: List[Batch] = []
        self._rev = 0
        self._saved_rev = 0

    def set_base(self, X: Any, y: Sequence[str]) -> None:
        """Keep the samples of an existing model (--append); not undoable, not dirty."""
        Xb = _rows(X, self.n_features)
        yb = [str(v) for v in y]
        if Xb.shape[0] != len(yb):
            raise ValueError('base samples: %d rows but %d labels' % (Xb.shape[0], len(yb)))
        self.base_X, self.base_y = Xb, yb

    def add(self, batch: Batch) -> Batch:
        """Append a batch (one undo step); raise ValueError for bad rows, labels or a relabel."""
        X = _rows(batch.X, self.n_features)
        labels = [str(v) for v in batch.labels]
        if X.shape[0] == 0 or X.shape[0] != len(labels):
            raise ValueError('a batch needs at least one sample and one label per sample')
        if not np.all(np.isfinite(X)):
            raise ValueError('sample features must be finite')
        if any(not v or v == 'unknown' for v in labels):
            raise ValueError("labels must be non-empty and not 'unknown'")
        if batch.view >= 0:
            if len(batch.indices) != len(labels):
                raise ValueError('a live batch needs one instance number per sample')
            again = set(batch.indices) & set(self.labeled_in_view(batch.view))
            if again:
                raise ValueError('instance #%d of view %d is labeled already'
                                 % (min(again), batch.view))
        b = dataclasses.replace(batch, X=X, labels=labels, indices=list(batch.indices))
        self.batches.append(b)
        self._rev += 1
        return b

    def undo(self) -> Optional[Batch]:
        """Remove and return the last batch (None when there is none)."""
        if not self.batches:
            return None
        self._rev += 1
        return self.batches.pop()

    def labeled_in_view(self, view: int) -> Dict[int, str]:
        """Return {instance number: label} of the instances labeled in live view view."""
        out: Dict[int, str] = {}
        if view < 0:
            return out
        for b in self.batches:
            if b.view == view:
                out.update(zip(b.indices, b.labels))
        return out

    def session_arrays(self) -> Tuple[np.ndarray, List[str]]:
        """Return the samples added in this run (without the base samples)."""
        if not self.batches:
            return np.zeros((0, self.n_features)), []
        return (np.vstack([b.X for b in self.batches]),
                [v for b in self.batches for v in b.labels])

    def arrays(self) -> Tuple[np.ndarray, List[str]]:
        """Return base + session samples without exact duplicates (what gets written)."""
        Xs, ys = self.session_arrays()
        X, y, _dup = merge_samples(self.base_X, self.base_y, Xs, ys, n_features=self.n_features)
        return X, y

    def counts(self) -> Dict[str, int]:
        """Return the samples per class of arrays()."""
        return dict(sorted(Counter(self.arrays()[1]).items()))

    @property
    def n_session(self) -> int:
        """Return the number of samples added in this run."""
        return sum(len(b.labels) for b in self.batches)

    def source_counts(self) -> Dict[str, int]:
        """Return the sample count per source ('appended' = samples of the old model)."""
        out: Dict[str, int] = {}
        if self.base_y:
            out['appended'] = len(self.base_y)
        for b in self.batches:
            out[b.source] = out.get(b.source, 0) + len(b.labels)
        return out

    @property
    def dirty(self) -> bool:
        """Return True when the samples changed since the last mark_saved()."""
        return self._rev != self._saved_rev

    def mark_saved(self) -> None:
        """Record that the current samples were written."""
        self._saved_rev = self._rev


def merge_samples(X_a: Any, y_a: Sequence[str], X_b: Any, y_b: Sequence[str],
                  names_a: Optional[Sequence[str]] = None,
                  names_b: Optional[Sequence[str]] = None,
                  n_features: Optional[int] = None) -> Tuple[np.ndarray, List[str], int]:
    """
    Merge two sample sets, e.g. the samples of an existing model and new ones.

    Exact duplicates (same label, identical features) are kept once, so
    appending the same data twice changes nothing. Returns (X, y, number of
    duplicates dropped). Raises ValueError when the feature names differ or
    rows and labels do not match.
    """
    if names_a is not None and names_b is not None and list(names_a) != list(names_b):
        raise ValueError('the sample sets use different features; retrain instead of merging')
    n = n_features if n_features is not None else len(names_a or names_b or FEATURE_NAMES)
    A, B = _rows(X_a, n), _rows(X_b, n)
    ya, yb = [str(v) for v in y_a], [str(v) for v in y_b]
    if A.shape[0] != len(ya) or B.shape[0] != len(yb):
        raise ValueError('every sample needs exactly one label')
    X = np.vstack([A, B])
    y = ya + yb
    seen: Set[Tuple[str, bytes]] = set()
    keep: List[int] = []
    for i, lab in enumerate(y):
        key = (lab, X[i].tobytes())
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return X[keep], [y[i] for i in keep], len(y) - len(keep)


# ------------------------------------------------------------------ evaluation
def weak_classes(counts: Dict[str, int],
                 min_samples: int = MIN_SAMPLES) -> List[Tuple[str, int]]:
    """Return [(label, count)] of the classes with fewer than min_samples samples."""
    return [(str(k), int(v)) for k, v in sorted(counts.items()) if int(v) < int(min_samples)]


def confused_pairs(confusion: Dict[str, Dict[str, int]],
                   min_rate: float = CONFUSE_RATE) -> List[Dict[str, Any]]:
    """
    Return the class pairs leave-one-out mixes up in at least min_rate of their samples.

    confusion: {true: {predicted: count}}. Each entry is {'a', 'b', 'a_as_b',
    'b_as_a', 'rate'} with rate = (a_as_b + b_as_a) / (samples of a and b),
    sorted by decreasing rate.
    """
    totals = {t: sum(int(c) for c in row.values()) for t, row in confusion.items()}
    labels = sorted(totals)
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ab = int(confusion.get(a, {}).get(b, 0))
            ba = int(confusion.get(b, {}).get(a, 0))
            total = totals[a] + totals[b]
            if ab + ba == 0 or total == 0:
                continue
            rate = (ab + ba) / float(total)
            if rate >= min_rate:
                out.append({'a': a, 'b': b, 'a_as_b': ab, 'b_as_a': ba, 'rate': rate})
    out.sort(key=lambda p: (-p['rate'], p['a'], p['b']))
    return out


def confusion_table(confusion: Dict[str, Dict[str, int]]) -> str:
    """
    Return the leave-one-out confusion matrix as text.

    Rows: true classes ('<i> <label>'); columns: predicted class numbers
    ('unk' = unknown, '.' = none) and the per-class accuracy.
    """
    if not confusion:
        return '(no confusion table yet)'
    true = sorted(confusion)
    preds = sorted(set(true) | {p for row in confusion.values() for p in row if p != 'unknown'})
    cols = preds + (['unknown'] if any('unknown' in r for r in confusion.values()) else [])
    num = {lab: i for i, lab in enumerate(preds)}
    headers = (['true \\ predicted'] + ['unk' if c == 'unknown' else str(num[c]) for c in cols]
               + ['accuracy'])
    rows = []
    for t in true:
        row = confusion[t]
        total = sum(int(v) for v in row.values())
        cells = [str(int(row[c])) if row.get(c) else '.' for c in cols]
        acc = '%.0f %%' % (100.0 * row.get(t, 0) / total) if total else '-'
        rows.append(['%d %s' % (num[t], t)] + cells + [acc])
    return format_table(headers, rows)


def counts_text(counts: Dict[str, int], known: Sequence[str] = (),
                min_samples: int = MIN_SAMPLES) -> str:
    """Return 'apple 9, lemon 3 (<8)': samples per class, listing task labels without any."""
    labels = [str(k) for k in known] + sorted(k for k in counts if k not in known)
    if not labels:
        return 'no samples yet'
    parts = []
    for lab in labels:
        n = int(counts.get(lab, 0))
        parts.append('%s %d%s' % (lab, n, '' if n >= min_samples else ' (<%d)' % min_samples))
    return ', '.join(parts)


def evaluate_samples(X: Any, y: Sequence[str], k: int = 3, max_distance: float = 3.0,
                     known: Sequence[str] = (), task: Optional[str] = None,
                     min_samples: int = MIN_SAMPLES,
                     confuse_rate: float = CONFUSE_RATE) -> Dict[str, Any]:
    """
    Evaluate samples with KnnClassifier.leave_one_out and collect warnings.

    Returns {'n', 'classes', 'samples_per_class', 'k', 'max_distance',
    'accuracy' (None with fewer than two classes), 'n_unknown', 'per_class',
    'confusion', 'weak', 'confused', 'missing_labels' (task labels without
    samples), 'unlisted_labels', 'warnings', 'text'}.
    """
    y = [str(v) for v in y]
    counts = dict(sorted(Counter(y).items()))
    known = [str(v) for v in known]
    key = 'tasks.%s.labels' % task if task else 'the task labels'
    rep: Dict[str, Any] = {
        'n': len(y), 'classes': list(counts), 'samples_per_class': counts, 'k': int(k),
        'max_distance': float(max_distance), 'accuracy': None, 'n_unknown': 0,
        'per_class': {}, 'confusion': {}, 'weak': weak_classes(counts, min_samples),
        'confused': [], 'missing_labels': [v for v in known if v not in counts],
        'unlisted_labels': [v for v in counts if known and v not in known], 'warnings': []}
    warn = rep['warnings']
    if len(counts) >= 2:
        knn = KnnClassifier(k=k, max_distance=max_distance).fit(_rows(X), y)
        loo = knn.leave_one_out()
        rep.update(accuracy=float(loo['accuracy']), n_unknown=int(loo['n_unknown']),
                   per_class=loo['per_class'], confusion=loo['confusion'])
        rep['confused'] = confused_pairs(loo['confusion'], confuse_rate)
    elif counts:
        warn.append('only one class (%s): the evaluation needs two or more; label the other '
                    'classes too (or --append to a model that has them)' % y[0])
    else:
        warn.append('no samples yet')
    if rep['weak']:
        warn.append('fewer than %d samples: %s; label them in more views (other positions and '
                    'orientations)' % (min_samples, ', '.join('%s %d' % w for w in rep['weak'])))
    for p in rep['confused']:
        warn.append('%s and %s are confused in %.0f %% of their samples (%d + %d): add views '
                    'of both and check that each label names the right object'
                    % (p['a'], p['b'], 100.0 * p['rate'], p['a_as_b'], p['b_as_a']))
    for t, row in sorted(rep['confusion'].items()):
        n_unk = int(row.get('unknown', 0))
        if n_unk:
            warn.append('%d of %d %s samples are farther than max_distance %.1f from all the '
                        'others (unknown): add more views of it'
                        % (n_unk, counts[t], t, float(max_distance)))
    if rep['missing_labels']:
        warn.append('%s without samples: %s (the classifier can never output them)'
                    % (key, ', '.join(rep['missing_labels'])))
    if rep['unlisted_labels']:
        warn.append('labels not in %s: %s' % (key, ', '.join(rep['unlisted_labels'])))
    rep['text'] = evaluation_text(rep)
    return rep


def evaluation_text(rep: Dict[str, Any]) -> str:
    """Return the evaluation printout: accuracy, samples, confusion table, warnings."""
    if rep.get('accuracy') is None:
        lines = ['leave-one-out: not available (%d samples, %d classes)'
                 % (rep['n'], len(rep['classes']))]
    else:
        lines = ['leave-one-out: %.1f %% of %d samples correct, %d unknown (k %d, '
                 'max_distance %.1f)' % (100.0 * rep['accuracy'], rep['n'], rep['n_unknown'],
                                         rep['k'], rep['max_distance'])]
    lines.append('samples per class: %s' % counts_text(rep['samples_per_class']))
    if rep.get('confusion'):
        lines.append(confusion_table(rep['confusion']))
    lines.extend('WARNING: %s' % w for w in rep['warnings'])
    if not rep['warnings']:
        lines.append('no warnings')
    return '\n'.join(lines)


# ------------------------------------------------------------------ capture sessions
_DATE_PREFIX = re.compile(r'^\d{8}(_\d{6})?_')
_INDEX_SUFFIX = re.compile(r'_\d+$')


def clean_label(text: Optional[str]) -> str:
    """Return text as a label: characters other than letters, digits, _ and - become _."""
    return re.sub(r'[^\w\-]+', '_', str(text or '').strip()).strip('_')


def session_label(session: Optional[str], dir_name: str, known: Sequence[str] = (),
                  explicit: Optional[str] = None) -> Tuple[str, str]:
    """
    Return (label, where it came from) for a frame of a one-class capture session.

    Order: explicit (LABEL=DIR), the meta extra 'session' (unless generic such
    as 'capture'), the directory name without a capture date prefix. A
    candidate matching a known label (case-insensitive, also after dropping
    a trailing _N) wins; otherwise the first candidate without its _N
    suffix is returned ('' when there is none).
    """
    if explicit:
        return clean_label(explicit), 'LABEL=DIR'
    cands: List[Tuple[str, str]] = []
    s = clean_label(session)
    if s and s.lower() not in GENERIC_SESSIONS:
        cands.append((s, 'session %r' % str(session)))
    d = clean_label(_DATE_PREFIX.sub('', str(dir_name)))
    if d:
        cands.append((d, 'directory %r' % str(dir_name)))
    if not cands:
        return '', 'no session name'
    low = {str(k).lower(): str(k) for k in known}
    for name, how in cands:
        for c in (name, _INDEX_SUFFIX.sub('', name)):
            if c and c.lower() in low:
                return low[c.lower()], how
    name, how = cands[0]
    return _INDEX_SUFFIX.sub('', name) or name, how


def parse_dir_spec(spec: str) -> Tuple[Optional[str], Path]:
    """Split '[LABEL=]DIR' into (label or None, directory path)."""
    s = str(spec)
    if '=' in s:
        lab, rest = s.split('=', 1)
        if lab and rest and re.match(r'^[\w\-]+$', lab):
            return lab, Path(rest).expanduser()
    return None, Path(s).expanduser()


def dir_summary(summ: Dict[str, Any]) -> str:
    """Return the one-line summary of a learned session directory."""
    got = ', '.join('%s %d' % kv for kv in sorted(summ.get('samples', {}).items()))
    s = '%s: %d frames -> %s' % (summ['dir'], summ.get('frames', 0), got or 'no samples')
    if summ.get('skipped'):
        s += ' | skipped: ' + ', '.join('%s x%d' % kv for kv in sorted(summ['skipped'].items()))
    if summ.get('error'):
        s += ' | %s' % summ['error']
    return s


# ------------------------------------------------------------------ output
def model_out_path(cfg, mode: str, out: Optional[str] = None) -> Path:
    """
    Return the model file to write.

    --out wins; else perception.classifier.knn.model_file (resolved with
    cfg.path), which sim mode redirects into <logging.dir>/sim_site.
    """
    if out:
        return Path(out).expanduser()
    default = cfg.path(MODEL_KEY) or cfg.resolve(DEFAULT_MODEL)
    return output_path(cfg, mode, None, default)


def report_path(model_path: Path) -> Path:
    """Return the JSON report written next to a model: <stem>_report.json."""
    p = Path(model_path)
    return p.with_name(p.stem + '_report.json')


def knn_params(cfg) -> Tuple[int, float]:
    """Return (k, max_distance) of perception.classifier.knn."""
    return (int(cfg.get('perception.classifier.knn.k', 3)),
            float(cfg.get('perception.classifier.knn.max_distance', 3.0)))


def build_report(evaluation: Dict[str, Any], created: str, source: str, model_file: Path,
                 **extra: Any) -> Dict[str, Any]:
    """Return the JSON report written next to the model (plain types, no NaN)."""
    acc = evaluation.get('accuracy')
    rep: Dict[str, Any] = {
        'model_file': str(model_file), 'created': str(created), 'source': str(source),
        'n_samples': int(evaluation['n']),
        'samples_per_class': {str(k): int(v)
                              for k, v in evaluation['samples_per_class'].items()},
        'accuracy': None if acc is None or not math.isfinite(acc) else round(float(acc), 4),
        'n_unknown': int(evaluation.get('n_unknown', 0)),
        'per_class': {str(t): {'correct': int(v['correct']), 'total': int(v['total']),
                               'accuracy': round(float(v['accuracy']), 4)}
                      for t, v in evaluation.get('per_class', {}).items()},
        'confusion': {str(t): {str(p): int(c) for p, c in row.items()}
                      for t, row in evaluation.get('confusion', {}).items()},
        'confused_pairs': [dict(p, rate=round(float(p['rate']), 4))
                           for p in evaluation.get('confused', [])],
        'weak_classes': {str(k): int(v) for k, v in evaluation.get('weak', [])},
        'warnings': [str(w) for w in evaluation.get('warnings', [])],
        'feature_names': list(FEATURE_NAMES),
        'k': int(evaluation.get('k', 3)),
        'max_distance': float(evaluation.get('max_distance', 3.0)),
    }
    rep.update(extra)
    return rep


def config_hints(cfg, task: Optional[str], mode: str, path: Path, classes: Sequence[str],
                 known: Sequence[str]) -> List[str]:
    """Return what to set in the config so that the task run uses the written model."""
    hints = [use_hint(cfg, mode, path, MODEL_KEY)]
    if not task:
        return hints
    tasks_yaml = overlay_file(cfg, 'tasks.yaml')
    try:
        backend = classifier_backend(cfg, task)
    except ValueError as e:
        backend = str(e)
    if 'knn' not in backend:
        hints.append('tasks.%s.classifier is %r: set tasks.%s.classifier: knn in %s to use this '
                     'model' % (task, backend, task, tasks_yaml))
    if not known:
        hints.append('tasks.%s.labels is empty: set tasks.%s.labels: [%s] in %s'
                     % (task, task, ', '.join(classes), tasks_yaml))
    if task != 'L3':
        ctz = cfg.get('tasks.%s.class_to_zone' % task, {}) or {}
        nozone = [c for c in classes if c not in ctz]
        if nozone:
            hints.append('tasks.%s.class_to_zone has no zone for %s (never picked); set it in %s'
                         % (task, ', '.join(nozone), tasks_yaml))
    return hints


# ------------------------------------------------------------------ views
INSTANCE_HEADERS = ('#', 'LxWxH mm', 'center xy mm', 'label')


def view_features(frame, dets: Sequence[Any]) -> Tuple[np.ndarray, List[bool]]:
    """Return (feature rows N x D, usable flags = all features finite) of the detections."""
    if not dets:
        return np.zeros((0, N_FEATURES)), []
    X = np.vstack([feature_vector(frame, d)[0] for d in dets]).astype(np.float64)
    return X, [bool(v) for v in np.all(np.isfinite(X), axis=1)]


def instance_rows(dets: Sequence[Any], usable: Sequence[bool],
                  labeled: Dict[int, str]) -> List[List[str]]:
    """Return INSTANCE_HEADERS rows for the valid instances of a view."""
    rows = []
    for i, d in enumerate(dets):
        if d.length is not None and d.width is not None and d.height is not None:
            size = '%.0fx%.0fx%.0f' % (d.length * 1e3, d.width * 1e3, d.height * 1e3)
        else:
            size = '-'
        c = '-' if d.center_base is None else fmt_mm(np.asarray(d.center_base)[:2], 0)
        if i in labeled:
            lab = labeled[i]
        else:
            lab = '-' if usable[i] else '(incomplete features: cannot be labeled)'
        rows.append([str(i), size, c, lab])
    return rows


def rejected_summary(dets: Sequence[Any]) -> str:
    """Return 'occluded x1, too_small x2' for the rejected instances ('' when none)."""
    c = Counter(d.reject_reason or 'invalid' for d in dets if not d.valid)
    return ', '.join('%s x%d' % kv for kv in sorted(c.items()))


def layout_of(dets: Sequence[Any]) -> List[Tuple[float, float, float]]:
    """Return [(x, y, yaw)] in the base frame of the instances with a position."""
    return [(float(d.center_base[0]), float(d.center_base[1]), float(d.yaw or 0.0))
            for d in dets if d.center_base is not None]


def layout_unchanged(prev: Sequence[Tuple[float, float, float]],
                     cur: Sequence[Tuple[float, float, float]], tol_m: float = 0.005,
                     tol_yaw_deg: float = 5.0) -> bool:
    """Return True when every instance of cur sits where one of prev was (same xy and yaw)."""
    if not prev or len(prev) != len(cur):
        return False
    tol_yaw = math.radians(tol_yaw_deg)
    for x, y, yaw in cur:
        if not any(math.hypot(x - px, y - py) <= tol_m
                   and abs((yaw - pyaw + math.pi / 2) % math.pi - math.pi / 2) <= tol_yaw
                   for px, py, pyaw in prev):
            return False
    return True


def draw_instances(img: np.ndarray, dets: Sequence[Any], labeled: Dict[int, str],
                   usable: Sequence[bool]) -> np.ndarray:
    """Write '#i' and the label given in this view at each instance (in place); return img."""
    import cv2
    for i, d in enumerate(dets):
        u, v = d.center_px
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        if i in labeled:
            text, color = '#%d %s' % (i, labeled[i]), (0, 255, 0)
        elif usable[i]:
            text, color = '#%d' % i, (0, 255, 255)
        else:
            text, color = '#%d ?' % i, (0, 0, 255)
        (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # a dark box keeps the number readable over the 'unknown 0.00' overlay text,
        # clamped so that a label near the border is not cut off
        x = int(min(max(2, int(u) + 8), max(2, img.shape[1] - w - 4)))
        y = int(min(max(h + 4, int(v) - 8), max(h + 4, img.shape[0] - base - 2)))
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def wrap_lines(lines: Sequence[str], width_px: int, char_px: int = 9) -> List[str]:
    """Wrap text panel lines so that they fit width_px (about char_px pixels per character)."""
    import textwrap
    n = max(20, int(width_px) // max(1, int(char_px)))
    out: List[str] = []
    for line in lines:
        out.extend(textwrap.wrap(line, n, subsequent_indent='  ') or [''])
    return out


def frames_per_view(cfg, mode: str) -> int:
    """Return the frames averaged per view: camera.frames_to_average (replay: <= repeat)."""
    n = max(1, int(cfg.get('camera.frames_to_average', 3) or 1))
    if mode == 'replay':
        n = min(n, max(1, int(cfg.get('camera.file.repeat', 1) or 1)))
    return n


# ------------------------------------------------------------------ operator I/O
def confirm_labels(ui, labels: Sequence[str], known: Sequence[str], task: Optional[str],
                   accepted: Set[str]) -> Optional[Dict[str, str]]:
    """
    Map labels to the spelling of tasks.<task>.labels, asking about the others (operator I/O).

    A label outside a non-empty list, or any new label when the list is
    empty, needs a confirmation once; accepted holds the labels confirmed
    so far (updated). Returns None when the operator declines one.
    """
    key = 'tasks.%s.labels' % task
    out: Dict[str, str] = {}
    for lab in labels:
        canon, status, close = check_label(lab, known)
        if status == 'ok' or canon in accepted:
            out[lab] = canon
            continue
        if status == 'unlisted':
            ui.say('WARNING: label %r is not in %s (%s)%s' % (
                canon, key, ', '.join(known),
                '; did you mean %s?' % ' / '.join(close) if close else ''))
            ok = ui.confirm('Use the label %r anyway?' % canon, default=False)
        else:
            ui.say('WARNING: %s is empty, so labels cannot be checked: new label %r (labels so '
                   'far: %s)' % (key, canon, ', '.join(sorted(accepted)) or 'none'))
            ok = ui.confirm('Is the new label %r spelled right?' % canon, default=True)
        if not ok:
            ui.say('label %r not used' % canon)
            return None
        accepted.add(canon)
        out[lab] = canon
    return out


def load_existing(ui, book: SampleBook, path: Path, append: bool) -> bool:
    """
    Report an existing model at path; with append keep its samples (operator I/O).

    Returns False when the operator aborts (unreadable model with --append).
    """
    if not Path(path).is_file():
        if append:
            ui.say('--append: no model at %s yet; starting empty' % path)
        return True
    try:
        old = KnnClassifier.load(path)
    except (OSError, ValueError, KeyError) as e:
        ui.say('WARNING: the existing model %s cannot be used (%s)' % (path, e))
        if append:
            return ui.confirm('Continue without its samples (w replaces it, a backup is kept)?',
                              default=False)
        return True
    desc = '%d samples: %s' % (old.n_samples, counts_text(dict(Counter(old.y))))
    if append:
        book.set_base(old.X, old.y)
        ui.say('--append: keeping the %s of %s' % (desc, path))
    else:
        ui.say('existing model %s (%s) is replaced when you write (a backup is kept); --append '
               'keeps its samples' % (path, desc))
    return True


def write_model(path: Path, X: Any, y: Sequence[str], k: int, max_distance: float,
                report: Dict[str, Any]) -> Tuple[Path, Optional[Path], Path]:
    """
    Back up the previous model and report, then write both atomically.

    Returns (model path, backup of the previous model or None, report path).
    """
    knn = KnnClassifier(k=k, max_distance=max_distance).fit(_rows(X), list(y), FEATURE_NAMES)
    path = Path(path)
    rpath = report_path(path)
    stamp = wall_stamp()
    backup = backup_file(path, stamp)
    backup_file(rpath, stamp)
    knn.save(path, created=str(report.get('created', '')) or None)
    atomic_write_text(rpath, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
                      + '\n')
    try:    # mkstemp files are owner-only; make the report as readable as the model
        os.chmod(str(rpath), os.stat(str(path)).st_mode & 0o777)
    except OSError:
        pass
    return path, backup, rpath


def save_model(ui, cfg, task: Optional[str], mode: str, book: SampleBook, path: Path,
               source: str, known: Sequence[str], extra: Optional[Dict[str, Any]] = None,
               evaluation: Optional[Dict[str, Any]] = None) -> bool:
    """Evaluate and write the samples of book with the JSON report; print what to set (I/O)."""
    X, y = book.arrays()
    if not y:
        ui.say('no samples to write')
        return False
    k, md = knn_params(cfg)
    ev = evaluation or evaluate_samples(X, y, k, md, known, task)
    created = datetime.datetime.now().isoformat(timespec='seconds')
    rep = build_report(ev, created, source, path, mode=mode, task=task,
                       sources=book.source_counts(), **(extra or {}))
    try:
        path, backup, rpath = write_model(path, X, y, k, md, rep)
    except (OSError, ValueError) as e:
        ui.say('ERROR: could not write the model %s: %s' % (path, e))
        return False
    book.mark_saved()
    acc = ev.get('accuracy')
    lines = ['wrote %s: %d samples, %d classes (%s)'
             % (path, len(y), len(ev['classes']), counts_text(ev['samples_per_class'])),
             '  leave-one-out accuracy: %s' % ('n/a' if acc is None else '%.1f %%' % (100 * acc))]
    if backup is not None:
        lines.append('  previous model kept as %s' % backup)
    lines.append('  report: %s' % rpath)
    lines.append('  config key: %s' % MODEL_KEY)
    lines.extend('  %s' % h for h in config_hints(cfg, task, mode, path, ev['classes'], known))
    lines.extend('  WARNING: %s' % w for w in ev['warnings'])
    ui.say('\n'.join(lines))
    return True


def add_sim_ground_truth(ui, book: SampleBook, task: str, mount: str, n_scenes: int,
                         first_seed: int = SIM_FIRST_SEED) -> int:
    """Sim rehearsal: add the samples of n_scenes simulated scenes with ground-truth labels."""
    from ..sim.teach import collect_sim_samples
    seeds = list(range(int(first_seed), int(first_seed) + int(n_scenes)))
    X, y, names = collect_sim_samples(seeds, mount=mount, task=task)
    if not y or (names is not None and list(names) != list(FEATURE_NAMES)):
        ui.say('[sim] no ground-truth samples collected')
        return 0
    book.add(Batch(text='sim ground truth (%d scenes)' % len(seeds), labels=list(y), X=X,
                   source='sim_gt'))
    ui.say('[sim] learned %d samples of %d classes from %d simulated scenes with ground-truth '
           'labels (seeds %d-%d; u removes them)'
           % (len(y), len(set(y)), len(seeds), seeds[0], seeds[-1]))
    return len(y)


# ------------------------------------------------------------------ live loop
class _Teacher:
    """State of one live teaching run (thin I/O around the pure helpers)."""

    def __init__(self, args, s, ui, book: SampleBook, known: List[str], out: Path, ext):
        from ..perception.pipeline import Perception
        self.args, self.s, self.ui, self.book = args, s, ui, book
        self.cfg, self.task, self.known, self.out, self.ext = s.cfg, s.task, known, out, ext
        self.accepted = accepted_labels(book, known)
        self.perception = Perception(s.cfg, classifier=None)
        self.T_ft = flange_to_tcp(s.cfg)
        self.n_avg = frames_per_view(s.cfg, args.mode)
        self.sim = args.mode == 'sim' and s.world is not None
        self.source = 'sim' if self.sim else 'live'
        self.view = 0
        self.frame = None
        self.overlay: Optional[np.ndarray] = None
        self.dets: List[Any] = []
        self.X = np.zeros((0, N_FEATURES))
        self.usable: List[bool] = []
        self.layout: List[Tuple[float, float, float]] = []
        self.last_stamp: Optional[float] = None
        self.relayouts = 0

    def intro(self) -> None:
        """Print the instructions."""
        ui = self.ui
        ui.say('Put the objects inside the pick area (perception.roi.pick_polygon), at least '
               '2 cm apart and away from the place zones; keep hands and the arm out of view.')
        ui.say('Label the numbered instances, then move / rotate the objects and press c for '
               'the next view. Aim for >= %d samples per class; e evaluates, w writes %s.'
               % (MIN_SAMPLES, self.out))
        if self.ext is not None and self.ext.mount == 'eye_in_hand':
            ui.say('eye_in_hand: the arm pose is read at every view (the arm is never moved); '
                   'hold the camera over the pick area like the observe pose and keep the arm '
                   'still while c captures.')
            if self.s.arm is not None and not self.sim:
                teach_hint(ui, self.s.arm)
        if ui.headless:
            ui.say('preview image: %s' % ui.preview_path('teach_classes'))
        self.help()

    def help(self) -> None:
        """Print the command help."""
        self.ui.say(HEADLESS_HELP if self.ui.headless else WINDOW_HELP)
        if self.sim:
            self.ui.say(SIM_HELP)

    def _with_arm_pose(self, frame):
        """Eye-in-hand: attach the current flange pose to the frame (read only)."""
        if (self.ext is None or self.ext.mount != 'eye_in_hand' or self.s.arm is None
                or recorded_flange_pose(frame) is not None):
            return frame
        pose = read_pose(self.s.arm)
        if pose is None:
            return frame
        meta = dict(frame.meta or {})
        extra = dict(meta.get('extra') or {})
        extra['T_base_flange'] = np.asarray(pose.T_base_flange, dtype=float).tolist()
        meta['extra'] = extra
        return dataclasses.replace(frame, meta=meta)

    def capture(self) -> bool:
        """Grab an averaged view, segment it and show the numbered valid instances."""
        frame = capture_view(self.s.camera, self.cfg, self.n_avg, self.last_stamp)
        if frame is None:
            self.ui.say('no camera frame: is the camera driver running? (check_setup)')
            return False
        self.last_stamp = float(frame.stamp)
        frame = self._with_arm_pose(frame)
        T_bc, why = camera_pose_for_frame(self.ext, frame, self.s.arm, self.T_ft)
        self.perception.reset_table()
        res = self.perception.process(frame, T_bc)
        dets = res.valid_detections()
        prev = self.layout
        self.view += 1
        self.frame, self.dets = frame, dets
        self.X, self.usable = view_features(frame, dets)
        self.layout = layout_of(dets)
        self.overlay = res.debug_image if res.debug_image is not None else frame.color.copy()
        rej = rejected_summary(res.detections)
        self.ui.say('view %d: %d valid instance%s%s' % (
            self.view, len(dets), '' if len(dets) == 1 else 's',
            ' (rejected: %s)' % rej if rej else ''))
        if T_bc is None:
            self.ui.say('WARNING: no camera pose (%s): no geometry, so no instances' % why)
        if res.warnings:
            self.ui.say('perception warnings: %s' % ', '.join(res.warnings))
        if layout_unchanged(prev, self.layout):
            self.ui.say('NOTE: the objects did not move since the last view; move / rotate them '
                        'for new samples (c), or label this view anyway')
        if dets:
            self.ui.say(format_table(INSTANCE_HEADERS, instance_rows(dets, self.usable, {})))
        else:
            self.ui.say('no valid instance: put the objects inside the pick area, apart from '
                        'each other and away from the place zones, then c')
        self.show()
        return True

    def show(self) -> None:
        """Show (headless: save) the overlay with the numbered instances and their labels."""
        if self.frame is None or self.overlay is None:
            return
        labeled = self.book.labeled_in_view(self.view)
        img = draw_instances(self.overlay.copy(), self.dets, labeled, self.usable)
        lines = ['view %d: %d valid instances, %d labeled' % (self.view, len(self.dets),
                                                              len(labeled)),
                 'samples: %s' % counts_text(self.book.counts(), self.known)]
        view = compose_view(self.frame, img)
        self.ui.show('teach_classes', text_panel(view, wrap_lines(lines, view.shape[1])))

    def new_view(self) -> bool:
        """Capture the next view; sim: the simulated operator rearranges the objects first."""
        if self.sim:
            self.relayouts += 1
            seed = int(self.args.seed) + SIM_RELAYOUT_SEED + self.relayouts
            world, task, mount = self.s.world, self.task, self.args.mount
            if not operator_action(self.ui, 'sim', 'rearrange the objects (layout of seed %d)'
                                   % seed, lambda: sim_relayout(world, task, seed, mount) > 0):
                return False
        return self.capture()

    def label(self, text: str) -> bool:
        """Apply an operator labeling command to the current view."""
        if self.view == 0:
            self.ui.say('no view yet: c captures one')
            return False
        parts, err = parse_label_command(text, len(self.dets))
        if parts is None:
            self.ui.say('not labeled: %s' % err)
            return False
        assign, notes, err = plan_labeling(parts, self.book.labeled_in_view(self.view),
                                           self.usable)
        for n in notes:
            self.ui.say(n)
        if err:
            self.ui.say('not labeled: %s' % err)
            return False
        names = confirm_labels(self.ui, sorted(set(assign.values())), self.known, self.task,
                               self.accepted)
        if names is None:
            self.ui.say('not labeled (nothing changed)')
            return False
        return self._add(text, {i: names[lab] for i, lab in assign.items()}, self.source)

    def ground_truth(self) -> bool:
        """Sim only: label the unlabeled instances of this view with the ground truth."""
        if not self.sim:
            self.ui.say('g (ground truth) works in --mode sim only')
            return False
        done = self.book.labeled_in_view(self.view)
        assign: Dict[int, str] = {}
        for i, d in enumerate(self.dets):
            if i in done or not self.usable[i]:
                continue
            lab = ground_truth_label(self.s.world, d)
            if lab:
                assign[i] = lab
        if not assign:
            self.ui.say('nothing to label from the ground truth in this view')
            return False
        return self._add('g', assign, 'sim_gt')

    def _add(self, text: str, assign: Dict[int, str], source: str) -> bool:
        idx = sorted(assign)
        batch = Batch(text=text, labels=[assign[i] for i in idx], X=self.X[idx],
                      view=self.view, indices=idx, source=source)
        try:
            self.book.add(batch)
        except ValueError as e:
            self.ui.say('not labeled: %s' % e)
            return False
        self.ui.say('labeled %s | samples: %s' % (
            ', '.join('#%d %s' % (i, assign[i]) for i in idx),
            counts_text(self.book.counts(), self.known)))
        self.show()
        return True

    def undo(self) -> bool:
        """Undo the last labeling (also of an earlier view)."""
        b = self.book.undo()
        if b is None:
            self.ui.say('nothing to undo')
            return False
        where = ' of view %d' % b.view if b.view >= 0 else ''
        self.ui.say('undid %r%s: %d samples removed | samples: %s' % (
            b.text, where, len(b.labels), counts_text(self.book.counts(), self.known)))
        self.show()
        return True

    def evaluate(self) -> Dict[str, Any]:
        """Print the leave-one-out evaluation of all samples (base + this run)."""
        X, y = self.book.arrays()
        k, md = knn_params(self.cfg)
        rep = evaluate_samples(X, y, k, md, self.known, self.task)
        self.ui.say(rep['text'])
        return rep

    def write(self) -> bool:
        """Write the model and its report."""
        return save_model(self.ui, self.cfg, self.task, self.args.mode, self.book, self.out,
                          self.source, self.known, extra={'views': self.view})

    def status(self) -> None:
        """Print the samples per class and the instances of this view."""
        self.ui.say('samples: %s' % counts_text(self.book.counts(), self.known))
        self.ui.say('sources: %s | %d labelings (u undoes the last) | %s' % (
            ', '.join('%s %d' % kv for kv in sorted(self.book.source_counts().items()))
            or 'none', len(self.book.batches),
            'unsaved changes' if self.book.dirty else 'nothing unsaved'))
        if self.view and self.dets:
            self.ui.say('view %d:\n%s' % (self.view, format_table(INSTANCE_HEADERS, instance_rows(
                self.dets, self.usable, self.book.labeled_in_view(self.view)))))

    def finish(self) -> None:
        """At quit: offer to write unsaved samples."""
        if not self.book.dirty or not self.book.arrays()[1]:
            return
        self.ui.say('unsaved samples: %s' % counts_text(self.book.counts(), self.known))
        if self.ui.confirm('Write the model to %s before quitting?' % self.out, default=True):
            self.write()
        else:
            self.ui.say('samples discarded (not written)')

    def loop(self) -> None:
        """Read and run operator commands until q (or the end of the input)."""
        actions = {'capture': self.new_view, 'undo': self.undo, 'evaluate': self.evaluate,
                   'write': self.write, 'status': self.status, 'help': self.help,
                   'ground_truth': self.ground_truth}
        prompt = SIM_PROMPT if self.sim else PROMPT
        try:
            while True:
                cmd = next_command(self.ui, prompt)
                kind = command_kind(cmd)
                if kind == 'quit':
                    break
                if kind == 'label' and not self.ui.headless and len(cmd.strip()) == 1:
                    self.ui.say('unknown key %r (h = help)' % cmd)
                elif kind == 'label':
                    self.label(cmd)
                elif kind in actions:
                    actions[kind]()
        except KeyboardInterrupt:
            self.ui.say('\ninterrupted')
        self.finish()


def run_live(args, cfg_world_info, ui, book: SampleBook, known: List[str], out: Path) -> int:
    """Run the live labeling loop (real camera, replay or the sim rig); return the exit code."""
    cfg = cfg_world_info[0]
    ext = load_extrinsics_or_none(cfg, ui)
    if ext is None:
        ui.say('teach_classes needs the camera extrinsics (camera.extrinsics_file) for the '
               'geometry features: calibrate first, or pass --config / --set')
        return 2
    use_arm = args.mode == 'sim' or (args.mode == 'real' and ext.mount == 'eye_in_hand')
    try:
        s = open_session(args, arm=use_arm, camera=True, perception=False,
                         connect=args.mode == 'sim', cfg_world_info=cfg_world_info)
    except (RuntimeError, ImportError, OSError, ValueError) as e:
        ui.say('ERROR: could not start the %s: %s' % ('camera and arm' if use_arm else 'camera',
                                                      e))
        ui.say('check the camera driver and the CAN link: check_setup --task %s' % args.task)
        return 2
    with s:
        if not connect_arm_readonly(s, ui, required=True, why='eye_in_hand camera poses'):
            return 2
        if args.mode == 'sim':
            ok, why = go_observe(s.arm, s.cfg, MotionGuard(ui, 'sim', 100))
            if not ok:
                ui.say('sim: could not reach the observe pose: %s' % why)
            if args.sim_scenes > 0:
                add_sim_ground_truth(ui, book, args.task, args.mount, args.sim_scenes)
        try:
            t = _Teacher(args, s, ui, book, known, out, ext)
        except ValueError as e:
            ui.say('ERROR: invalid perception config: %s' % e)
            return 2
        t.intro()
        t.capture()
        t.loop()
    return 0


# ------------------------------------------------------------------ offline
def learn_dir(d: Path, explicit: Optional[str], perception, ext, T_ft: np.ndarray, ui,
              book: SampleBook, known: Sequence[str], task: Optional[str], accepted: Set[str],
              decisions: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Learn the frames of one capture session directory into book; return its summary (I/O).

    Every valid instance of a frame gets the session label (session_label;
    confirm_labels asks once per new label, the answer is kept in decisions).
    """
    from ..camera.recorder import list_frames, load_frame
    summ: Dict[str, Any] = {'dir': str(d), 'frames': 0, 'samples': {}, 'skipped': {}}
    skipped = summ['skipped']
    if not d.is_dir():
        summ['error'] = 'not a directory'
        ui.say(dir_summary(summ))
        return summ
    idx = list_frames(d)
    summ['frames'] = len(idx)
    rows: Dict[str, List[np.ndarray]] = {}
    for i in idx:
        try:
            frame, extra = load_frame(d, i)
        except (OSError, ValueError) as e:
            _LOG.warning('%s frame %d: %s', d, i, e)
            skipped['unreadable'] = skipped.get('unreadable', 0) + 1
            continue
        name, how = session_label(extra.get('session'), d.name, known, explicit)
        if name not in decisions:
            ok = confirm_labels(ui, [name], known, task, accepted) if name else None
            decisions[name] = None if ok is None else ok[name]
            ui.say('%s: %s' % (d, 'label %r (from %s)' % (decisions[name], how)
                               if decisions[name] else 'no usable label (%s); use LABEL=DIR'
                               % how))
        label = decisions[name]
        X, why = frame_samples(frame, extra, perception, ext, T_ft) if label else (None,
                                                                                   'no label')
        if X is None:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        rows.setdefault(label, []).append(X)
    for label, parts in sorted(rows.items()):
        Xl = np.vstack(parts)
        book.add(Batch(text='%s: %s' % (d, label), labels=[label] * Xl.shape[0], X=Xl,
                       source='from_dir'))
        summ['samples'][label] = int(Xl.shape[0])
    ui.say(dir_summary(summ))
    return summ


def frame_samples(frame, extra: Dict[str, Any], perception, ext,
                  T_ft: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """
    Return (feature rows of the usable valid instances, '') of a recorded frame.

    The camera pose comes from ext and, for eye_in_hand, the flange pose in
    the frame's extra. Returns (None, why) when nothing can be learned.
    """
    frame = dataclasses.replace(frame, meta=dict(frame.meta or {}, extra=extra))
    T_bc, why = camera_pose_for_frame(ext, frame, None, T_ft)
    if T_bc is None:
        return None, 'no camera pose (%s)' % why
    perception.reset_table()
    res = perception.process(frame, T_bc)
    X, usable = view_features(frame, res.valid_detections())
    if not any(usable):
        return None, 'no valid instance%s' % (' (%s)' % ', '.join(res.warnings)
                                              if res.warnings else '')
    return X[np.asarray(usable, dtype=bool)], ''


def run_from_dirs(args, cfg, ui, book: SampleBook, known: List[str], out: Path) -> int:
    """Learn from capture session directories and write the model (no hardware); exit code."""
    from ..perception.pipeline import Perception
    ext = load_extrinsics_or_none(cfg, ui)
    if ext is None:
        ui.say('--from-dir needs the camera extrinsics of the capture (camera.extrinsics_file, '
               'or --config / --set camera.extrinsics_inline=...) for the geometry features')
        return 2
    try:
        perception = Perception(cfg, classifier=None)
    except ValueError as e:
        ui.say('ERROR: invalid perception config: %s' % e)
        return 2
    T_ft = flange_to_tcp(cfg)
    accepted = accepted_labels(book, known)
    decisions: Dict[str, Optional[str]] = {}
    summaries = []
    for spec in args.from_dir:
        explicit, d = parse_dir_spec(spec)
        summaries.append(learn_dir(d, explicit, perception, ext, T_ft, ui, book, known,
                                   args.task, accepted, decisions))
    if book.n_session == 0:
        ui.say('no samples learned from %s; nothing written' % ', '.join(args.from_dir))
        return 1
    X, y = book.arrays()
    k, md = knn_params(cfg)
    ev = evaluate_samples(X, y, k, md, known, args.task)
    ui.say(ev['text'])
    ok = save_model(ui, cfg, args.task, args.mode, book, out, 'from_dir', known,
                    extra={'dirs': summaries}, evaluation=ev)
    return 0 if ok else 1


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of teach_classes."""
    p = argparse.ArgumentParser(prog='teach_classes', description=__doc__.split('\n\n')[0],
                                epilog=__doc__.split('\n\n', 1)[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, default_task='L2')
    add_onsite_args(p, site=False)
    p.add_argument('--out', default=None, metavar='FILE',
                   help='model file to write (default %s; sim mode: <logging.dir>/sim_site/'
                        '<its name>)' % MODEL_KEY)
    p.add_argument('--append', action='store_true',
                   help='keep the samples of the existing model in the output file')
    p.add_argument('--from-dir', nargs='+', default=None, metavar='[LABEL=]DIR',
                   help='offline: learn from capture session directories (one class each) '
                        'and write the model; no hardware')
    p.add_argument('--sim-scenes', type=int, default=12, metavar='N',
                   help='sim mode: first learn from N simulated ground-truth scenes (0 = none)')
    return p


def _run(args, cfg_world_info, ui) -> int:
    """Pick the offline or the live flow after the common start-up checks."""
    cfg = cfg_world_info[0]
    task = args.task
    known = task_labels(cfg, task)
    out = model_out_path(cfg, args.mode, args.out)
    ui.say('teach_classes: task %s, mode %s, model -> %s' % (task, args.mode, out))
    if known:
        ui.say('labels (tasks.%s.labels): %s' % (task, ', '.join(known)))
    else:
        ui.say('WARNING: tasks.%s.labels is empty, so labels cannot be checked (every new label '
               'is confirmed once); set the list in %s' % (task, overlay_file(cfg, 'tasks.yaml')))
    if args.mode == 'sim' and args.sim_scenes < 0:
        ui.say('--sim-scenes must be >= 0')
        return 2
    book = SampleBook()
    if not load_existing(ui, book, out, args.append):
        return 2
    if args.from_dir:
        return run_from_dirs(args, cfg, ui, book, known, out)
    return run_live(args, cfg_world_info, ui, book, known, out)


def main(argv=None, ui=None) -> int:
    """Run teach_classes; return the process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    cfg_world_info = load_cfg_for(args)
    ui = make_ui(args, cfg_world_info[0], 'teach_classes', ui)
    try:
        return _run(args, cfg_world_info, ui)
    finally:
        ui.close()


if __name__ == '__main__':
    raise SystemExit(main())
