"""
Recorded teach_classes sessions and their offline review.

Recording: every live view is saved as a frame (camera/recorder.py, with the
flange pose for eye_in_hand) plus <i>_instances.json (the numbered valid
instances) into one session directory; labels.json holds the labels per view
and is rewritten whenever a labeling or an undo changes them.

Review (teach_classes --review DIR ...): the recorded frames are segmented
again with the current perception config, the labeled instances are matched
by their pixel center, and every sample gets an id ('<view>#<n>', with several
directories '<k>:<view>#<n>'). A contact sheet per class is written (and
shown) with the leave-one-out suspects marked; labels are fixed or samples
dropped by id, then w writes the model and each directory's labels.json.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.logger import atomic_write_text, to_jsonable

_LOG = logging.getLogger('cube_task.tools.teach_record')

LABELS_FILE = 'labels.json'
MATCH_PX = 25.0           # a re-segmented instance matches a recorded one within this (px)
TILE_PX = 112             # contact sheet tile size
SHEET_COLS = 8
DROP_WORDS = ('x', 'drop', 'del', 'delete', '-')
REVIEW_HELP = ("'<id> <label>' relabel, '<id> x' drop (several: separate with ';'), l list "
               "suspects, r redraw the sheets, e evaluate, w write the model and labels, "
               "q quit")
_ID_RE = re.compile(r'^(?:(\d+):)?(\d+)#(\d+)$')


# ------------------------------------------------------------------ recording
def session_dir(cfg, mode: str, task: Optional[str], stamp: Optional[str] = None) -> Path:
    """Return the recording directory: ~/vision_data/<stamp>_teach_<task> (sim: log dir)."""
    from .onsite import logging_dir
    stamp = stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if mode == 'real':
        return Path('~/vision_data').expanduser() / ('%s_teach_%s' % (stamp, task or 'L2'))
    return logging_dir(cfg) / 'teach_classes' / stamp


def instance_records(dets: Sequence[Any]) -> List[Dict[str, Any]]:
    """Return the JSON records of the numbered valid instances of a view."""
    out = []
    for n, d in enumerate(dets):
        out.append({'n': n, 'center_px': [float(v) for v in d.center_px],
                    'bbox': [int(v) for v in d.bbox],
                    'center_base': None if d.center_base is None
                    else [float(v) for v in np.asarray(d.center_base)[:3]],
                    'size': [None if v is None else float(v)
                             for v in (d.length, d.width, d.height)]})
    return out


def save_view(d: Path, view: int, frame, dets: Sequence[Any]) -> None:
    """Save one live view: the frame (with its flange pose) and its numbered instances."""
    from ..camera.recorder import save_frame
    extra = dict((frame.meta or {}).get('extra') or {})
    extra.update(session='teach_classes', view=int(view))
    save_frame(d, view, frame, extra=extra)
    atomic_write_text(Path(d) / ('%06d_instances.json' % view),
                      json.dumps(instance_records(dets), indent=1) + '\n')


def save_labels(d: Path, labels: Dict[int, Dict[int, str]]) -> Path:
    """Write labels.json ({'views': {view: {n: label}}}) atomically; return its path."""
    views = {str(v): {str(n): str(lab) for n, lab in sorted(m.items())}
             for v, m in sorted(labels.items()) if m}
    path = Path(d) / LABELS_FILE
    atomic_write_text(path, json.dumps({'views': views, 'saved': datetime.datetime.now()
                                        .isoformat(timespec='seconds')}, indent=1) + '\n')
    return path


def load_labels(d: Path) -> Dict[int, Dict[int, str]]:
    """Return {view: {n: label}} of labels.json ({} when the file is missing)."""
    path = Path(d) / LABELS_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return {int(v): {int(n): str(lab) for n, lab in m.items()}
            for v, m in (data.get('views') or {}).items()}


def load_instances(d: Path, view: int) -> List[Dict[str, Any]]:
    """Return the recorded instances of a view ([] when the file is missing)."""
    path = Path(d) / ('%06d_instances.json' % view)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []


def match_instances(recorded: Sequence[Dict[str, Any]], dets: Sequence[Any],
                    max_px: float = MATCH_PX) -> Dict[int, int]:
    """Return {recorded n: index in dets} by the nearest pixel center (one to one)."""
    pairs = []
    for r in recorded:
        u, v = r['center_px']
        for j, d in enumerate(dets):
            du, dv = float(d.center_px[0]) - u, float(d.center_px[1]) - v
            dist = math.hypot(du, dv)
            if dist <= max_px:
                pairs.append((dist, int(r['n']), j))
    out: Dict[int, int] = {}
    used = set()
    for _dist, n, j in sorted(pairs):
        if n not in out and j not in used:
            out[n] = j
            used.add(j)
    return out


# ------------------------------------------------------------------ review samples
@dataclass
class Sample:
    """One labeled instance of a recorded view."""

    sid: str
    dir_index: int
    view: int
    n: int
    label: str
    x: np.ndarray
    crop: np.ndarray
    dropped: bool = False


def sample_id(dir_index: int, view: int, n: int, n_dirs: int) -> str:
    """Return '<view>#<n>' (one directory) or '<k>:<view>#<n>' (k = 1-based directory)."""
    return ('%d#%d' % (view, n)) if n_dirs == 1 else ('%d:%d#%d' % (dir_index + 1, view, n))


def parse_sample_id(text: str, n_dirs: int) -> Optional[Tuple[int, int, int]]:
    """Return (directory index, view, n) of an id, or None when it is malformed."""
    m = _ID_RE.match(str(text).strip())
    if not m:
        return None
    k = int(m.group(1)) - 1 if m.group(1) else 0
    if m.group(1) is None and n_dirs != 1:
        return None
    if not 0 <= k < n_dirs:
        return None
    return k, int(m.group(2)), int(m.group(3))


def crop_of(color: np.ndarray, bbox: Sequence[int], pad: int = 8) -> np.ndarray:
    """Return the padded bbox crop of a BGR image."""
    h, w = color.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((8, 8, 3), np.uint8)
    return np.ascontiguousarray(color[y0:y1, x0:x1])


def load_samples(dirs: Sequence[Path], perception, ext, T_ft: np.ndarray,
                 ui) -> Tuple[List[Sample], Dict[str, int]]:
    """Re-segment the labeled views of the recorded dirs; return (samples, skip counts)."""
    from ..camera.recorder import load_frame
    from .onsite import camera_pose_for_frame
    from .teach_classes import view_features
    import dataclasses
    samples: List[Sample] = []
    skipped: Dict[str, int] = {}

    def skip(why: str, n: int = 1) -> None:
        skipped[why] = skipped.get(why, 0) + n

    for k, d in enumerate(dirs):
        labels = load_labels(d)
        if not labels:
            ui.say('%s: no %s (nothing labeled there)' % (d, LABELS_FILE))
        for view, lab in sorted(labels.items()):
            try:
                frame, extra = load_frame(d, view)
            except (OSError, ValueError) as e:
                _LOG.warning('%s view %d: %s', d, view, e)
                skip('unreadable frame', len(lab))
                continue
            frame = dataclasses.replace(frame, meta=dict(frame.meta or {}, extra=extra))
            T_bc, why = camera_pose_for_frame(ext, frame, None, T_ft)
            if T_bc is None:
                skip('no camera pose', len(lab))
                continue
            perception.reset_table()
            dets = perception.process(frame, T_bc).valid_detections()
            X, usable = view_features(frame, dets)
            match = match_instances(load_instances(d, view), dets)
            for n, label in sorted(lab.items()):
                j = match.get(n)
                if j is None:
                    skip('instance not found again')
                    continue
                if not usable[j]:
                    skip('incomplete features')
                    continue
                samples.append(Sample(sample_id(k, view, n, len(dirs)), k, view, n, label,
                                      X[j].copy(), crop_of(frame.color, dets[j].bbox)))
    return samples, skipped


def loo_predictions(samples: Sequence[Sample], k: int, max_distance: float) -> List[str]:
    """Return the leave-one-out prediction of every kept sample ('' for dropped ones)."""
    from ..perception.classify_knn import KnnClassifier
    keep = [i for i, s in enumerate(samples) if not s.dropped]
    out = [''] * len(samples)
    if len(keep) < 2 or len({samples[i].label for i in keep}) < 2:
        return out
    X = np.vstack([samples[i].x for i in keep])
    y = [samples[i].label for i in keep]
    for row, i in enumerate(keep):
        others = [r for r in range(len(keep)) if r != row]
        knn = KnnClassifier(k=k, max_distance=max_distance).fit(X[others], [y[r] for r in others])
        out[i] = knn.predict(X[row:row + 1])[0][0]
    return out


def suspects(samples: Sequence[Sample], preds: Sequence[str]) -> List[int]:
    """Return the kept samples whose leave-one-out prediction differs from their label."""
    return [i for i, s in enumerate(samples)
            if not s.dropped and preds[i] and preds[i] != s.label]


def contact_sheet(samples: Sequence[Sample], idx: Sequence[int], preds: Sequence[str],
                  title: str, tile: int = TILE_PX, cols: int = SHEET_COLS) -> np.ndarray:
    """Return a grid of the crops idx with their ids; suspects get a red frame + prediction."""
    import cv2
    rows = max(1, int(math.ceil(len(idx) / float(cols))))
    head = 28
    sheet = np.full((head + rows * (tile + 18), cols * tile, 3), 40, np.uint8)
    cv2.putText(sheet, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    for t, i in enumerate(idx):
        s = samples[i]
        r, c = divmod(t, cols)
        y0, x0 = head + r * (tile + 18), c * tile
        img = s.crop
        f = (tile - 4) / float(max(img.shape[:2]))
        img = cv2.resize(img, (max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f))))
        sheet[y0 + 2:y0 + 2 + img.shape[0], x0 + 2:x0 + 2 + img.shape[1]] = img
        bad = bool(preds[i]) and preds[i] != s.label
        if bad:
            cv2.rectangle(sheet, (x0 + 1, y0 + 1), (x0 + tile - 2, y0 + tile - 2), (0, 0, 255), 2)
        text = s.sid + (' ->%s' % preds[i] if bad else '')
        cv2.putText(sheet, text, (x0 + 3, y0 + tile + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 255) if bad else (220, 220, 220), 1, cv2.LINE_AA)
    return sheet


def parse_review_command(text: str, n_dirs: int,
                         ) -> Tuple[Optional[List[Tuple[Tuple[int, int, int], str]]], str]:
    """Parse '<id> <label|x>[; ...]'; return ([(id key, label or '')], '') or (None, error)."""
    out = []
    for part in [p.strip() for p in str(text).split(';') if p.strip()]:
        words = part.split()
        if len(words) != 2:
            return None, '%r: expected <id> <label> or <id> x' % part
        key = parse_sample_id(words[0], n_dirs)
        if key is None:
            form = "'<view>#<n>'" if n_dirs == 1 else "'<k>:<view>#<n>'"
            return None, '%r: bad id (%s)' % (words[0], form)
        out.append((key, '' if words[1].lower() in DROP_WORDS else words[1]))
    if not out:
        return None, 'empty command'
    return out, ''


# ------------------------------------------------------------------ review loop
class Reviewer:
    """State of one review run (thin I/O around the helpers above)."""

    def __init__(self, ui, cfg, task, known, dirs, samples, out_dir: Path):
        from .teach_classes import knn_params
        self.ui, self.cfg, self.task, self.known = ui, cfg, task, list(known)
        self.dirs, self.samples, self.out_dir = list(dirs), list(samples), out_dir
        self.k, self.md = knn_params(cfg)
        self.preds: List[str] = []
        self.dirty = False
        self.by_id = {(s.dir_index, s.view, s.n): i for i, s in enumerate(self.samples)}

    def kept(self) -> List[int]:
        return [i for i, s in enumerate(self.samples) if not s.dropped]

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for i in self.kept():
            c[self.samples[i].label] = c.get(self.samples[i].label, 0) + 1
        return c

    def refresh(self) -> None:
        """Recompute the leave-one-out predictions, write and show the contact sheets."""
        import cv2
        self.preds = loo_predictions(self.samples, self.k, self.md)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        sheets = []
        for label in sorted(self.counts()):
            idx = [i for i in self.kept() if self.samples[i].label == label]
            bad = sum(1 for i in idx if self.preds[i] and self.preds[i] != label)
            sheet = contact_sheet(self.samples, idx, self.preds, '%s: %d samples, %d suspect'
                                  % (label, len(idx), bad))
            cv2.imwrite(str(self.out_dir / ('%s.jpg' % label)), sheet)
            sheets.append(sheet)
        sus = suspects(self.samples, self.preds)
        if sus:
            cv2.imwrite(str(self.out_dir / 'suspects.jpg'),
                        contact_sheet(self.samples, sus, self.preds, 'suspects: %d' % len(sus)))
        if sheets and not self.ui.headless:
            w = max(s.shape[1] for s in sheets)
            tall = np.vstack([np.pad(s, ((0, 0), (0, w - s.shape[1]), (0, 0))) for s in sheets])
            self.ui.show('teach_classes review', tall[:1400])
        self.ui.say('contact sheets: %s (<label>.jpg, suspects.jpg)' % self.out_dir)
        self.list_suspects()

    def list_suspects(self) -> None:
        """Print the samples whose leave-one-out prediction differs from their label."""
        from .teach_classes import counts_text
        sus = suspects(self.samples, self.preds)
        self.ui.say('samples: %s' % counts_text(self.counts(), self.known))
        if not sus:
            self.ui.say('no suspect: every sample agrees with its neighbors')
            return
        items = ['%s %s->%s' % (self.samples[i].sid, self.samples[i].label, self.preds[i])
                 for i in sus]
        self.ui.say('%d suspect(s) (label -> what the others say): %s'
                    % (len(sus), ', '.join(items)))

    def apply(self, text: str) -> bool:
        """Relabel / drop samples by id."""
        from .teach_classes import check_label
        parts, err = parse_review_command(text, len(self.dirs))
        if parts is None:
            self.ui.say('not changed: %s (%s)' % (err, REVIEW_HELP))
            return False
        changes = []
        for key, label in parts:
            i = self.by_id.get(key)
            if i is None:
                self.ui.say('not changed: no sample %s' % sample_id(*key, len(self.dirs)))
                return False
            if label:
                canon, status, close = check_label(label, self.known)
                if status == 'unlisted':
                    self.ui.say('not changed: %r is not a task label%s' % (
                        label, ' (did you mean %s?)' % ' / '.join(close) if close else ''))
                    return False
                label = canon
            changes.append((i, label))
        for i, label in changes:
            s = self.samples[i]
            if label:
                self.ui.say('%s: %s -> %s' % (s.sid, s.label, label))
                s.label, s.dropped = label, False
            else:
                self.ui.say('%s (%s) dropped' % (s.sid, s.label))
                s.dropped = True
        self.dirty = True
        return True

    def evaluate(self) -> None:
        from .teach_classes import evaluate_samples
        X, y = self.arrays()
        self.ui.say(evaluate_samples(X, y, self.k, self.md, self.known, self.task)['text'])

    def arrays(self) -> Tuple[np.ndarray, List[str]]:
        keep = self.kept()
        if not keep:
            from .teach_classes import N_FEATURES
            return np.zeros((0, N_FEATURES)), []
        return np.vstack([self.samples[i].x for i in keep]), [self.samples[i].label
                                                              for i in keep]

    def write_labels(self) -> None:
        """Write each directory's labels.json from the reviewed samples (old file backed up)."""
        from .onsite import backup_file
        for k, d in enumerate(self.dirs):
            old = load_labels(d)
            new: Dict[int, Dict[int, str]] = {}
            for v, m in old.items():
                for n, lab in m.items():
                    i = self.by_id.get((k, v, n))
                    if i is None:
                        new.setdefault(v, {})[n] = lab          # not re-found: kept as is
                    elif not self.samples[i].dropped:
                        new.setdefault(v, {})[n] = self.samples[i].label
            if new != old:
                backup_file(Path(d) / LABELS_FILE)
                save_labels(d, new)
                self.ui.say('%s: %s rewritten' % (d, LABELS_FILE))


def review_summary(samples: Sequence[Sample], skipped: Dict[str, int]) -> str:
    """Return a one-line summary of the loaded samples."""
    skip = ', '.join('%s %d' % kv for kv in sorted(skipped.items()))
    return 'loaded %d labeled samples%s' % (len(samples), '; skipped: %s' % skip if skip else '')


def to_report(samples: Sequence[Sample]) -> List[Dict[str, Any]]:
    """Return the sample ids and labels for the model report."""
    return to_jsonable([{'id': s.sid, 'label': s.label} for s in samples if not s.dropped])
