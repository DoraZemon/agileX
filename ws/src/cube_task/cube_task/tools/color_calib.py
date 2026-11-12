"""
HSV color calibration of the L1 blocks under the venue lighting.

Captures a view of the pick area, finds the object instances without a
classifier and numbers the valid ones. The operator names the color of
each numbered block. The tool collects the HSV pixels of the upper surface
of every labeled instance (features.upper_mask, eroded by --erode-px) and
derives the ranges with classify_color.hsv_ranges_from_samples. It warns
when two classes overlap (fraction of each class's samples inside the
other class's ranges), when a class has samples from fewer than
--min-views views and when a labeled instance would not reach the
classifier threshold. Labels must be in tasks.<task>.labels when that list
is non-empty.

Commands (headless: type them; window: keys in the image window, Enter
opens a terminal command line for commands with arguments):
  <n> <color>   label instance n; several: '0 red; 1 green' or '0,2 red'
  all <color>   label every instance in the view (only one color in view)
  c             capture a new view (move the blocks or change the light first)
  u             undo the last labeling line
  p             preview: classify this view with the ranges computed so far
  s             summary: samples, ranges and warnings
  m H [SV]      set the hue (and S / V) margin and recompute the ranges
  w             write colors.yaml (perception.classifier.color.classes)
  q             quit

Writing keeps the classes not recalibrated in this session unless
--replace and keeps the previous file as a timestamped backup. Sim mode
writes to <logging.dir>/sim_site/colors.yaml, and 'c' simulates the
operator rearranging the blocks. The arm is only read in real mode; with
--observe it first moves to the observe pose after a confirmation.

Example commands
    color_calib
    color_calib --headless --replace
    color_calib --observe --speed 20
    color_calib --mode sim --headless
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .common import add_common_args, load_cfg_for, open_session, setup_logging
from .onsite import (add_depth_range_arg, add_onsite_args, assign_command, build_perception,
                     camera_pose_for_frame, capture_view, compose_view, connect_arm_readonly,
                     ensure_motion_ready, flange_to_tcp, format_table, go_observe,
                     ground_truth_label, load_extrinsics_or_none, make_ui, MotionGuard,
                     next_command, operator_action, output_path, overlay_file, parse_floats,
                     read_yaml_or_empty, resolve_speed, sim_relayout, text_panel, wall_stamp,
                     write_report, write_site_yaml)
from ..perception.classify_color import ColorClassifier, hsv_ranges_from_samples, in_ranges
from ..perception.features import bgr_to_hsv_pixels, instance_pixels, upper_mask

_LOG = logging.getLogger('cube_task.tools.color_calib')

CLASSES_KEY = 'perception.classifier.color.classes'
#: a labeled instance needs at least this many upper-surface pixels.
MIN_INSTANCE_PIXELS = 20
#: hue spread (OpenCV units, 5-95 % of the saturated pixels) above which an
#: instance probably holds two colors (touching blocks).
MIXED_HUE_SPREAD = 30.0
#: an instance whose mean hue is this far from the other samples of its label.
LABEL_HUE_JUMP = 20.0
#: a new view whose instance centers all lie this close to the previous view's (m).
SAME_LAYOUT_TOL = 0.01

COMMAND_WORDS = {
    'c': 'capture', 'capture': 'capture', 'u': 'undo', 'undo': 'undo',
    'p': 'preview', 'preview': 'preview', 's': 'summary', 'summary': 'summary',
    'w': 'write', 'write': 'write', 'q': 'quit', 'quit': 'quit', 'exit': 'quit',
    'esc': 'quit', 'h': 'help', 'help': 'help', '?': 'help',
}
HEADLESS_HELP = ("'<n> <color>' (';' between several), 'all <color>', c capture, u undo, "
                 "p preview, s summary, m H [SV] margins, w write, q quit, h help")
WINDOW_HELP = ('keys in the image window: c capture, u undo, p preview, s summary, w write, '
               'q quit; Enter = type a command (0 red; 1 green | all red | m 3 20)')
HELP_TEXT = '\n'.join([
    'commands:',
    "  <n> <color>   label instance n; several: '0 red; 1 green' or '0,2 red'",
    '  all <color>   label every instance in the view (only one color in view)',
    '  c             capture a new view (move the blocks or change the light first)',
    '  u             undo the last labeling line',
    '  p             preview: classify this view with the ranges computed so far',
    '  s             summary: samples, ranges and warnings',
    '  m H [SV]      set the hue (and S / V) margin and recompute the ranges',
    '  w             write the ranges (the previous file is kept as a backup)',
    '  q             quit',
    '  Enter         (headless) show the current view again'])

VIEW_HEADERS = ('#', 'pixels', 'hue', 'spread', 'S', 'V', 'LxW mm', 'label')
SUMMARY_HEADERS = ('label', 'instances', 'views', 'pixels', 'worst %', 'ranges')
PREVIEW_HEADERS = ('#', 'assigned', 'predicted', 'fraction', 'runner-up')


# ------------------------------------------------------------------ options / commands
@dataclasses.dataclass
class RangeOptions:
    """Options of classify_color.hsv_ranges_from_samples (same names and meaning)."""

    lo_pct: float = 2.0
    hi_pct: float = 98.0
    hue_margin: float = 4.0
    sv_margin: float = 25.0
    min_saturation: int = 40
    min_chromatic_fraction: float = 0.3

    def validate(self) -> 'RangeOptions':
        """Return self; raise ValueError when an option is out of range."""
        if not 0.0 <= self.lo_pct < self.hi_pct <= 100.0:
            raise ValueError('need 0 <= --lo-pct < --hi-pct <= 100')
        if not 0.0 <= self.hue_margin <= 45.0:
            raise ValueError('the hue margin must be in [0, 45] (OpenCV hue units of 2 deg)')
        if not 0.0 <= self.sv_margin <= 255.0:
            raise ValueError('the S / V margin must be in [0, 255]')
        if not 0 <= int(self.min_saturation) <= 255:
            raise ValueError('--min-saturation must be in [0, 255]')
        if not 0.0 <= self.min_chromatic_fraction <= 1.0:
            raise ValueError('--min-chromatic-fraction must be in [0, 1]')
        return self

    def kwargs(self) -> Dict[str, Any]:
        """Return the keyword arguments for hsv_ranges_from_samples."""
        d = dataclasses.asdict(self)
        d['min_saturation'] = int(d['min_saturation'])
        return d

    def text(self) -> str:
        """Return a one-line description (operator output and file header)."""
        return ('percentiles %g-%g, hue margin %g, S/V margin %g, min saturation %d'
                % (self.lo_pct, self.hi_pct, self.hue_margin, self.sv_margin,
                   int(self.min_saturation)))


@dataclasses.dataclass
class Command:
    """One parsed operator command (see parse_command)."""

    kind: str
    pairs: List[Tuple[int, str]] = dataclasses.field(default_factory=list)
    values: List[float] = dataclasses.field(default_factory=list)
    error: str = ''


def normalize_line(text: Optional[str]) -> str:
    """Return text stripped, with full-width ';' ',' and spaces (Chinese input) as ASCII."""
    s = str(text or '')
    for a, b in (('；', ';'), ('，', ','), ('　', ' ')):
        s = s.replace(a, b)
    return s.strip()


def match_label(label: str, allowed: Sequence[str],
                labels_key: str = 'tasks.L1.labels') -> Tuple[Optional[str], str]:
    """
    Check a typed label against the task labels (case-insensitive).

    Returns (label as configured, '') or (None, error). An empty allowed
    list accepts any label as typed.
    """
    if not allowed:
        return label, ''
    if label in allowed:
        return label, ''
    low = {str(a).lower(): str(a) for a in allowed}
    if label.lower() in low:
        return low[label.lower()], ''
    return None, 'label %r is not in %s (%s)' % (label, labels_key, ', '.join(allowed))


def _parse_margins(tokens: Sequence[str]) -> Command:
    vals = parse_floats(' '.join(tokens[1:]), len(tokens) - 1) if len(tokens) in (2, 3) else None
    if vals is None or min(vals) < 0.0:
        return Command('error', error='usage: m <hue margin> [<S/V margin>] (numbers >= 0)')
    return Command('margins', values=vals)


def parse_command(text: Optional[str], n_instances: int, allowed: Sequence[str] = (),
                  labels_key: str = 'tasks.L1.labels') -> Command:
    """
    Parse one operator command line.

    Returns a Command whose kind is 'assign' (pairs = sorted [(instance,
    label)]), 'capture', 'undo', 'preview', 'summary', 'margins' (values =
    [hue] or [hue, sv]), 'write', 'quit', 'help', 'refresh' (empty line) or
    'error' (error = message). Labeling: '<n> <label>', '<n>,<m> <label>' or
    'all <label>', several separated by ';'. The whole line is rejected when
    one part is invalid, so a typo never labels half of the instances.
    """
    line = normalize_line(text)
    if not line:
        return Command('refresh')
    tokens = line.split()
    head = tokens[0].lower()
    if head in ('m', 'margin', 'margins'):
        return _parse_margins(tokens)
    if len(tokens) == 1 and ';' not in line:
        kind = COMMAND_WORDS.get(head)
        if kind is None:
            return Command('error', error="unknown command %r: label with '<n> <color>' or "
                                          "'all <color>', h = help" % tokens[0])
        return Command(kind)
    seen: Dict[int, str] = {}
    for part in line.split(';'):
        part = part.strip()
        if not part:
            continue
        idx, label, err = assign_command(part, n_instances)
        if idx is None:
            return Command('error', error='%r: %s' % (part, err))
        name, err = match_label(label, allowed, labels_key)
        if name is None:
            return Command('error', error=err)
        for i in idx:
            if seen.get(i, name) != name:
                return Command('error', error='instance %d labeled twice (%s and %s)'
                               % (i, seen[i], name))
            seen[i] = name
    if not seen:
        return Command('error', error='nothing to label (h = help)')
    return Command('assign', pairs=sorted(seen.items()))


# ------------------------------------------------------------------ pixel statistics
def circular_mean_hue(h: Any) -> float:
    """Return the circular mean of OpenCV hues (0-180 wraps) in [0, 180); NaN when empty."""
    ang = np.asarray(h, dtype=np.float64).ravel() * (2.0 * math.pi / 180.0)
    if ang.size == 0:
        return float('nan')
    m = math.atan2(float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang))))
    return (m * 180.0 / (2.0 * math.pi)) % 180.0


def hue_distance(a: float, b: float) -> float:
    """Return the circular distance of two OpenCV hues (0..90)."""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def hue_stats(hsv: Any, min_saturation: int = 40,
              min_chromatic_fraction: float = 0.3) -> Dict[str, Any]:
    """
    Summarize Nx3 OpenCV HSV pixels.

    Returns {'n', 'hue' (circular mean of the pixels with S >=
    min_saturation), 'spread' (5-95 % width of their hue around that mean),
    's', 'v' (medians)}. hue / spread are None for an achromatic sample
    (fewer than min_chromatic_fraction saturated pixels, the same rule as
    hsv_ranges_from_samples); s / v are None without pixels.
    """
    px = np.asarray(hsv).reshape(-1, 3)
    n = int(px.shape[0])
    out: Dict[str, Any] = {'n': n, 'hue': None, 'spread': None, 's': None, 'v': None}
    if n == 0:
        return out
    out['s'] = float(np.median(px[:, 1]))
    out['v'] = float(np.median(px[:, 2]))
    chrom = px[:, 1].astype(np.int32) >= int(min_saturation)
    if np.count_nonzero(chrom) < max(10, float(min_chromatic_fraction) * n):
        return out
    h = px[chrom, 0].astype(np.float64)
    mu = circular_mean_hue(h)
    lo, hi = np.percentile((h - mu + 90.0) % 180.0, [5.0, 95.0])
    out['hue'] = mu
    out['spread'] = float(hi - lo)
    return out


def _eroded_upper_pixels(frame: Any, det: Any, erode_px: int) -> np.ndarray:
    """Return the BGR pixels of features.upper_mask eroded by erode_px (fallback: uneroded)."""
    empty = np.zeros((0, 3), dtype=np.uint8)
    color = getattr(frame, 'color', None)
    if color is None or color.ndim != 3 or color.shape[2] != 3:
        return empty
    m = upper_mask(det)
    if m is None:
        return empty
    x0, y0, x1, y1 = (int(v) for v in det.bbox)
    crop = color[y0:y1, x0:x1]
    if crop.shape[:2] != m.shape:
        return empty
    k = 2 * int(erode_px) + 1
    er = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8),
                   borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if np.count_nonzero(er) >= MIN_INSTANCE_PIXELS:
        m = er
    return np.ascontiguousarray(crop[m], dtype=np.uint8)


def instance_hsv(frame: Any, det: Any, erode_px: int = 2, max_pixels: int = 4000) -> np.ndarray:
    """
    Return the Nx3 uint8 HSV pixels of an instance's upper surface.

    Uses features.upper_mask (features.instance_pixels when erode_px is 0)
    eroded by erode_px pixels, which drops edge pixels mixed with the table
    or a neighbor (the uneroded mask is used when fewer than
    MIN_INSTANCE_PIXELS remain). Keeps at most max_pixels evenly spaced
    pixels so one large or near instance does not dominate the percentiles.
    """
    if int(erode_px) <= 0:
        bgr = instance_pixels(frame, det, upper=True)
    else:
        bgr = _eroded_upper_pixels(frame, det, int(erode_px))
    hsv = bgr_to_hsv_pixels(bgr)
    n = int(hsv.shape[0])
    if max_pixels and n > int(max_pixels):
        hsv = hsv[np.linspace(0, n - 1, int(max_pixels)).round().astype(np.int64)]
    return hsv


@dataclasses.dataclass
class InstanceSample:
    """HSV pixels of one labeled instance (view number, instance number in that view)."""

    view: int
    index: int
    label: str
    hsv: np.ndarray


class SampleBook:
    """
    Labeled instance samples with undo (pure state, no I/O).

    One sample per (view, instance): labeling an instance again replaces its
    label. undo() reverts the last assign() call (one command line).
    """

    def __init__(self) -> None:
        self.samples: Dict[Tuple[int, int], InstanceSample] = {}
        self._history: List[List[Tuple[Tuple[int, int], Optional[InstanceSample]]]] = []

    def assign(self, view: int, items: Sequence[Tuple[int, str, Any]]) -> List[str]:
        """Label instances of a view, items = [(instance, label, Nx3 HSV)]; return notes."""
        record: List[Tuple[Tuple[int, int], Optional[InstanceSample]]] = []
        notes: List[str] = []
        for idx, label, hsv in items:
            key = (int(view), int(idx))
            prev = self.samples.get(key)
            record.append((key, prev))
            px = np.asarray(hsv, dtype=np.uint8).reshape(-1, 3)
            self.samples[key] = InstanceSample(key[0], key[1], str(label), px)
            if prev is not None and prev.label != label:
                notes.append('#%d was %s, now %s' % (key[1], prev.label, label))
        if record:
            self._history.append(record)
        return notes

    def undo(self) -> Optional[List[str]]:
        """Revert the last assign(); return what was undone, None when there is nothing."""
        if not self._history:
            return None
        done = []
        for key, prev in self._history.pop():
            cur = self.samples.pop(key, None)
            text = 'view %d #%d %s' % (key[0], key[1], cur.label if cur is not None else '?')
            if prev is not None:
                self.samples[key] = prev
                text += ' (back to %s)' % prev.label
            done.append(text)
        return done

    def label_of(self, view: int, index: int) -> Optional[str]:
        """Return the label of instance index of view, or None."""
        s = self.samples.get((int(view), int(index)))
        return None if s is None else s.label

    def labels(self) -> List[str]:
        """Return the labels that have samples, sorted."""
        return sorted({s.label for s in self.samples.values()})

    def by_label(self, label: str) -> List[InstanceSample]:
        """Return the samples of label ordered by (view, instance)."""
        return [s for _k, s in sorted(self.samples.items()) if s.label == label]

    def pixels_by_label(self) -> Dict[str, np.ndarray]:
        """Return label -> all its HSV pixels (Nx3 uint8)."""
        return {lab: np.concatenate([s.hsv for s in self.by_label(lab)], axis=0)
                for lab in self.labels()}

    def views_by_label(self) -> Dict[str, List[int]]:
        """Return label -> sorted view numbers it has samples from."""
        return {lab: sorted({s.view for s in self.by_label(lab)}) for lab in self.labels()}

    def state(self) -> Tuple[Tuple[int, int, str], ...]:
        """Return a hashable snapshot of the labels (to detect unsaved changes)."""
        return tuple((k[0], k[1], s.label) for k, s in sorted(self.samples.items()))


# ------------------------------------------------------------------ ranges / overlap
def compute_ranges(pixels_by_label: Mapping[str, Any], opts: RangeOptions
                   ) -> Tuple[Dict[str, list], Dict[str, str]]:
    """
    Return (label -> ranges, label -> error) from hsv_ranges_from_samples.

    Each label is computed on its own, so a label with too few pixels does
    not block the others.
    """
    ranges: Dict[str, list] = {}
    errors: Dict[str, str] = {}
    for label, px in pixels_by_label.items():
        try:
            ranges.update(hsv_ranges_from_samples({label: px}, **opts.kwargs()))
        except ValueError as e:
            errors[str(label)] = str(e)
    return ranges, errors


def ranges_array(ranges: Any) -> Optional[np.ndarray]:
    """Return ranges as a Kx2x3 float array, or None when they are malformed."""
    try:
        arr = np.asarray(ranges, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 2 and arr.shape == (2, 3):
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1:] != (2, 3) or arr.shape[0] < 1:
        return None
    return arr if np.all(np.isfinite(arr)) else None


def inside_fraction(hsv: Any, ranges: Any) -> float:
    """Return the fraction of Nx3 HSV pixels inside ranges (0 without pixels / bad ranges)."""
    px = np.asarray(hsv).reshape(-1, 3)
    arr = ranges_array(ranges)
    if px.shape[0] == 0 or arr is None:
        return 0.0
    return float(np.count_nonzero(in_ranges(px, arr))) / float(px.shape[0])


def overlap_fractions(samples: Mapping[str, Any], ranges: Mapping[str, Any]
                      ) -> Dict[str, Dict[str, float]]:
    """
    Return out[a][b] = fraction of class a's samples inside class b's ranges.

    a runs over the classes with samples, b over the other classes in ranges
    (calibrated now or kept from the config).
    """
    out: Dict[str, Dict[str, float]] = {}
    for a, px in samples.items():
        if np.asarray(px).size == 0:
            continue
        out[a] = {b: inside_fraction(px, rr) for b, rr in ranges.items() if b != a}
    return out


def overlap_pairs(fractions: Mapping[str, Mapping[str, float]], threshold: float = 0.05
                  ) -> List[Tuple[str, str, float, Optional[float]]]:
    """
    Return the class pairs whose overlap reaches threshold, worst first.

    Each entry is (a, b, fraction of a's samples inside b's ranges, fraction
    of b's samples inside a's ranges or None when b has no samples). When
    both classes have samples, a < b alphabetically.
    """
    out = []
    done = set()
    for a in sorted(fractions):
        for b in sorted(fractions[a]):
            if (b, a) in done:
                continue
            done.add((a, b))
            f_ab = float(fractions[a][b])
            f_ba = fractions.get(b, {}).get(a)
            worst = max(f_ab, 0.0 if f_ba is None else float(f_ba))
            if worst >= threshold:
                out.append((a, b, f_ab, None if f_ba is None else float(f_ba)))
    out.sort(key=lambda p: (-max(p[2], p[3] or 0.0), p[0], p[1]))
    return out


def overlap_warnings(pairs: Sequence[Tuple[str, str, float, Optional[float]]]) -> List[str]:
    """Return one warning per overlapping class pair (see overlap_pairs)."""
    out = []
    for a, b, f_ab, f_ba in pairs:
        if f_ba is None:
            other = '%s has no samples in this session (its ranges come from the config)' % b
        else:
            other = '%.0f %% of the %s samples inside the %s ranges' % (100.0 * f_ba, b, a)
        advice = ('label %s blocks too' % b) if f_ba is None else 'capture more views'
        out.append('%s / %s overlap: %.0f %% of the %s samples inside the %s ranges, %s. They '
                   'may be confused: check the labels, %s, or narrow the ranges '
                   '(m <hue margin>, --lo-pct / --hi-pct)'
                   % (a, b, 100.0 * f_ab, a, b, other, advice))
    return out


def view_warnings(views_by_label: Mapping[str, Sequence[int]], min_views: int = 2) -> List[str]:
    """Return a warning for each label with samples from fewer than min_views views."""
    out = []
    for lab in sorted(views_by_label):
        n = len(set(views_by_label[lab]))
        if n < int(min_views):
            out.append('%s: samples from %d view%s only (want %d): move the %s blocks or change '
                       'the light, press c and label them again'
                       % (lab, n, '' if n == 1 else 's', int(min_views), lab))
    return out


def weak_instance_warnings(samples: Sequence[InstanceSample], ranges: Mapping[str, Any],
                           threshold: float) -> List[str]:
    """Return a warning for each sample with less than threshold of its pixels in its ranges."""
    out = []
    for s in samples:
        rr = ranges.get(s.label)
        if rr is None:
            continue
        f = inside_fraction(s.hsv, rr)
        if f < threshold:
            out.append('view %d #%d (%s): only %.0f %% of its pixels are inside the %s ranges '
                       '(the classifier needs %.0f %%): wrong label or unusual light?'
                       % (s.view, s.index, s.label, 100.0 * f, s.label, 100.0 * threshold))
    return out


def effective_classes(base: Optional[Mapping[str, Any]], new: Optional[Mapping[str, Any]],
                      allowed: Sequence[str] = ()) -> Dict[str, Any]:
    """
    Return the classes the color classifier would use after writing.

    base: classes of the loaded config; new: ranges computed now (they win).
    Restricted to allowed (tasks.<task>.labels) when that is non-empty, like
    perception.factory.make_classifier.
    """
    eff = dict(base or {})
    eff.update(new or {})
    if allowed:
        return {a: eff[a] for a in allowed if a in eff}
    return eff


def missing_label_warnings(allowed: Sequence[str], classes: Mapping[str, Any]) -> List[str]:
    """Return a warning when task labels have no ranges (the color classifier refuses them)."""
    missing = [a for a in allowed if a not in classes]
    if not missing:
        return []
    return ['labels without HSV ranges: %s; the color classifier refuses to start until they '
            'are calibrated' % ', '.join(missing)]


# ------------------------------------------------------------------ colors.yaml
class _FlowList(list):
    """A list that YAML writes in flow style, like [[0, 90, 60], [8, 255, 255]]."""


def _represent_flow(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', list(data), flow_style=True)


# Only affects _FlowList values: one class's ranges stay on one readable line.
yaml.SafeDumper.add_representer(_FlowList, _represent_flow)


def plain_ranges(ranges: Any) -> List[List[List[int]]]:
    """Return ranges as nested Python int lists (Kx2x3); raise ValueError when malformed."""
    arr = ranges_array(ranges)
    if arr is None:
        raise ValueError('ranges must be a list of [[h, s, v], [h, s, v]], got %r' % (ranges,))
    return [[[int(round(v)) for v in row] for row in r] for r in arr]


def build_colors_yaml(existing: Optional[Mapping[str, Any]], new: Mapping[str, Any],
                      replace: bool = False
                      ) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    """
    Merge newly computed class ranges into the content of an existing colors.yaml.

    Returns (data to write, dotted keys written, classes kept from the
    existing file, classes removed). Other keys of the existing file stay.
    replace=True drops the existing classes that are not in new.
    """
    data: Dict[str, Any] = copy.deepcopy(dict(existing)) if isinstance(existing, Mapping) else {}
    node = data
    for k in CLASSES_KEY.split('.')[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    old = node.get('classes')
    old = dict(old) if isinstance(old, dict) else {}
    others = [str(k) for k in old if str(k) not in new]
    classes: Dict[str, Any] = {}
    if not replace:
        for k, v in old.items():
            classes[str(k)] = _FlowList(v) if isinstance(v, list) else v
    for k, v in new.items():
        classes[str(k)] = _FlowList(plain_ranges(v))
    node['classes'] = classes
    written = ['%s.%s' % (CLASSES_KEY, k) for k in new]
    return data, written, ([] if replace else others), (others if replace else [])


def overridden_classes(loaded: Any, written: Mapping[str, Any]) -> List[str]:
    """Return the written classes whose ranges differ in a reloaded config (overridden)."""
    loaded = loaded if isinstance(loaded, Mapping) else {}
    out = []
    for k, v in written.items():
        try:
            same = plain_ranges(loaded.get(k)) == plain_ranges(v)
        except ValueError:
            same = False
        if not same:
            out.append(str(k))
    return out


# ------------------------------------------------------------------ formatting
def fmt_ranges(ranges: Any) -> str:
    """Return ranges as 'H 170-180|0-8  S 90-255  V 60-255' (hue parts joined by '|')."""
    arr = ranges_array(ranges)
    if arr is None:
        return 'invalid ranges %r' % (ranges,)
    groups: List[Tuple[str, List[str]]] = []
    for lo, hi in arr:
        sv = 'S %d-%d  V %d-%d' % (lo[1], hi[1], lo[2], hi[2])
        h = '%d-%d' % (lo[0], hi[0])
        if groups and groups[-1][0] == sv:
            groups[-1][1].append(h)
        else:
            groups.append((sv, [h]))
    return ' ; '.join('H %s  %s' % ('|'.join(hs), sv) for sv, hs in groups)


def same_layout(prev_xy: Optional[Sequence[Sequence[float]]], xy: Sequence[Sequence[float]],
                tol: float = SAME_LAYOUT_TOL) -> bool:
    """Return True when both views hold as many instances, each within tol of one in prev."""
    if prev_xy is None:
        return False
    a = np.asarray(prev_xy, dtype=float).reshape(-1, 2)
    b = np.asarray(xy, dtype=float).reshape(-1, 2)
    if a.shape[0] == 0 or a.shape[0] != b.shape[0]:
        return False
    d = np.linalg.norm(b[:, None, :] - a[None, :, :], axis=2)
    return bool(np.all(d.min(axis=1) <= tol))


def calib_header(stamp: str, task: Optional[str], mode: str, n_views: int,
                 opts: RangeOptions, book: SampleBook, ranges: Mapping[str, Any]) -> str:
    """Return the comment block written at the top of colors.yaml."""
    views = book.views_by_label()
    lines = ['written by color_calib %s (task %s, mode %s, %d views)'
             % (stamp, task, mode, n_views),
             'HSV ranges, OpenCV scale: H 0-180 (wraps around), S / V 0-255; %s' % opts.text()]
    for lab in sorted(ranges):
        inst = book.by_label(lab)
        lines.append('%s: %d instances from %d views, %d px: %s'
                     % (lab, len(inst), len(views.get(lab, [])),
                        sum(int(s.hsv.shape[0]) for s in inst), fmt_ranges(ranges[lab])))
    return '\n'.join(lines)


def _num(v: Optional[float], fmt: str = '%.0f') -> str:
    return '-' if v is None else fmt % v


def view_rows(dets: Sequence[Any], hsv_list: Sequence[Any], labels: Sequence[Optional[str]],
              opts: RangeOptions, truth: Optional[Sequence[Optional[str]]] = None
              ) -> List[List[str]]:
    """Return VIEW_HEADERS rows (+ 'sim truth' when truth is given) for a view's instances."""
    rows = []
    for i, (d, px) in enumerate(zip(dets, hsv_list)):
        st = hue_stats(px, opts.min_saturation, opts.min_chromatic_fraction)
        spread = _num(st['spread'])
        if st['spread'] is not None and st['spread'] > MIXED_HUE_SPREAD:
            spread += ' !'
        size = '-'
        if d.length is not None and d.width is not None:
            size = '%.0fx%.0f' % (d.length * 1e3, d.width * 1e3)
        hue = None if st['hue'] is None else round(st['hue']) % 180
        row = [str(i), str(st['n']), _num(hue), spread, _num(st['s']), _num(st['v']),
               size, labels[i] or '-']
        if truth is not None:
            row.append(truth[i] or '?')
        rows.append(row)
    return rows


def summary_rows(book: SampleBook, ranges: Mapping[str, Any],
                 errors: Mapping[str, str]) -> List[List[str]]:
    """Return SUMMARY_HEADERS rows: per label instances, views, pixels, worst %, ranges."""
    rows = []
    views = book.views_by_label()
    for lab in book.labels():
        inst = book.by_label(lab)
        rr = ranges.get(lab)
        if rr is None:
            worst, text = '-', 'not computed: %s' % errors.get(lab, 'unknown error')
        else:
            worst = '%.0f' % (100.0 * min(inside_fraction(s.hsv, rr) for s in inst))
            text = fmt_ranges(rr)
        rows.append([lab, str(len(inst)), ','.join(str(v) for v in views[lab]),
                     str(sum(int(s.hsv.shape[0]) for s in inst)), worst, text])
    return rows


def preview_rows(dets: Sequence[Any], assigned: Sequence[Optional[str]], min_score: float,
                 truth: Optional[Sequence[Optional[str]]] = None
                 ) -> Tuple[List[List[str]], List[str], int]:
    """
    Return (PREVIEW_HEADERS rows, problems, instances checked) for classified dets.

    dets were labeled by ColorClassifier.classify (label, score,
    extra['color_fractions']); the label becomes 'unknown' below min_score
    like the pipeline's min_score filter. An instance is checked against its
    assigned label, else against truth (sim); problems lists the mismatches.
    """
    rows, problems, checked = [], [], 0
    for i, d in enumerate(dets):
        fr = dict(d.extra.get('color_fractions') or {})
        ranked = sorted(fr.items(), key=lambda kv: (-kv[1], kv[0]))
        score = float(d.score)
        pred = d.label if d.label != 'unknown' and score >= min_score else 'unknown'
        second = '%s %.2f' % ranked[1] if len(ranked) > 1 else '-'
        row = [str(i), assigned[i] or '-', pred, '%.2f' % score, second]
        want = assigned[i]
        if truth is not None:
            row.append(truth[i] or '?')
            want = want or truth[i]
        if want is not None:
            checked += 1
            if pred != want:
                problems.append('#%d: expected %s, classified %s (fraction %.2f)'
                                % (i, want, pred, score))
        rows.append(row)
    return rows, problems, checked


def draw_texts(img: np.ndarray, dets: Sequence[Any], texts: Sequence[str],
               colors: Optional[Sequence[Tuple[int, int, int]]] = None) -> np.ndarray:
    """Write texts[i] next to dets[i]'s image center (in place); return img."""
    for i, (d, t) in enumerate(zip(dets, texts)):
        u, v = d.center_px
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        col = colors[i] if colors is not None else (0, 255, 255)
        org = (int(u) + 8, int(v) - 8)
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ session
@dataclasses.dataclass
class _View:
    """One captured view: frame, perception result, numbered instances and their pixels."""

    vid: int
    frame: Any
    result: Any
    dets: List[Any]
    hsv: List[np.ndarray]
    centers: List[List[float]]
    note: str = ''


class _ColorCalib:
    """State of one color_calib run (thin I/O around the pure helpers)."""

    def __init__(self, args, s, ui, opts: RangeOptions, allowed: Sequence[str], out: Path):
        self.args, self.s, self.ui, self.opts = args, s, ui, opts
        self.cfg = s.cfg
        self.allowed = [str(a) for a in allowed]
        self.labels_key = 'tasks.%s.labels' % s.task
        self.out = Path(out)
        self.book = SampleBook()
        self.ext = load_extrinsics_or_none(s.cfg, ui)
        self.perception = build_perception(s.cfg, s.task, args, ui, classify=False)
        self.T_ft = flange_to_tcp(s.cfg)
        base = s.cfg.get(CLASSES_KEY, {})
        self.base_classes = dict(base) if isinstance(base, dict) else {}
        self.min_fraction = float(s.cfg.get('perception.classifier.color.min_fraction', 0.35))
        self.min_score = float(s.cfg.get('perception.classifier.min_score', 0.55))
        self.threshold = max(self.min_fraction, self.min_score)
        self.view: Optional[_View] = None
        self.n_views = 0
        self.last_stamp: Optional[float] = None
        self.live_stamp: Optional[float] = None
        self.writes = 0
        self.saved = self.state()

    def state(self) -> tuple:
        """Return what decides the written file (labels and range options)."""
        return self.book.state(), dataclasses.astuple(self.opts)

    def unsaved(self) -> bool:
        """Return True when the labeled samples changed since the last write."""
        return bool(self.book.samples) and self.state() != self.saved

    def n_instances(self) -> int:
        """Return the number of numbered instances in the current view."""
        return 0 if self.view is None else len(self.view.dets)

    def truth(self) -> Optional[List[Optional[str]]]:
        """Sim: return the ground-truth label of each instance in the view, else None."""
        if self.s.world is None or self.view is None:
            return None
        return [ground_truth_label(self.s.world, d) for d in self.view.dets]

    # -------------------------------------------------------------- views

    def capture(self) -> bool:
        """Grab a view, number its valid instances, print and show it; False without frame."""
        frame = capture_view(self.s.camera, self.cfg, self.args.average, self.last_stamp)
        if frame is None:
            return False
        self.last_stamp = float(frame.stamp)
        T_bc, why = camera_pose_for_frame(self.ext, frame, self.s.arm, self.T_ft)
        res = self.perception.process(frame, T_bc)
        dets = [d for d in res.detections if d.valid]
        hsv = [instance_hsv(frame, d, self.args.erode_px, self.args.max_pixels) for d in dets]
        centers = [[float(d.center_base[0]), float(d.center_base[1])] for d in dets
                   if d.center_base is not None]
        prev = None if self.view is None else self.view.centers
        self.view = _View(self.n_views, frame, res, dets, hsv, centers,
                          '' if T_bc is not None else 'no camera pose: %s' % why)
        self.n_views += 1
        self.print_view()
        self.show_view()
        if same_layout(prev, centers):
            self.ui.say('note: the blocks are where they were in the previous view. A useful new '
                        'view has them moved or the light changed (ignore this if you changed '
                        'the light).')
        return True

    def new_view(self) -> None:
        """Capture a new view; sim: the simulated operator rearranges the blocks first."""
        if self.s.mode == 'sim' and self.s.world is not None:
            seed = int(self.args.seed) + 1000 + self.n_views
            operator_action(self.ui, 'sim', 'rearrange the blocks (layout of scenario seed %d)'
                            % seed, sim_action=lambda: sim_relayout(
                                self.s.world, self.s.task, seed, self.args.mount) > 0)
        if not self.capture():
            self.ui.say('no camera frame; the previous view stays')

    def print_view(self) -> None:
        """Print the numbered instance table of the current view."""
        v = self.view
        rejected: Dict[str, int] = {}
        for d in v.result.detections:
            if not d.valid:
                r = d.reject_reason or 'invalid'
                rejected[r] = rejected.get(r, 0) + 1
        rej = ''
        if rejected:
            rej = ' (rejected, not numbered: %s)' % ', '.join(
                '%s x%d' % kv for kv in sorted(rejected.items()))
        self.ui.say('view %d: %d instances%s' % (v.vid, len(v.dets), rej))
        if v.note:
            self.ui.say(v.note)
        if v.result.warnings:
            self.ui.say('perception warnings: %s' % ', '.join(v.result.warnings))
        if not v.dets:
            self.ui.say('no valid instances: put the blocks inside the pick area '
                        '(perception.roi.pick_polygon), away from the zones, then press c')
            return
        truth = self.truth()
        labels = [self.book.label_of(v.vid, i) for i in range(len(v.dets))]
        headers = VIEW_HEADERS + (('sim truth',) if truth is not None else ())
        self.ui.say(format_table(headers, view_rows(v.dets, v.hsv, labels, self.opts, truth)))
        if truth is not None:
            self.ui.say('[sim] ground truth: %s' % ', '.join(
                '#%d %s' % (i, t or '?') for i, t in enumerate(truth)))

    def show_view(self) -> None:
        """Show the view image: overlay with '#n label' | depth, status panel below."""
        v = self.view
        base = v.result.debug_image if v.result.debug_image is not None else v.frame.color
        overlay = np.array(base, copy=True)
        labels = [self.book.label_of(v.vid, i) for i in range(len(v.dets))]
        draw_texts(overlay, v.dets, ['#%d %s' % (i, lab) if lab else '#%d' % i
                                     for i, lab in enumerate(labels)])
        counts = ', '.join('%s x%d' % (lab, len(self.book.by_label(lab)))
                           for lab in self.book.labels())
        lines = ['view %d: %d instances | samples: %s' % (v.vid, len(v.dets), counts or 'none'),
                 HEADLESS_HELP if self.ui.headless else WINDOW_HELP]
        img = compose_view(v.frame, overlay, self.args.depth_range)
        self.ui.show('color_calib', text_panel(img, lines))

    def live(self) -> None:
        """Window mode, no key pressed: refresh the live camera window."""
        if self.ui.headless:
            return
        f = self.s.camera.get_frame(timeout=0.05, newer_than=self.live_stamp)
        if f is None:
            return
        self.live_stamp = float(f.stamp)
        img = compose_view(f, None, self.args.depth_range)
        self.ui.show('color_calib live', text_panel(
            img, ['live camera: arrange the blocks, then press c in the color_calib window']))

    # -------------------------------------------------------------- labeling
    def assign(self, pairs: Sequence[Tuple[int, str]]) -> None:
        """Label instances of the current view (the whole line or nothing)."""
        v = self.view
        items, problems, notes = [], [], []
        for idx, label in pairs:
            px = v.hsv[idx]
            if px.shape[0] < MIN_INSTANCE_PIXELS:
                problems.append('#%d has only %d usable pixels (need %d): is it a block? Capture '
                                'again with it nearer the image center'
                                % (idx, px.shape[0], MIN_INSTANCE_PIXELS))
                continue
            items.append((idx, label, px))
            notes.extend(self._label_checks(idx, label, px))
        if problems:
            self.ui.say('nothing labeled:')
            for p in problems:
                self.ui.say('  ' + p)
            return
        changed = self.book.assign(v.vid, items)
        self.ui.say('labeled in view %d: %s' % (
            v.vid, ', '.join('#%d %s' % (i, lab) for i, lab, _px in items)))
        for n in changed + notes:
            self.ui.say('  ' + n)
        self.show_view()
        self.summary()

    def _label_checks(self, idx: int, label: str, px: np.ndarray) -> List[str]:
        """Return notes on a new sample: mixed colors, hue far from the label's other samples."""
        o = self.opts
        st = hue_stats(px, o.min_saturation, o.min_chromatic_fraction)
        notes = []
        if st['spread'] is not None and st['spread'] > MIXED_HUE_SPREAD:
            notes.append('#%d: hue spread %.0f: two touching blocks of different colors? '
                         'Separate them and capture again (u undoes)' % (idx, st['spread']))
        others = [s.hsv for s in self.book.by_label(label)
                  if (s.view, s.index) != (self.view.vid, idx)]
        if others and st['hue'] is not None:
            ref = hue_stats(np.concatenate(others, axis=0), o.min_saturation,
                            o.min_chromatic_fraction)
            if ref['hue'] is not None and hue_distance(st['hue'], ref['hue']) > LABEL_HUE_JUMP:
                notes.append('#%d: hue %.0f is far from the other %s samples (hue %.0f): '
                             'wrong label? (u undoes)' % (idx, st['hue'], label, ref['hue']))
        return notes

    def undo(self) -> None:
        """Revert the last labeling line."""
        done = self.book.undo()
        if done is None:
            self.ui.say('nothing to undo')
            return
        self.ui.say('undone: %s' % ', '.join(done))
        self.show_view()
        self.summary()

    def set_margins(self, values: Sequence[float]) -> None:
        """Change the hue (and S / V) margin and print the new summary."""
        sv = values[1] if len(values) > 1 else self.opts.sv_margin
        new = dataclasses.replace(self.opts, hue_margin=float(values[0]), sv_margin=float(sv))
        try:
            self.opts = new.validate()
        except ValueError as e:
            self.ui.say('error: %s' % e)
            return
        self.ui.say('margins: hue %g, S/V %g' % (self.opts.hue_margin, self.opts.sv_margin))
        self.summary()

    # -------------------------------------------------------------- ranges
    def ranges(self) -> Tuple[Dict[str, list], Dict[str, str]]:
        """Return (label -> ranges, label -> error) from the samples so far."""
        return compute_ranges(self.book.pixels_by_label(), self.opts)

    def warnings(self, new: Mapping[str, Any], errors: Mapping[str, str]) -> List[str]:
        """Return every warning about the ranges that would be written."""
        eff = effective_classes(self.base_classes, new, self.allowed)
        out = ['%s: %s' % (k, e) for k, e in sorted(errors.items())]
        out += view_warnings(self.book.views_by_label(), self.args.min_views)
        samples = {k: v for k, v in self.book.pixels_by_label().items() if k in eff}
        out += overlap_warnings(overlap_pairs(overlap_fractions(samples, eff),
                                              self.args.overlap_warn))
        out += weak_instance_warnings([s for _k, s in sorted(self.book.samples.items())], new,
                                      self.threshold)
        out += missing_label_warnings(self.allowed, eff)
        return out

    def summary(self) -> None:
        """Print the samples, the ranges computed so far and the warnings."""
        if not self.book.samples:
            self.ui.say('no labeled instances yet')
            return
        new, errors = self.ranges()
        self.ui.say('ranges (%s):' % self.opts.text())
        self.ui.say(format_table(SUMMARY_HEADERS, summary_rows(self.book, new, errors)))
        kept = [k for k in effective_classes(self.base_classes, {}, self.allowed) if k not in new]
        if kept:
            self.ui.say('not recalibrated (ranges from the config): %s' % ', '.join(kept))
        ws = self.warnings(new, errors)
        if ws:
            self.ui.say('WARNINGS:')
            for w in ws:
                self.ui.say('  - ' + w)
        else:
            self.ui.say('no warnings')

    def preview(self) -> None:
        """Classify the current view with the ranges so far (ColorClassifier); print and show."""
        v = self.view
        new, _errors = self.ranges()
        classes = effective_classes(self.base_classes, new, self.allowed)
        if not classes:
            self.ui.say('no color ranges to preview with: label some instances first')
            return
        try:
            clf = ColorClassifier(classes, self.min_fraction)
        except ValueError as e:
            self.ui.say('cannot build the color classifier: %s' % e)
            return
        dets = [dataclasses.replace(d, extra=dict(d.extra)) for d in v.dets]
        clf.classify(v.frame, dets)
        assigned = [self.book.label_of(v.vid, i) for i in range(len(dets))]
        truth = self.truth()
        rows, problems, checked = preview_rows(dets, assigned, self.min_score, truth)
        self.ui.say('preview of view %d with %s (* = calibrated in this session); the label is '
                    'unknown below a fraction of %.2f'
                    % (v.vid, ', '.join(k + ('*' if k in new else '') for k in classes),
                       self.threshold))
        if not dets:
            self.ui.say('(no instances in this view)')
            return
        headers = PREVIEW_HEADERS + (('sim truth',) if truth is not None else ())
        self.ui.say(format_table(headers, rows))
        if problems:
            self.ui.say('MISMATCH:')
            for p in problems:
                self.ui.say('  ' + p)
        elif checked:
            self.ui.say('preview OK: %d of %d instances get the expected label'
                        % (checked, len(dets)))
        else:
            self.ui.say('label the instances to compare the preview with your labels')
        colors = []
        for i, r in enumerate(rows):
            want = assigned[i] or (truth[i] if truth is not None else None)
            colors.append((0, 255, 255) if want is None else
                          ((0, 220, 0) if r[2] == want else (0, 0, 255)))
        base = v.result.debug_image if v.result.debug_image is not None else v.frame.color
        overlay = draw_texts(np.array(base, copy=True), dets,
                             ['#%d %s %s' % (i, r[2], r[3]) for i, r in enumerate(rows)], colors)
        self.ui.show('color_calib_preview', text_panel(
            compose_view(v.frame, overlay, self.args.depth_range),
            ['preview of view %d: green = expected label, red = mismatch' % v.vid]))

    # -------------------------------------------------------------- writing
    def write(self) -> bool:
        """Write the computed ranges (backup + atomic write); return True when written."""
        new, errors = self.ranges()
        if not new:
            self.ui.say('nothing to write: label some instances first')
            return False
        ws = self.warnings(new, errors)
        if ws:
            self.ui.say('WARNINGS:')
            for w in ws:
                self.ui.say('  - ' + w)
            if not self.ui.confirm('Write %s anyway?' % self.out, default=False):
                self.ui.say('not written')
                return False
        try:
            existing = read_yaml_or_empty(self.out)
        except (OSError, ValueError, yaml.YAMLError) as e:
            self.ui.say('cannot read the existing %s: %s' % (self.out, e))
            if not self.ui.confirm('Replace it (the old file is kept as a backup)?',
                                   default=False):
                self.ui.say('not written')
                return False
            existing = {}
        data, keys, kept, removed = build_colors_yaml(existing, new, self.args.replace)
        header = calib_header(wall_stamp(), self.s.task, self.s.mode, self.n_views, self.opts,
                              self.book, new)
        try:
            backup = write_site_yaml(self.out, data, header=header)
        except OSError as e:
            self.ui.say('writing %s failed: %s' % (self.out, e))
            return False
        self.ui.say(write_report(self.out, keys, backup))
        if kept:
            self.ui.say('  classes kept from the previous file: %s' % ', '.join(kept))
        if removed:
            self.ui.say('  classes removed (--replace): %s' % ', '.join(removed))
        self.ui.say(self.use_hint())
        self.check_reload(new)
        self.saved = self.state()
        self.writes += 1
        return True

    def _is_configured_path(self) -> bool:
        try:
            return self.out.resolve() == Path(overlay_file(self.cfg, 'colors.yaml')).resolve()
        except OSError:
            return False

    def use_hint(self) -> str:
        """Return how the other tools pick up the written file."""
        configured = overlay_file(self.cfg, 'colors.yaml')
        if self._is_configured_path():
            hint = 'the tools merge it automatically (site_overlays in config/default.yaml)'
            return hint + (' in real mode; sim tools only with --site' if self.s.mode == 'sim'
                           else '')
        return ('the tools do not read this file automatically: pass --config %s, or copy it '
                'to %s' % (self.out, configured))

    def check_reload(self, new: Mapping[str, Any]) -> None:
        """Real mode: reload the config and check that the new ranges are the ones used."""
        if self.s.mode == 'sim' or not self._is_configured_path():
            return
        from ..core.config import load_config
        try:
            cfg2 = load_config(extra=self.args.config, overrides=self.args.set)
        except (OSError, ValueError, yaml.YAMLError) as e:
            self.ui.say('WARNING: reloading the config failed: %s' % e)
            return
        bad = overridden_classes(cfg2.get(CLASSES_KEY, {}), new)
        if bad:
            self.ui.say('WARNING: the reloaded config has other ranges for %s: a later site '
                        'overlay (local.yaml), --config or --set overrides them'
                        % ', '.join(bad))
        else:
            self.ui.say('  checked: the reloaded config uses the new ranges')

    # -------------------------------------------------------------- loop
    def intro(self) -> None:
        """Print what the run does and how to use it."""
        a = self.args
        labels = ', '.join(self.allowed) if self.allowed else 'any (%s is empty)' % self.labels_key
        self.ui.say('color_calib: task %s, mode %s, labels %s' % (self.s.task, self.s.mode,
                                                                  labels))
        self.ui.say('output: %s%s' % (self.out, ' (--replace: classes not calibrated now are '
                                                'dropped from it)' if a.replace else ''))
        self.ui.say('ranges: %s; instance masks eroded by %d px; warnings at %.0f %% overlap and '
                    'fewer than %d views' % (self.opts.text(), a.erode_px,
                                             100.0 * a.overlap_warn, a.min_views))
        from ..perception.factory import classifier_backend
        try:
            backend = classifier_backend(self.cfg, self.s.task)
        except ValueError as e:
            backend = 'invalid (%s)' % e
        if backend != 'color':
            self.ui.say('note: the %s classifier backend is %s; these ranges are only used by '
                        'the color backend' % (self.s.task, backend))
        self.ui.say("Put the round's blocks in the pick area under the competition light. Label "
                    "each numbered instance, then capture more views (c) with the blocks moved "
                    "or the light changed: every color needs %d views." % a.min_views)
        if self.ui.headless:
            self.ui.say('preview images: %s, %s' % (self.ui.preview_path('color_calib'),
                                                    self.ui.preview_path('color_calib_preview')))
        self.ui.say(HEADLESS_HELP if self.ui.headless else WINDOW_HELP)

    def handle(self, cmd: Command) -> None:
        """Run one parsed command other than 'quit'."""
        k = cmd.kind
        if k == 'error':
            self.ui.say('error: %s' % cmd.error)
        elif k == 'help':
            self.ui.say(HELP_TEXT)
        elif k == 'refresh':
            self.print_view()
        elif k == 'assign':
            self.assign(cmd.pairs)
        elif k == 'undo':
            self.undo()
        elif k == 'capture':
            self.new_view()
        elif k == 'preview':
            self.preview()
        elif k == 'summary':
            self.summary()
        elif k == 'margins':
            self.set_margins(cmd.values)
        elif k == 'write':
            self.write()

    def finish(self) -> None:
        """Print the final state when the operator quits."""
        if self.unsaved():
            self.ui.say('quit: the ranges of this session were NOT written (w writes them)')
        elif self.writes:
            self.ui.say('done: %s written %d time%s'
                        % (self.out, self.writes, '' if self.writes == 1 else 's'))
        else:
            self.ui.say('quit: nothing written')


# ------------------------------------------------------------------ main
def prepare_pose(args, s, ui) -> bool:
    """
    Bring the camera to its task view before the first capture (thin wrapper).

    Sim: move to the observe pose (simulated). Real with --observe: check
    the arm accepts commands, then move there at the resolved speed after
    the operator confirms (MotionGuard). Returns False to abort the run.
    """
    if args.mode == 'sim':
        ok, why = go_observe(s.arm, s.cfg, MotionGuard(ui, 'sim', 100))
        if not ok:
            ui.say('sim: could not reach the observe pose: %s' % why)
        return True
    if not args.observe:
        return True
    if args.mode != 'real':
        ui.say('--observe only moves the real arm; ignored in %s mode' % args.mode)
        return True
    if s.arm is None:
        ui.say('--observe needs the arm (it did not connect)')
        return False
    try:
        speed, note = resolve_speed(s.cfg, args.speed)
    except ValueError as e:
        ui.say('error: %s' % e)
        return False
    if note:
        ui.say(note)
    if not ensure_motion_ready(ui, s.arm, 'move to the observe pose'):
        ui.say('not moving; run without --observe to calibrate from the current camera view')
        return False
    s.arm.set_speed_percent(speed)
    ok, why = go_observe(s.arm, s.cfg, MotionGuard(ui, 'real', speed))
    if not ok:
        ui.say('observe pose not reached: %s' % why)
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser of color_calib."""
    p = argparse.ArgumentParser(prog='color_calib',
                                description=(__doc__ or '').strip().split('\n\n')[0],
                                epilog=HELP_TEXT,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    add_onsite_args(p, site=True)
    add_depth_range_arg(p)
    p.add_argument('--replace', action='store_true',
                   help='drop the classes of the existing colors.yaml that are not calibrated '
                        'in this session (default: keep them)')
    p.add_argument('--average', type=int, default=None,
                   help='frames per view, median depth (default camera.frames_to_average)')
    p.add_argument('--observe', action='store_true',
                   help='real mode: move to the observe pose first (asks before moving)')
    p.add_argument('--speed', type=int, default=None,
                   help='--observe speed percent; can only lower robot.speed_percent')
    g = p.add_argument_group('HSV ranges (classify_color.hsv_ranges_from_samples)')
    d = RangeOptions()
    g.add_argument('--lo-pct', type=float, default=d.lo_pct,
                   help='lower percentile of the samples (default %(default)s)')
    g.add_argument('--hi-pct', type=float, default=d.hi_pct,
                   help='upper percentile of the samples (default %(default)s)')
    g.add_argument('--hue-margin', type=float, default=d.hue_margin,
                   help='widen the hue range by this, OpenCV units of 2 deg (default %(default)s)')
    g.add_argument('--sv-margin', type=float, default=d.sv_margin,
                   help='widen the S / V ranges by this (default %(default)s)')
    g.add_argument('--min-saturation', type=int, default=d.min_saturation,
                   help='pixels below this saturation do not count for the hue (default '
                        '%(default)s)')
    g.add_argument('--min-chromatic-fraction', type=float, default=d.min_chromatic_fraction,
                   help='fewer saturated pixels than this fraction: full hue range (white / '
                        'black blocks; default %(default)s)')
    g.add_argument('--erode-px', type=int, default=2,
                   help='erode each upper-surface mask by this many pixels against edge and '
                        'table pixels (default %(default)s)')
    g.add_argument('--max-pixels', type=int, default=4000,
                   help='use at most this many pixels per instance (default %(default)s)')
    g.add_argument('--min-views', type=int, default=2,
                   help='warn when a color has samples from fewer views (default %(default)s)')
    g.add_argument('--overlap-warn', type=float, default=0.05,
                   help="warn when this fraction of one color's samples falls inside another "
                        "color's ranges (default %(default)s)")
    return p


def options_from_args(args: argparse.Namespace) -> RangeOptions:
    """Return the validated RangeOptions of args; raise ValueError for any bad option."""
    if not 0 <= args.erode_px <= 10:
        raise ValueError('--erode-px must be in [0, 10]')
    if args.max_pixels < 100:
        raise ValueError('--max-pixels must be >= 100')
    if args.min_views < 1:
        raise ValueError('--min-views must be >= 1')
    if not 0.0 < args.overlap_warn <= 1.0:
        raise ValueError('--overlap-warn must be in (0, 1]')
    if args.average is not None and args.average < 1:
        raise ValueError('--average must be >= 1')
    return RangeOptions(args.lo_pct, args.hi_pct, args.hue_margin, args.sv_margin,
                        args.min_saturation, args.min_chromatic_fraction).validate()


def main(argv=None, ui=None) -> int:
    """Run the color calibration; return the process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    cfg_world_info = load_cfg_for(args)
    cfg = cfg_world_info[0]
    ui = make_ui(args, cfg, 'color_calib', ui)
    try:
        opts = options_from_args(args)
    except ValueError as e:
        ui.say('error: %s' % e)
        return 2
    from ..perception.factory import task_labels
    allowed = task_labels(cfg, args.task)
    out = output_path(cfg, args.mode, args.site_dir, overlay_file(cfg, 'colors.yaml'))
    try:
        return _run(args, ui, cfg_world_info, opts, allowed, out)
    finally:
        ui.close()


def _run(args, ui, cfg_world_info, opts: RangeOptions, allowed: Sequence[str],
         out: Path) -> int:
    """Open the session and run the command loop; return the exit code."""
    with open_session(args, arm=True, camera=True, perception=False,
                      connect=args.mode != 'real', cfg_world_info=cfg_world_info) as s:
        if not connect_arm_readonly(s, ui, required=True,
                                    why='the observe motion' if args.observe
                                    else 'wrist camera poses'):
            return 2
        if not prepare_pose(args, s, ui):
            return 2
        cal = _ColorCalib(args, s, ui, opts, allowed, out)
        cal.intro()
        if not cal.capture():
            ui.say('no camera frame: check the camera driver and topics (check_setup)')
            return 1
        while True:
            line = next_command(ui, HEADLESS_HELP)
            if line is None:
                cal.live()
                continue
            cmd = parse_command(line, cal.n_instances(), cal.allowed, cal.labels_key)
            if cmd.kind == 'quit':
                if (cal.unsaved() and not ui.eof
                        and not ui.confirm('The ranges of this session are not written. '
                                           'Quit anyway?', default=False)):
                    continue
                break
            cal.handle(cmd)
        cal.finish()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
