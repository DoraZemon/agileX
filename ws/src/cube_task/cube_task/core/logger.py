"""
Per-run log directory: JSONL events, debug images, JSON / text files, summary.

Layout of one run (``<base_dir>/<YYYYmmdd_HHMMSS>_<task>/``)::

    events.jsonl   one JSON object per line:
                   {"t": seconds since the logger was created (injected Clock),
                    "wall": local ISO wall time, "kind": ..., <fields>}
    *.jpg / *.png  images from save_image()
    *.json / *.txt files from save_json() / save_text()
    summary.json   written by close()

The wall clock (datetime) is only used for naming and for the human-readable
``wall`` field; all durations use the injected Clock so simulated runs log
simulated time.

Logging must never stop the robot. Every file operation is wrapped: a failure
(disk full, permissions, unserializable value) logs a rate-limited warning and
the run continues.
"""
from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import logging
import math
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .interfaces import Clock

_LOG = logging.getLogger('cube_task.logger')
# Child of the 'cube_task' logger, so handlers attached to 'cube_task' see events.
_EVENT_LOG = logging.getLogger('cube_task.events')

#: Arrays with more elements than this are summarized instead of dumped.
MAX_ARRAY_ELEMENTS = 4096
_RESERVED = ('t', 'wall', 'kind')
_MAX_DEPTH = 32


def to_jsonable(obj: Any, max_array: int = MAX_ARRAY_ELEMENTS, _depth: int = 0) -> Any:
    """
    Convert obj into plain JSON types (dict, list, str, int, float, bool, None).

    - numpy arrays: nested lists; arrays with more than ``max_array`` elements
      become ``{"ndarray": shape, "dtype": name}`` so a mask or image never
      lands in a log line.
    - numpy scalars: int / float / bool. Non-finite floats (NaN, inf): None,
      because strict JSON has no NaN.
    - dataclasses: ``obj.summary()`` when the class defines it (Detection),
      else a dict of the fields. Enum: its value. Path: str. bytes: length.
    - tuples / sets: lists. Dict keys: str. Anything else: ``str(obj)``.
    """
    if _depth > _MAX_DEPTH:
        return str(obj)[:200]
    d = _depth + 1
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        if obj.size > max_array:
            return {'ndarray': [int(s) for s in obj.shape], 'dtype': str(obj.dtype)}
        return to_jsonable(obj.tolist(), max_array, d)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_array, d) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, max_array, d) for v in obj]
    if isinstance(obj, (set, frozenset)):
        items = [to_jsonable(v, max_array, d) for v in obj]
        try:
            return sorted(items)
        except TypeError:
            return items
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value, max_array, d)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {'bytes': len(obj)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        summary = getattr(obj, 'summary', None)
        if callable(summary):
            try:
                return to_jsonable(summary(), max_array, d)
            except Exception:
                # A broken summary() must not break logging: use the fields.
                pass
        return {f.name: to_jsonable(getattr(obj, f.name, None), max_array, d)
                for f in dataclasses.fields(obj)}
    return str(obj)


def dumps_compact(obj: Any) -> str:
    """Serialize obj to one strict JSON line (UTF-8 text, no NaN)."""
    return json.dumps(to_jsonable(obj), ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False)


def safe_filename(name: str, default: str = 'unnamed') -> str:
    """Make a single path component from name (no separators, no '..')."""
    s = re.sub(r'[^\w.\-]+', '_', str(name)).strip('._')
    return s or default


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    """Write text to path through a temporary file and os.replace (raises OSError)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp',
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates 0600 files; give the result the usual rw-r--r-- mode.
        os.chmod(tmp, 0o644)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _short(value: Any, limit: int = 120) -> str:
    try:
        s = dumps_compact(value)
    except (TypeError, ValueError):
        s = str(value)
    return s if len(s) <= limit else s[:limit - 3] + '...'


class RunLogger:
    """
    Thread-safe writer for one run directory. Never raises on I/O errors.

    Parameters
    ----------
    base_dir
        Parent directory (``~`` is expanded; None = system temp dir).
    task
        Task name ('L1', 'L2', 'L3', 'sim_bench', ...), used in the name.
    clock
        Injected Clock; event times are ``clock.now() - start`` seconds.
    save_images
        False makes save_image() a no-op that returns None.
    name
        Directory name to use instead of ``<YYYYmmdd_HHMMSS>_<task>``. An
        existing directory is never reused: ``_1``, ``_2``, ... is appended.

    Attributes
    ----------
    dir
        The run directory (Path). If it cannot be created a temporary
        directory is used, and if that fails too ``enabled`` is False and
        only the Python logger mirror works.

    """

    def __init__(self, base_dir: Optional[Union[str, Path]], task: str, clock: Clock,
                 save_images: bool = True, name: Optional[str] = None):
        self.task = str(task)
        self.clock = clock
        self.save_images = bool(save_images)
        self.enabled = True
        self._lock = threading.RLock()
        self._t0 = float(clock.now())
        self._n_events = 0
        self._n_failures = 0
        self._closed = False
        self._fh = None
        if name:
            dirname = safe_filename(name)
        else:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            dirname = '%s_%s' % (stamp, safe_filename(self.task, 'run'))
        base = Path(base_dir).expanduser() if base_dir else Path(tempfile.gettempdir())
        self.dir = self._make_dir(base, dirname)
        self.events_path = self.dir / 'events.jsonl'
        _LOG.info('run log directory: %s', self.dir)

    # ------------------------------------------------------------------ setup
    def _make_dir(self, base: Path, dirname: str) -> Path:
        try:
            base.mkdir(parents=True, exist_ok=True)
            for i in range(1000):
                cand = base / (dirname if i == 0 else '%s_%d' % (dirname, i))
                try:
                    cand.mkdir()
                    return cand
                except FileExistsError:
                    continue
            raise OSError('no free directory name for %s in %s' % (dirname, base))
        except OSError as e:
            _LOG.warning('cannot create run log directory in %s (%s); using a temp dir',
                         base, e)
        try:
            return Path(tempfile.mkdtemp(prefix='cube_task_%s_' % dirname))
        except OSError as e:
            _LOG.warning('cannot create a temp run log directory either (%s); '
                         'file logging disabled', e)
            self.enabled = False
            return base / dirname

    def _fail(self, what: str, exc: BaseException) -> None:
        with self._lock:
            self._n_failures += 1
            n = self._n_failures
        if n <= 5 or n % 100 == 0:
            _LOG.warning('run log: %s failed (%s: %s); continuing [%d failures]',
                         what, type(exc).__name__, exc, n)

    # ------------------------------------------------------------- accessors
    def elapsed(self) -> float:
        """Return Clock seconds since the logger was created."""
        return float(self.clock.now()) - self._t0

    def path(self, name: str) -> Path:
        """Return the path of file ``name`` inside the run directory."""
        return self.dir / safe_filename(name)

    @property
    def n_events(self) -> int:
        """Return the number of events recorded so far."""
        return self._n_events

    @property
    def n_failures(self) -> int:
        """Return the number of logging operations that failed."""
        return self._n_failures

    # ---------------------------------------------------------------- events
    def event(self, kind: str, **fields: Any) -> None:
        """
        Append one event to events.jsonl and mirror it to the Python logger.

        Field names 't', 'wall' and 'kind' are reserved; such fields are stored
        with a trailing underscore. numpy values are converted (see
        to_jsonable()).
        """
        try:
            t = round(self.elapsed(), 4)
            rec: Dict[str, Any] = {'t': t,
                                   'wall': datetime.datetime.now().isoformat(
                                       timespec='milliseconds'),
                                   'kind': str(kind)}
            for k, v in fields.items():
                rec[k + '_' if k in _RESERVED else k] = v
            line = dumps_compact(rec)
        except Exception as e:  # unserializable field: keep the kind at least
            self._fail('event %r serialization' % (kind,), e)
            return
        with self._lock:
            self._n_events += 1
            if self.enabled:
                try:
                    if self._fh is None:
                        self._fh = open(self.events_path, 'a', encoding='utf-8')
                    self._fh.write(line + '\n')
                    self._fh.flush()
                except Exception as e:
                    self._close_fh()
                    self._fail('writing %s' % self.events_path.name, e)
        try:
            if _EVENT_LOG.isEnabledFor(logging.INFO):
                text = ' '.join('%s=%s' % (k, _short(v)) for k, v in fields.items())
                msg = '[%8.3f] %s %s' % (t, kind, text)
                _EVENT_LOG.info(msg if len(msg) <= 600 else msg[:597] + '...')
        except Exception as e:
            self._fail('event mirror', e)

    def _close_fh(self) -> None:
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                # Closing a broken handle (disk full) can fail again; it is gone anyway.
                pass

    # ----------------------------------------------------------------- files
    def save_image(self, name: str, bgr: Any) -> Optional[Path]:
        """
        Write an image into the run directory and return its path.

        ``name`` ending in '.png' writes a lossless PNG (uint8 or uint16);
        '.jpg' / '.jpeg' or no image extension writes JPEG quality 90 (uint8
        only; '.jpg' is appended). bool images are written as 0 / 255.
        An existing file of the same name is overwritten. Returns None when
        save_images is False or on failure.
        """
        if not self.save_images or not self.enabled:
            return None
        try:
            import cv2
            fname = safe_filename(name, 'image')
            low = fname.lower()
            if not (low.endswith('.png') or low.endswith('.jpg') or low.endswith('.jpeg')):
                fname += '.jpg'
                low = fname.lower()
            if bgr is None:
                raise ValueError('image is None')
            img = np.asarray(bgr)
            if img.dtype == np.bool_:
                img = img.astype(np.uint8) * 255
            if img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] not in (1, 3, 4)):
                raise ValueError('unsupported image shape %s' % (img.shape,))
            if img.size == 0:
                raise ValueError('empty image')
            is_png = low.endswith('.png')
            if img.dtype != np.uint8 and not (is_png and img.dtype == np.uint16):
                raise ValueError('unsupported dtype %s for %s (uint8, or uint16 for PNG)'
                                 % (img.dtype, fname))
            path = self.dir / fname
            params = [] if is_png else [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            if not cv2.imwrite(str(path), np.ascontiguousarray(img), params):
                raise OSError('cv2.imwrite returned False for %s' % path)
            return path
        except Exception as e:
            self._fail('save_image %r' % (name,), e)
            return None

    def save_json(self, name: str, obj: Any) -> Optional[Path]:
        """Write obj (converted with to_jsonable) as indented JSON; '.json' is added."""
        if not self.enabled:
            return None
        try:
            fname = safe_filename(name, 'data')
            if not fname.lower().endswith('.json'):
                fname += '.json'
            text = json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2,
                              allow_nan=False)
            path = self.dir / fname
            atomic_write_text(path, text + '\n')
            return path
        except Exception as e:
            self._fail('save_json %r' % (name,), e)
            return None

    def save_text(self, name: str, text: str) -> Optional[Path]:
        """Write a UTF-8 text file; '.txt' is added when name has no extension."""
        if not self.enabled:
            return None
        try:
            fname = safe_filename(name, 'text')
            if '.' not in fname:
                fname += '.txt'
            path = self.dir / fname
            atomic_write_text(path, str(text))
            return path
        except Exception as e:
            self._fail('save_text %r' % (name,), e)
            return None

    def close(self, summary: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Write summary.json, record a 'close' event and close events.jsonl.

        The summary is stored as given plus a ``_logger`` entry (elapsed Clock
        seconds, event and failure counts). Calling close() again rewrites
        summary.json; events after close() reopen events.jsonl.
        """
        data: Dict[str, Any] = {}
        if summary is not None:
            conv = to_jsonable(summary)
            data = conv if isinstance(conv, dict) else {'summary': conv}
        if not self._closed:
            self.event('close', elapsed_s=round(self.elapsed(), 3))
        data['_logger'] = {'elapsed_s': round(self.elapsed(), 3),
                           'events': self._n_events, 'failures': self._n_failures,
                           'wall_end': datetime.datetime.now().isoformat(timespec='seconds'),
                           'dir': str(self.dir)}
        path = self.save_json('summary.json', data)
        with self._lock:
            self._close_fh()
            self._closed = True
        return path

    def __enter__(self) -> 'RunLogger':
        """Return self for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the logger (with the exception text in the summary, if any)."""
        if not self._closed:
            self.close({'exception': None if exc is None else '%s: %s'
                        % (exc_type.__name__, exc)})
