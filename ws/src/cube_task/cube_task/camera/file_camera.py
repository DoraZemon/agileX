"""Replay frames recorded with camera/recorder.py as a CameraSource."""
from __future__ import annotations

import dataclasses
import logging
import math
import threading
from pathlib import Path
from typing import List, Optional, Union

from ..core.interfaces import CameraError, CameraSource, Clock
from ..core.types import CameraIntrinsics, Frame
from .recorder import list_frames, load_frame

_LOG = logging.getLogger('cube_task.camera.file')
#: Simulated frame period (s) used to advance the clock between frames.
FRAME_PERIOD_S = 1.0 / 30.0


class FileCamera(CameraSource):
    """
    Serve recorded frames in index order, stamped with the injected clock.

    Each get_frame() returns the next recorded frame (``repeat`` times each;
    use repeat = camera.frames_to_average to replay single-shot captures
    through capture_averaged). Stamps are ``clock.now()`` and strictly
    increase: when the clock has not moved past the previous stamp or
    ``newer_than``, the camera calls ``clock.sleep(1/30 s)`` (virtual with
    SimClock), at most ``timeout`` seconds of clock time, then returns None.
    After the last frame it wraps around when loop is True, else returns
    None. ``meta`` holds file_index, orig_source, orig_stamp and the recorded
    ``extra`` (e.g. T_base_flange of an eye-in-hand capture).
    """

    name = 'file'

    def __init__(self, dir: Union[str, Path], clock: Clock, loop: bool = True,
                 repeat: int = 1):
        if dir is None or str(dir).strip() == '':
            raise CameraError('FileCamera needs a directory (camera.file.dir)')
        self.dir = Path(dir).expanduser()
        self._clock = clock
        self.loop = bool(loop)
        self.repeat = int(repeat)
        if self.repeat < 1:
            raise CameraError('FileCamera repeat must be >= 1, got %r' % (repeat,))
        self._lock = threading.Lock()
        self._indices: List[int] = []
        self._pos = 0
        self._served = 0
        self._last_stamp = -math.inf
        self._intr: Optional[CameraIntrinsics] = None
        self._started = False
        self._ended_logged = False

    def start(self) -> None:
        """List the recorded frames; raise CameraError when there are none."""
        if not self.dir.is_dir():
            raise CameraError('FileCamera: directory %s does not exist' % self.dir)
        idx = list_frames(self.dir)
        if not idx:
            raise CameraError('FileCamera: no frames (*_meta.json) in %s' % self.dir)
        try:
            first, _extra = load_frame(self.dir, idx[0])
        except (OSError, ValueError) as e:
            raise CameraError('FileCamera: cannot load frame %d from %s: %s'
                              % (idx[0], self.dir, e))
        with self._lock:
            self._indices = idx
            self._pos = 0
            self._served = 0
            self._intr = first.intrinsics
            self._started = True
            self._ended_logged = False
        _LOG.info('FileCamera: %d frames in %s (loop=%s, repeat=%d)',
                  len(idx), self.dir, self.loop, self.repeat)

    def stop(self) -> None:
        """Stop serving frames (safe to call twice)."""
        with self._lock:
            self._started = False

    def intrinsics(self) -> Optional[CameraIntrinsics]:
        """Return the intrinsics of the first recorded frame once started."""
        return self._intr

    def __len__(self) -> int:
        """Return the number of recorded frames."""
        return len(self._indices)

    def _wait_stamp(self, newer_than: Optional[float], timeout: float) -> Optional[float]:
        thresh = self._last_stamp
        if newer_than is not None:
            thresh = max(thresh, float(newer_than))
        t_end = self._clock.now() + max(0.0, float(timeout))
        for _ in range(int(max(0.0, float(timeout)) / FRAME_PERIOD_S) + 3):
            now = self._clock.now()
            if now > thresh:
                return now
            remaining = t_end - now
            if remaining <= 1e-9:
                return None
            self._clock.sleep(min(FRAME_PERIOD_S, remaining))
        now = self._clock.now()
        return now if now > thresh else None

    def get_frame(self, timeout: float = 1.0,
                  newer_than: Optional[float] = None) -> Optional[Frame]:
        """Return the next recorded frame, or None (end without loop, or timeout)."""
        with self._lock:
            if not self._started:
                raise CameraError('FileCamera.get_frame() called before start()')
            if self._pos >= len(self._indices):
                if not self.loop:
                    if not self._ended_logged:
                        _LOG.info('FileCamera: end of recording in %s', self.dir)
                        self._ended_logged = True
                    return None
                self._pos = 0
            file_idx = self._indices[self._pos]
            stamp = self._wait_stamp(newer_than, timeout)
            if stamp is None:
                return None
            try:
                frame, extra = load_frame(self.dir, file_idx)
            except (OSError, ValueError) as e:
                _LOG.warning('FileCamera: skipping frame %d: %s', file_idx, e)
                self._served = 0
                self._pos += 1
                return None
            self._last_stamp = stamp
            self._advance()
        meta = dict(frame.meta)
        meta.update({'file_index': file_idx, 'orig_source': frame.source,
                     'orig_stamp': frame.stamp, 'extra': extra})
        return dataclasses.replace(frame, stamp=stamp, source='file', meta=meta)

    def _advance(self) -> None:
        self._served += 1
        if self._served >= self.repeat:
            self._served = 0
            self._pos += 1
