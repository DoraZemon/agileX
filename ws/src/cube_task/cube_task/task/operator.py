"""
Operator keyboard controls for a competition round.

Keys: Enter starts the timed run; 'p' or space aborts (stop issuing commands,
hold the position); 'e' calls the e-stop callback (normally
``arm.emergency_stop``) and aborts. A disabled operator (tests, simulation)
starts immediately and only aborts when ``abort()`` is called.

Waiting here uses real wall time on purpose: it is operator I/O, not robot logic.
"""
from __future__ import annotations

import atexit
import logging
import sys
import threading
from typing import Callable, Optional

_LOG = logging.getLogger('cube_task.task.operator')

START_KEYS = ('\n', '\r')
ABORT_KEYS = ('p', 'P', ' ')
ESTOP_KEYS = ('e', 'E')


class Operator:
    """
    Background stdin reader with start / abort / e-stop events.

    stdin: stream to read (default sys.stdin). The terminal is put in cbreak mode
    only when it is a TTY, and is always restored by ``close()``, on reader
    errors and at interpreter exit.
    """

    def __init__(self, enabled: bool = True, estop_callback: Optional[Callable[[], None]] = None,
                 stdin=None, out=None):
        self.enabled = bool(enabled)
        self.abort_event = threading.Event()
        self.start_event = threading.Event()
        self.estopped = False
        self.abort_reason = ''
        self._estop_cb = estop_callback
        self._stdin = stdin
        self._out = out
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._saved_tty = None
        self._tty_fd: Optional[int] = None
        self._lock = threading.Lock()
        if not self.enabled:
            self.start_event.set()

    # ------------------------------------------------------------------ public
    def set_estop_callback(self, fn: Optional[Callable[[], None]]) -> None:
        """Set the function called on 'e' (for example arm.emergency_stop)."""
        self._estop_cb = fn

    def start(self) -> None:
        """Start the background key reader (no-op when disabled or running)."""
        if not self.enabled or self._thread is not None:
            return
        stream = self._stdin if self._stdin is not None else sys.stdin
        try:
            fd = stream.fileno()
        except (AttributeError, OSError, ValueError):
            fd = None
        if fd is not None:
            # Before the prompt is printed: cbreak flushes keys typed earlier.
            self._setup_tty(stream, fd)
        self._thread = threading.Thread(target=self._reader, args=(stream, fd),
                                        name='operator-keys', daemon=True)
        self._thread.start()

    def wait_start(self, prompt: str = 'Press Enter to start the timed run '
                   '(p / space = abort, e = e-stop)', discard_earlier: bool = False) -> bool:
        """
        Block until Enter. Return False when aborted first. Disabled: return at once.

        discard_earlier: ignore an Enter read before the prompt (for example one
        typed while the arm moved to the observe pose with the reader running).
        """
        if not self.enabled:
            return not self.abort_event.is_set()
        self.start()
        if discard_earlier:
            self.start_event.clear()
        self._print(prompt)
        while not self.start_event.is_set() and not self.abort_event.is_set():
            self.start_event.wait(0.1)
            if self._thread is not None and not self._thread.is_alive() and \
                    not self.start_event.is_set():
                self._print('stdin closed: cannot wait for Enter, aborting')
                self.abort('stdin closed before start')
        return self.start_event.is_set() and not self.abort_event.is_set()

    def abort(self, reason: str = 'operator abort') -> None:
        """Set the abort event (motion waits return 'aborted')."""
        if not self.abort_event.is_set():
            self.abort_reason = reason
            _LOG.warning('ABORT: %s', reason)
        self.abort_event.set()

    def estop(self) -> None:
        """Call the e-stop callback, then abort."""
        self.estopped = True
        cb = self._estop_cb
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: B902 - still abort if the e-stop call fails
                _LOG.exception('emergency stop callback raised')
        else:
            _LOG.error('e-stop pressed but no callback set')
        self.abort('emergency stop')

    def feed(self, ch: str) -> None:
        """Handle one key (used by the reader thread and by tests)."""
        if ch in START_KEYS:
            self.start_event.set()
        elif ch in ABORT_KEYS:
            self.abort('key %r' % ch)
        elif ch in ESTOP_KEYS:
            self.estop()

    def close(self) -> None:
        """Stop the reader and restore the terminal. Safe to call twice."""
        self._stop.set()
        self._restore_tty()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=0.5)

    def __enter__(self) -> 'Operator':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------------- internal
    def _print(self, msg: str) -> None:
        out = self._out if self._out is not None else sys.stderr
        try:
            out.write(msg + '\n')
            out.flush()
        except (OSError, ValueError):
            pass

    def _reader(self, stream, fd: Optional[int]) -> None:
        try:
            if fd is None:
                self._read_stream(stream)
            else:
                self._read_fd(fd)
        except Exception:  # noqa: B902 - the reader must never take the process down
            _LOG.exception('operator key reader failed')
        finally:
            self._restore_tty()

    def _read_stream(self, stream) -> None:
        while not self._stop.is_set():
            ch = stream.read(1)
            if not ch:
                return
            self.feed(ch)

    def _read_fd(self, fd: int) -> None:
        import os
        import select
        while not self._stop.is_set():
            ready, _w, _x = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            data = os.read(fd, 64)
            if not data:
                return
            for ch in data.decode('utf-8', 'ignore'):
                self.feed(ch)

    def _setup_tty(self, stream, fd: int) -> None:
        try:
            is_tty = bool(stream.isatty())
        except (AttributeError, OSError, ValueError):
            is_tty = False
        if not is_tty:
            return
        import termios
        import tty
        with self._lock:
            self._saved_tty = termios.tcgetattr(fd)
            self._tty_fd = fd
        atexit.register(self._restore_tty)
        tty.setcbreak(fd)

    def _restore_tty(self) -> None:
        with self._lock:
            saved, fd = self._saved_tty, self._tty_fd
            self._saved_tty = None
        if saved is None or fd is None:
            return
        try:
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except Exception:  # noqa: B902
            _LOG.exception('could not restore the terminal')
