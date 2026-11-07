"""Tests for task/operator.py (keyboard start / abort / e-stop)."""
from __future__ import annotations

import io
import os
import threading

from cube_task.task.operator import Operator
import pytest


def test_disabled_operator_starts_immediately_and_aborts_only_on_request():
    op = Operator(enabled=False)
    assert op.wait_start() is True
    assert not op.abort_event.is_set()
    op.abort('test')
    assert op.abort_event.is_set() and op.abort_reason == 'test'
    assert op.wait_start() is False


def test_enter_starts_from_a_stream():
    op = Operator(enabled=True, stdin=io.StringIO('\n'), out=io.StringIO())
    assert op.wait_start() is True
    op.close()
    op.close()                                    # safe twice


@pytest.mark.parametrize('key', ['p', ' '])
def test_abort_keys(key):
    op = Operator(enabled=True, stdin=io.StringIO(key), out=io.StringIO())
    assert op.wait_start() is False
    assert op.abort_event.is_set()
    op.close()


def test_estop_key_calls_callback_and_aborts():
    called = []
    op = Operator(enabled=True, stdin=io.StringIO('e'), out=io.StringIO())
    op.set_estop_callback(lambda: called.append(1))
    assert op.wait_start() is False
    assert called == [1] and op.estopped and op.abort_event.is_set()


def test_estop_callback_error_still_aborts():
    def boom():
        raise RuntimeError('can bus down')
    op = Operator(enabled=False, estop_callback=boom)
    op.feed('e')
    assert op.abort_event.is_set()


def test_closed_stdin_before_enter_aborts():
    op = Operator(enabled=True, stdin=io.StringIO(''), out=io.StringIO())
    assert op.wait_start() is False
    assert 'stdin closed' in op.abort_reason


def test_pipe_reader_handles_keys_during_run():
    r, w = os.pipe()
    with os.fdopen(r, 'r') as rf:
        op = Operator(enabled=True, stdin=rf, out=io.StringIO())
        os.write(w, b'\n')
        assert op.wait_start() is True
        done = threading.Event()

        def watch():
            op.abort_event.wait(2.0)
            done.set()
        t = threading.Thread(target=watch)
        t.start()
        os.write(w, b'p')
        assert done.wait(2.0) and op.abort_event.is_set()
        t.join()
        op.close()
    os.close(w)


def test_tty_is_restored():
    pty = pytest.importorskip('pty')
    import termios
    master, slave = pty.openpty()
    try:
        before = termios.tcgetattr(slave)
        with os.fdopen(os.dup(slave), 'r') as sf:
            op = Operator(enabled=True, stdin=sf, out=io.StringIO())
            op.start()                                  # cbreak is active on return
            assert termios.tcgetattr(slave)[3] & termios.ICANON == 0
            os.write(master, b'\r')
            assert op.wait_start() is True
            op.close()
        assert termios.tcgetattr(slave) == before
    finally:
        os.close(master)
        os.close(slave)
