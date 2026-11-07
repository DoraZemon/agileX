"""
Smoke test of PyAgxArm with the real pyAgxArm SDK on a python-can virtual bus.

Uses PiperCanSlave from the pyAgxArm repository (tests/slaves). That slave only
answers host frames and sends feedback once per state change, so a feeder
thread here repeats its standard feedback burst (plus an AgxGripper 0x2A8 frame)
at 50 Hz, like the real controller's periodic feedback. The slave's arm never
moves and its gripper reports disabled, so the motion and gripper calls must
come back with a failure result instead of hanging.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import uuid

import pytest

from cube_task.core import transforms as tf
from cube_task.core.clock import RealClock
from cube_task.core.config import load_config
from cube_task.core.types import GripperResult, MotionResult

SDK_ROOT = '/home/zph/vision/third_party/pyAgxArm'


def _load_slave():
    pytest.importorskip('can')
    pytest.importorskip('pyAgxArm')
    if not os.path.isfile(os.path.join(SDK_ROOT, 'tests', 'slaves', 'piper_can_slave.py')):
        pytest.skip('pyAgxArm repository with tests/slaves not found')
    if 'tests' in sys.modules and SDK_ROOT not in str(getattr(sys.modules['tests'], '__path__',
                                                              '')):
        pytest.skip('another "tests" package is already imported')
    if SDK_ROOT not in sys.path:
        sys.path.insert(0, SDK_ROOT)
    from tests.slaves import _can_payloads as pl
    from tests.slaves.piper_can_slave import PiperCanSlave
    return PiperCanSlave, pl


class Feeder(threading.Thread):
    def __init__(self, slave, pl, period=0.02):
        super().__init__(daemon=True)
        self.slave, self.pl, self.period = slave, pl, period
        self.stop_event = threading.Event()

    def run(self):
        import can
        while not self.stop_event.is_set():
            frames = self.slave._standard_feedback_burst()
            frames.append(can.Message(is_extended_id=False, arbitration_id=0x2A8,
                                      data=self.pl.pack_gripper_feedback_2a8(0, 0)))
            for m in frames:
                try:
                    self.slave._bus.send(m, timeout=0.2)
                except Exception:
                    return
            time.sleep(self.period)


def test_connect_enable_move_and_gripper_on_virtual_can():
    PiperCanSlave, pl = _load_slave()
    from cube_task.robot.pyagxarm_arm import PyAgxArm

    channel = 'cube_task_%s' % uuid.uuid4().hex[:8]
    slave = PiperCanSlave(channel=channel)
    slave.start()
    feeder = Feeder(slave, pl)
    feeder.start()
    arm = None
    try:
        cfg = load_config(use_overlays=False)
        for k, v in (('robot.can_interface', 'virtual'), ('robot.can_channel', channel),
                     ('robot.connect_timeout_s', 3.0), ('robot.motion.timeout_base_s', 3.0),
                     ('robot.gripper.timeout_s', 1.5)):
            cfg.set(k, v)
        arm = PyAgxArm(cfg, RealClock())
        t0 = time.monotonic()
        arm.connect()
        assert arm.health().comm_ok
        # The slave reports enable through the low-speed foc flags, so enable() works.
        assert arm.enable(timeout=3.0)
        assert arm.health().enabled
        res = arm.move_tcp(tf.top_down_T(0.30, 0.0, 0.05, 1.5708))
        assert isinstance(res, MotionResult) and not res.ok
        assert res.error in ('stalled', 'timeout')       # the simulated arm never moves
        g = arm.gripper(0.05)
        assert isinstance(g, GripperResult) and not g.ok and g.error in ('fault', 'timeout')
        assert time.monotonic() - t0 < 10.0
        host_ids = {f.arbitration_id for f in slave.host_frames}
        # 0x471 enable, 0x151 mode, 0x152-0x154 move_p pose, 0x155-0x157 hold (move_j),
        # 0x159 gripper
        for cid in (0x471, 0x151, 0x152, 0x153, 0x154, 0x155, 0x156, 0x157, 0x159):
            assert cid in host_ids, hex(cid)
    finally:
        if arm is not None:
            arm.close()
        feeder.stop_event.set()
        feeder.join(timeout=1.0)
        slave.stop()
