"""
MP4 video of a MuJoCo run: overview camera, perception inset and a HUD.

VideoRecorder registers a frame hook on the scene's MujocoSim, so frames are
taken at a fixed rate of virtual time (the physics clock), not wall time:
the video plays at real speed (speed=1) whatever the simulation speed was.
Each frame is the 'overview_cam' render (4x MSAA, own renderer) with

- the zone names drawn at the zones,
- bottom right: the latest perception debug image (wrap the Perception
  with wrap_perception(); its process() results are tapped),
- bottom left, a HUD: task / seed / mount, time since the recording and the round
  started, placed / required (runner) and the physics ground truth
  (MujocoWorld.score), and the current pick and skill step (wrap the run
  logger with wrap_logger(); the runner and skill events are tapped).

Encoding: H.264 (libx264, yuv420p, crf 23, +faststart) through
imageio-ffmpeg, which plays in browsers and VS Code. close() finalizes the
file and writes a poster JPG (the last frame) next to it.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .engine import require_mujoco

_LOG = logging.getLogger('cube_task.sim.mujoco.video')

#: skill step that follows a finished one (PickPlaceSkill._pick_place order).
NEXT_STEP = {'open': 'pre-grasp', 'pre': 'descend', 'descend': 'close', 'close': 'lift',
             'lift': 'transit', 'transit': 'release', 'release': 'open (release)',
             'open_release': 'retreat', 'retreat': 'observe'}
_FONT = 0                         # cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_GREEN = (120, 230, 120)
_RED = (255, 110, 110)
_YELLOW = (255, 220, 90)
_MSAA = 4


class _PerceptionTap:
    """Delegate to a Perception; hand every result's debug image to the recorder."""

    def __init__(self, inner: Any, recorder: 'VideoRecorder'):
        self._inner = inner
        self._rec = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def process(self, frame, T_base_camera, zones=None):
        res = self._inner.process(frame, T_base_camera, zones)
        try:
            img = res.debug_image if res.debug_image is not None else frame.color
            self._rec.set_inset(img)
        except Exception:  # noqa: B902 - the video must never break perception
            _LOG.exception('video inset update failed')
        return res


class _NullLogger:
    dir = ''

    def event(self, kind: str, **fields: Any) -> None:
        pass

    def save_image(self, name: str, bgr: Any) -> None:
        return None

    def save_json(self, name: str, obj: Any) -> None:
        return None


class _LoggerTap:
    """Delegate to a RunLogger (or nothing); show runner / skill events on the HUD."""

    def __init__(self, inner: Any, recorder: 'VideoRecorder'):
        self._inner = inner if inner is not None else _NullLogger()
        self._rec = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def event(self, kind: str, **fields: Any) -> None:
        try:
            self._rec.on_event(kind, fields)
        except Exception:  # noqa: B902 - the video must never break logging
            _LOG.exception('video HUD update failed')
        self._inner.event(kind, **fields)


class VideoRecorder:
    """
    Record an MP4 of a MujocoWorld while the clock advances.

    world: MujocoWorld attached to its clock. path: output .mp4. fps: frames
    per second of the video; speed: virtual seconds per video second (2 =
    twice as fast, fewer renders). size: (width, height), even numbers.
    camera: MuJoCo camera to film. inset: show the perception debug image.
    hud: draw the text overlay. title: first HUD line (default from
    world.info). crf: x264 quality (lower = better, 23 default).
    """

    def __init__(self, world: Any, path: Any, fps: float = 30.0,
                 size: Tuple[int, int] = (1280, 720), camera: str = 'overview_cam',
                 inset: bool = True, hud: bool = True, speed: float = 1.0,
                 title: Optional[str] = None, crf: int = 23):
        self._mj = require_mujoco()
        w, h = int(size[0]), int(size[1])
        if w <= 0 or h <= 0 or w % 2 or h % 2:
            raise ValueError('video size must be positive even numbers, got %r' % (size,))
        if not (fps > 0 and math.isfinite(fps)) or not (speed > 0 and math.isfinite(speed)):
            raise ValueError('fps and speed must be positive')
        m = world.model
        if w > m.vis.global_.offwidth or h > m.vis.global_.offheight:
            raise ValueError('video size %dx%d exceeds the offscreen buffer %dx%d'
                             % (w, h, m.vis.global_.offwidth, m.vis.global_.offheight))
        cid = self._mj.mj_name2id(m, self._mj.mjtObj.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise ValueError('no camera named %r in the scene' % camera)
        self.world = world
        self.path = Path(path).expanduser()
        self.poster_path = self.path.with_suffix('.jpg')
        self.fps = float(fps)
        self.speed = float(speed)
        self.size = (w, h)
        self.camera = camera
        self._cam_id = int(cid)
        self.inset = bool(inset)
        self.hud = bool(hud)
        self.crf = int(crf)
        info = world.info or {}
        self.title = title or '%s seed %s | %s %s | MuJoCo physics' % (
            info.get('task', '?'), info.get('seed', '?'), info.get('mount', '?'),
            info.get('difficulty', 'normal'))
        task = info.get('task')
        self.required = int(len(info.get('instructions') or []) if task == 'L3'
                            else info.get('required', 0) or 0)
        self.class_to_zone = dict(info.get('class_to_zone') or {}) if task != 'L3' else {}
        self.time_limit = float(info.get('time_limit_s', 0.0) or 0.0)
        self.frames = 0
        self._renderer = None
        self._writer = None
        self._started = False
        self._closed = False
        self._t_start: Optional[float] = None
        self._t_round: Optional[float] = None
        self._t_end: Optional[float] = None
        self._inset_img: Optional[np.ndarray] = None
        self._placed = 0
        self._pick = ''
        self._status = 'preparing'
        self._status_color = _WHITE
        self._last_rgb: Optional[np.ndarray] = None
        self._zone_text = self._zone_labels()

    # ------------------------------------------------------------ wiring
    def wrap_perception(self, perception: Any) -> Any:
        """Return a Perception proxy whose results feed the inset."""
        return _PerceptionTap(perception, self)

    def wrap_logger(self, logger: Any) -> Any:
        """Return a logger proxy whose events feed the HUD (logger may be None)."""
        return _LoggerTap(logger, self)

    def start(self) -> 'VideoRecorder':
        """Start recording at the current virtual time (the first frame is the next step)."""
        if self._started:
            return self
        sim = self.world.sim
        if sim is None:
            raise RuntimeError('VideoRecorder: attach the MujocoWorld to its clock first')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t_start = float(sim.time)
        sim.add_frame_hook(self._on_frame, self.fps / self.speed)
        self._started = True
        return self

    # ----------------------------------------------------------- HUD input
    def set_inset(self, bgr: Optional[np.ndarray]) -> None:
        """Use bgr (HxWx3 uint8) as the picture-in-picture from now on."""
        if bgr is None:
            return
        self._inset_img = np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1])     # -> RGB

    def on_event(self, kind: str, fields: Dict[str, Any]) -> None:
        """Update the HUD state from a runner / skill log event."""
        t = self.world.sim.time if self.world.sim is not None else None
        if kind == 'start':
            self._t_round = t
            self._status, self._status_color = 'round started', _WHITE
        elif kind == 'observe_move':
            self._pick = ''
            self._status, self._status_color = 'observe pose', _WHITE
        elif kind == 'observation':
            if fields.get('ok'):
                n = len(fields.get('detections') or [])
                self._status = 'perception: %d detection%s' % (n, '' if n == 1 else 's')
                self._status_color = _WHITE
            else:
                self._status, self._status_color = ('observation failed: %s'
                                                    % fields.get('reason', '?')), _RED
        elif kind == 'decision':
            act = fields.get('action')
            if act == 'pick':
                self._pick = '%s -> %s' % (fields.get('label', '?'), fields.get('zone', '?'))
                self._status, self._status_color = 'open gripper', _YELLOW
            elif act == 'nothing_to_pick':
                self._status, self._status_color = 'nothing left to pick', _WHITE
        elif kind == 'skill_step':
            stage = str(fields.get('stage', ''))
            if fields.get('ok'):
                self._status = NEXT_STEP.get(stage, stage + ' done')
                self._status_color = _YELLOW
            else:
                self._status = '%s FAILED: %s' % (stage, fields.get('error', '?'))
                self._status_color = _RED
        elif kind == 'pick_result':
            if fields.get('ok'):
                self._placed += 1
                self._status, self._status_color = 'released in %s' % fields.get('zone'), _GREEN
            else:
                self._status = 'pick failed: %s %s' % (fields.get('stage', ''),
                                                       fields.get('error', ''))
                self._status_color = _RED
        elif kind == 'report':
            self._t_end = t
            self._pick = ''
            self._status = 'round over: %s' % (fields.get('stop_reason') or '?')
            self._status_color = _GREEN if fields.get('ok') else _RED

    # ------------------------------------------------------------- frames
    def _zone_labels(self) -> List[Tuple[np.ndarray, str]]:
        out = []
        for name, z in self.world.zones.items():
            p = np.array([float(z.center[0]), float(z.center[1]),
                          float(z.center[2]) + max(float(z.wall_height), 0.0) + 0.01])
            out.append((p, str(z.label or name)))
        return out

    def _open(self) -> None:
        import imageio_ffmpeg
        m = self.world.model
        old = int(m.vis.quality.offsamples)
        m.vis.quality.offsamples = _MSAA           # smoother edges for this renderer only
        try:
            self._renderer = self._mj.Renderer(m, self.size[1], self.size[0])
        finally:
            m.vis.quality.offsamples = old
        self._writer = imageio_ffmpeg.write_frames(
            str(self.path), self.size, fps=self.fps, codec='libx264', pix_fmt_in='rgb24',
            pix_fmt_out='yuv420p', quality=None, macro_block_size=2,
            ffmpeg_log_level='error',
            output_params=['-crf', str(self.crf), '-preset', 'veryfast',
                           '-movflags', '+faststart'])
        self._writer.send(None)

    def _on_frame(self, sim: Any, t: float) -> None:
        if self._closed:
            return
        if self._renderer is None:
            self._open()
        r = self._renderer
        r.update_scene(self.world.data, camera=self._cam_id)
        rgb = r.render()
        img = np.array(rgb, dtype=np.uint8)
        self._draw_zone_names(img)
        if self.inset:
            self._draw_inset(img)
        if self.hud:
            self._draw_hud(img, t)
        self._writer.send(np.ascontiguousarray(img))
        self._last_rgb = img
        self.frames += 1

    def _project(self, p: np.ndarray) -> Optional[Tuple[int, int]]:
        T = self.world.camera_pose(self.camera)
        q = T[:3, :3].T @ (p - T[:3, 3])
        if q[2] <= 0.05:
            return None
        h = float(self.size[1])
        f = (h / 2.0) / math.tan(math.radians(float(self.world.model.cam_fovy[self._cam_id])) / 2)
        u = f * q[0] / q[2] + (self.size[0] - 1) / 2.0
        v = f * q[1] / q[2] + (h - 1) / 2.0
        return int(round(u)), int(round(v))

    def _draw_zone_names(self, img: np.ndarray) -> None:
        import cv2
        for p, text in self._zone_text:
            uv = self._project(p)
            if uv is None:
                continue
            (tw, th), _b = cv2.getTextSize(text, _FONT, 0.6, 2)
            org = (uv[0] - tw // 2, uv[1] - 12)
            cv2.putText(img, text, org, _FONT, 0.6, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(img, text, org, _FONT, 0.6, _WHITE, 2, cv2.LINE_AA)

    def _draw_inset(self, img: np.ndarray) -> None:
        """Draw the perception image in the bottom-right corner (empty table there)."""
        import cv2
        W, H = self.size
        iw = int(W * 0.30) // 2 * 2
        ih = iw * 3 // 4
        x0, y0 = W - iw - 12, H - ih - 12
        if self._inset_img is None:
            box = np.full((ih, iw, 3), 40, dtype=np.uint8)
            cv2.putText(box, 'waiting for the first observation', (12, ih // 2), _FONT, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)
        else:
            box = cv2.resize(self._inset_img, (iw, ih), interpolation=cv2.INTER_AREA)
        img[y0 - 2:y0 + ih + 2, x0 - 2:x0 + iw + 2] = 230
        img[y0:y0 + ih, x0:x0 + iw] = box
        cap = 'perception (%s)' % self.world.task_camera
        org = (x0, y0 - 10)
        cv2.putText(img, cap, org, _FONT, 0.55, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(img, cap, org, _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, t: float) -> None:
        """Draw the text lines on a dark band in the bottom-left corner."""
        import cv2
        t_rec = t - (self._t_start or t)
        if self._t_round is None:
            rnd = 'round not started'
        else:
            t_r = (self._t_end if self._t_end is not None else t) - self._t_round
            rnd = 'round %.1f / %.0f s' % (t_r, self.time_limit)
        score = self.world.score(self.class_to_zone) if self.class_to_zone else None
        if score is not None:
            truth = 'physics: %d correct, %d wrong, %d dropped' % (
                score['correct'], score['wrong'], score['dropped'])
        else:
            truth = 'physics: %s' % ', '.join(
                '%s %d' % (k, len(v)) for k, v in sorted(self.world.placements().items()))
        lines = [(self.title, _WHITE),
                 ('t %.1f s | %s | placed %d/%d' % (t_rec, rnd, self._placed, self.required),
                  _WHITE),
                 (truth, _RED if score is not None and (score['wrong'] or score['dropped'])
                  else _GREEN),
                 (('%s: %s' % (self._pick, self._status)) if self._pick else self._status,
                  self._status_color)]
        scale, lh = 0.6, 26
        width = max(cv2.getTextSize(text, _FONT, scale, 2)[0][0] for text, _c in lines)
        band_w = min(width + 20, int(self.size[0] * 0.66))
        band_h = lh * len(lines) + 12
        x0, y0 = 12, self.size[1] - band_h - 12
        sub = img[y0:y0 + band_h, x0:x0 + band_w]
        sub[:] = (sub.astype(np.uint16) * 2 // 5).astype(np.uint8)
        for i, (text, col) in enumerate(lines):
            cv2.putText(img, text, (x0 + 10, y0 + 24 + i * lh), _FONT, scale, col, 2,
                        cv2.LINE_AA)

    # -------------------------------------------------------------- close
    def close(self) -> Dict[str, Any]:
        """Stop recording, finalize the MP4 and write the poster; return a summary."""
        if self._closed:
            return self.summary()
        self._closed = True
        sim = self.world.sim
        if sim is not None:
            sim.remove_frame_hook(self._on_frame)
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: B902 - report, keep closing
                _LOG.exception('finishing the video failed')
            self._writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._last_rgb is not None:
            import cv2
            cv2.imwrite(str(self.poster_path), self._last_rgb[:, :, ::-1],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        """Return {path, poster, frames, duration_s (video), virtual_s (simulated)}."""
        return {'path': str(self.path), 'poster': str(self.poster_path)
                if self.poster_path.exists() else None, 'frames': self.frames,
                'duration_s': round(self.frames / self.fps, 2),
                'virtual_s': round(self.frames * self.speed / self.fps, 2)}
