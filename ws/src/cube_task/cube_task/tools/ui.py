"""
Minimal operator UI for the on-site tools: OpenCV windows or a headless fallback.

Headless (no display, SSH, or --headless): images are written to a preview
directory instead of shown, keys are read as terminal lines.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_LOG = logging.getLogger('cube_task.tools.ui')


def display_available() -> bool:
    """Return True when an X / Wayland display is set."""
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


class Ui:
    """Show images and read operator input."""

    def __init__(self, headless: bool = False, preview_dir: Optional[str] = None,
                 input_fn=None, output=None):
        self.headless = bool(headless) or not display_available()
        self.preview_dir = Path(preview_dir or '~/vision_runs/preview').expanduser()
        self._input = input_fn or input
        self._out = output or sys.stdout
        self._windows = set()
        self._n = 0
        #: True once the input stream ended (EOF); loops should stop asking.
        self.eof = False
        if not self.headless:
            try:
                import cv2
                cv2.namedWindow('cube_task', cv2.WINDOW_NORMAL)
                cv2.destroyWindow('cube_task')
            except Exception:  # noqa: B902 - no GUI support in this OpenCV
                _LOG.warning('OpenCV has no window support, switching to headless mode')
                self.headless = True

    def say(self, text: str) -> None:
        self._out.write(text + '\n')
        self._out.flush()

    def show(self, name: str, img: Optional[np.ndarray], save_every: int = 1) -> Optional[Path]:
        """
        Show img in window name (headless: save every save_every-th image).

        Returns the preview file written in headless mode, else None.
        """
        if img is None:
            return None
        if self.headless:
            self._n += 1
            if save_every > 0 and self._n % save_every == 0:
                import cv2
                self.preview_dir.mkdir(parents=True, exist_ok=True)
                path = self.preview_dir / ('%s.jpg' % name)
                cv2.imwrite(str(path), img)
                return path
            return None
        import cv2
        cv2.imshow(name, img)
        self._windows.add(name)
        return None

    def preview_path(self, name: str) -> Path:
        """Return the file a headless show(name, ...) writes."""
        return self.preview_dir / ('%s.jpg' % name)

    def command(self, text: str, wait_ms: int = 30) -> Optional[str]:
        """
        Return the next operator command.

        Window mode: the pressed key (None when no key within wait_ms).
        Headless: a terminal line (prompt text), '' for a bare Enter.
        """
        if self.headless:
            return self.prompt(text)
        return self.key(wait_ms)

    def set_mouse_callback(self, name: str, fn) -> bool:
        """Call fn(event, x, y, flags, param) for mouse events in window name (not headless)."""
        if self.headless or name not in self._windows:
            return False
        import cv2
        cv2.setMouseCallback(name, fn)
        return True

    def key(self, wait_ms: int = 1) -> Optional[str]:
        """Return the pressed key as a 1-char string, or None. Headless: always None."""
        if self.headless:
            return None
        import cv2
        k = cv2.waitKey(max(1, int(wait_ms))) & 0xFF
        if k == 255:
            return None
        if k in (10, 13):
            return '\n'
        if k == 27:
            return 'esc'
        return chr(k)

    def prompt(self, text: str, default: str = '') -> str:
        """Ask for a line of text in the terminal."""
        suffix = ' [%s]' % default if default else ''
        try:
            ans = self._input('%s%s: ' % (text, suffix))
        except EOFError:
            ans = ''
            self.eof = True
        ans = (ans or '').strip()
        return ans if ans else default

    def confirm(self, text: str, default: bool = False) -> bool:
        d = 'y' if default else 'n'
        ans = self.prompt('%s (y/n)' % text, d).lower()
        return ans.startswith('y')

    def confirm_word(self, text: str, word: str = 'YES') -> bool:
        """Require the operator to type an exact word (for risky actions)."""
        return self.prompt('%s  Type %s to continue' % (text, word)) == word

    def close(self) -> None:
        if self.headless or not self._windows:
            return
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:  # noqa: B902
            pass


def depth_colormap(depth: Optional[np.ndarray], zmin: float = 0.2,
                   zmax: float = 1.5) -> Optional[np.ndarray]:
    """Return a BGR visualization of a float meter depth image (invalid = black)."""
    if depth is None:
        return None
    import cv2
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    n = np.zeros(d.shape, dtype=np.uint8)
    n[valid] = np.clip((d[valid] - zmin) / max(1e-6, zmax - zmin) * 255.0, 0, 255).astype(
        np.uint8)
    img = cv2.applyColorMap(n, cv2.COLORMAP_JET)
    img[~valid] = 0
    return img


def side_by_side(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Stack two BGR images horizontally (b resized to a's height)."""
    if a is None:
        return b
    if b is None:
        return a
    import cv2
    if b.shape[0] != a.shape[0]:
        s = a.shape[0] / float(b.shape[0])
        b = cv2.resize(b, (int(round(b.shape[1] * s)), a.shape[0]))
    return np.hstack([a, b])
