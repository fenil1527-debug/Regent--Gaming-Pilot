# gaming-agent-lab/OSROKBOT/perception/window_capture.py
import ctypes
import os
import platform
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mss
import cv2

_DPI_AWARENESS_INITIALIZED = False


def enable_windows_dpi_awareness() -> None:
    """
    Prevents Windows 10/11 display scaling from shifting capture coordinates.
    Guarded with idempotency flag to prevent Win32 error on repeated calls.
    """
    global _DPI_AWARENESS_INITIALIZED
    if _DPI_AWARENESS_INITIALIZED:
        return

    if platform.system() == "Windows":
        try:
            # 2 = PROCESS_PER_MONITOR_DPI_AWARE
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
            _DPI_AWARENESS_INITIALIZED = True
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
                _DPI_AWARENESS_INITIALIZED = True
            except Exception:
                pass


class WindowCapture:
    """
    High-performance DPI-aware screen capture utility using MSS.
    Supports full monitor grabs, window-targeted grabs, and safe relative ROI slicing.
    """

    def __init__(self, monitor_index: int = 1):
        enable_windows_dpi_awareness()
        self.sct = mss.mss()
        self.monitor_index = monitor_index

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self) -> None:
        """Frees OS GDI handles allocated by MSS to prevent resource exhaustion."""
        if hasattr(self, "sct") and self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass

    def _get_safe_monitor(self) -> Dict[str, int]:
        """Safely returns monitor geometry, guarding against IndexError."""
        monitors = self.sct.monitors
        if not monitors:
            raise RuntimeError("MSS detected no active display monitors.")

        # monitors[0] is virtual desktop (all screens); monitors[1] is primary
        if 0 <= self.monitor_index < len(monitors):
            return monitors[self.monitor_index]
        elif len(monitors) > 1:
            return monitors[1]
        return monitors[0]

    def grab_full_frame(self) -> np.ndarray:
        """Captures the active screen in contiguous BGR format for OpenCV."""
        monitor = self._get_safe_monitor()
        raw_shot = self.sct.grab(monitor)
        frame = np.array(raw_shot, dtype=np.uint8)
        # Convert BGRA buffer from MSS to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(bgr)

    @staticmethod
    def slice_relative_rois(
        frame: Optional[np.ndarray], roi_map: Dict[str, Union[List[float], Tuple[float, ...]]]
    ) -> Dict[str, np.ndarray]:
        """
        Slices relative percentage boxes [x_min, y_min, x_max, y_max] from any frame.

        Safeguards:
        - Validates that frame is not None and not empty.
        - Clamps coordinates strictly within [0, width] and [0, height].
        - Returns contiguous array copies to prevent C-extension memory alignment errors.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            print("[WARN] WindowCapture.slice_relative_rois received an invalid or None frame.")
            return {}

        if not roi_map or not isinstance(roi_map, dict):
            return {}

        height, width = frame.shape[:2]
        crops: Dict[str, np.ndarray] = {}

        for name, coords in roi_map.items():
            if not coords or len(coords) < 4:
                print(f"[WARN] Skipping ROI '{name}': Expected 4 coordinates [x_min, y_min, x_max, y_max].")
                continue

            try:
                x_min = max(0, min(width - 1, int(float(coords[0]) * width)))
                y_min = max(0, min(height - 1, int(float(coords[1]) * height)))
                x_max = max(x_min + 1, min(width, int(float(coords[2]) * width)))
                y_max = max(y_min + 1, min(height, int(float(coords[3]) * height)))
            except (ValueError, TypeError) as exc:
                print(f"[WARN] Invalid coordinates for ROI '{name}': {coords} ({exc})")
                continue

            if x_max > x_min and y_max > y_min:
                # Return a contiguous memory copy so OpenCV/torch C-extensions do not fail
                crop_view = frame[y_min:y_max, x_min:x_max]
                crops[name] = np.ascontiguousarray(crop_view.copy())

        return crops
