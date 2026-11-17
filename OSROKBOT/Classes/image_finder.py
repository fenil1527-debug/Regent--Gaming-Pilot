# OSROKBOT/Classes/image_finder.py
import json
import os
import time
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import pyautogui
from termcolor import colored

try:
    import Classes.global_vars as global_vars
except ImportError:
    try:
        import global_vars
    except ImportError:
        global_vars = None

try:
    from window_handler import WindowHandler
except ImportError:
    try:
        from Classes.window_handler import WindowHandler
    except ImportError:
        WindowHandler = None


class ImageFinder:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.template_resolution = (1280, 720)  # Reference resolution for templates
        self._window_handler = None

    def _get_window_handler(self):
        if self._window_handler is None and WindowHandler is not None:
            try:
                self._window_handler = WindowHandler()
            except Exception:
                self._window_handler = None
        return self._window_handler

    def _get_scaling_factor(self, screenshot: np.ndarray) -> Tuple[float, float]:
        if screenshot is None or screenshot.shape[0] < 31 or screenshot.shape[1] < 8:
            return (1.0, 1.0)
        screen_resolution = (screenshot.shape[1] - 8, screenshot.shape[0] - 31)
        scale_x = screen_resolution[0] / self.template_resolution[0]
        scale_y = screen_resolution[1] / self.template_resolution[1]
        return (max(scale_x, 0.1), max(scale_y, 0.1))

    def _to_cv2_image(self, screenshot: Any) -> Optional[np.ndarray]:
        if screenshot is None:
            return None
        if isinstance(screenshot, np.ndarray):
            if screenshot.ndim == 2:
                return cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
            elif screenshot.shape[2] == 4:
                return cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            return screenshot
        try:
            return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _match_image(
        self, target_image_path: str, screenshot: Any
    ) -> Tuple[Tuple[float, float], List[Tuple[int, int]], int, float, Optional[np.ndarray], Optional[np.ndarray]]:
        screenshot_cv = self._to_cv2_image(screenshot)
        if screenshot_cv is None:
            return ((1.0, 1.0), [], 0, 0.0, None, None)

        if not os.path.exists(target_image_path):
            return ((1.0, 1.0), [], 0, 0.0, None, screenshot_cv)

        target_image = cv2.imread(target_image_path)
        if target_image is None or target_image.size == 0:
            return ((1.0, 1.0), [], 0, 0.0, None, screenshot_cv)

        scaling_factor = self._get_scaling_factor(screenshot_cv)
        try:
            resized_img = cv2.resize(
                target_image,
                None,
                fx=scaling_factor[0],
                fy=scaling_factor[1],
                interpolation=cv2.INTER_AREA if (scaling_factor[0] < 1.0 or scaling_factor[1] < 1.0) else cv2.INTER_CUBIC,
            )
        except Exception:
            resized_img = target_image

        if (
            resized_img.shape[0] > screenshot_cv.shape[0]
            or resized_img.shape[1] > screenshot_cv.shape[1]
        ):
            return (scaling_factor, [], 0, 0.0, target_image, screenshot_cv)

        try:
            result = cv2.matchTemplate(screenshot_cv, resized_img, cv2.TM_CCOEFF_NORMED)
        except Exception:
            return (scaling_factor, [], 0, 0.0, target_image, screenshot_cv)

        locations = np.where(result >= self.threshold)
        matches = list(zip(*locations[::-1]))

        w, h = (resized_img.shape[1], resized_img.shape[0])
        for pt in matches:
            cv2.rectangle(screenshot_cv, pt, (int(pt[0] + w), int(pt[1] + h)), (255, 0, 255), 4)

        best_val = float(np.max(result)) if result.size > 0 else 0.0
        return scaling_factor, matches, len(matches), best_val, target_image, screenshot_cv

    @staticmethod
    def non_max_suppression_fast(boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick]

    def _sample_consensus(
        self, target_image_path: str, initial_screenshot: Any, win: Any = None
    ) -> Tuple[bool, Any, List[Any], int, float, Any, Any]:
        """
        2-of-3 Multi-Frame Temporal Consensus (Part 18, FLAW-11).
        Evaluates up to 3 frames with 80ms delay to eliminate transient UI flicker.
        """
        consensus_frames = getattr(global_vars, "CONSENSUS_FRAMES", 3) if global_vars else 3
        required_votes = getattr(global_vars, "CONSENSUS_REQUIRED_VOTES", 2) if global_vars else 2
        delay_sec = getattr(global_vars, "CONSENSUS_DELAY_SEC", 0.08) if global_vars else 0.08

        votes = 0
        last_match_data = None
        wh = self._get_window_handler()

        for frame_idx in range(consensus_frames):
            if frame_idx == 0:
                cur_screen = initial_screenshot
            else:
                if wh is not None and win is not None and hasattr(wh, "screenshot_window"):
                    title = getattr(win, "title", "Rise of Kingdoms")
                    cur_screen, _ = wh.screenshot_window(title)
                else:
                    cur_screen = None

            if cur_screen is None:
                # If cannot capture subsequent frames, vote based on the initial frame only
                if frame_idx == 0 and last_match_data is not None and last_match_data[3] >= self.threshold:
                    return (True, *last_match_data)
                break

            scaling_factor, matches, num_matches, best_val, target_img, screen_cv = self._match_image(
                target_image_path, cur_screen
            )
            last_match_data = (scaling_factor, matches, num_matches, best_val, target_img, screen_cv)

            if best_val >= self.threshold:
                votes += 1

            if votes >= required_votes:
                return (True, scaling_factor, matches, num_matches, best_val, target_img, screen_cv)

            if frame_idx < consensus_frames - 1:
                time.sleep(delay_sec)

        is_consensus = votes >= required_votes
        if last_match_data is None:
            last_match_data = ((1.0, 1.0), [], 0, 0.0, None, None)
        return (is_consensus, *last_match_data)

    def find_image(self, target_image_path: str, screenshot: Any) -> bool:
        """
        Checks if template exists in screenshot with multi-frame temporal validation.
        """
        is_match, _, _, _, best_max_val, _, _ = self._sample_consensus(target_image_path, screenshot)
        if is_match and best_max_val >= self.threshold:
            print(colored(f"found {target_image_path} {best_max_val:.2f}%", "green"))
            return True
        else:
            if target_image_path != "Media/captchachest.png":
                print(colored(f"No matches for {target_image_path} found in screenshot.", "red"))
            return False

    def find_and_click_image(
        self,
        target_image_path: str,
        screenshot: Any,
        win: Any,
        x_offset: int,
        y_offset: int,
        max_matches: int,
    ) -> bool:
        """
        Matches image with 2-of-3 consensus, validates bounding boxes,
        and respects LIVE_ADVISORY_MODE safety bounds.
        """
        is_match, best_scale, best_loc, num_matches, best_max_val, target_image, screenshot_cv = (
            self._sample_consensus(target_image_path, screenshot, win)
        )

        if is_match and best_max_val >= self.threshold and target_image is not None:
            rects = np.array(
                [
                    (
                        pt[0],
                        pt[1],
                        pt[0] + target_image.shape[1] * best_scale[0],
                        pt[1] + target_image.shape[0] * best_scale[1],
                    )
                    for pt in best_loc
                ]
            )

            pick = ImageFinder.non_max_suppression_fast(rects, 0.3)

            center_x, center_y = 0, 0
            x_offset_scaled, y_offset_scaled = 0, 0
            win_left = getattr(win, "left", 0) if win else 0
            win_top = getattr(win, "top", 0) if win else 0

            for startX, startY, endX, endY in pick:
                if screenshot_cv is not None:
                    cv2.rectangle(
                        screenshot_cv,
                        (int(startX), int(startY)),
                        (int(endX), int(endY)),
                        (255, 0, 255),
                        2,
                    )
                center_x = int(startX + (endX - startX) // 2 + win_left)
                center_y = int(startY + (endY - startY) // 2 + win_top)
                x_offset_scaled = int(x_offset * best_scale[0])
                y_offset_scaled = int(y_offset * best_scale[1])

            print(colored(f"found {target_image_path} {len(pick)}x at {best_max_val:.2f}%", "green"))

            if max_matches != 0 and len(pick) >= max_matches:
                return False
            if max_matches != 0 and len(pick) < max_matches:
                return True

            target_click_x = center_x + x_offset_scaled
            target_click_y = center_y + y_offset_scaled

            # Part 76 & FLAW-02: Strict Live Advisory Mode Guard (Default-Deny)
            mode = getattr(global_vars, "OPERATING_MODE", "LIVE_ADVISORY_MODE") if global_vars else "LIVE_ADVISORY_MODE"
            if mode == "LIVE_ADVISORY_MODE":
                print(
                    colored(
                        f"[CO-PILOT ADVISORY] Template matched '{target_image_path}' at ({target_click_x}, {target_click_y}) "
                        f"- Input click suppressed in LIVE_ADVISORY_MODE.",
                        "cyan",
                    )
                )
                return True

            # In SIM_MODE or autonomous test mode:
            try:
                prev_active_window = pyautogui.getActiveWindow()
                prev_mouse_x, prev_mouse_y = pyautogui.position()
                pyautogui.click(target_click_x, target_click_y)
                if prev_active_window is not None:
                    prev_active_window.activate()
                pyautogui.moveTo(prev_mouse_x, prev_mouse_y)
            except Exception as e:
                print(f"[ImageFinder] Click dispatch error: {e}")
            return True
        else:
            if target_image_path != "Media/captchachest.png":
                print(colored(f"No matches for {target_image_path} found in screenshot.", "red"))
            if max_matches != 0:
                return True
            return False
