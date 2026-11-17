# OSROKBOT/Classes/Actions/extract_text_action.py
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import mss

# Safe relative imports for both OSROKBOT/ and Classes/ working directories
try:
    from Classes.Actions.action import Action
except ImportError:
    try:
        from Actions.action import Action
    except ImportError:
        from action import Action

try:
    import Classes.global_vars as global_vars
except ImportError:
    import global_vars

# Optional EasyOCR import with automatic graceful fallback
try:
    import easyocr  # type: ignore
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

# Fallback to PyTesseract (already installed in virtual environment)
try:
    import pytesseract  # type: ignore
    _default_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and Path(_default_tess).is_file():
        pytesseract.pytesseract.tesseract_cmd = _default_tess
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    PYTESSERACT_AVAILABLE = False


class PytesseractReaderAdapter:
    """
    Provides an EasyOCR-compatible readtext() API using PyTesseract.
    Allows multi-frame consensus and confidence-gating to work seamlessly
    even without PyTorch or EasyOCR installed.
    """

    def readtext(
        self,
        image: np.ndarray,
        allowlist: Optional[str] = None,
        detail: int = 1,
    ) -> List[Tuple[Any, str, float]]:
        if image is None or image.size == 0 or pytesseract is None:
            return []

        config_parts = ["--oem 3 --psm 6"]
        if allowlist:
            config_parts.append(f"-c tessedit_char_whitelist={allowlist}")
        config = " ".join(config_parts)

        try:
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config=config
            )
        except Exception as exc:
            return []

        results = []
        n_boxes = len(data.get("text", []))
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            if conf > 0 and word:
                # Normalize confidence from [0..100] to [0.0..1.0] to match EasyOCR
                results.append((None, word, round(conf / 100.0, 4)))

        return results


# Shared singleton OCR reader to avoid GPU/VRAM handle leaks (FLAW-105 & FLAW-05)
_SHARED_OCR_READER = None


def get_shared_ocr_reader():
    """
    Returns the singleton OCR reader instance.
    Prefers EasyOCR (GPU/CPU) if installed; seamlessly falls back to PyTesseract.
    """
    global _SHARED_OCR_READER
    if _SHARED_OCR_READER is not None:
        return _SHARED_OCR_READER

    if EASYOCR_AVAILABLE and easyocr is not None:
        try:
            _SHARED_OCR_READER = easyocr.Reader(["en"], gpu=True, verbose=False)
            return _SHARED_OCR_READER
        except Exception as exc:
            try:
                _SHARED_OCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
                return _SHARED_OCR_READER
            except Exception:
                pass

    if PYTESSERACT_AVAILABLE:
        _SHARED_OCR_READER = PytesseractReaderAdapter()
        return _SHARED_OCR_READER

    raise RuntimeError("No OCR engine available. Install 'pytesseract' or 'easyocr'.")


class ExtractTextAction(Action):
    """
    Upgraded ExtractTextAction supporting:
    - Step 3 Multi-Frame Consensus Voting (FLAW-08 & FLAW-11)
    - Part 12 (Unknown != 0) and Part 18 (Observation Uncertainty)
    - Dual constructor support for both legacy action_sets and new state machines
    """

    def __init__(
        self,
        id: str = "extract_text",
        name: str = "ExtractText",
        data: Optional[dict] = None,
        True_Id: Optional[str] = None,
        False_Id: Optional[str] = None,
        image_path: str = "test.png",
        description: str = "",
        aggregate: bool = False,
        delay: float = 0,
        retard: float = 0,
        skip_check_first_time: bool = False,
    ):
        # Support both legacy Action(skip_check_first_time) and modern Action(id, name, data, True_Id, False_Id)
        try:
            super().__init__(id, name, data or {}, True_Id, False_Id)
        except TypeError:
            super().__init__(skip_check_first_time)
            self.id = id
            self.name = name
            self.data = data or {}
            self.True_Id = True_Id
            self.False_Id = False_Id

        self.image_path = image_path
        self.description = description
        self.aggregate = aggregate
        self.delay = delay
        self.retard = retard

        self.extracted_value = None  # Strictly None when Unknown (Part 12: Unknown != 0)
        self.observation_status = "UNKNOWN"
        self.confidence = 0.0

    @staticmethod
    def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
        """Fixes Section 1.5 & Flaw 2A: 3x cubic upscale + bilateral filter + Otsu threshold."""
        upscaled = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def _grab_roi_frame(self, region: dict) -> np.ndarray:
        with mss.mss() as sct:
            shot = np.array(sct.grab(region))
            return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    def execute(self) -> bool:
        """
        Fixes FLAW-11 & FLAW-08:
        Samples up to 3 frames (80ms apart) and requires 2 matching high-confidence
        reads before accepting the text.
        """
        reader = get_shared_ocr_reader()
        region = self.data.get("region", {"left": 0, "top": 0, "width": 200, "height": 150})

        consensus_frames = getattr(global_vars, "CONSENSUS_FRAMES", 3)
        consensus_delay = getattr(global_vars, "CONSENSUS_DELAY_SEC", 0.08)
        min_uncertain_thresh = getattr(global_vars, "OCR_UNCERTAIN_THRESHOLD", 0.60)
        verified_thresh = getattr(global_vars, "OCR_VERIFIED_THRESHOLD", 0.90)
        required_votes = getattr(global_vars, "CONSENSUS_REQUIRED_VOTES", 2)

        valid_reads = []
        confidences = []

        # Legacy file reading mode if image_path exists and region is default
        if self.image_path and os.path.exists(self.image_path) and "region" not in self.data:
            frame = cv2.imread(self.image_path)
            if frame is not None:
                crops = [frame]
            else:
                crops = []
        else:
            crops = None

        for i in range(consensus_frames):
            if crops is not None and len(crops) > 0:
                crop = crops[0]
            else:
                crop = self._grab_roi_frame(region)

            processed = self._preprocess_crop(crop)

            # If reader is EasyOCR, CRAFT requires 3 channels
            if EASYOCR_AVAILABLE and easyocr and isinstance(reader, easyocr.Reader):
                ocr_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                ocr_input = processed

            results = reader.readtext(ocr_input)

            if results:
                text = " ".join([item[1].strip() for item in results if item[1].strip()])
                avg_conf = float(np.mean([item[2] for item in results]))

                if text and avg_conf >= min_uncertain_thresh:
                    valid_reads.append(text)
                    confidences.append(avg_conf)

            # Early exit if first 2 frames already agree with high confidence
            if len(valid_reads) >= 2 and valid_reads[0] == valid_reads[1]:
                break

            time.sleep(consensus_delay)

        # Case 1: No consensus reached -> Strictly set UNKNOWN and None (never 0)
        if not valid_reads:
            self.extracted_value = None
            self.observation_status = "UNKNOWN"
            self.confidence = 0.0
            if hasattr(self, "log_compact_event"):
                self.log_compact_event("OCR_UNKNOWN", {"region": region})
            return False

        most_common_text, vote_count = Counter(valid_reads).most_common(1)[0]
        mean_conf = float(np.mean(confidences))

        # Propagate to legacy GlobalVars if description (Q, A, B, C, D, marchcount) is provided
        try:
            gv = getattr(global_vars, "GlobalVars", None)
            if gv and hasattr(gv, "_instance"):
                inst = gv()
                clean_txt = most_common_text.replace(",", "").replace('"', "")
                if self.description in ("Q", "A", "B", "C", "D"):
                    setattr(inst, self.description, clean_txt)
        except Exception:
            pass

        if vote_count >= required_votes and mean_conf >= verified_thresh:
            self.extracted_value = most_common_text
            self.observation_status = "VERIFIED"
            self.confidence = round(mean_conf, 4)

            # Handle marchcount legacy check
            if self.description == "marchcount":
                try:
                    parts = most_common_text.split("/")
                    if len(parts) >= 2 and int(parts[0]) < int(parts[1]):
                        return True
                    return False
                except Exception:
                    return True

            return True

        # Case 2: Uncertain read (0.60 - 0.89 or insufficient votes) -> Flag as UNCERTAIN
        self.extracted_value = None
        self.observation_status = "UNCERTAIN"
        self.confidence = round(mean_conf, 4)
        if hasattr(self, "log_compact_event"):
            self.log_compact_event(
                "OCR_UNCERTAIN",
                {"candidate": most_common_text, "votes": vote_count, "conf": self.confidence},
            )
        return False
