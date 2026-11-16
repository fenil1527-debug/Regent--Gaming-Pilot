# gaming-agent-lab/OSROKBOT/perception/ocr_engine.py
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import cv2

# Check for EasyOCR availability
try:
    import easyocr  # type: ignore
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Check for PyTesseract availability as a robust fallback
try:
    import pytesseract  # type: ignore
    # Configure well-known default path on Windows if present
    _win_tess = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.name == "nt" and Path(_win_tess).is_file():
        pytesseract.pytesseract.tesseract_cmd = _win_tess
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


@dataclass
class ObservationResult:
    roi_name: str
    raw_text: str
    parsed_value: Optional[int]
    confidence: float
    status: str  # "VERIFIED", "UNCERTAIN", or "UNKNOWN"
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfidenceGatedOCR:
    """
    Confidence-gated OCR Engine enforcing Unknown != 0 and Observation Uncertainty.
    Supports EasyOCR with automatic GPU/CPU fallback, and PyTesseract fallback if EasyOCR
    is not installed.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        verified_threshold: float = 0.90,
        uncertain_threshold: float = 0.60,
    ):
        self.verified_threshold = verified_threshold
        self.uncertain_threshold = uncertain_threshold
        self.use_gpu = use_gpu
        self.engine_name = "none"
        self.reader = None

        if EASYOCR_AVAILABLE:
            try:
                # Attempt initialization with requested GPU setting
                self.reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                self.engine_name = "easyocr"
            except Exception as exc:
                # Fallback to CPU if GPU/CUDA initialization fails
                print(f"[OCR] EasyOCR GPU initialization failed ({exc}). Falling back to CPU...")
                try:
                    self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                    self.engine_name = "easyocr_cpu"
                except Exception as inner_exc:
                    print(f"[OCR] EasyOCR CPU initialization also failed: {inner_exc}")
                    self.reader = None

        if self.reader is None:
            if PYTESSERACT_AVAILABLE:
                self.engine_name = "pytesseract"
                print("[OCR] Using PyTesseract engine.")
            else:
                print("[WARN] Neither EasyOCR nor PyTesseract is available. Install 'easyocr' or 'pytesseract'.")

    @staticmethod
    def _parse_game_number(raw_text: str) -> Optional[int]:
        """
        Converts game shorthand like '1.5M', '240K', or '12,450' into an integer.
        Returns None (UNKNOWN) if text cannot be parsed cleanly. NEVER co-erces to 0!
        """
        if not raw_text:
            return None

        # Clean string: strip whitespace, trailing punctuation (. , : ;)
        cleaned = raw_text.strip().rstrip(".,;:_-").upper().replace(" ", "")
        if not cleaned:
            return None

        multiplier = 1
        if cleaned.endswith("K"):
            multiplier = 1_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("B"):
            multiplier = 1_000_000_000
            cleaned = cleaned[:-1]

        # Handle European decimal commas like 1,5M -> 1.5M
        if "," in cleaned and "." not in cleaned:
            parts = cleaned.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2:
                cleaned = f"{parts[0]}.{parts[1]}"
            else:
                cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", "")

        # Strip any remaining trailing dots (e.g. from OCR artifacts "12450.")
        cleaned = cleaned.strip().rstrip(".")

        # Match clean number pattern (e.g. "12450" or "1.5")
        match = re.match(r"^(\d+(?:\.\d+)?)$", cleaned)
        if not match:
            return None

        try:
            return int(round(float(match.group(1)) * multiplier))
        except (ValueError, OverflowError):
            return None

    def _read_with_easyocr(
        self, image: np.ndarray
    ) -> List[tuple]:
        """
        Prepares image buffer and executes EasyOCR reader.readtext.
        CRAFT detector requires 3-channel (RGB) images to reliably detect text boxes.
        """
        if image is None or image.size == 0:
            return []

        # Convert 1-channel binary/gray image to 3-channel RGB for EasyOCR CRAFT model
        if len(image.shape) == 2:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = image

        try:
            results = self.reader.readtext(
                rgb_img,
                allowlist="0123456789.,KMBkmb",
                detail=1,
            )
            # Sort detected boxes left-to-right by horizontal coordinate
            results.sort(key=lambda item: item[0][0][0] if (item and len(item) > 0 and len(item[0]) > 0) else 0)
            return results
        except Exception as exc:
            print(f"[OCR] EasyOCR read error: {exc}")
            return []

    def _read_with_pytesseract(
        self, image: np.ndarray
    ) -> List[tuple]:
        """Fallback OCR using PyTesseract image_to_data."""
        if image is None or image.size == 0:
            return []

        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,KMBkmb"
        try:
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=config,
            )
        except Exception as exc:
            print(f"[OCR] PyTesseract read error: {exc}")
            return []

        results = []
        n_boxes = len(data.get("text", []))
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            if conf > 0 and word:
                # Normalize tesseract confidence [0..100] to [0.0..1.0] to match EasyOCR
                norm_conf = conf / 100.0
                results.append((None, word, norm_conf))

        return results

    def read_numeric_roi(
        self, roi_name: str, processed_image: np.ndarray
    ) -> ObservationResult:
        """
        Enforces Part 12 (Unknown != 0) and Part 18 (Observation Uncertainty).
        """
        now = time.time()

        if processed_image is None or processed_image.size == 0:
            return ObservationResult(
                roi_name=roi_name,
                raw_text="",
                parsed_value=None,
                confidence=0.0,
                status="UNKNOWN",
                timestamp=now,
            )

        # 1. Query the active OCR engine
        if self.engine_name.startswith("easyocr") and self.reader is not None:
            results = self._read_with_easyocr(processed_image)
        elif PYTESSERACT_AVAILABLE:
            results = self._read_with_pytesseract(processed_image)
        else:
            results = []

        # Case 1: Nothing detected -> Strictly return UNKNOWN and None (never 0)
        if not results:
            return ObservationResult(
                roi_name=roi_name,
                raw_text="",
                parsed_value=None,
                confidence=0.0,
                status="UNKNOWN",
                timestamp=now,
            )

        # Combine detected text blocks and average their confidence
        combined_text = "".join([str(item[1]).strip() for item in results])
        confidences = [float(item[2]) for item in results if len(item) > 2]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        parsed_val = self._parse_game_number(combined_text)

        # If text couldn't be parsed as a valid number, mark as UNKNOWN
        if parsed_val is None:
            return ObservationResult(
                roi_name=roi_name,
                raw_text=combined_text,
                parsed_value=None,
                confidence=round(avg_confidence, 4),
                status="UNKNOWN",
                timestamp=now,
            )

        # Gate by confidence thresholds (Flaw-10 resolution)
        if avg_confidence >= self.verified_threshold:
            status = "VERIFIED"
        elif avg_confidence >= self.uncertain_threshold:
            status = "UNCERTAIN"
        else:
            # Below threshold is rejected as UNKNOWN so bad reads don't corrupt state
            return ObservationResult(
                roi_name=roi_name,
                raw_text=combined_text,
                parsed_value=None,
                confidence=round(avg_confidence, 4),
                status="UNKNOWN",
                timestamp=now,
            )

        return ObservationResult(
            roi_name=roi_name,
            raw_text=combined_text,
            parsed_value=parsed_val,
            confidence=round(avg_confidence, 4),
            status=status,
            timestamp=now,
        )
