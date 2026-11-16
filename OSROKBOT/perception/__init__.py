"""
OSROKBOT Perception Package
Step 2: Perception Pipeline (Capture, Preprocessing, Confidence-Gated OCR)
"""

from perception.window_capture import (
    WindowCapture,
    enable_windows_dpi_awareness,
)
from perception.preprocessor import (
    preprocess_game_ui_crop,
)
from perception.ocr_engine import (
    ConfidenceGatedOCR,
    ObservationResult,
)

# Aliases for backward compatibility
enable_dpi_awareness = enable_windows_dpi_awareness
preprocess_for_ocr = preprocess_game_ui_crop
OCREngine = ConfidenceGatedOCR
OCRResult = ObservationResult

__all__ = [
    "WindowCapture",
    "enable_windows_dpi_awareness",
    "enable_dpi_awareness",
    "preprocess_game_ui_crop",
    "preprocess_for_ocr",
    "ConfidenceGatedOCR",
    "ObservationResult",
    "OCREngine",
    "OCRResult",
]
