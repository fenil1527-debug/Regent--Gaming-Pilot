# gaming-agent-lab/OSROKBOT/perception/preprocessor.py
import cv2
import numpy as np
from typing import Optional


def preprocess_game_ui_crop(
    crop: np.ndarray,
    upscale_factor: int = 3,
    save_debug_path: Optional[str] = None,
) -> np.ndarray:
    """
    Fixes Section 1.5 & Flaw 2A:
    1. Upscales small game UI crops by 3x using cubic interpolation.
    2. Converts to grayscale.
    3. Applies a bilateral filter (removes background noise while keeping text edges sharp).
    4. Applies Otsu binary thresholding so bright game text stands out cleanly.
    """
    if crop is None or crop.size == 0:
        raise ValueError("Received an empty image crop for preprocessing.")

    # 1. Upscale 3x so tiny 200x150 crops become 600x450
    upscaled = cv2.resize(
        crop,
        None,
        fx=upscale_factor,
        fy=upscale_factor,
        interpolation=cv2.INTER_CUBIC,
    )

    # 2. Convert to grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # 3. Remove background game texture while keeping font borders sharp
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Binarize text (invert so bright white/yellow game numbers become dark text on white)
    _, binary = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Optional: Save processed snapshot selectively for diagnosis (Part 2)
    if save_debug_path:
        cv2.imwrite(save_debug_path, binary)

    return binary