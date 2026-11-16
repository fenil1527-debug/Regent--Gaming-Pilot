# gaming-agent-lab/OSROKBOT/main.py
import os
import sys
import yaml
import cv2

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from perception.window_capture import WindowCapture
from perception.preprocessor import preprocess_game_ui_crop
from perception.ocr_engine import ConfidenceGatedOCR


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def verify_mode_and_safety(runtime_cfg: dict) -> str:
    """Step 1 Architecture Guard: Enforces SIM_MODE vs. LIVE_ADVISORY_MODE (Part 76)."""
    mode = runtime_cfg["system"]["operating_mode"]
    allow_live_input = runtime_cfg["safety_gates"]["allow_input_automation_in_live_mode"]

    if mode == "LIVE_ADVISORY_MODE" and allow_live_input:
        raise PermissionError(
            "CRITICAL SAFETY BLOCK (Part 76): Automated input is forbidden in LIVE_ADVISORY_MODE."
        )

    max_ram = runtime_cfg["hardware_budget"]["max_system_ram_percent"]
    if PSUTIL_AVAILABLE:
        ram_usage = psutil.virtual_memory().percent
        ram_str = f"{ram_usage:.1f}%"
    else:
        ram_str = "N/A (psutil not installed)"
    print(f"[BOOT] Mode: {mode} | System RAM Usage: {ram_str} (Limit: {max_ram}%)")
    return mode


def run_single_image_test(image_path: str, runtime_cfg: dict, roi_cfg: dict) -> None:
    """Replaces Section 1.5 analyze_screen.py with the new 4-stage pipeline."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    rois = roi_cfg["games"]["Test_Screenshot"]["rois"]
    crops = WindowCapture.slice_relative_rois(frame, rois)

    ocr = ConfidenceGatedOCR(
        use_gpu=runtime_cfg["hardware_budget"]["ocr_use_gpu"],
        verified_threshold=runtime_cfg["safety_gates"]["ocr_verified_threshold"],
        uncertain_threshold=runtime_cfg["safety_gates"]["ocr_uncertain_threshold"],
    )

    for roi_name, crop in crops.items():
        processed = preprocess_game_ui_crop(
            crop, upscale_factor=3, save_debug_path="screenshot_processed.png"
        )
        obs = ocr.read_numeric_roi(roi_name, processed)
        print(f"[OBSERVATION] {obs.to_dict()}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    runtime_config = load_yaml(os.path.join(config_dir, "runtime_mode.yaml"))
    roi_config = load_yaml(os.path.join(config_dir, "roi_maps.yaml"))

    active_mode = verify_mode_and_safety(runtime_config)

    # Test on a screenshot if passed via command line: python main.py screenshot.png
    if len(sys.argv) > 1:
        run_single_image_test(sys.argv[1], runtime_config, roi_config)
    else:
        print("Ready! Run 'python main.py screenshot.png' to test your OCR pipeline.")