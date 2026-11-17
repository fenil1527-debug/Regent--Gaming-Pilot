# OSROKBOT/Classes/global_vars.py
import sys

# Ensure module singleton across both 'global_vars' and 'Classes.global_vars' imports
if __name__ != "global_vars" and "global_vars" not in sys.modules:
    sys.modules["global_vars"] = sys.modules[__name__]
if __name__ != "Classes.global_vars" and "Classes.global_vars" not in sys.modules:
    sys.modules["Classes.global_vars"] = sys.modules[__name__]


class GlobalVars:
    _instance = None
    UI=None
    Q=None
    A=None
    B=None
    C=None
    D=None
    def __new__(cls, UI_INSTANCE=None):
        if cls._instance is None:
            cls._instance = super(GlobalVars, cls).__new__(cls)
            cls._instance.UI_INSTANCE = UI_INSTANCE
            print(cls._instance.UI_INSTANCE)
        return cls._instance

# Usage
GLOBAL_VARS = GlobalVars("Your UI Instance")
another_instance = GlobalVars()  # Will return the same instance as GLOBAL_VARS

# Both references point to the same object
assert GLOBAL_VARS is another_instance
# --- STEP 3 ERROR-REDUCTION & GOVERNANCE UPGRADE (Parts 12, 18, 48, 76) ---

# Operating Mode: "LIVE_ADVISORY_MODE" (read-only coach) or "SIM_MODE" (autonomous simulator)
OPERATING_MODE = "LIVE_ADVISORY_MODE"

# Strict Numerical State Precedence (Fixes FLAW-04: Higher number always wins)
STATE_PRECEDENCE = {
    "RUNNING": 10,
    "DEGRADED": 20,      # Read-only observations allowed; risky actions blocked
    "RECOVERING": 30,    # Re-verifying window/state consistency
    "PAUSED": 40,        # User pause
    "SAFE_STOPPED": 50,  # Critical conflict or oversight failure (Part 46 & 51)
    "TERMINATED": 60,    # Emergency kill-switch or terminal episode end
}
CURRENT_OPERATIONAL_STATE = "RUNNING"

# Multi-Frame Consensus & OCR Thresholds (Fixes FLAW-08, FLAW-10, FLAW-11)
OCR_VERIFIED_THRESHOLD = 0.90
OCR_UNCERTAIN_THRESHOLD = 0.60
CONSENSUS_FRAMES = 3          # Sample up to 3 frames
CONSENSUS_REQUIRED_VOTES = 2  # Require 2 matching frames to eliminate UI flicker
CONSENSUS_DELAY_SEC = 0.08    # 80ms between frames

# Fault Recovery & Safe-Stop Escalation (Fixes FLAW-12, LOG ENTRY #005)
MAX_CONSECUTIVE_FAILURES = 3
MAX_RECOVERY_ATTEMPTS = 2

# Compact Event Log Path (Part 2: <= 1KB per JSONL line)
COMPACT_EVENT_LOG_PATH = "compact_events.jsonl"
