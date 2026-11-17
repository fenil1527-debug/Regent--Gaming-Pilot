# OSROKBOT/Classes/state_machine.py
import json
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

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


class StateMachine:
    """
    Upgraded State Machine (Step 3C, LOG ENTRY #005, FLAW-12, FLAW-04).
    Features:
      - Operational state governance (RUNNING, DEGRADED, RECOVERING, PAUSED, SAFE_STOPPED, TERMINATED).
      - Pre-flight execution via current_action.execute_safely().
      - Bounded failure tracking & consecutive failure counter (Part 3).
      - Degraded state recovery with auto-escalation to SAFE_STOPPED on repeated errors (Part 47, 51).
      - Compact JSONL logging (<1KB/event) for episodic traceability (Part 2).
    """

    def __init__(self):
        self.states: Dict[str, Tuple[Any, Any, Any]] = {}
        self.current_state: Optional[str] = None
        self.consecutive_failures: int = 0
        self.failure_counts: Dict[str, int] = {}
        self.recovery_attempts: int = 0
        self._window_handler: Optional[Any] = None

    def _get_window_handler(self) -> Optional[Any]:
        if self._window_handler is None and WindowHandler is not None:
            try:
                self._window_handler = WindowHandler()
            except Exception:
                self._window_handler = None
        return self._window_handler

    def _get_op_state(self) -> str:
        if global_vars and hasattr(global_vars, "CURRENT_OPERATIONAL_STATE"):
            return getattr(global_vars, "CURRENT_OPERATIONAL_STATE", "RUNNING")
        return "RUNNING"

    def _set_op_state(self, new_state: str) -> None:
        if global_vars and hasattr(global_vars, "STATE_PRECEDENCE"):
            precedence = getattr(global_vars, "STATE_PRECEDENCE", {"RUNNING": 10})
            curr = getattr(global_vars, "CURRENT_OPERATIONAL_STATE", "RUNNING")
            curr_rank = precedence.get(curr, 10)
            new_rank = precedence.get(new_state, 10)
            if new_rank >= curr_rank or new_state == "RUNNING":
                global_vars.CURRENT_OPERATIONAL_STATE = new_state
        elif global_vars:
            global_vars.CURRENT_OPERATIONAL_STATE = new_state

    def _log_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Part 2 & FLAW-10: Logs state machine events compactly to JSONL (<1KB)."""
        log_path = getattr(global_vars, "COMPACT_EVENT_LOG_PATH", "compact_events.jsonl") if global_vars else "compact_events.jsonl"
        record = {
            "ts": round(time.time(), 3),
            "component": "StateMachine",
            "current_state": self.current_state,
            "op_state": self._get_op_state(),
            "event": event_type,
            "consecutive_failures": self.consecutive_failures,
            "details": details or {},
        }
        try:
            line = json.dumps(record, separators=(",", ":"))[:1000]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def add_state(
        self,
        name: str,
        state: Any,
        next_state_on_success: Optional[Union[str, Callable]] = None,
        next_state_on_failure: Optional[Union[str, Callable]] = None,
    ) -> None:
        self.states[name] = (state, next_state_on_success, next_state_on_failure)

    def set_initial_state(self, name: str) -> None:
        self.current_state = name

    def _attempt_degraded_recovery(self) -> bool:
        """
        Bounded recovery attempt in DEGRADED mode (Part 47, Step 3C).
        Validates window presence/focus and verifies system integrity.
        """
        self.recovery_attempts += 1
        max_recovery = getattr(global_vars, "MAX_RECOVERY_ATTEMPTS", 2) if global_vars else 2
        self._set_op_state("RECOVERING")
        self._log_event("RECOVERY_ATTEMPT_STARTED", {"attempt": self.recovery_attempts})

        wh = self._get_window_handler()
        window_ok = False
        op_mode = getattr(global_vars, "OPERATING_MODE", "LIVE_ADVISORY_MODE") if global_vars else "LIVE_ADVISORY_MODE"

        if op_mode == "SIM_MODE":
            window_ok = True
        elif wh is not None and hasattr(wh, "get_window"):
            try:
                win = wh.get_window("Rise of Kingdoms")
                if win is not None:
                    window_ok = True
                    try:
                        wh.activate_window("Rise of Kingdoms")
                    except Exception:
                        pass
            except Exception:
                window_ok = False
        else:
            window_ok = True

        if window_ok and self.recovery_attempts <= max_recovery:
            self._set_op_state("RUNNING")
            self.recovery_attempts = 0
            self._log_event("RECOVERY_SUCCEEDED", {"state": "RUNNING"})
            return True
        else:
            self._set_op_state("SAFE_STOPPED")
            self._log_event("RECOVERY_FAILED_ESCALATED_SAFE_STOPPED", {
                "attempts": self.recovery_attempts,
                "window_ok": window_ok,
            })
            print(f"[StateMachine] Recovery failed after {self.recovery_attempts} attempts. Escalated to SAFE_STOPPED.")
            return False

    def execute(self) -> bool:
        """
        Executes one tick of the state machine.
        Returns True if action succeeded, False otherwise.
        """
        if self.current_state is None:
            raise Exception("Initial state is not set")

        op_state = self._get_op_state()

        # 1. Terminal / Halt Guard (Part 20, 51)
        if op_state in ("TERMINATED", "SAFE_STOPPED"):
            self._log_event("EXECUTION_HALTED_BY_OP_STATE", {"op_state": op_state})
            return False

        # 2. Pause Guard (Part 20)
        if op_state == "PAUSED":
            self._log_event("EXECUTION_PAUSED", {"holding_state": self.current_state})
            return False

        # 3. Degraded Recovery (Part 47)
        if op_state == "DEGRADED":
            if not self._attempt_degraded_recovery():
                return False

        current_state_info = self.states.get(self.current_state)
        if not current_state_info:
            print(f"[StateMachine] Unknown state: {self.current_state}")
            self.consecutive_failures += 1
            max_consecutive = getattr(global_vars, "MAX_CONSECUTIVE_FAILURES", 3) if global_vars else 3
            if self.consecutive_failures >= max_consecutive:
                self._set_op_state("SAFE_STOPPED")
            return False

        action = current_state_info[0]

        # 4. Safe Pre-Flight Execution (FLAW-11, Part 17)
        try:
            if hasattr(action, "execute_safely"):
                result = action.execute_safely()
            elif hasattr(action, "perform"):
                result = action.perform()
            else:
                result = action.execute()
        except Exception as exc:
            self._log_event("ACTION_UNHANDLED_EXCEPTION", {"error": str(exc)[:200]})
            result = False

        max_consecutive = getattr(global_vars, "MAX_CONSECUTIVE_FAILURES", 3) if global_vars else 3

        # 5. Outcome Processing & State Transition
        if result:
            self.consecutive_failures = 0
            self.recovery_attempts = 0
            if self._get_op_state() in ("DEGRADED", "RECOVERING"):
                self._set_op_state("RUNNING")

            next_state_on_success = current_state_info[1]
            if callable(next_state_on_success):
                next_state = next_state_on_success()
            else:
                next_state = next_state_on_success

            self._log_event("TRANSITION_SUCCESS", {"from": self.current_state, "to": next_state})
            self.current_state = next_state
            return True
        else:
            self.consecutive_failures += 1
            self.failure_counts[self.current_state] = self.failure_counts.get(self.current_state, 0) + 1

            self._log_event("ACTION_FAILED", {
                "state": self.current_state,
                "consecutive_failures": self.consecutive_failures,
            })

            # Check consecutive failure threshold (FLAW-12)
            if self.consecutive_failures >= max_consecutive:
                self._set_op_state("SAFE_STOPPED")
                self._log_event("ESCALATED_SAFE_STOPPED", {
                    "consecutive_failures": self.consecutive_failures,
                    "threshold": max_consecutive,
                })
                print(
                    f"[StateMachine] CRITICAL: Reached {self.consecutive_failures} consecutive failures. "
                    f"State machine escalated to SAFE_STOPPED to prevent unbounded false-branch loops."
                )
                return False

            next_state_on_failure = current_state_info[2]
            if callable(next_state_on_failure):
                next_state = next_state_on_failure()
            else:
                next_state = next_state_on_failure

            self._log_event("TRANSITION_FAILURE", {"from": self.current_state, "to": next_state})
            self.current_state = next_state
            return False
