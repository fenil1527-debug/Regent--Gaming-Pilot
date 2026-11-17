# OSROKBOT/Classes/Actions/action.py
from abc import ABC, abstractmethod
import json
import time

try:
    import Classes.global_vars as global_vars
except ImportError:
    import global_vars

try:
    from window_handler import WindowHandler
except ImportError:
    try:
        from Classes.window_handler import WindowHandler
    except ImportError:
        WindowHandler = None


class Action(ABC):
    # Actions that inject mouse/keyboard input into the game window
    INPUT_ACTIONS = {
        "FindAndClickImageAction",
        "ManualClickAction",
        "ManualMoveAction",
        "ManualScrollAction",
        "PressKeyAction",
    }

    def __init__(
        self,
        id=None,
        name=None,
        data=None,
        True_Id=None,
        False_Id=None,
        skip_check_first_time: bool = False,
    ):
        self.id = id or self.__class__.__name__
        self.name = name or self.__class__.__name__
        self.data = data or {}
        self.True_Id = True_Id
        self.False_Id = False_Id
        self.skip_check_first_time = skip_check_first_time
        self.first_run = True
        self.performance_multiplier = 1
        self.last_status = "UNKNOWN"  # Enforces Part 12: Unknown != 0
        self.delay = 0
        self.retard = 0

    def is_input_action(self) -> bool:
        return self.__class__.__name__ in self.INPUT_ACTIONS

    @staticmethod
    def set_operational_state(requested_state: str) -> str:
        """
        Fixes FLAW-04 (Part 48): Prevents state oscillation by enforcing
        strict numerical precedence unless explicitly reset by human/recovery.
        """
        current = getattr(global_vars, "CURRENT_OPERATIONAL_STATE", "RUNNING")
        precedence = getattr(global_vars, "STATE_PRECEDENCE", {"RUNNING": 10})
        curr_rank = precedence.get(current, 10)
        req_rank = precedence.get(requested_state, 10)

        if req_rank >= curr_rank or requested_state == "RUNNING":
            global_vars.CURRENT_OPERATIONAL_STATE = requested_state
        return getattr(global_vars, "CURRENT_OPERATIONAL_STATE", "RUNNING")

    def log_compact_event(self, outcome: str, details: dict = None) -> None:
        """Part 2 & FLAW-10: Logs every action compactly (<1KB JSONL)."""
        mode = getattr(global_vars, "OPERATING_MODE", "LIVE_ADVISORY_MODE")
        op_state = getattr(global_vars, "CURRENT_OPERATIONAL_STATE", "RUNNING")
        log_path = getattr(global_vars, "COMPACT_EVENT_LOG_PATH", "compact_events.jsonl")

        record = {
            "ts": round(time.time(), 3),
            "mode": mode,
            "op_state": op_state,
            "action_id": self.id,
            "action_type": self.__class__.__name__,
            "name": self.name,
            "outcome": outcome,
            "details": details or {},
        }
        try:
            line = json.dumps(record, separators=(",", ":"))[:1000]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def perform(self):
        """Legacy runner hook calling into execute_safely."""
        delay = getattr(self, "delay", 0)
        retard = getattr(self, "retard", 0)

        tempString = ""
        try:
            if hasattr(self, "image") and self.image != "Media/captchachest.png":
                tempString = f"{self.__class__.__name__}\n{self.image}\n{delay}s delay\n{retard}s retard\n"
        except Exception:
            pass

        try:
            gv = getattr(global_vars, "GlobalVars", None)
            if gv and hasattr(gv, "_instance") and gv().UI and tempString:
                clean_name = (
                    tempString.replace("action", "")
                    .replace("FindAnd", "")
                    .replace(".png", "")
                    .replace("Media/", "")
                    .replace("Action", "")
                )
                gv().UI.currentState(clean_name)
        except Exception:
            pass

        if delay > 0:
            time.sleep(delay)
        result = self.execute_safely()
        if retard > 0:
            time.sleep(retard)
        return result

    def execute_safely(self, signal_emitter=None, window_handler=None):
        """
        Fixes FLAW-11, FLAW-01, FLAW-02:
        Centralized pre-flight validator for all OSROKBOT actions.
        """
        op_state = getattr(global_vars, "CURRENT_OPERATIONAL_STATE", "RUNNING")
        precedence = getattr(global_vars, "STATE_PRECEDENCE", {"RUNNING": 10, "PAUSED": 40})
        op_mode = getattr(global_vars, "OPERATING_MODE", "LIVE_ADVISORY_MODE")

        # 1. Hard Operational State Gate (Part 20, 48, 51)
        if precedence.get(op_state, 10) >= precedence.get("PAUSED", 40):
            self.log_compact_event("BLOCKED_BY_OP_STATE", {"state": op_state})
            return False

        # 2. Degraded State Gate: Allow read-only checks, block mutating input actions (Part 47)
        if op_state == "DEGRADED" and self.is_input_action():
            self.log_compact_event("BLOCKED_IN_DEGRADED_MODE")
            return False

        # 3. Live Advisory Mode Gate (Part 76 & FLAW-02): Never click in live commercial games
        if op_mode == "LIVE_ADVISORY_MODE" and self.is_input_action():
            msg = f"[CO-PILOT ADVICE] Recommended Action: {self.name} (ID: {self.id})"
            if signal_emitter and hasattr(signal_emitter, "emit_log"):
                signal_emitter.emit_log(msg)
            else:
                print(msg)
            self.log_compact_event("ADVISORY_EMITTED", {"recommended": self.name})
            return True

        # 4. Execute the child action with exception isolation (Part 47)
        try:
            result = self.execute()
            status_label = "CONFIRMED_SUCCESS" if result else "CONFIRMED_FALSE"
            self.last_status = status_label
            self.log_compact_event(status_label)
            return result
        except Exception as exc:
            self.last_status = "UNCERTAIN"
            self.set_operational_state("DEGRADED")
            self.log_compact_event("ACTION_EXCEPTION", {"error": str(exc)[:200]})
            return False

    @abstractmethod
    def execute(self):
        """Overridden by each subclass in Classes/Actions/."""
        raise NotImplementedError("Subclasses must implement execute()")
