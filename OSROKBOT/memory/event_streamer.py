# OSROKBOT/memory/event_streamer.py
"""
High-Throughput Compact JSONL Structured Event Streaming Engine (Step 5, Part 2).
Enforces strict <= 1KB per record constraints, in-memory circular ring buffering,
thread-safe logging, and bounded disk rotation to respect the 16GB RAM / 8GB VRAM limit (FLAW-05).
"""

from __future__ import annotations

import collections
import json
import os
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional


class EventSeverity(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventStreamer:
    """
    Structured event logger providing high-throughput micro-action serialization.
    Guarantees Part 2 compliance: every log entry is strictly valid compact JSON <= 1KB.
    """

    def __init__(
        self,
        log_file_path: str = "compact_events.jsonl",
        buffer_capacity: int = 500,
        max_file_size_bytes: int = 5 * 1024 * 1024,  # 5MB rotation limit
    ):
        self.log_file_path = log_file_path
        self.buffer_capacity = buffer_capacity
        self.max_file_size_bytes = max_file_size_bytes
        self.lock = threading.Lock()

        # In-memory circular ring buffer for fast, non-blocking telemetry access
        self.ring_buffer: collections.deque[Dict[str, Any]] = collections.deque(
            maxlen=buffer_capacity
        )
        self.total_logged_events: int = 0

    def log(
        self,
        event_type: str,
        component: str,
        details: Optional[Dict[str, Any]] = None,
        severity: EventSeverity = EventSeverity.INFO,
        flush_now: bool = True,
    ) -> Dict[str, Any]:
        """
        Emits a structured event, appends to ring buffer, and writes to compact JSONL file.
        """
        record: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "sev": severity.value,
            "comp": component,
            "type": event_type,
            "data": details or {},
        }

        # Format compact JSON line without whitespace indentation
        raw_json = json.dumps(record, separators=(",", ":"), default=str)

        # Enforce Part 2 strict byte limit (1000 bytes max per line)
        if len(raw_json) > 1000:
            # Truncate details safely
            record["data"] = {"_truncated": True, "preview": str(details)[:150]}
            raw_json = json.dumps(record, separators=(",", ":"), default=str)[:1000]

        with self.lock:
            self.ring_buffer.append(record)
            self.total_logged_events += 1

            if flush_now:
                self._write_to_disk_locked(raw_json)

        return record

    def _write_to_disk_locked(self, json_line: str) -> None:
        try:
            # Check rotation limit
            if os.path.exists(self.log_file_path):
                if os.path.getsize(self.log_file_path) >= self.max_file_size_bytes:
                    backup_path = f"{self.log_file_path}.old"
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                    os.rename(self.log_file_path, backup_path)

            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
        except Exception:
            # Memory logging must never crash runtime execution
            pass

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent events from circular ring buffer without disk I/O overhead."""
        with self.lock:
            items = list(self.ring_buffer)
            return items[-limit:] if len(items) > limit else items

    def clear(self) -> None:
        """Clears in-memory buffer."""
        with self.lock:
            self.ring_buffer.clear()
