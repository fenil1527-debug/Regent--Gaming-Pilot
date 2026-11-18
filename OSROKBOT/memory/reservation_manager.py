# OSROKBOT/memory/reservation_manager.py
"""
Four-Tier Transactional Resource Reservation Subsystem (Step 5, Part 52, Part 53).
Directly resolves FLAW-03 (Resource Reservation Deadlock on Ambiguous Outcomes).

Mechanics:
  - Bounded reservation lifecycle with hard expiration timeout (T_timeout = 3.0s).
  - Four-tier state accounting: Available -> Reserved -> Committed -> Expended.
  - Thread-safe locking (Part 50 Concurrency Control).
  - Deadlock prevention engine: automatically detects and recovers stale locks
    caused by dropped frames, OCR misreads, or interrupted actions.
"""

from __future__ import annotations

import threading
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple

from schema.game_state import (
    FourTierResourceAccount,
    ObservationStatus,
    ResourceType,
)


class ReservationStatus(str, Enum):
    ACTIVE = "ACTIVE"          # Soft-reserved, pending action confirmation
    COMMITTED = "COMMITTED"    # Action succeeded; funds hard-committed/expended
    RELEASED = "RELEASED"      # Action failed or cancelled; balance restored to Available
    EXPIRED = "EXPIRED"        # Timed out without confirmation; automatically recovered


class ResourceReservation:
    def __init__(
        self,
        reservation_id: str,
        resource_type: ResourceType,
        amount: int,
        timeout_sec: float = 3.0,
        purpose: str = "ACTION_DISPATCH",
    ):
        self.reservation_id = reservation_id
        self.resource_type = resource_type
        self.amount = amount
        self.created_at = time.time()
        self.timeout_sec = timeout_sec
        self.purpose = purpose
        self.status = ReservationStatus.ACTIVE
        self.resolved_at: Optional[float] = None

    def is_expired(self, current_time: Optional[float] = None) -> bool:
        if self.status != ReservationStatus.ACTIVE:
            return False
        now = current_time if current_time is not None else time.time()
        return (now - self.created_at) >= self.timeout_sec


class ReservationManager:
    """
    Manages transactional multi-tier resource reservations with deadlock recovery.
    Fixes FLAW-03: Prevents permanent resource starvation when action outcomes are ambiguous.
    """

    def __init__(
        self,
        default_timeout_sec: float = 3.0,
        reconcile_interval_sec: float = 1.0,
    ):
        self.default_timeout_sec = default_timeout_sec
        self.reconcile_interval_sec = reconcile_interval_sec
        self.lock = threading.Lock()

        # Storage for accounts and active/historical reservations
        self.accounts: Dict[ResourceType, FourTierResourceAccount] = {}
        self.reservations: Dict[str, ResourceReservation] = {}

        # Telemetry
        self.deadlocks_prevented_count: int = 0
        self.total_reservations_created: int = 0

    def register_account(self, account: FourTierResourceAccount) -> None:
        """Registers a four-tier resource account to be governed by the manager."""
        with self.lock:
            self.accounts[account.resource_type] = account

    def reserve(
        self,
        resource_type: ResourceType,
        amount: int,
        timeout_sec: Optional[float] = None,
        purpose: str = "ACTION_DISPATCH",
    ) -> Optional[str]:
        """
        Attempts to reserve 'amount' of a resource.
        Returns a reservation_id string on success, or None if insufficient balance.
        """
        if amount <= 0:
            return None

        actual_timeout = timeout_sec if timeout_sec is not None else self.default_timeout_sec

        with self.lock:
            # First, trigger background cleanup of any expired reservations
            self._cleanup_expired_reservations_locked()

            acct = self.accounts.get(resource_type)
            if acct is None or acct.available is None:
                return None

            # Check effective liquid headroom: available - reserved - committed
            effective_liquid = acct.available - acct.reserved - acct.committed
            if effective_liquid < amount:
                return None

            # Execute transactional reserve
            acct.reserved += amount
            acct.last_updated = time.time()

            res_id = str(uuid.uuid4())[:8]
            res = ResourceReservation(
                reservation_id=res_id,
                resource_type=resource_type,
                amount=amount,
                timeout_sec=actual_timeout,
                purpose=purpose,
            )
            self.reservations[res_id] = res
            self.total_reservations_created += 1
            return res_id

    def commit(self, reservation_id: str, expend_immediately: bool = True) -> bool:
        """
        Commits an active reservation upon confirmed action success.
        Moves funds from Reserved to Committed, and optionally Expended.
        """
        with self.lock:
            res = self.reservations.get(reservation_id)
            if not res or res.status != ReservationStatus.ACTIVE:
                return False

            acct = self.accounts.get(res.resource_type)
            if not acct:
                return False

            # Transfer reserved -> committed
            acct.reserved = max(0, acct.reserved - res.amount)
            if expend_immediately:
                if acct.available is not None:
                    acct.available = max(0, acct.available - res.amount)
                acct.expended += res.amount
            else:
                acct.committed += res.amount

            acct.last_updated = time.time()
            res.status = ReservationStatus.COMMITTED
            res.resolved_at = time.time()
            return True

    def release(self, reservation_id: str, reason: str = "CANCELLED") -> bool:
        """
        Releases an active reservation back to available balance.
        Called on confirmed action failure or user cancellation.
        """
        with self.lock:
            res = self.reservations.get(reservation_id)
            if not res or res.status != ReservationStatus.ACTIVE:
                return False

            acct = self.accounts.get(res.resource_type)
            if acct:
                acct.reserved = max(0, acct.reserved - res.amount)
                acct.last_updated = time.time()

            res.status = ReservationStatus.RELEASED
            res.resolved_at = time.time()
            return True

    def reconcile_and_recover_deadlocks(
        self, current_time: Optional[float] = None
    ) -> List[str]:
        """
        Explicit deadlock recovery routine (FLAW-03).
        Scans all reservations, rolls back any active reservations exceeding T_timeout (3.0s),
        and returns list of recovered reservation IDs.
        """
        with self.lock:
            return self._cleanup_expired_reservations_locked(current_time=current_time)

    def _cleanup_expired_reservations_locked(
        self, current_time: Optional[float] = None
    ) -> List[str]:
        now = current_time if current_time is not None else time.time()
        recovered_ids: List[str] = []

        for res_id, res in self.reservations.items():
            if res.is_expired(now):
                # Rollback balance from reserved back to liquid available
                acct = self.accounts.get(res.resource_type)
                if acct:
                    acct.reserved = max(0, acct.reserved - res.amount)
                    acct.last_updated = now

                res.status = ReservationStatus.EXPIRED
                res.resolved_at = now
                self.deadlocks_prevented_count += 1
                recovered_ids.append(res_id)

        return recovered_ids

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic telemetry of current accounting tiers."""
        with self.lock:
            summary = {
                "deadlocks_prevented": self.deadlocks_prevented_count,
                "total_created": self.total_reservations_created,
                "active_reservations": sum(
                    1 for r in self.reservations.values() if r.status == ReservationStatus.ACTIVE
                ),
                "tiers": {},
            }
            for r_type, acct in self.accounts.items():
                summary["tiers"][r_type.value] = {
                    "available": acct.available,
                    "reserved": acct.reserved,
                    "committed": acct.committed,
                    "expended": acct.expended,
                }
            return summary
