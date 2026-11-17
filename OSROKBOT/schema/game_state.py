# OSROKBOT/schema/game_state.py
"""
Hierarchical Game-State Schema & Uncertainty Quantification (Step 3 & Step 4 Foundation).
Covers:
  - Part 12 (Unknown != 0): All unobserved or low-confidence telemetry evaluates to None / UNKNOWN.
  - Part 18 (Observation Uncertainty & Freshness Decay): Timestamped state freshness tracking.
  - Part 51 (State Consistency): Strict validation & contradiction prevention.
  - Part 52 (Resource Accounting Schema): Four-tier transactional accounting
    (Available, Reserved, Committed, Expended).
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ObservationStatus(str, Enum):
    VERIFIED = "VERIFIED"    # High-confidence, cross-validated read (>= 0.90)
    UNCERTAIN = "UNCERTAIN"  # Ambiguous read (0.60 - 0.89) or partial consensus
    UNKNOWN = "UNKNOWN"      # Unobserved or failed read (< 0.60) - NEVER default to 0


class ResourceType(str, Enum):
    FOOD = "FOOD"
    WOOD = "WOOD"
    STONE = "STONE"
    GOLD = "GOLD"
    GEMS = "GEMS"


class ResourceTier(str, Enum):
    AVAILABLE = "AVAILABLE"  # Liquid balance free to be reserved
    RESERVED = "RESERVED"    # Soft-locked for pending tactical decisions
    COMMITTED = "COMMITTED"  # Hard-locked for dispatched march / building upgrade
    EXPENDED = "EXPENDED"    # Consumed cumulative accounting total


class FourTierResourceAccount(BaseModel):
    """
    Part 52: Four-Tier Resource Accounting.
    Tracks liquid available, soft reserved, hard committed, and spent totals.
    Part 12: available can be None (UNKNOWN), but NEVER defaulted to 0 unless verified!
    """
    resource_type: ResourceType
    available: Optional[int] = Field(
        default=None,
        description="Liquid resource amount. None represents UNKNOWN (Part 12).",
    )
    reserved: int = Field(default=0, ge=0, description="Soft-locked amount")
    committed: int = Field(default=0, ge=0, description="Hard-committed amount")
    expended: int = Field(default=0, ge=0, description="Cumulative consumed total")
    status: ObservationStatus = Field(default=ObservationStatus.UNKNOWN)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated: float = Field(default_factory=time.time)

    def is_fresh(self, max_staleness_sec: float = 15.0) -> bool:
        """Part 18: Telemetry freshness evaluation."""
        return (time.time() - self.last_updated) <= max_staleness_sec

    def reserve(self, amount: int) -> bool:
        """Transactional reservation: moves liquid available into reserved tier."""
        if self.available is None or amount <= 0:
            return False
        effective_available = self.available - self.reserved - self.committed
        if effective_available >= amount:
            self.reserved += amount
            return True
        return False

    def release_reservation(self, amount: int) -> bool:
        """Releases soft-reserved tier back to available."""
        if amount <= 0 or self.reserved < amount:
            return False
        self.reserved -= amount
        return True

    def commit(self, amount: int) -> bool:
        """Transitions soft-reserved resources to hard-committed tier."""
        if amount <= 0 or self.reserved < amount:
            return False
        self.reserved -= amount
        self.committed += amount
        return True

    def expend(self, amount: int) -> bool:
        """Finalizes expenditure: deducts from available/committed and increments expended."""
        if amount <= 0:
            return False
        if self.committed >= amount:
            self.committed -= amount
            if self.available is not None:
                self.available = max(0, self.available - amount)
            self.expended += amount
            return True
        elif self.available is not None and self.available >= amount:
            self.available -= amount
            self.expended += amount
            return True
        return False


class TroopUnit(BaseModel):
    unit_type: str = Field(..., description="infantry, cavalry, archer, siege")
    tier: int = Field(default=1, ge=1, le=5)
    count: Optional[int] = Field(default=None, description="None = UNKNOWN (Part 12)")
    health: float = Field(default=100.0, ge=0.0, le=100.0)


class MarchStatus(str, Enum):
    IDLE = "IDLE"
    MOVING_TO_TARGET = "MOVING_TO_TARGET"
    GATHERING = "GATHERING"
    BATTLING = "BATTLING"
    RETURNING = "RETURNING"


class MarchState(BaseModel):
    march_id: int
    status: MarchStatus = Field(default=MarchStatus.IDLE)
    commander: Optional[str] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    current_x: float = Field(default=0.0)
    current_y: float = Field(default=0.0)
    speed: float = Field(default=1.0, gt=0.0)
    troops: List[TroopUnit] = Field(default_factory=list)
    load_capacity: int = Field(default=10000, ge=0)
    current_load: int = Field(default=0, ge=0)
    target_resource_type: Optional[ResourceType] = None
    eta_sec: float = Field(default=0.0, ge=0.0)
    last_updated: float = Field(default_factory=time.time)


class TileType(str, Enum):
    EMPTY = "EMPTY"
    RESOURCE_FOOD = "RESOURCE_FOOD"
    RESOURCE_WOOD = "RESOURCE_WOOD"
    RESOURCE_STONE = "RESOURCE_STONE"
    RESOURCE_GOLD = "RESOURCE_GOLD"
    BARBARIAN_CAMP = "BARBARIAN_CAMP"
    PLAYER_CITY = "PLAYER_CITY"
    ALLIANCE_PASS = "ALLIANCE_PASS"


class GridTile(BaseModel):
    x: int
    y: int
    tile_type: TileType = Field(default=TileType.EMPTY)
    level: int = Field(default=1, ge=1, le=10)
    remaining_resources: Optional[int] = Field(default=None)
    occupant_id: Optional[str] = None
    danger_level: float = Field(default=0.0, ge=0.0, le=10.0)
    last_scouted: float = Field(default_factory=time.time)


class GameState(BaseModel):
    """
    Unified Hierarchical Game-State Schema (Step 3 & Step 4).
    Domain-invariant representation shared across Simulator and Live Game Adapters.
    """
    timestamp: float = Field(default_factory=time.time)
    operational_mode: str = Field(default="SIM_MODE")
    freshness_decay_sec: float = Field(default=15.0, gt=0.0)

    # City & Macro Information
    city_level: Optional[int] = Field(default=None)
    power: Optional[int] = Field(default=None)
    active_marches_count: Optional[int] = Field(default=None)
    max_marches_count: int = Field(default=5, ge=1, le=10)

    # Four-Tier Resources
    resources: Dict[ResourceType, FourTierResourceAccount] = Field(default_factory=dict)

    # Tactical Marches
    marches: List[MarchState] = Field(default_factory=list)

    # World Grid
    tiles: Dict[str, GridTile] = Field(default_factory=dict)

    # Observation metadata
    global_status: ObservationStatus = Field(default=ObservationStatus.UNKNOWN)
    provenance: str = Field(default="SIMULATOR")  # "SIMULATOR" or "OCR_LIVE_INGESTION"

    @model_validator(mode="after")
    def populate_default_resources(self) -> GameState:
        """Ensures all standard resource types exist in the dictionary."""
        for r_type in ResourceType:
            if r_type not in self.resources:
                self.resources[r_type] = FourTierResourceAccount(
                    resource_type=r_type,
                    available=None,  # Part 12: UNKNOWN, not 0
                    status=ObservationStatus.UNKNOWN,
                )
        return self

    def is_fresh(self, max_age: Optional[float] = None) -> bool:
        """Part 18: Checks whether overall state observation has decayed."""
        threshold = max_age if max_age is not None else self.freshness_decay_sec
        return (time.time() - self.timestamp) <= threshold

    def get_effective_liquid_resource(self, r_type: ResourceType) -> Optional[int]:
        """
        Calculates usable liquid resources considering soft reservations and commits.
        Returns None if resource balance is UNKNOWN (Part 12).
        """
        acct = self.resources.get(r_type)
        if acct is None or acct.available is None:
            return None
        return max(0, acct.available - acct.reserved - acct.committed)

    def to_compact_dict(self) -> Dict[str, Union[float, str, int, dict]]:
        """Compact dictionary for <= 1KB event serialization (Part 2)."""
        res_summary = {}
        for k, v in self.resources.items():
            res_summary[k.value] = {
                "avail": v.available if v.available is not None else "UNK",
                "resv": v.reserved,
                "comm": v.committed,
            }
        return {
            "ts": round(self.timestamp, 2),
            "prov": self.provenance,
            "status": self.global_status.value,
            "marches": f"{len(self.marches)}/{self.max_marches_count}",
            "res": res_summary,
        }
