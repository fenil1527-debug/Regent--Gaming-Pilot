# OSROKBOT/simulator/adapters.py
"""
Game-Specific Ingestion & Translation Adapters (Part 16, Part 27, FLAW-07).
Decouples core strategic planning from environment-specific perceptual formats.
Translates both live OCR perception feeds and simulator outputs into the unified GameState schema.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from schema.game_state import (
    FourTierResourceAccount,
    GameState,
    ObservationStatus,
    ResourceType,
)
from simulator.engine import SimAction, SimActionType, StrategySimulator


class BaseGameAdapter(ABC):
    """Abstract interface for game state ingestion (Part 16 & Part 27)."""

    @abstractmethod
    def ingest(self, raw_input: Any) -> GameState:
        """Translates raw sensor telemetry into unified GameState."""
        raise NotImplementedError

    @abstractmethod
    def translate_action(self, action: SimAction) -> Dict[str, Any]:
        """Translates abstract simulator decision into game-specific instructions/advice."""
        raise NotImplementedError


class ROKLiveAdapter(BaseGameAdapter):
    """
    Ingestion adapter for Rise of Kingdoms live game client.
    Converts OCR and visual recognition data into the unified GameState schema.
    Enforces Part 12 (Unknown != 0) and Part 76 (Advisory Translation).
    """

    def _parse_game_number(self, text: Optional[str]) -> Optional[int]:
        if not text:
            return None
        clean = text.strip().upper().replace(",", "").replace(" ", "")
        mult = 1.0
        if clean.endswith("K"):
            mult = 1_000.0
            clean = clean[:-1]
        elif clean.endswith("M"):
            mult = 1_000_000.0
            clean = clean[:-1]
        elif clean.endswith("B"):
            mult = 1_000_000_000.0
            clean = clean[:-1]

        try:
            val = float(clean)
            return int(val * mult)
        except ValueError:
            return None

    def ingest(self, raw_observations: Dict[str, Any]) -> GameState:
        """
        Parses live perception outputs:
        Expected keys in raw_observations:
          - 'march_count': str e.g. '2/5'
          - 'food': {'text': str, 'conf': float}
          - 'wood': {'text': str, 'conf': float}
          - 'stone': {'text': str, 'conf': float}
          - 'gold': {'text': str, 'conf': float}
          - 'city_level': int
        """
        current_time = time.time()
        resources: Dict[ResourceType, FourTierResourceAccount] = {}

        # 1. Parse Resource Telemetry with Strict Part 12 Gating
        res_mapping = {
            ResourceType.FOOD: "food",
            ResourceType.WOOD: "wood",
            ResourceType.STONE: "stone",
            ResourceType.GOLD: "gold",
        }

        for r_type, key in res_mapping.items():
            entry = raw_observations.get(key, {})
            raw_text = entry.get("text", "")
            conf = float(entry.get("conf", 0.0))

            parsed_val = self._parse_game_number(raw_text)

            if parsed_val is not None and conf >= 0.90:
                status = ObservationStatus.VERIFIED
                available = parsed_val
            elif parsed_val is not None and conf >= 0.60:
                status = ObservationStatus.UNCERTAIN
                # Enforce Part 12: UNCERTAIN values must NOT corrupt liquid available balance
                available = None
            else:
                status = ObservationStatus.UNKNOWN
                available = None

            resources[r_type] = FourTierResourceAccount(
                resource_type=r_type,
                available=available,
                reserved=0,
                committed=0,
                expended=0,
                status=status,
                confidence=round(conf, 4),
                last_updated=current_time,
            )

        # 2. Parse March Counters (e.g., '2/5')
        active_marches: Optional[int] = None
        max_marches = 5
        raw_marches = raw_observations.get("march_count")
        if isinstance(raw_marches, str):
            match = re.search(r"(\d+)\s*/\s*(\d+)", raw_marches)
            if match:
                active_marches = int(match.group(1))
                max_marches = int(match.group(2))

        return GameState(
            timestamp=current_time,
            operational_mode="LIVE_ADVISORY_MODE",
            freshness_decay_sec=15.0,
            city_level=raw_observations.get("city_level"),
            active_marches_count=active_marches,
            max_marches_count=max_marches,
            resources=resources,
            global_status=ObservationStatus.VERIFIED if active_marches is not None else ObservationStatus.UNKNOWN,
            provenance="OCR_LIVE_INGESTION",
        )

    def translate_action(self, action: SimAction) -> Dict[str, Any]:
        """
        Translates a strategic simulator decision into advisory instructions for the player.
        Part 76: In LIVE_ADVISORY_MODE, never inject inputs; emit human advisory only.
        """
        if action.action_type == SimActionType.DISPATCH_GATHER_MARCH:
            advice = (
                f"[CO-PILOT ADVISORY] Recommend dispatching March #{action.march_id} "
                f"to gather {action.resource_type.value if action.resource_type else 'Resources'} "
                f"at coordinates ({action.target_x}, {action.target_y})."
            )
        elif action.action_type == SimActionType.DISPATCH_ATTACK_MARCH:
            advice = (
                f"[CO-PILOT ADVISORY] Recommend dispatching March #{action.march_id} "
                f"to attack Barbarians at coordinates ({action.target_x}, {action.target_y})."
            )
        elif action.action_type == SimActionType.UPGRADE_BUILDING:
            advice = f"[CO-PILOT ADVISORY] Safe to proceed with City Hall upgrade (Resource margin verified)."
        elif action.action_type == SimActionType.RECALL_MARCH:
            advice = f"[CO-PILOT ADVISORY] Recommend recalling March #{action.march_id} to base."
        else:
            advice = "[CO-PILOT ADVISORY] Hold current position. No urgent action recommended."

        return {
            "mode": "LIVE_ADVISORY_MODE",
            "action_type": action.action_type.value,
            "advisory_text": advice,
            "input_injection_blocked": True,
        }


class SimulatorAdapter(BaseGameAdapter):
    """Bridge adapter extracting GameState directly from StrategySimulator."""

    def ingest(self, sim: StrategySimulator) -> GameState:
        return sim.get_state()

    def translate_action(self, action: SimAction) -> Dict[str, Any]:
        return {
            "mode": "SIM_MODE",
            "action_type": action.action_type.value,
            "march_id": action.march_id,
            "target": (action.target_x, action.target_y),
            "input_injection_blocked": False,
        }
