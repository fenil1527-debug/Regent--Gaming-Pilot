# OSROKBOT/simulator/engine.py
"""
Deterministic Controlled Strategy Simulator Engine (Step 4, v0.1 Prototype).
Resolves FLAW-07 (Simulator-to-Live Transfer Gap) and enforces Part 30, Part 36, Part 52.

Features:
  - 100% deterministic discrete-event simulation engine with controllable seed.
  - 2D tactical grid with resource nodes, barbarian encampments, and city headquarters.
  - 4-Tier transactional resource accounting (Part 52: Available, Reserved, Committed, Expended).
  - Tactical march movement, resource gathering, combat resolution, and return cycles.
  - Standard step() / reset() interface for reinforcement learning and heuristic evaluation.
"""

from __future__ import annotations

import math
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from schema.game_state import (
    FourTierResourceAccount,
    GameState,
    GridTile,
    MarchState,
    MarchStatus,
    ObservationStatus,
    ResourceType,
    TileType,
    TroopUnit,
)


class SimActionType(str, Enum):
    NOOP = "NOOP"
    DISPATCH_GATHER_MARCH = "DISPATCH_GATHER_MARCH"
    DISPATCH_ATTACK_MARCH = "DISPATCH_ATTACK_MARCH"
    RECALL_MARCH = "RECALL_MARCH"
    UPGRADE_BUILDING = "UPGRADE_BUILDING"
    TRAIN_TROOPS = "TRAIN_TROOPS"


class SimAction:
    def __init__(
        self,
        action_type: SimActionType,
        march_id: Optional[int] = None,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None,
        resource_type: Optional[ResourceType] = None,
        amount: int = 0,
        troop_count: int = 1000,
    ):
        self.action_type = action_type
        self.march_id = march_id
        self.target_x = target_x
        self.target_y = target_y
        self.resource_type = resource_type
        self.amount = amount
        self.troop_count = troop_count


class StrategySimulator:
    """
    Lightweight, deterministic strategy simulator engine.
    Mirrors core mechanics of 4X mobile strategy titles (Rise of Kingdoms).
    """

    def __init__(
        self,
        grid_radius: int = 10,
        max_marches: int = 5,
        gather_rate_per_sec: float = 60.0,
        base_march_speed: float = 1.5,
        seed: Optional[int] = 42,
    ):
        self.grid_radius = grid_radius
        self.max_marches = max_marches
        self.gather_rate_per_sec = gather_rate_per_sec
        self.base_march_speed = base_march_speed
        self.rng = random.Random(seed)
        self.seed = seed

        # Simulation clock
        self.sim_time: float = 0.0
        self.tick_count: int = 0

        # State storage
        self.city_x = 0
        self.city_y = 0
        self.city_level = 1
        self.idle_reserve_troops = 10000

        self.resources: Dict[ResourceType, FourTierResourceAccount] = {}
        self.marches: Dict[int, MarchState] = {}
        self.tiles: Dict[str, GridTile] = {}

        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> GameState:
        """Resets simulator to clean initial state with optional deterministic seed."""
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        self.sim_time = 0.0
        self.tick_count = 0
        self.city_level = 1
        self.idle_reserve_troops = 10000

        # Initialize four-tier resource pools with starting balances
        self.resources = {
            ResourceType.FOOD: FourTierResourceAccount(
                resource_type=ResourceType.FOOD,
                available=50000,
                reserved=0,
                committed=0,
                expended=0,
                status=ObservationStatus.VERIFIED,
                confidence=1.0,
                last_updated=self.sim_time,
            ),
            ResourceType.WOOD: FourTierResourceAccount(
                resource_type=ResourceType.WOOD,
                available=50000,
                reserved=0,
                committed=0,
                expended=0,
                status=ObservationStatus.VERIFIED,
                confidence=1.0,
                last_updated=self.sim_time,
            ),
            ResourceType.STONE: FourTierResourceAccount(
                resource_type=ResourceType.STONE,
                available=25000,
                reserved=0,
                committed=0,
                expended=0,
                status=ObservationStatus.VERIFIED,
                confidence=1.0,
                last_updated=self.sim_time,
            ),
            ResourceType.GOLD: FourTierResourceAccount(
                resource_type=ResourceType.GOLD,
                available=10000,
                reserved=0,
                committed=0,
                expended=0,
                status=ObservationStatus.VERIFIED,
                confidence=1.0,
                last_updated=self.sim_time,
            ),
            ResourceType.GEMS: FourTierResourceAccount(
                resource_type=ResourceType.GEMS,
                available=500,
                reserved=0,
                committed=0,
                expended=0,
                status=ObservationStatus.VERIFIED,
                confidence=1.0,
                last_updated=self.sim_time,
            ),
        }

        # Initialize Marches (all initially IDLE)
        self.marches = {}
        for m_id in range(1, self.max_marches + 1):
            self.marches[m_id] = MarchState(
                march_id=m_id,
                status=MarchStatus.IDLE,
                current_x=float(self.city_x),
                current_y=float(self.city_y),
                speed=self.base_march_speed,
                load_capacity=15000,
                current_load=0,
                troops=[TroopUnit(unit_type="infantry", tier=1, count=2000, health=100.0)],
                last_updated=self.sim_time,
            )

        # Generate World Grid Map
        self._generate_grid()

        return self.get_state()

    def _tile_key(self, x: int, y: int) -> str:
        return f"{x}_{y}"

    def _generate_grid(self) -> None:
        """Procedurally populates tactical grid nodes deterministically."""
        self.tiles = {}
        # City at center
        self.tiles[self._tile_key(0, 0)] = GridTile(
            x=0,
            y=0,
            tile_type=TileType.PLAYER_CITY,
            level=self.city_level,
            last_scouted=self.sim_time,
        )

        # Distribute resource nodes & barbarian encampments
        for x in range(-self.grid_radius, self.grid_radius + 1):
            for y in range(-self.grid_radius, self.grid_radius + 1):
                if x == 0 and y == 0:
                    continue
                roll = self.rng.random()
                dist = math.hypot(x, y)

                if roll < 0.12:
                    # Food Field
                    level = max(1, min(5, int(dist // 2) + 1))
                    self.tiles[self._tile_key(x, y)] = GridTile(
                        x=x,
                        y=y,
                        tile_type=TileType.RESOURCE_FOOD,
                        level=level,
                        remaining_resources=level * 20000,
                        last_scouted=self.sim_time,
                    )
                elif roll < 0.24:
                    # Wood Forest
                    level = max(1, min(5, int(dist // 2) + 1))
                    self.tiles[self._tile_key(x, y)] = GridTile(
                        x=x,
                        y=y,
                        tile_type=TileType.RESOURCE_WOOD,
                        level=level,
                        remaining_resources=level * 20000,
                        last_scouted=self.sim_time,
                    )
                elif roll < 0.32:
                    # Stone Quarry
                    level = max(1, min(5, int(dist // 2) + 1))
                    self.tiles[self._tile_key(x, y)] = GridTile(
                        x=x,
                        y=y,
                        tile_type=TileType.RESOURCE_STONE,
                        level=level,
                        remaining_resources=level * 15000,
                        last_scouted=self.sim_time,
                    )
                elif roll < 0.38:
                    # Gold Mine
                    level = max(1, min(5, int(dist // 2) + 1))
                    self.tiles[self._tile_key(x, y)] = GridTile(
                        x=x,
                        y=y,
                        tile_type=TileType.RESOURCE_GOLD,
                        level=level,
                        remaining_resources=level * 10000,
                        last_scouted=self.sim_time,
                    )
                elif roll < 0.44:
                    # Barbarian Camp
                    barb_level = max(1, min(10, int(dist) + 1))
                    self.tiles[self._tile_key(x, y)] = GridTile(
                        x=x,
                        y=y,
                        tile_type=TileType.BARBARIAN_CAMP,
                        level=barb_level,
                        remaining_resources=barb_level * 5000,  # Loot upon defeat
                        danger_level=float(barb_level),
                        last_scouted=self.sim_time,
                    )
                else:
                    self.tiles[self._tile_key(x, y)] = GridTile(
                        x=x,
                        y=y,
                        tile_type=TileType.EMPTY,
                        level=1,
                        last_scouted=self.sim_time,
                    )

    def tick(self, dt: float = 1.0) -> None:
        """
        Discrete simulation step advancing environmental clock and mechanics by dt seconds.
        """
        self.sim_time += dt
        self.tick_count += 1

        # 1. Base city passive income (10 Food/Wood per sec)
        if ResourceType.FOOD in self.resources and self.resources[ResourceType.FOOD].available is not None:
            self.resources[ResourceType.FOOD].available += int(10 * dt)
        if ResourceType.WOOD in self.resources and self.resources[ResourceType.WOOD].available is not None:
            self.resources[ResourceType.WOOD].available += int(10 * dt)

        # 2. Advance active marches
        for march in self.marches.values():
            march.last_updated = self.sim_time

            # Status: Moving to Target
            if march.status == MarchStatus.MOVING_TO_TARGET:
                if march.target_x is None or march.target_y is None:
                    march.status = MarchStatus.IDLE
                    continue

                dx = march.target_x - march.current_x
                dy = march.target_y - march.current_y
                dist = math.hypot(dx, dy)
                step_dist = march.speed * dt

                if dist <= step_dist:
                    march.current_x = float(march.target_x)
                    march.current_y = float(march.target_y)
                    march.eta_sec = 0.0

                    # Arrived at destination -> Transition based on tile type
                    t_key = self._tile_key(int(march.target_x), int(march.target_y))
                    tile = self.tiles.get(t_key)

                    if tile and tile.tile_type in (
                        TileType.RESOURCE_FOOD,
                        TileType.RESOURCE_WOOD,
                        TileType.RESOURCE_STONE,
                        TileType.RESOURCE_GOLD,
                    ):
                        march.status = MarchStatus.GATHERING
                    elif tile and tile.tile_type == TileType.BARBARIAN_CAMP:
                        march.status = MarchStatus.BATTLING
                    else:
                        # Empty tile or unrecognized -> return home
                        self._order_march_return(march)
                else:
                    march.current_x += (dx / dist) * step_dist
                    march.current_y += (dy / dist) * step_dist
                    march.eta_sec = (dist - step_dist) / march.speed

            # Status: Gathering
            elif march.status == MarchStatus.GATHERING:
                t_key = self._tile_key(int(march.current_x), int(march.current_y))
                tile = self.tiles.get(t_key)

                if not tile or tile.remaining_resources is None or tile.remaining_resources <= 0:
                    self._order_march_return(march)
                    continue

                gather_amount = min(
                    int(self.gather_rate_per_sec * dt),
                    tile.remaining_resources,
                    march.load_capacity - march.current_load,
                )

                tile.remaining_resources -= gather_amount
                march.current_load += gather_amount

                # If march capacity reached or tile depleted -> return home
                if march.current_load >= march.load_capacity or tile.remaining_resources <= 0:
                    self._order_march_return(march)

            # Status: Battling Barbarians
            elif march.status == MarchStatus.BATTLING:
                t_key = self._tile_key(int(march.current_x), int(march.current_y))
                tile = self.tiles.get(t_key)

                if not tile or tile.tile_type != TileType.BARBARIAN_CAMP:
                    self._order_march_return(march)
                    continue

                # Deterministic combat resolution
                barb_power = tile.level * 150
                player_power = sum(
                    (u.count or 0) * (u.health / 100.0) * (u.tier * 1.2) for u in march.troops
                )

                if player_power > barb_power:
                    # Player Victory: secure loot and return
                    loot = tile.remaining_resources or (tile.level * 2500)
                    march.current_load = min(march.load_capacity, loot)
                    # Convert tile to empty
                    tile.tile_type = TileType.EMPTY
                    tile.remaining_resources = 0
                    tile.danger_level = 0.0
                    self._order_march_return(march)
                else:
                    # Player Defeat: take casualties and retreat
                    for u in march.troops:
                        u.health = max(20.0, u.health - 40.0)
                        if u.count is not None:
                            u.count = max(0, int(u.count * 0.75))
                    self._order_march_return(march)

            # Status: Returning to City
            elif march.status == MarchStatus.RETURNING:
                dx = self.city_x - march.current_x
                dy = self.city_y - march.current_y
                dist = math.hypot(dx, dy)
                step_dist = march.speed * dt

                if dist <= step_dist:
                    march.current_x = float(self.city_x)
                    march.current_y = float(self.city_y)
                    march.eta_sec = 0.0

                    # Deposit collected load into Available tier (Part 52)
                    if march.target_resource_type and march.current_load > 0:
                        r_acct = self.resources.get(march.target_resource_type)
                        if r_acct and r_acct.available is not None:
                            r_acct.available += march.current_load
                            r_acct.last_updated = self.sim_time

                    march.current_load = 0
                    march.status = MarchStatus.IDLE
                    march.target_x = None
                    march.target_y = None
                    march.target_resource_type = None
                else:
                    march.current_x += (dx / dist) * step_dist
                    march.current_y += (dy / dist) * step_dist
                    march.eta_sec = (dist - step_dist) / march.speed

    def _order_march_return(self, march: MarchState) -> None:
        march.status = MarchStatus.RETURNING
        march.target_x = float(self.city_x)
        march.target_y = float(self.city_y)

    def dispatch_gather(self, march_id: int, target_x: int, target_y: int) -> bool:
        """Dispatches an IDLE march to a specific resource tile."""
        march = self.marches.get(march_id)
        if not march or march.status != MarchStatus.IDLE:
            return False

        t_key = self._tile_key(target_x, target_y)
        tile = self.tiles.get(t_key)
        if not tile or tile.tile_type not in (
            TileType.RESOURCE_FOOD,
            TileType.RESOURCE_WOOD,
            TileType.RESOURCE_STONE,
            TileType.RESOURCE_GOLD,
        ):
            return False

        march.status = MarchStatus.MOVING_TO_TARGET
        march.target_x = float(target_x)
        march.target_y = float(target_y)

        # Map tile type to resource type
        type_mapping = {
            TileType.RESOURCE_FOOD: ResourceType.FOOD,
            TileType.RESOURCE_WOOD: ResourceType.WOOD,
            TileType.RESOURCE_STONE: ResourceType.STONE,
            TileType.RESOURCE_GOLD: ResourceType.GOLD,
        }
        march.target_resource_type = type_mapping.get(tile.tile_type)
        return True

    def dispatch_attack(self, march_id: int, target_x: int, target_y: int) -> bool:
        """Dispatches an IDLE march to attack a barbarian encampment."""
        march = self.marches.get(march_id)
        if not march or march.status != MarchStatus.IDLE:
            return False

        t_key = self._tile_key(target_x, target_y)
        tile = self.tiles.get(t_key)
        if not tile or tile.tile_type != TileType.BARBARIAN_CAMP:
            return False

        march.status = MarchStatus.MOVING_TO_TARGET
        march.target_x = float(target_x)
        march.target_y = float(target_y)
        march.target_resource_type = ResourceType.FOOD  # Barbarian loot defaults to Food
        return True

    def upgrade_building(self, food_cost: int = 10000, wood_cost: int = 10000) -> bool:
        """
        Upgrades city using 4-tier transactional reservation (Part 52).
        Reserves -> Commits -> Expends.
        """
        f_acct = self.resources.get(ResourceType.FOOD)
        w_acct = self.resources.get(ResourceType.WOOD)

        if not f_acct or not w_acct:
            return False

        # Phase 1: Reserve
        if not (f_acct.reserve(food_cost) and w_acct.reserve(wood_cost)):
            # Rollback any partial reservation
            f_acct.release_reservation(food_cost)
            w_acct.release_reservation(wood_cost)
            return False

        # Phase 2: Commit & Expend
        f_acct.commit(food_cost)
        w_acct.commit(wood_cost)

        f_acct.expend(food_cost)
        w_acct.expend(wood_cost)

        self.city_level += 1
        return True

    def step(
        self, action: SimAction, dt: float = 1.0
    ) -> Tuple[GameState, float, bool, bool, Dict[str, Any]]:
        """
        Executes action and advances simulation time by dt.
        Returns: (next_state, reward, terminated, truncated, info)
        """
        reward = 0.0
        success = False

        if action.action_type == SimActionType.DISPATCH_GATHER_MARCH:
            if action.march_id and action.target_x is not None and action.target_y is not None:
                success = self.dispatch_gather(action.march_id, action.target_x, action.target_y)
                reward += 1.0 if success else -0.5

        elif action.action_type == SimActionType.DISPATCH_ATTACK_MARCH:
            if action.march_id and action.target_x is not None and action.target_y is not None:
                success = self.dispatch_attack(action.march_id, action.target_x, action.target_y)
                reward += 1.5 if success else -0.5

        elif action.action_type == SimActionType.RECALL_MARCH:
            if action.march_id and action.march_id in self.marches:
                self._order_march_return(self.marches[action.march_id])
                success = True

        elif action.action_type == SimActionType.UPGRADE_BUILDING:
            success = self.upgrade_building(food_cost=action.amount or 10000, wood_cost=action.amount or 10000)
            reward += 10.0 if success else -1.0

        # Run internal simulation tick
        self.tick(dt=dt)

        next_state = self.get_state()
        terminated = False
        truncated = False
        info = {"success": success, "sim_time": self.sim_time, "tick": self.tick_count}

        return next_state, reward, terminated, truncated, info

    def get_state(self) -> GameState:
        """Projects current internal state into the unified Pydantic GameState model."""
        active_marches = sum(1 for m in self.marches.values() if m.status != MarchStatus.IDLE)
        return GameState(
            timestamp=self.sim_time,
            operational_mode="SIM_MODE",
            freshness_decay_sec=15.0,
            city_level=self.city_level,
            active_marches_count=active_marches,
            max_marches_count=self.max_marches,
            resources=dict(self.resources),
            marches=list(self.marches.values()),
            tiles=dict(self.tiles),
            global_status=ObservationStatus.VERIFIED,
            provenance="SIMULATOR",
        )
