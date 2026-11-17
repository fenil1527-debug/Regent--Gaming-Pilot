# OSROKBOT/simulator/scenarios.py
"""
Scenario Generation Fixtures & Benchmarking Suite (Step 4, Part 30, Part 33).
Provides reproducible test fixtures for strategy evaluation and policy benchmarking.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from schema.game_state import (
    GameState,
    GridTile,
    ObservationStatus,
    ResourceType,
    TileType,
)
from simulator.engine import SimAction, SimActionType, StrategySimulator


class ScenarioGenerator:
    """Factory creating standardized simulation scenario fixtures with deterministic seeds."""

    @staticmethod
    def make_resource_gathering_scenario(
        seed: int = 42,
        grid_radius: int = 10,
        rich_nodes_count: int = 12,
    ) -> StrategySimulator:
        """
        Creates an economic throughput benchmark:
        Surrounds the city with rich resource deposits to benchmark march allocation efficiency.
        """
        sim = StrategySimulator(grid_radius=grid_radius, seed=seed)

        # Inject dense, high-yield nodes within radius 4 of city
        offsets = [
            (1, 1, TileType.RESOURCE_FOOD),
            (-1, 1, TileType.RESOURCE_WOOD),
            (1, -1, TileType.RESOURCE_STONE),
            (-1, -1, TileType.RESOURCE_GOLD),
            (2, 0, TileType.RESOURCE_FOOD),
            (-2, 0, TileType.RESOURCE_WOOD),
            (0, 2, TileType.RESOURCE_STONE),
            (0, -2, TileType.RESOURCE_GOLD),
            (3, 1, TileType.RESOURCE_FOOD),
            (-3, 1, TileType.RESOURCE_WOOD),
            (1, 3, TileType.RESOURCE_STONE),
            (-1, 3, TileType.RESOURCE_GOLD),
        ]

        for x, y, t_type in offsets[:rich_nodes_count]:
            sim.tiles[sim._tile_key(x, y)] = GridTile(
                x=x,
                y=y,
                tile_type=t_type,
                level=4,
                remaining_resources=100000,
                last_scouted=0.0,
            )

        return sim

    @staticmethod
    def make_barbarian_hunt_scenario(
        seed: int = 101,
        grid_radius: int = 10,
        num_camps: int = 6,
    ) -> StrategySimulator:
        """
        Creates a tactical combat scenario:
        Spawns barbarian encampments of ascending difficulty to benchmark risk management.
        """
        sim = StrategySimulator(grid_radius=grid_radius, seed=seed)

        camp_coords = [(2, 2), (-2, 2), (3, -2), (-3, -3), (4, 1), (-4, -2)]
        for idx, (x, y) in enumerate(camp_coords[:num_camps]):
            level = idx + 1
            sim.tiles[sim._tile_key(x, y)] = GridTile(
                x=x,
                y=y,
                tile_type=TileType.BARBARIAN_CAMP,
                level=level,
                remaining_resources=level * 5000,
                danger_level=float(level),
                last_scouted=0.0,
            )

        return sim

    @staticmethod
    def make_resource_starvation_scenario(
        seed: int = 777,
        grid_radius: int = 10,
    ) -> StrategySimulator:
        """
        Creates a recovery scenario:
        Starts with critical resource deficits (500 units each) to test recovery heuristics.
        """
        sim = StrategySimulator(grid_radius=grid_radius, seed=seed)

        for r_type in (ResourceType.FOOD, ResourceType.WOOD, ResourceType.STONE, ResourceType.GOLD):
            if r_type in sim.resources:
                sim.resources[r_type].available = 500

        return sim


def run_scenario_benchmark(
    sim: StrategySimulator,
    policy_fn: Callable[[GameState], SimAction],
    max_steps: int = 100,
    dt: float = 1.0,
) -> Dict[str, Any]:
    """
    Executes a reproducible scenario benchmark for a given decision policy.
    Collects performance telemetry: economic throughput, survival rate, and step efficiency.
    """
    initial_food = sim.resources[ResourceType.FOOD].available or 0
    initial_wood = sim.resources[ResourceType.WOOD].available or 0

    total_reward = 0.0
    action_counts: Dict[str, int] = {}
    idle_march_ticks = 0

    for step_num in range(max_steps):
        current_state = sim.get_state()

        # Count idle marches for efficiency metric
        idle_count = sum(1 for m in current_state.marches if m.status.value == "IDLE")
        idle_march_ticks += idle_count

        # Query policy
        action = policy_fn(current_state)
        act_key = action.action_type.value
        action_counts[act_key] = action_counts.get(act_key, 0) + 1

        # Advance simulator
        _, reward, terminated, truncated, _ = sim.step(action, dt=dt)
        total_reward += reward

        if terminated or truncated:
            break

    final_food = sim.resources[ResourceType.FOOD].available or 0
    final_wood = sim.resources[ResourceType.WOOD].available or 0

    return {
        "steps_executed": step_num + 1,
        "sim_time_sec": round(sim.sim_time, 2),
        "total_reward": round(total_reward, 2),
        "action_distribution": action_counts,
        "food_delta": final_food - initial_food,
        "wood_delta": final_wood - initial_wood,
        "idle_march_ratio": round(idle_march_ticks / (max(1, step_num + 1) * sim.max_marches), 3),
        "city_level": sim.city_level,
    }
