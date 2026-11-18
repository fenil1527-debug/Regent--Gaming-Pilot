# OSROKBOT/memory/hybrid_retriever.py
"""
Multi-Index Hybrid Retrieval Module (Step 5, Part 4, Part 15).
Combines deterministic heuristic rule matching with feature vector similarity search
over episodic experience replay buffers.
"""

from __future__ import annotations

import collections
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from schema.game_state import GameState, ResourceType


class EpisodicTransition:
    def __init__(
        self,
        transition_id: str,
        state_features: List[float],
        action_name: str,
        outcome: str,
        reward: float,
        timestamp: float,
        rule_tag: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.transition_id = transition_id
        self.state_features = state_features
        self.action_name = action_name
        self.outcome = outcome
        self.reward = reward
        self.timestamp = timestamp
        self.rule_tag = rule_tag
        self.context = context or {}


class HybridRetriever:
    """
    Dual-Index Retrieval Engine combining:
      1. Exact Heuristic Signature Lookups (O(1) rule index for deterministic patterns).
      2. Normalized State Vector Similarity (Cosine similarity over game states).
    """

    def __init__(self, max_buffer_size: int = 1000):
        self.max_buffer_size = max_buffer_size

        # In-memory bounded replay buffer (FLAW-05 memory constraint)
        self.buffer: collections.deque[EpisodicTransition] = collections.deque(
            maxlen=max_buffer_size
        )

        # Index 1: Heuristic Rule Index {rule_tag: list of transition references}
        self.rule_index: Dict[str, List[EpisodicTransition]] = collections.defaultdict(list)

    @staticmethod
    def extract_state_vector(state: GameState) -> List[float]:
        """
        Extracts a normalized numerical feature vector representing game state:
          - Food, Wood, Stone, Gold ratios (normalized against reference 100k)
          - City Level (normalized against 25)
          - Active march ratio (active / max)
        """
        f = (state.get_effective_liquid_resource(ResourceType.FOOD) or 0) / 100_000.0
        w = (state.get_effective_liquid_resource(ResourceType.WOOD) or 0) / 100_000.0
        s = (state.get_effective_liquid_resource(ResourceType.STONE) or 0) / 100_000.0
        g = (state.get_effective_liquid_resource(ResourceType.GOLD) or 0) / 100_000.0

        city_lvl = (state.city_level or 1) / 25.0
        march_ratio = (
            (state.active_marches_count or 0) / max(1, state.max_marches_count)
        )

        return [
            min(f, 5.0),
            min(w, 5.0),
            min(s, 5.0),
            min(g, 5.0),
            min(city_lvl, 1.0),
            min(march_ratio, 1.0),
        ]

    def record_transition(
        self,
        state: GameState,
        action_name: str,
        outcome: str,
        reward: float,
        rule_tag: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stores a state-action-outcome transition in the dual-index buffer."""
        t_id = f"tx_{int(time.time() * 1000)}_{len(self.buffer)}"
        vec = self.extract_state_vector(state)

        tx = EpisodicTransition(
            transition_id=t_id,
            state_features=vec,
            action_name=action_name,
            outcome=outcome,
            reward=reward,
            timestamp=time.time(),
            rule_tag=rule_tag,
            context=context,
        )

        self.buffer.append(tx)
        if rule_tag:
            self.rule_index[rule_tag].append(tx)
            # Prune index to last 100 for any given tag
            if len(self.rule_index[rule_tag]) > 100:
                self.rule_index[rule_tag] = self.rule_index[rule_tag][-100:]

        return t_id

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def query_by_similarity(
        self, query_state: GameState, top_k: int = 3
    ) -> List[Tuple[EpisodicTransition, float]]:
        """
        Finds the top-K most similar historical transitions using cosine similarity.
        Returns list of (transition, similarity_score).
        """
        if not self.buffer:
            return []

        query_vec = self.extract_state_vector(query_state)
        scored: List[Tuple[EpisodicTransition, float]] = []

        for tx in self.buffer:
            sim = self._cosine_similarity(query_vec, tx.state_features)
            scored.append((tx, round(sim, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def query_by_rule(self, rule_tag: str, limit: int = 5) -> List[EpisodicTransition]:
        """Queries historical transitions matching a deterministic heuristic rule tag."""
        matches = self.rule_index.get(rule_tag, [])
        return matches[-limit:] if len(matches) > limit else matches

    def hybrid_retrieve(
        self,
        query_state: GameState,
        rule_tag: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Combines rule matching and vector similarity.
        Prioritizes exact rule hits (Part 9 & Part 15), enriched by nearest historical states.
        """
        rule_hits = self.query_by_rule(rule_tag, limit=top_k) if rule_tag else []
        similar_hits = self.query_by_similarity(query_state, top_k=top_k)

        return {
            "query_timestamp": time.time(),
            "rule_tag": rule_tag,
            "rule_matches_count": len(rule_hits),
            "rule_matches": [
                {
                    "tx_id": t.transition_id,
                    "action": t.action_name,
                    "outcome": t.outcome,
                    "reward": t.reward,
                }
                for t in rule_hits
            ],
            "similarity_matches": [
                {
                    "tx_id": t.transition_id,
                    "action": t.action_name,
                    "outcome": t.outcome,
                    "similarity": score,
                    "reward": t.reward,
                }
                for t, score in similar_hits
            ],
        }
