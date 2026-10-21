"""
models.py

Tier + RealMemoryItem, split out from the store so both the store and the
storage backends can import them without a circular import.
"""

from dataclasses import dataclass
from enum import Enum


class Tier(Enum):
    ACTIVE = 3
    VECTOR = 2
    COLD = 1
    DELETED = 0


@dataclass
class RealMemoryItem:
    id: str
    prompt: str
    content: str
    provider: str
    model: str
    created_at: int
    last_accessed_at: int
    access_count: int = 1
    regeneration_cost: float = 0.0
    regeneration_tokens: int = 0
    regeneration_latency_s: float = 0.0
    tier: Tier = Tier.ACTIVE
    score: float = 0.0

    def touch(self, t):
        self.last_accessed_at = t
        self.access_count += 1
