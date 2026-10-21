"""
Regression + stress tests for RealCostAwareMemoryStore, now backed by REAL
storage (ActiveBackend dict / ChromaDB VectorBackend / SQLite ColdBackend)
instead of one shared in-memory dict of dataclass objects.

WHY THIS EXISTS
----------------
The store's tiering/eviction/regen-cost logic is pure Python with two side
effects: it calls a real, billed LLM API via `call_llm`, and it writes to
real storage (Chroma, SQLite). This file:
  - replaces `call_llm` with a deterministic, free, in-memory mock (still
    varies cost/tokens per prompt, so cost-weighted eviction is exercised
    realistically) via the `call_llm_fn=` constructor param;
  - replaces the VECTOR tier's embedding model with a deterministic, offline
    FakeEmbeddingFunction (see conftest.py) so tests don't need network
    access or a slow real model download;
  - points COLD/DELETED storage at a fresh pytest tmp_path per test, so
    tests can't see each other's SQLite rows.

Everything else -- tiering math, capacity enforcement, eviction, regen-cost
multipliers, hit/miss counters -- is the real code, unmodified, and every
item really does live in a real dict / real ChromaDB collection / real
SQLite table as it moves between tiers.

NOTE ON MUTATING ITEMS IN TESTS: with real backends, `store.items[id].x = y`
no longer persists for VECTOR/COLD items (see memory_store.py's docstring).
Tests that need to force a field use `store.update_item(id, x=y)` instead.

NOTE ON STRESS-TEST SCALE: real Chroma/SQLite round-trips are meaningfully
slower than a plain dict (see conftest.py / README for a measured
comparison). Step counts here are scaled down from the original
pure-in-memory suite so the suite still runs in reasonable time; the perf
assertions are correspondingly loosened, not removed.

RUN:
    pytest test_eviction_regression.py -v
"""

import math
import random
import time

import pytest

from models import Tier
from memory_store import RealCostAwareMemoryStore, default_mock_call_llm
from conftest import make_backends, FakeEmbeddingFunction


# ---------------------------------------------------------------------------
# Mock LLM layer (deterministic, offline, cost/tokens vary by prompt)
# ---------------------------------------------------------------------------

FAKE_CLIENTS = {"groq": None, "gemini": None, "openrouter": None}

# Re-exported so other modules (lru_baseline tests, print_comparison) can
# `from test_eviction_regression import call_llm` the same way they used to.
call_llm = default_mock_call_llm


def make_store(tmp_path, **kwargs):
    defaults = dict(active_capacity=3, vector_capacity=4, cold_capacity=5)
    defaults.update(kwargs)
    backends = make_backends(tmp_path)
    return RealCostAwareMemoryStore(FAKE_CLIENTS, call_llm_fn=call_llm, **backends, **defaults)


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

def test_add_creates_active_item_with_real_cost(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "prompt A", "groq", "llama-3.1-8b-instant")
    assert item.tier == Tier.ACTIVE
    assert item.regeneration_cost > 0
    assert store.generation_cost_paid == pytest.approx(item.regeneration_cost)
    assert item.id in store.items


def test_item_really_lives_in_the_active_dict(tmp_path):
    """Direct proof ACTIVE tier is the real ActiveBackend dict, not just a
    tier label -- the item must be gettable straight off the backend."""
    store = make_store(tmp_path)
    item = store.add(0, "prompt A", "groq", "m")
    assert store.active_backend.get(item.id) is not None
    assert store.vector_backend.get(item.id) is None
    assert store.cold_backend.get(item.id) is None


def test_item_pushed_to_vector_really_lives_in_chroma(tmp_path):
    """Direct proof VECTOR tier is a real ChromaDB collection: once an item
    is ranked into VECTOR, it must be absent from ACTIVE's dict and present
    in the Chroma collection (queryable by id, with its metadata intact)."""
    store = make_store(tmp_path, active_capacity=1, vector_capacity=3, cold_capacity=3)
    first = store.add(0, "first prompt", "groq", "m")
    for i in range(1, 3):
        store.add(i, f"filler-{i}", "groq", "m")
    assert store.get_item(first.id).tier == Tier.VECTOR
    assert store.active_backend.get(first.id) is None
    chroma_item = store.vector_backend.get(first.id)
    assert chroma_item is not None
    assert chroma_item.prompt == "first prompt"


def test_item_pushed_to_cold_really_lives_in_sqlite(tmp_path):
    """Direct proof COLD tier is a real SQLite table: query the table
    directly (bypassing the store entirely) and find the row."""
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=3)
    first = store.add(0, "first prompt", "groq", "m")
    store.active_backend.delete(first.id)
    first.tier = Tier.COLD
    store.cold_backend.put(first)
    assert store.get_item(first.id).tier == Tier.COLD
    cur = store.cold_backend._conn.execute(
        "SELECT prompt FROM cold_items WHERE id = ?", (first.id,)
    )
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "first prompt"


def test_capacity_limits_never_exceeded(tmp_path):
    store = make_store(tmp_path, active_capacity=2, vector_capacity=2, cold_capacity=2)
    for i in range(20):
        store.add(i, f"prompt {i}", "groq", "llama-3.1-8b-instant")
    counts = store.tier_counts()
    assert counts[Tier.ACTIVE] <= 2
    assert counts[Tier.VECTOR] <= 2
    assert counts[Tier.COLD] <= 2
    assert len(store.items) <= 6


def test_items_beyond_all_tiers_are_deleted_and_removed(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    for i in range(10):
        store.add(i, f"prompt {i}", "groq", "m")
    # only the 3 most-recent/highest-scoring items should remain resident
    assert len(store.items) == 3
    assert len(store.deleted_log) == 7
    for d in store.deleted_log:
        assert d.id not in store.items


def test_access_on_deleted_item_triggers_real_regeneration(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    for i in range(10):
        store.add(i, f"prompt {i}", "groq", "m")
    evicted_id = store.deleted_log[0].id
    before_misses = store.misses
    before_regen_events = store.regeneration_events
    result = store.access(evicted_id, 100)
    assert store.misses == before_misses + 1
    assert store.regeneration_events == before_regen_events + 1
    assert result.tier == Tier.ACTIVE
    assert result.id in store.items
    assert store.regeneration_cost_paid > 0


def test_regenerated_item_uses_original_prompt_not_placeholder(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    store.add(0, "the one true prompt", "groq", "m")
    for i in range(1, 10):
        store.add(i, f"filler {i}", "groq", "m")
    evicted = next(d for d in store.deleted_log if d.prompt == "the one true prompt")
    result = store.access(evicted.id, 999)
    assert result.prompt == "the one true prompt"
    assert "the one true prompt"[:30] in result.content or result.prompt == "the one true prompt"


def test_deleted_item_survives_a_fresh_store_instance(tmp_path):
    """DeletedBackend is real SQLite, so a *new* store instance pointed at
    the same db file must still be able to regenerate an item another
    instance evicted -- proof the deleted log is durable, not in-process
    only, unlike the original list-based deleted_log."""
    backends = make_backends(tmp_path)
    store1 = RealCostAwareMemoryStore(
        FAKE_CLIENTS, call_llm_fn=call_llm, active_capacity=1, vector_capacity=1,
        cold_capacity=1, **backends,
    )
    for i in range(10):
        store1.add(i, f"prompt {i}", "groq", "m")
    evicted_id = store1.deleted_log[0].id
    evicted_prompt = store1.deleted_log[0].prompt

    # Fresh store, same backends (same SQLite file / same Chroma dir) --
    # simulates a process restart.
    store2 = RealCostAwareMemoryStore(
        FAKE_CLIENTS, call_llm_fn=call_llm, active_capacity=1, vector_capacity=1,
        cold_capacity=1, **backends,
    )
    result = store2.access(evicted_id, 500)
    assert result.prompt == evicted_prompt
    assert store2.misses == 1


def test_vector_hit_charges_five_percent_of_original_cost(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=5, cold_capacity=5)
    item = store.add(0, "p0", "groq", "m")
    store.active_backend.delete(item.id)
    item.tier = Tier.VECTOR
    store.vector_backend.put(item)
    original_cost = store.get_item(item.id).regeneration_cost
    before = store.regeneration_cost_paid
    store.access(item.id, 50)
    assert store.regeneration_cost_paid == pytest.approx(before + 0.05 * original_cost)
    assert store.vector_hits == 1


def test_cold_hit_charges_twenty_five_percent_of_original_cost(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=5)
    item = store.add(0, "p0", "groq", "m")
    store.active_backend.delete(item.id)
    item.tier = Tier.COLD
    store.cold_backend.put(item)
    original_cost = store.get_item(item.id).regeneration_cost
    before = store.regeneration_cost_paid
    store.access(item.id, 50)
    assert store.regeneration_cost_paid == pytest.approx(before + 0.25 * original_cost)
    assert store.cold_hits == 1


def test_active_hit_is_free(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "p0", "groq", "m")
    before = store.regeneration_cost_paid
    store.access(item.id, 1)
    assert store.regeneration_cost_paid == before
    assert store.active_hits == 1


def test_access_promotes_item_back_to_active(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=5)
    item = store.add(0, "p0", "groq", "m")
    store.active_backend.delete(item.id)
    item.tier = Tier.COLD
    store.cold_backend.put(item)
    store.access(item.id, 50)
    assert store.get_item(item.id).tier == Tier.ACTIVE
    assert store.active_backend.get(item.id) is not None


def test_cost_aware_scoring_protects_expensive_items_over_cheap_ones(tmp_path):
    """
    Direct proof the cost-aware score genuinely favors an expensive-to-
    regenerate item over an equally-recent, equally-frequent but cheap one
    when eviction pressure forces a choice -- and that this holds across
    the real ACTIVE/VECTOR/COLD backends, not just an in-memory dict.
    """
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    cheap = store.add(0, "cheap prompt", "groq", "m")
    expensive = store.add(1, "expensive prompt", "groq", "m")
    store.update_item(cheap.id, regeneration_cost=0.0001)
    store.update_item(expensive.id, regeneration_cost=10.0)
    store._cost_norm = max(store._cost_norm, 10.0)

    store.update_item(cheap.id, last_accessed_at=1)
    store.update_item(expensive.id, last_accessed_at=1)
    for i in range(2, 8):
        store.add(i, f"filler-{i}", "groq", "m")

    assert expensive.id in store.items, "cost-aware store evicted the expensive item -- scoring regression"
    assert cheap.id not in store.items, "cheap item should have been evicted first"


def test_touch_updates_recency_and_access_count(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "p0", "groq", "m")
    assert item.access_count == 1
    store.access(item.id, 10)
    updated = store.get_item(item.id)
    assert updated.access_count == 2
    assert updated.last_accessed_at == 10


def test_cost_norm_never_decreases(tmp_path):
    store = make_store(tmp_path, active_capacity=10, vector_capacity=10, cold_capacity=10)
    norms = []
    for i in range(30):
        store.add(i, f"p{i}", "groq", "m")
        norms.append(store._cost_norm)
    assert all(norms[i] <= norms[i + 1] for i in range(len(norms) - 1))


def test_zero_capacity_active_tier(tmp_path):
    store = make_store(tmp_path, active_capacity=0, vector_capacity=2, cold_capacity=2)
    store.add(0, "p0", "groq", "m")
    assert store.tier_counts()[Tier.ACTIVE] == 0


def test_zero_capacity_everywhere_deletes_immediately(tmp_path):
    store = make_store(tmp_path, active_capacity=0, vector_capacity=0, cold_capacity=0)
    item = store.add(0, "p0", "groq", "m")
    assert item.id not in store.items
    assert len(store.deleted_log) == 1


def test_single_item_store_stable_across_many_accesses(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    item = store.add(0, "solo", "groq", "m")
    for t in range(1, 50):
        store.access(item.id, t)
    assert store.misses == 0
    assert item.id in store.items


def test_hit_and_miss_counters_sum_to_total_accesses(tmp_path):
    store = make_store(tmp_path, active_capacity=2, vector_capacity=2, cold_capacity=2)
    ids = [store.add(i, f"p{i}", "groq", "m").id for i in range(15)]
    rng = random.Random(7)
    n_accesses = 80
    for t in range(15, 15 + n_accesses):
        store.access(rng.choice(ids), t)
    total = store.active_hits + store.vector_hits + store.cold_hits + store.misses
    assert total == n_accesses


def test_unknown_item_id_falls_back_gracefully(tmp_path):
    store = make_store(tmp_path)
    result = store.access("never-existed", 0)
    assert result.prompt == "unknown prompt"
    assert result.tier == Tier.ACTIVE
    assert store.misses == 1


def test_vector_backend_supports_real_semantic_query(tmp_path):
    """Bonus capability that only exists because VECTOR tier is a real
    vector DB now: query_similar() against whatever's currently resident."""
    store = make_store(tmp_path, active_capacity=1, vector_capacity=5, cold_capacity=5)
    store.add(0, "resume of a backend engineer with Python experience", "groq", "m")
    for i in range(1, 3):
        store.add(i, f"filler-{i}", "groq", "m")
    results = store.vector_backend.query_similar("backend engineer", top_k=3)
    assert len(results) > 0
    for item, distance in results:
        assert isinstance(item.prompt, str)
        assert distance >= 0


# ---------------------------------------------------------------------------
# Latency tracking tests
# ---------------------------------------------------------------------------

def test_latency_stats_shape_when_empty(tmp_path):
    store = make_store(tmp_path)
    stats = store.latency_stats()
    for key in ("active_hit", "vector_hit", "cold_hit", "miss_regen", "generation"):
        assert key in stats
        assert stats[key] == dict(count=0, min_ms=0.0, avg_ms=0.0, max_ms=0.0)


def test_generation_latency_recorded_on_add(tmp_path):
    store = make_store(tmp_path)
    store.add(0, "p0", "groq", "m")
    stats = store.latency_stats()
    assert stats["generation"]["count"] == 1
    assert stats["generation"]["avg_ms"] >= 0.0
    assert stats["generation"]["min_ms"] <= stats["generation"]["avg_ms"] <= stats["generation"]["max_ms"]


def test_active_hit_latency_recorded_on_access(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "p0", "groq", "m")
    store.access(item.id, 1)
    store.access(item.id, 2)
    stats = store.latency_stats()
    assert stats["active_hit"]["count"] == 2
    assert stats["vector_hit"]["count"] == 0
    assert stats["cold_hit"]["count"] == 0


def test_vector_hit_latency_recorded_when_item_is_in_vector_tier(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=5, cold_capacity=5)
    item = store.add(0, "p0", "groq", "m")
    store.active_backend.delete(item.id)
    item.tier = Tier.VECTOR
    store.vector_backend.put(item)
    store.access(item.id, 50)
    stats = store.latency_stats()
    assert stats["vector_hit"]["count"] == 1
    assert stats["vector_hit"]["avg_ms"] >= 0.0


def test_cold_hit_latency_recorded_when_item_is_in_cold_tier(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=5)
    item = store.add(0, "p0", "groq", "m")
    store.active_backend.delete(item.id)
    item.tier = Tier.COLD
    store.cold_backend.put(item)
    store.access(item.id, 50)
    stats = store.latency_stats()
    assert stats["cold_hit"]["count"] == 1


def test_miss_regen_latency_recorded_on_full_miss(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    for i in range(10):
        store.add(i, f"prompt {i}", "groq", "m")
    evicted_id = store.deleted_log[0].id
    store.access(evicted_id, 100)
    stats = store.latency_stats()
    assert stats["miss_regen"]["count"] == 1


def test_latency_min_avg_max_are_consistent(tmp_path):
    """min <= avg <= max must hold for every populated category, across a
    workload that exercises all three tiers plus at least one miss."""
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    ids = [store.add(i, f"p{i}", "groq", "m").id for i in range(15)]
    rng = random.Random(3)
    for t in range(15, 100):
        store.access(rng.choice(ids), t)
    stats = store.latency_stats()
    for key, s in stats.items():
        if s["count"] > 0:
            assert s["min_ms"] <= s["avg_ms"] <= s["max_ms"], f"{key} violates min<=avg<=max"


# ---------------------------------------------------------------------------
# Token-based reconstruction cost tests
# ---------------------------------------------------------------------------

def _zero_cost_mock_call_llm(clients, provider, model, prompt, max_tokens=None):
    import hashlib
    h = int(hashlib.md5(f"{provider}|{model}|{prompt}".encode()).hexdigest(), 16) % 1000
    in_tok = 20 + (h % 80)
    out_tok = 10 + (h % 150)
    return dict(
        content=f"mock response for: {prompt[:30]}",
        input_tokens=in_tok, output_tokens=out_tok, cost=0.0,
        latency_s=0.0, provider=provider, model=model,
    )


def make_zero_cost_store(tmp_path, **kwargs):
    defaults = dict(active_capacity=3, vector_capacity=4, cold_capacity=5)
    defaults.update(kwargs)
    backends = make_backends(tmp_path)
    return RealCostAwareMemoryStore(
        FAKE_CLIENTS, call_llm_fn=_zero_cost_mock_call_llm, **backends, **defaults,
    )


def test_add_tracks_generation_tokens(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "prompt A", "groq", "llama-3.1-8b-instant")
    assert item.regeneration_tokens > 0
    assert store.generation_tokens_paid == item.regeneration_tokens


def test_miss_regeneration_tracks_tokens(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    for i in range(10):
        store.add(i, f"prompt {i}", "groq", "m")
    evicted_id = store.deleted_log[0].id
    before = store.regeneration_tokens_paid
    result = store.access(evicted_id, 100)
    assert store.regeneration_tokens_paid > before
    assert result.regeneration_tokens > 0


def test_vector_hit_charges_five_percent_of_token_cost(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=5, cold_capacity=5)
    item = store.add(0, "p0", "groq", "m")
    store.active_backend.delete(item.id)
    item.tier = Tier.VECTOR
    store.vector_backend.put(item)
    original_tokens = store.get_item(item.id).regeneration_tokens
    before = store.regeneration_tokens_paid
    store.access(item.id, 50)
    assert store.regeneration_tokens_paid == pytest.approx(before + 0.05 * original_tokens)


def test_cold_hit_charges_twenty_five_percent_of_token_cost(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=5)
    item = store.add(0, "p0", "groq", "m")
    store.active_backend.delete(item.id)
    item.tier = Tier.COLD
    store.cold_backend.put(item)
    original_tokens = store.get_item(item.id).regeneration_tokens
    before = store.regeneration_tokens_paid
    store.access(item.id, 50)
    assert store.regeneration_tokens_paid == pytest.approx(before + 0.25 * original_tokens)


def test_active_hit_charges_no_tokens(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "p0", "groq", "m")
    before = store.regeneration_tokens_paid
    store.access(item.id, 1)
    assert store.regeneration_tokens_paid == before


def test_effective_cost_uses_dollar_cost_when_nonzero(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "p0", "groq", "m")
    item.regeneration_cost = 5.0
    item.regeneration_tokens = 999999  # would dominate if wrongly used
    assert store._effective_cost(item) == pytest.approx(5.0)


def test_effective_cost_falls_back_to_tokens_when_dollar_cost_is_zero(tmp_path):
    store = make_store(tmp_path)
    item = store.add(0, "p0", "groq", "m")
    item.regeneration_cost = 0.0
    item.regeneration_tokens = 100
    assert store._effective_cost(item) == pytest.approx(100 * store.TOKEN_COST_RATE)


def test_cost_norm_uses_effective_cost_never_decreases_under_zero_dollar_workload(tmp_path):
    store = make_zero_cost_store(tmp_path, active_capacity=10, vector_capacity=10, cold_capacity=10)
    norms = []
    for i in range(30):
        store.add(i, f"p{i}", "ollama", "llama3.2:1b")
        norms.append(store._cost_norm)
    assert all(n > 1e-6 for n in norms), "cost_norm never left its floor -- token fallback isn't firing"
    assert all(norms[i] <= norms[i + 1] for i in range(len(norms) - 1))


def test_cost_aware_scoring_protects_expensive_items_under_zero_dollar_workload(tmp_path):
    store = make_zero_cost_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    cheap = store.add(0, "cheap prompt", "ollama", "m")
    expensive = store.add(1, "expensive prompt", "ollama", "m")
    assert store.get_item(cheap.id).regeneration_cost == 0.0
    assert store.get_item(expensive.id).regeneration_cost == 0.0
    store.update_item(cheap.id, regeneration_tokens=1)
    store.update_item(expensive.id, regeneration_tokens=100000)
    store._cost_norm = max(store._cost_norm, 100000 * store.TOKEN_COST_RATE)

    store.update_item(cheap.id, last_accessed_at=1)
    store.update_item(expensive.id, last_accessed_at=1)
    for i in range(2, 8):
        store.add(i, f"filler-{i}", "ollama", "m")

    assert expensive.id in store.items, "token-based cost fallback isn't protecting the expensive item"
    assert cheap.id not in store.items


def test_reconstruction_cost_in_tokens_matches_notebook_free_tier_guidance(tmp_path):
    store = make_zero_cost_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    for i in range(10):
        store.add(i, f"p{i}", "ollama", "llama3.2:1b")
    evicted_id = store.deleted_log[0].id
    store.access(evicted_id, 100)
    assert store.regeneration_cost_paid == 0.0
    assert store.regeneration_tokens_paid > 0


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------
# Step counts are reduced from the original pure-in-memory suite: every
# add()/access() now does real ChromaDB + SQLite I/O on rebalance, which is
# legitimately slower than mutating a dict. See README's "Performance"
# section for measured numbers and the trade-off this represents.

def _run_random_workload(store, n_steps, new_item_prob=0.3, seed=0, prompt_pool_size=40):
    rng = random.Random(seed)
    id_pool = []
    for t in range(n_steps):
        if not id_pool or (rng.random() < new_item_prob and len(id_pool) < prompt_pool_size):
            item = store.add(t, f"prompt-{len(id_pool)}", "groq", "m")
            id_pool.append(item.id)
        else:
            item_id = rng.choice(id_pool)
            store.access(item_id, t)
    return store


def _assert_invariants(store):
    counts = store.tier_counts()
    assert counts[Tier.ACTIVE] <= store.active_capacity
    assert counts[Tier.VECTOR] <= store.vector_capacity
    assert counts[Tier.COLD] <= store.cold_capacity
    assert sum(counts.values()) == len(store.items)
    assert all(item.tier != Tier.DELETED for item in store.items.values())
    assert store.generation_cost_paid >= 0
    assert store.regeneration_cost_paid >= 0
    assert store.generation_tokens_paid >= 0
    assert store.regeneration_tokens_paid >= 0
    assert store._cost_norm > 0
    assert math.isfinite(store.generation_cost_paid)
    assert math.isfinite(store.regeneration_cost_paid)
    assert math.isfinite(store.generation_tokens_paid)
    assert math.isfinite(store.regeneration_tokens_paid)
    for item in store.items.values():
        assert math.isfinite(item.score)


@pytest.mark.parametrize("active,vector,cold", [
    (1, 1, 1), (0, 0, 5), (5, 0, 0), (3, 4, 5), (0, 1, 0),
])
def test_stress_many_configs_hold_invariants(tmp_path, active, vector, cold):
    store = make_store(tmp_path, active_capacity=active, vector_capacity=vector, cold_capacity=cold)
    _run_random_workload(store, n_steps=150, seed=active * 100 + vector * 10 + cold)
    _assert_invariants(store)


def test_stress_moderate_workload_performance_and_correctness(tmp_path):
    """Real-backend analogue of the old 10k-step in-memory perf test.
    Scaled to 400 steps -- large enough to exercise sustained eviction
    pressure and catch an algorithmic regression (e.g. accidental O(n^2) in
    _rebalance), small enough to run in seconds against real Chroma/SQLite."""
    store = make_store(tmp_path, active_capacity=4, vector_capacity=6, cold_capacity=8)
    t0 = time.time()
    _run_random_workload(store, n_steps=400, new_item_prob=0.2, seed=42, prompt_pool_size=100)
    elapsed = time.time() - t0
    _assert_invariants(store)
    assert elapsed < 60.0, f"400-step workload took {elapsed:.2f}s -- possible perf regression"


def test_stress_high_churn_all_new_items_never_reused(tmp_path):
    store = make_store(tmp_path, active_capacity=2, vector_capacity=2, cold_capacity=2)
    n = 200
    for t in range(n):
        store.add(t, f"unique-{t}", "groq", "m")
    _assert_invariants(store)
    assert len(store.deleted_log) == n - 6  # 6 = 2+2+2 resident capacity


def test_stress_repeated_regeneration_of_same_evicted_item(tmp_path):
    store = make_store(tmp_path, active_capacity=1, vector_capacity=1, cold_capacity=1)
    total_capacity = 3
    target = store.add(0, "hot potato", "groq", "m").id
    for i in range(1, total_capacity + 6):
        store.add(i, f"filler-{i}", "groq", "m")
    assert target not in store.items, "setup failed to evict target -- bury count too low"

    regen_count_before = store.regeneration_events
    actual_regen_cycles = 0
    for cycle in range(6):
        t = 100 + cycle * 20
        was_evicted = target not in store.items
        store.access(target, t)
        if was_evicted:
            actual_regen_cycles += 1
            assert target in store.items, "regeneration should re-admit the item as ACTIVE"
        for i in range(total_capacity + 6):
            store.add(t + i + 1, f"bury-{cycle}-{i}", "groq", "m")

    assert actual_regen_cycles > 0
    assert store.regeneration_events == regen_count_before + actual_regen_cycles
    assert store.regeneration_cost_paid > 0
    _assert_invariants(store)


def test_stress_extreme_weight_settings_dont_crash(tmp_path):
    configs = [
        dict(w_recency=1.0, w_frequency=0.0, w_cost=0.0),
        dict(w_recency=0.0, w_frequency=1.0, w_cost=0.0),
        dict(w_recency=0.0, w_frequency=0.0, w_cost=1.0),
        dict(w_recency=0.0, w_frequency=0.0, w_cost=0.0),
        dict(w_recency=5.0, w_frequency=-2.0, w_cost=3.0),
    ]
    for i, cfg in enumerate(configs):
        store = make_store(tmp_path / f"cfg{i}", active_capacity=3, vector_capacity=3, cold_capacity=3, **cfg)
        _run_random_workload(store, n_steps=100, seed=1)
        _assert_invariants(store)


def test_stress_tiny_recency_halflife_and_huge_halflife(tmp_path):
    for halflife in (0.001, 1, 10, 100000):
        store = make_store(tmp_path / str(halflife), recency_halflife=halflife)
        _run_random_workload(store, n_steps=100, seed=int(halflife * 10) % (2**31))
        _assert_invariants(store)


def test_stress_zero_dollar_cost_workload_holds_invariants_and_uses_tokens(tmp_path):
    store = make_zero_cost_store(tmp_path, active_capacity=4, vector_capacity=8, cold_capacity=12)
    _run_random_workload(store, n_steps=300, new_item_prob=0.2, seed=99, prompt_pool_size=80)
    _assert_invariants(store)
    assert store.generation_cost_paid == 0.0
    assert store.regeneration_cost_paid == 0.0
    assert store.generation_tokens_paid > 0
    if store.misses > 0 or store.vector_hits > 0 or store.cold_hits > 0:
        assert store.regeneration_tokens_paid > 0


@pytest.mark.parametrize("active,vector,cold", [
    (1, 1, 1), (0, 0, 5), (3, 4, 5),
])
def test_stress_zero_dollar_cost_many_configs_hold_invariants(tmp_path, active, vector, cold):
    store = make_zero_cost_store(tmp_path, active_capacity=active, vector_capacity=vector, cold_capacity=cold)
    _run_random_workload(store, n_steps=150, seed=active * 100 + vector * 10 + cold)
    _assert_invariants(store)


if __name__ == "__main__":
    import sys
    print("Running regression + stress suite directly (no pytest CLI)...\n")
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
