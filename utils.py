"""
utils.py

Two small, standalone helpers pipeline.py depends on, extracted from the
original research repo's capacity_sweep.py and multi_store_report.py (the
rest of those two files is experiment-specific and not needed here).
"""


def split_capacity(total):
    """Splits `total` resident-item capacity into (active, vector, cold) in
    a 3:4:5 ratio, rounding so the three parts still sum to exactly
    `total`. Capacity is intentionally NOT scaled up with a larger question
    set -- see pipeline.py's module docstring for why."""
    raw = [total * 3 / 12, total * 4 / 12, total * 5 / 12]
    parts = [max(1, round(r)) for r in raw]
    diff = total - sum(parts)
    parts[2] += diff
    return dict(active_capacity=parts[0], vector_capacity=parts[1], cold_capacity=parts[2])


def compute_row_stats(store):
    """Summary stats for one RealCostAwareMemoryStore -- hit/miss counts,
    hit rate, tokens spent, and per-tier latency. Used by pipeline.py's
    printed report."""
    total_hits = store.active_hits + store.vector_hits + store.cold_hits
    total_lookups = total_hits + store.misses
    hit_rate = total_hits / total_lookups if total_lookups else 0.0
    total_spend = store.generation_cost_paid + store.regeneration_cost_paid
    total_tokens = store.generation_tokens_paid + store.regeneration_tokens_paid
    lat = store.latency_stats()

    regen_tokens = store.regeneration_tokens_paid
    regen_latency_s = sum(store.miss_latencies)
    tier_latency_s = sum(sum(v) for v in store.latencies.values())

    return dict(
        active=store.active_hits, vector=store.vector_hits, cold=store.cold_hits,
        misses=store.misses, hit_rate=hit_rate, total_spend=total_spend,
        total_tokens=total_tokens,
        regen_tokens=regen_tokens,
        regen_latency_s=regen_latency_s,
        tier_latency_s=tier_latency_s,
        lookup_latency_s=regen_latency_s + tier_latency_s,
        avg_active_ms=lat["active_hit"]["avg_ms"], avg_vector_ms=lat["vector_hit"]["avg_ms"],
        avg_cold_ms=lat["cold_hit"]["avg_ms"], avg_miss_ms=lat["miss_regen"]["avg_ms"],
        avg_gen_ms=lat["generation"]["avg_ms"],
    )
