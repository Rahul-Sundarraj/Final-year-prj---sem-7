"""
workload.py

Shared, precomputable workload generator + replayer, used by both
print_comparison.py (mock LLM) and run_real_local_comparison.py (real
Ollama), and by baseline_stores.BeladyOptimalMemoryStore (which needs to
see the whole future before any store runs).

None of the random decisions here depend on store state -- only on step
count, seed, and how many logical items have been introduced so far -- so
the entire event sequence can be generated ONCE and replayed identically
against every store in a comparison. That guarantees every policy sees
literally the same add/access sequence (fair comparison), and lets an
oracle policy look ahead at what hasn't happened yet.
"""

import random


def generate_workload_events(n_steps, new_item_prob=0.3, seed=42, pool_size=40,
                              recency_bias=False, recency_bias_prob=0.7,
                              recency_fraction=1 / 3):
    """Returns a list of (t, kind, idx) tuples, kind in ("new", "access"),
    idx a logical index into whatever prompt pool the caller is using
    (0..pool_size-1, in first-introduced order). Pure bookkeeping -- no
    store, no LLM call -- so it's cheap to compute once and replay against
    many stores.

    recency_bias=True reproduces the original run_real_local_comparison.py
    workload (70% chance an access targets one of the most-recently
    introduced items); recency_bias=False (default) reproduces the
    original print_comparison.py workload (uniform choice across the
    whole pool introduced so far).
    """
    rng = random.Random(seed)
    events = []
    id_pool = []  # logical indices introduced so far, in introduction order
    for t in range(n_steps):
        if not id_pool or (rng.random() < new_item_prob and len(id_pool) < pool_size):
            idx = len(id_pool)
            id_pool.append(idx)
            events.append((t, "new", idx))
        else:
            if recency_bias and rng.random() < recency_bias_prob and len(id_pool) > 3:
                recent_pool = id_pool[-max(3, int(len(id_pool) * recency_fraction)):]
                idx = rng.choice(recent_pool)
            else:
                idx = rng.choice(id_pool)
            events.append((t, "access", idx))
    return events


def future_access_map(events):
    """logical_idx -> sorted list of every timestep that idx is touched
    (both 'new' and 'access' events count as a touch). This is exactly
    "the future" that Belady's optimal algorithm is allowed to see --
    build it once from the full precomputed event list, before any store
    has run."""
    m = {}
    for t, kind, idx in events:
        m.setdefault(idx, []).append(t)
    for idx in m:
        m[idx].sort()
    return m


def run_workload_events(store, events, prompt_source, provider="groq", model="m"):
    """Replays a precomputed event list against a store. `prompt_source`
    maps a logical index to prompt text (e.g. `lambda i: TASK_BANK[i]` or
    `lambda i: f"prompt-{i}"`). Every store given the same `events` list
    processes an identical add/access sequence -- the only thing that can
    differ between stores is how each one internally ranks/evicts items."""
    id_pool = {}  # logical idx -> this store's actual item id
    for t, kind, idx in events:
        if kind == "new":
            item = store.add(t, prompt_source(idx), provider, model)
            id_pool[idx] = item.id
        else:
            store.access(id_pool[idx], t)
    return store
