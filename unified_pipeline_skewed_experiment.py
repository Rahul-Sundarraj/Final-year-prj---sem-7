"""
unified_pipeline_skewed_experiment.py

The same four-configuration, three-test comparison as
unified_pipeline_experiment.py, but driven by a SKEWED (realistic,
recency-biased) access pattern instead of "N distinct questions, asked
once, then the identical set replayed verbatim" (the flat/uniform
pattern unified_pipeline_experiment.py uses).

WHY THIS EXISTS: unified_pipeline_experiment.py's pool_size=100/1000
results showed MODULE 1+2 ONLY's cache hit rate collapsing to 0.0% at
scale -- every question in that test is equally "cold" (asked once per
pass, no question is more popular than another), so a capacity of 78
resident items is instantly overwhelmed and nothing survives from pass 1
to pass 2. That is a genuine, correctly-measured result, but it only
tells you what happens under a workload with NO temporal locality. Real
multi-agent traffic is not like that -- a few sub-tasks get asked
repeatedly within a short window (the same clarifying question, the same
sub-goal re-checked by different agents) while a long tail of other
questions are asked once and never revisited. Module 1 (Cost-Aware Memory
Eviction)'s own standalone experiments already establish this distinction
(see CAME/experiments/02_workload_regime.md): "uniform" access (no
locality) and "skewed" access (70% of accesses hit the most-recently-
introduced third of the pool) produce very different hit rates for the
exact same fixed capacity.

This script reuses CAME's own `workload.py` event generator
(`generate_workload_events(..., recency_bias=True)`) to produce the
IDENTICAL kind of skewed event sequence, then replays it through this
project's full four-module pipeline (all four MODULE 1+2 ONLY /
MODULE 3+4 ONLY / TRUE ZERO MODULES / FULL configurations), instead of
a synthetic single-item memory-store microbenchmark. Each workload event
resolves to one of `--pool-size` real benchmark questions
(`small_query_bank.py`); every event -- "new" (first time this question
is asked) or "access" (a repeat) -- is run through the identical 4-stage
pipeline (Extractor/Analyzer/Writer/Reviewer) as the uniform experiment,
so the ONLY variable that changes between the two scripts is the
schedule of which question gets asked when, not the pipeline logic
itself.

RUN:
    python3 unified_pipeline_skewed_experiment.py
    python3 unified_pipeline_skewed_experiment.py --steps 600 --pool-size 300
"""
import argparse
import tempfile
import time
from pathlib import Path

import hooks
import llm_backends
import module3
import module4
import pipeline as pipeline_mod
from example_agents import PIPELINE_STAGES
from memory_store import RealCostAwareMemoryStore
from semantic_directory import SemanticSharedMemoryDirectory
from storage_backends import ActiveBackend, ColdBackend, DeletedBackend, VectorBackend
from unified_pipeline_experiment import (SMALL_BACKEND_WORDS, LARGE_BACKEND_WORDS,
                                         _uncapped_call_llm, print_comparison)
from utils import compute_row_stats
from workload import generate_workload_events

# Skewed-regime parameters, matching CAME's own "skewed" definition
# (recency_bias=True, 70% of accesses hit the most-recently-introduced
# third of the pool -- see CAME/experiments/02_workload_regime.md) rather
# than inventing a new skew model for this project.
DEFAULT_STEPS = 300
DEFAULT_POOL_SIZE = 200       # distinct questions AVAILABLE to be introduced
DEFAULT_NEW_ITEM_PROB = 0.3   # same default as workload.py's own signature
DEFAULT_SEED = 42


def build_prompt_bank(pool_size):
    from small_query_bank import SMALL_QUERY_BANK_1000
    if pool_size <= 1000:
        return list(SMALL_QUERY_BANK_1000[:pool_size])
    from small_query_bank import SMALL_QUERY_BANK_FULL
    return list(SMALL_QUERY_BANK_FULL[:pool_size])


def run_config_skewed(label, events, prompt_bank, use_caching,
                      use_routing_and_budgeting, capacities=None,
                      progress_every=200):
    """Same semantics as unified_pipeline_experiment.run_config(), but
    driven by a precomputed skewed event sequence instead of `passes`
    verbatim repeats of a flat question list. Every event resolves to one
    question, which is then run through all 4 pipeline stages exactly as
    in the uniform experiment -- the store/directory naturally determines
    hit-vs-miss from the event's actual position in the skewed schedule,
    so "new" vs "access" is not special-cased here."""
    module4.configure(
        "module4_budget.joblib",
        small=module4.MockBackend("small", SMALL_BACKEND_WORDS),
        large=module4.MockBackend("large", LARGE_BACKEND_WORDS),
    )

    if use_routing_and_budgeting:
        if not llm_backends.router_is_configured():
            llm_backends.configure_router("my_hybrid_trained.pkl")
        select_model = hooks.select_model
        call_llm = llm_backends.unified_call_llm
    else:
        def select_model(stage_name, question):
            return "module4", "large"
        call_llm = _uncapped_call_llm

    caps = capacities or pipeline_mod.DEFAULT_CAPACITIES
    routing_counts = {"small": 0, "large": 0}
    counts = dict(fresh=0, shared_cached=0, shared_evicted=0)

    t = 0
    wall0 = time.perf_counter()
    n_done = 0
    total_tokens_nocache = 0

    def run_loop(lookup_and_record):
        nonlocal t, n_done, total_tokens_nocache
        for _t_event, _kind, idx in events:
            question = prompt_bank[idx]
            context = {}
            for stage_name, prompt_fn, context_key in PIPELINE_STAGES:
                prompt = prompt_fn(snippet=question, **context)
                provider, model = select_model(stage_name, question)
                if use_routing_and_budgeting:
                    routing_counts[model] += 1
                content = lookup_and_record(stage_name, question, prompt,
                                            provider, model, t, counts)
                context[context_key] = content
                t += 1
                n_done += 1
                if progress_every and n_done % progress_every == 0:
                    elapsed = time.perf_counter() - wall0
                    print(f"    [{label}] {n_done} stage-calls done, "
                          f"{elapsed:.0f}s elapsed", flush=True)

    if use_caching:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            store = RealCostAwareMemoryStore(
                {"module4": None}, call_llm_fn=call_llm,
                active_capacity=caps["active_capacity"],
                vector_capacity=caps["vector_capacity"],
                cold_capacity=caps["cold_capacity"],
                active_backend=ActiveBackend(),
                vector_backend=VectorBackend(path=str(path / "chroma"), persistent=False),
                cold_backend=ColdBackend(db_path=str(path / "store.db")),
                deleted_backend=DeletedBackend(db_path=str(path / "store.db")),
            )
            directory = SemanticSharedMemoryDirectory()

            def lookup_and_record(stage_name, question, prompt, provider, model, t, counts):
                match = directory.lookup(stage_name, question)
                if match["hit"]:
                    misses_before = store.misses
                    item = store.access(match["item_id"], t)
                    evicted = store.misses > misses_before
                    counts["shared_evicted" if evicted else "shared_cached"] += 1
                else:
                    item = store.add(t, prompt, provider, model)
                    directory.record(stage_name, question, item.id, f"q{question[:12]}")
                    counts["fresh"] += 1
                return item.content

            run_loop(lookup_and_record)
            stats = compute_row_stats(store)
            total_tokens, hit_rate = stats["total_tokens"], stats["hit_rate"]
    else:
        def lookup_and_record(stage_name, question, prompt, provider, model, t, counts):
            nonlocal total_tokens_nocache
            result = call_llm({}, provider, model, prompt)
            total_tokens_nocache += result["input_tokens"] + result["output_tokens"]
            counts["fresh"] += 1
            return result["content"]

        run_loop(lookup_and_record)
        total_tokens = total_tokens_nocache
        hit_rate = 0.0

    wall_elapsed = time.perf_counter() - wall0
    return dict(label=label, counts=counts, total_tokens=total_tokens,
               hit_rate=hit_rate, wall_elapsed=wall_elapsed,
               routing_counts=routing_counts)


def print_config(result):
    print("=" * 78)
    print(result["label"])
    print("=" * 78)
    c = result["counts"]
    print(f"  fresh={c['fresh']}, shared_cached={c['shared_cached']}, "
          f"shared_evicted={c['shared_evicted']}")
    if result["routing_counts"]["small"] or result["routing_counts"]["large"]:
        print(f"  routing decisions: small={result['routing_counts']['small']}, "
              f"large={result['routing_counts']['large']}")
    print(f"  total tokens: {result['total_tokens']:.0f}")
    print(f"  hit rate: {result['hit_rate']:.1%}")
    print(f"  wall time: {result['wall_elapsed']:.2f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description="Skewed-workload 4-module comparison, 3 baselines")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Number of question-level workload events (each event = 1 question = 4 stage-calls).")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                        help="Distinct real benchmark questions available to be introduced.")
    parser.add_argument("--new-item-prob", type=float, default=DEFAULT_NEW_ITEM_PROB)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=200)
    args = parser.parse_args()

    prompt_bank = build_prompt_bank(args.pool_size)
    events = generate_workload_events(
        n_steps=args.steps, new_item_prob=args.new_item_prob, seed=args.seed,
        pool_size=args.pool_size, recency_bias=True,
    )
    n_new = sum(1 for _, kind, _ in events if kind == "new")
    n_access = len(events) - n_new
    caps = pipeline_mod.DEFAULT_CAPACITIES

    print(f"Skewed workload: {args.steps} question-events ({n_new} new, "
          f"{n_access} repeat accesses) x {len(PIPELINE_STAGES)} stages = "
          f"{args.steps * len(PIPELINE_STAGES)} stage-requests")
    print(f"Pool size (distinct questions available): {args.pool_size}, "
          f"new_item_prob={args.new_item_prob}, seed={args.seed}")
    print(f"Capacity (fixed): {caps}")
    print("Regime: SKEWED (recency_bias=True, 70% of accesses hit the "
          "most-recently-introduced third of the pool -- matching "
          "Cost-Aware Memory Eviction's own 'skewed' definition)\n")

    common_kwargs = dict(capacities=caps, progress_every=args.progress_every)

    module12_only = run_config_skewed("MODULE 1+2 ONLY (caching on, routing/budgeting off)",
                                      events, prompt_bank, use_caching=True,
                                      use_routing_and_budgeting=False, **common_kwargs)
    module34_only = run_config_skewed("MODULE 3+4 ONLY (caching off, routing/budgeting on)",
                                      events, prompt_bank, use_caching=False,
                                      use_routing_and_budgeting=True, **common_kwargs)
    true_zero = run_config_skewed("TRUE ZERO MODULES (plain orchestrator, one common LLM)",
                                  events, prompt_bank, use_caching=False,
                                  use_routing_and_budgeting=False, **common_kwargs)
    full = run_config_skewed("FULL (Module 1+2+3+4)", events, prompt_bank,
                             use_caching=True, use_routing_and_budgeting=True,
                             **common_kwargs)

    for result in (module12_only, module34_only, true_zero, full):
        print_config(result)

    print_comparison("TEST 1: MODULE 1+2 ONLY -> FULL "
                     "(what Module 3+4 add on top of Module 1+2)",
                     module12_only, full)
    print_comparison("TEST 2: MODULE 3+4 ONLY -> FULL "
                     "(what Module 1+2's caching adds on top of Module 3+4)",
                     module34_only, full)
    print_comparison("TEST 3: TRUE ZERO MODULES -> FULL "
                     "(what all four modules together save vs. no modules at all)",
                     true_zero, full)


if __name__ == "__main__":
    main()
