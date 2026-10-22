"""
unified_pipeline_experiment.py

The actual "what do you get from combining all four modules" experiment --
runs the IDENTICAL question set, through the IDENTICAL orchestration loop
(pipeline.py), in two configurations that differ in only one respect:

    BASELINE : no Module 3 routing (every stage forced to the "large"
               model), no Module 4 budgeting (every generation uncapped,
               via the same backend's own normal/full-length output).
               Module 1 + Module 2 (caching/eviction) still active --
               this isolates what Module 3 + Module 4 add ON TOP of an
               already-working Module 1+2 pipeline, not a from-scratch
               no-memory-at-all baseline (that comparison already exists,
               see Module 2's own README section).

    FULL     : Module 3's real trained router decides small/large per
               stage; Module 4 predicts a token budget for that choice
               and generates the answer with it (MockBackend text
               generation in both configurations, so the comparison
               isolates the routing+budgeting DECISION, not a difference
               in which text-generation backend was used).

QUESTION SET: by default, the same 10-question small set as before. Pass
`--pool-size N` to instead use the first N real benchmark queries from
`small_query_bank.SMALL_QUERY_BANK_1000` (or `_FULL` if N > 1000) --
matching this project's own established pool_size convention from Module
1's experiments, rather than inventing a new one.

CAPACITY IS DELIBERATELY KEPT FIXED regardless of pool size (not scaled
proportionally) -- scaling capacity up with a much larger pool is what
previously pushed a related experiment in this project into severe
system memory pressure. A fixed capacity against a larger candidate pool
is also how this project's own capacity-vs-pool-size experiments are
run, so this isolates "bigger candidate pool, same memory" as the one
new variable.

A NOTE ON COST AT LARGE POOL SIZES: the semantic directory's TF-IDF
re-fit is O(n) per lookup in how many questions have been recorded for
that stage so far, so a full pass's total directory-lookup cost grows
roughly QUADRATICALLY with pool size -- this is the dominant cost at
large pool sizes when using MockBackend (near-instant) rather than real
inference, not the LLM calls themselves. Module 3's real router adds a
roughly CONSTANT ~0.5-1s per stage-call (a real Longformer forward pass),
so its total cost grows only LINEARLY with pool size x stage count x
passes. Run a small `--pool-size` first to measure real per-item timing
on this machine before committing to a large run -- see the printed
per-config wall-clock time and extrapolate from that, rather than
guessing.

RUN:
    python3 unified_pipeline_experiment.py                    # original 10-question set
    python3 unified_pipeline_experiment.py --pool-size 100     # pilot run, measure timing
    python3 unified_pipeline_experiment.py --pool-size 1000    # full-scale run
"""
import argparse
import time
import tempfile
from pathlib import Path

import module3
import module4
import llm_backends
import hooks
from memory_store import RealCostAwareMemoryStore
from semantic_directory import SemanticSharedMemoryDirectory
from storage_backends import ActiveBackend, VectorBackend, ColdBackend, DeletedBackend
from example_agents import PIPELINE_STAGES
from utils import compute_row_stats
import pipeline as pipeline_mod

SMALL_QUESTION_SET = [
    # pipeline.py's own 4 example questions
    "What is the capital of France?",
    "What is 12 multiplied by 8?",
    "Name a primary color.",
    "What is the boiling point of water in Celsius?",
    # Module 3's own router.ipynb test set (3 "easy", 3 "difficult")
    "What is photosynthesis?",
    "What is a binary search tree?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain why the worst-case time complexity of quicksort is O(n^2), "
    "derive the recurrence for the worst case, and explain how randomized "
    "pivot selection changes the expected complexity.",
    "Explain how a Transformer uses self-attention to model relationships "
    "between tokens. Derive the scaled dot-product attention equation and "
    "explain the purpose of query, key, value, and the scaling factor.",
    "Explain the bias-variance tradeoff in machine learning. Relate it to "
    "underfitting and overfitting, explain how model complexity affects "
    "each component, and describe how cross-validation can be used to "
    "select an appropriate model complexity.",
]

SMALL_BACKEND_WORDS = 80   # MockBackend word budgets -- deliberately
LARGE_BACKEND_WORDS = 260  # different, like Module 4's own 60/300 test split

# Capacity kept FIXED regardless of pool size -- see module docstring.
DEFAULT_TOTAL_CAPACITY = 78


def build_questions(pool_size):
    if pool_size is None:
        return SMALL_QUESTION_SET
    from small_query_bank import SMALL_QUERY_BANK_1000
    if pool_size <= 1000:
        return list(SMALL_QUERY_BANK_1000[:pool_size])
    from small_query_bank import SMALL_QUERY_BANK_FULL
    return list(SMALL_QUERY_BANK_FULL[:pool_size])


def _baseline_call_llm(clients, provider, model, prompt, max_tokens=None):
    """No Module 3 routing (model is always forced to 'large' by
    baseline_select_model below), no Module 4 budgeting -- generates the
    backend's own full, uncapped answer (module4.NORMAL_MAX ceiling,
    i.e. effectively "as much as the backend has to say"), the same code
    path module4.run()'s own safe-mode fallback uses internally."""
    backend, _note = module4._pick_backend(module4.normalize_choice(model))
    r = backend.generate(module4.build_system(None), prompt, module4.NORMAL_MAX)
    return dict(content=r["text"], input_tokens=0, output_tokens=r["tokens"],
               cost=0.0, latency_s=0.0, provider=provider, model=f"baseline:{model}")


def run_config(label, questions, use_routing_and_budgeting, passes=2,
              capacities=None, progress_every=200):
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
        call_llm = _baseline_call_llm

    directory = SemanticSharedMemoryDirectory()
    caps = capacities or pipeline_mod.DEFAULT_CAPACITIES
    routing_counts = {"small": 0, "large": 0}

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

        t = 0
        wall0 = time.perf_counter()
        per_pass = []
        n_done = 0
        for p in range(1, passes + 1):
            counts = dict(fresh=0, shared_cached=0, shared_evicted=0)
            for question in questions:
                context = {}
                for stage_name, prompt_fn, context_key in PIPELINE_STAGES:
                    prompt = prompt_fn(snippet=question, **context)
                    provider, model = select_model(stage_name, question)
                    if use_routing_and_budgeting:
                        routing_counts[model] += 1
                    match = directory.lookup(stage_name, question)
                    if match["hit"]:
                        misses_before = store.misses
                        item = store.access(match["item_id"], t)
                        evicted = store.misses > misses_before
                        counts["shared_evicted" if evicted else "shared_cached"] += 1
                        content = item.content
                    else:
                        item = store.add(t, prompt, provider, model)
                        directory.record(stage_name, question, item.id, f"q{question[:12]}")
                        counts["fresh"] += 1
                        content = item.content
                    context[context_key] = content
                    t += 1
                    n_done += 1
                    if progress_every and n_done % progress_every == 0:
                        elapsed = time.perf_counter() - wall0
                        print(f"    [{label}] {n_done} stage-calls done, "
                              f"{elapsed:.0f}s elapsed", flush=True)
            per_pass.append(counts)
        wall_elapsed = time.perf_counter() - wall0

    store_stats = compute_row_stats(store)
    total_requests = passes * len(questions) * len(PIPELINE_STAGES)
    return dict(label=label, per_pass=per_pass, store_stats=store_stats,
               wall_elapsed=wall_elapsed, total_requests=total_requests,
               routing_counts=routing_counts, n_questions=len(questions))


def main():
    parser = argparse.ArgumentParser(description="Combined 4-module baseline-vs-full comparison")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Use the first N real benchmark queries from "
                             "small_query_bank instead of the default 10-question set.")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=200,
                        help="Print a progress line every N stage-calls (0 to disable).")
    args = parser.parse_args()

    questions = build_questions(args.pool_size)
    caps = pipeline_mod.DEFAULT_CAPACITIES  # fixed regardless of pool size, by design

    print(f"Question set: {len(questions)} questions x {len(PIPELINE_STAGES)} "
          f"stages x {args.passes} passes = "
          f"{len(questions) * len(PIPELINE_STAGES) * args.passes} stage-requests")
    print(f"Capacity (fixed): {caps}\n")

    baseline = run_config("BASELINE", questions, use_routing_and_budgeting=False,
                          passes=args.passes, capacities=caps,
                          progress_every=args.progress_every)
    full = run_config("FULL", questions, use_routing_and_budgeting=True,
                      passes=args.passes, capacities=caps,
                      progress_every=args.progress_every)

    for result in (baseline, full):
        s = result["store_stats"]
        print("=" * 78)
        print(result["label"])
        print("=" * 78)
        for p in result["per_pass"]:
            print(f"  pass: fresh={p['fresh']}, shared_cached={p['shared_cached']}, "
                  f"shared_evicted={p['shared_evicted']}")
        if result["routing_counts"]["small"] or result["routing_counts"]["large"]:
            print(f"  routing decisions: small={result['routing_counts']['small']}, "
                  f"large={result['routing_counts']['large']}")
        print(f"  total tokens: {s['total_tokens']:.0f}")
        print(f"  store hit rate: {s['hit_rate']:.1%}")
        print(f"  wall time: {result['wall_elapsed']:.2f}s")
        print()

    base_tokens = baseline["store_stats"]["total_tokens"]
    full_tokens = full["store_stats"]["total_tokens"]
    saved_pct = 100 * (base_tokens - full_tokens) / base_tokens if base_tokens else 0
    print("=" * 78)
    print("COMBINED RESULT")
    print("=" * 78)
    print(f"  Questions: {len(questions)}, capacity: {caps}")
    print(f"  Baseline tokens: {base_tokens:.0f}")
    print(f"  Full (all 4 modules) tokens: {full_tokens:.0f}")
    print(f"  Token saving from Module 3 + Module 4 (on top of Module 1+2): {saved_pct:.1f}%")


if __name__ == "__main__":
    main()
