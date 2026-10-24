"""
unified_pipeline_experiment.py

The actual "what do you get from combining all four modules" experiment --
runs the IDENTICAL question set, through the IDENTICAL orchestration loop
(pipeline.py), across FOUR configurations that turn Module 1+2 (caching/
eviction) and Module 3+4 (routing/budgeting) on or off independently:

    MODULE 1+2 ONLY   : caching/eviction ON, routing/budgeting OFF (every
                        stage forced to the "large" model, uncapped
                        generation).
    MODULE 3+4 ONLY   : caching/eviction OFF (every stage is a genuinely
                        fresh call, every pass -- nothing is ever served
                        from a cache), routing/budgeting ON.
    TRUE ZERO MODULES : caching/eviction OFF, routing/budgeting OFF -- a
                        plain orchestrator pipeline hitting ONE common
                        LLM, every stage, every pass. The actual
                        from-scratch starting point with none of the
                        four modules involved.
    FULL (1+2+3+4)    : caching/eviction ON, routing/budgeting ON -- all
                        four modules together.

Three comparisons are printed, each isolating a different question:
    1. MODULE 1+2 ONLY   -> FULL   : what do Module 3+4 add on top of an
                                     already-working Module 1+2 pipeline?
    2. MODULE 3+4 ONLY   -> FULL   : what does Module 1+2's caching add on
                                     top of an already-working Module 3+4
                                     pipeline?
    3. TRUE ZERO MODULES -> FULL   : what do all four modules together save
                                     against the real from-scratch baseline?

All four configurations use the SAME MockBackend text generation (so the
comparisons isolate the routing+budgeting+caching DECISIONS, not a
difference in which text-generation backend was used), and the SAME fixed
capacity (not scaled with pool size -- see below).

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


def _uncapped_call_llm(clients, provider, model, prompt, max_tokens=None):
    """No Module 4 budgeting -- generates the backend's own full, uncapped
    answer (module4.NORMAL_MAX ceiling, i.e. effectively "as much as the
    backend has to say"), the same code path module4.run()'s own safe-mode
    fallback uses internally. Used by any configuration with
    routing/budgeting OFF, regardless of whether caching is on."""
    backend, _note = module4._pick_backend(module4.normalize_choice(model))
    r = backend.generate(module4.build_system(None), prompt, module4.NORMAL_MAX)
    return dict(content=r["text"], input_tokens=0, output_tokens=r["tokens"],
               cost=0.0, latency_s=0.0, provider=provider, model=f"nobudget:{model}")


def build_groq_backends(api_key, small_model="llama-3.1-8b-instant",
                        large_model="llama-3.3-70b-versatile"):
    """Real backends via Groq's free, OpenAI-compatible API -- swap in for
    MockBackend to measure real inference instead of synthetic filler text.
    Get a free key at https://console.groq.com (API Keys)."""
    base = "https://api.groq.com/openai/v1"
    return (module4.OpenAIChatBackend(small_model, base, api_key),
           module4.OpenAIChatBackend(large_model, base, api_key))


def run_config(label, questions, use_caching, use_routing_and_budgeting,
              passes=2, capacities=None, progress_every=200, backends=None):
    """Runs `questions` through the 4-stage pipeline `passes` times.

    use_caching=True   -> Module 1 (eviction) + Module 2 (shared memory)
                           active: a Module 2 hit is served from Module 1's
                           store without a fresh LLM call.
    use_caching=False  -> no Module 1, no Module 2: every single stage
                           call, every pass, is a fresh LLM call. This is
                           what makes "MODULE 3+4 ONLY" and "TRUE ZERO
                           MODULES" genuinely have no memory system, not
                           just a relabeled Module 1+2 run.

    use_routing_and_budgeting=True  -> Module 3 (hooks.select_model) picks
                           small/large per stage, Module 4 predicts the
                           budget and generates the answer.
    use_routing_and_budgeting=False -> every stage forced to "large",
                           uncapped generation (_uncapped_call_llm) -- a
                           single common LLM with no routing decision.
    """
    small, large = backends or (module4.MockBackend("small", SMALL_BACKEND_WORDS),
                                module4.MockBackend("large", LARGE_BACKEND_WORDS))
    module4.configure("module4_budget.joblib", small=small, large=large)

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

    t = 0
    wall0 = time.perf_counter()
    per_pass = []
    n_done = 0
    total_tokens_nocache = 0  # only used when use_caching=False

    def run_loop(lookup_and_record):
        nonlocal t, n_done, total_tokens_nocache
        for p in range(1, passes + 1):
            counts = dict(fresh=0, shared_cached=0, shared_evicted=0)
            for question in questions:
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
            per_pass.append(counts)

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
    total_requests = passes * len(questions) * len(PIPELINE_STAGES)
    return dict(label=label, per_pass=per_pass, total_tokens=total_tokens,
               hit_rate=hit_rate, wall_elapsed=wall_elapsed,
               total_requests=total_requests, routing_counts=routing_counts,
               n_questions=len(questions))


def print_config(result):
    print("=" * 78)
    print(result["label"])
    print("=" * 78)
    for p in result["per_pass"]:
        print(f"  pass: fresh={p['fresh']}, shared_cached={p['shared_cached']}, "
              f"shared_evicted={p['shared_evicted']}")
    if result["routing_counts"]["small"] or result["routing_counts"]["large"]:
        print(f"  routing decisions: small={result['routing_counts']['small']}, "
              f"large={result['routing_counts']['large']}")
    print(f"  total tokens: {result['total_tokens']:.0f}")
    print(f"  hit rate: {result['hit_rate']:.1%}")
    print(f"  wall time: {result['wall_elapsed']:.2f}s")
    print()


def print_comparison(title, from_result, to_result):
    from_tokens = from_result["total_tokens"]
    to_tokens = to_result["total_tokens"]
    saved_pct = 100 * (from_tokens - to_tokens) / from_tokens if from_tokens else 0
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {from_result['label']}: {from_tokens:.0f} tokens")
    print(f"  {to_result['label']}: {to_tokens:.0f} tokens")
    print(f"  Token saving: {saved_pct:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Combined 4-module comparison, 3 baselines")
    parser.add_argument("--pool-size", type=int, default=None,
                        help="Use the first N real benchmark queries from "
                             "small_query_bank instead of the default 10-question set.")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=200,
                        help="Print a progress line every N stage-calls (0 to disable).")
    parser.add_argument("--backend", choices=["mock", "groq"], default="mock",
                        help="'mock' = instant synthetic filler text (default). "
                             "'groq' = real inference via Groq's free OpenAI-compatible "
                             "API -- requires GROQ_API_KEY env var, get one free at "
                             "https://console.groq.com.")
    args = parser.parse_args()

    questions = build_questions(args.pool_size)
    caps = pipeline_mod.DEFAULT_CAPACITIES  # fixed regardless of pool size, by design

    backends = None
    if args.backend == "groq":
        import os
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("--backend groq requires the GROQ_API_KEY environment "
                             "variable (free key: https://console.groq.com)")
        backends = build_groq_backends(api_key)
        print("Backend: Groq (real inference) -- small=llama-3.1-8b-instant, "
              "large=llama-3.3-70b-versatile\n")

    print(f"Question set: {len(questions)} questions x {len(PIPELINE_STAGES)} "
          f"stages x {args.passes} passes = "
          f"{len(questions) * len(PIPELINE_STAGES) * args.passes} stage-requests")
    print(f"Capacity (fixed, only applies when caching is on): {caps}\n")

    common_kwargs = dict(passes=args.passes, capacities=caps,
                         progress_every=args.progress_every, backends=backends)

    module12_only = run_config("MODULE 1+2 ONLY (caching on, routing/budgeting off)",
                               questions, use_caching=True,
                               use_routing_and_budgeting=False, **common_kwargs)
    module34_only = run_config("MODULE 3+4 ONLY (caching off, routing/budgeting on)",
                               questions, use_caching=False,
                               use_routing_and_budgeting=True, **common_kwargs)
    true_zero = run_config("TRUE ZERO MODULES (plain orchestrator, one common LLM)",
                           questions, use_caching=False,
                           use_routing_and_budgeting=False, **common_kwargs)
    full = run_config("FULL (Module 1+2+3+4)", questions, use_caching=True,
                      use_routing_and_budgeting=True, **common_kwargs)

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
