"""
pipeline.py

The unified Module 1 + 2 + 3 + 4 orchestration pipeline:

    Module 1 (memory_store.py)       -- cost-aware tiered memory eviction
    Module 2 (semantic_directory.py) -- shared / transactive memory
    Module 3 (module3.py)            -- trained query-complexity router
    Module 4 (module4.py)            -- adaptive token budget + generation

hooks.py is the integration surface between Module 1+2 (the orchestration
loop below) and Module 3+4 -- see that file's docstring for the exact
contract. llm_backends.py is where Module 3's routing decision and
Module 4's budgeting/generation actually get wired into Module 1's single
`call_llm_fn` slot.

PER-STAGE FLOW (run_stage(), below):

    1. hooks.select_model(stage, question)         <- MODULE 3
    2. hooks.select_token_budget(stage, question)   <- MODULE 4 (always
       None in the real 4-module path -- see hooks.py's docstring)
    3. Module 2 lookup: "has any agent already answered this?"
       -> HIT:  Module 1 access() -- real cache hit, or a stale "shared but
                evicted" miss if Module 1 had already thrown it away.
       -> MISS: Module 1 add() -- a real LLM call (Module 3's chosen model,
                Module 4's chosen budget, both via llm_backends.py), then
                record() the question in Module 2's directory for reuse.

RUN:
    python3 pipeline.py --llm mock               # instant, offline wiring check
    python3 pipeline.py --llm ollama             # real Ollama, Module 1+2 only
    python3 pipeline.py --llm module34           # full 4-module pipeline
                                                  # (Module 4 uses MockBackend
                                                  # text generation by default)
    python3 pipeline.py --baseline               # basic exact-match Module 2
                                                  # instead of semantic matching
"""

import argparse
import tempfile
import time
from pathlib import Path

from memory_store import RealCostAwareMemoryStore
from semantic_directory import SemanticSharedMemoryDirectory
from exact_match_directory import ExactMatchDirectory
from storage_backends import ActiveBackend, VectorBackend, ColdBackend, DeletedBackend
from example_agents import PIPELINE_STAGES
from utils import split_capacity, compute_row_stats
import hooks

# Capacity kept FIXED regardless of how many questions you run through this
# pipeline -- deliberately NOT scaled up with a larger question set. Scaling
# eviction capacity proportionally to a much larger candidate pool is what
# previously pushed a similar experiment into severe swap-thrashing in the
# original research repo (see that repo's experiment.md); keeping capacity
# fixed is also how that project demonstrates the capacity-vs-pool-size
# effect deliberately, rather than by accident. Override via
# `run_demo(capacities=...)` if you need a different size, but choose it
# deliberately, not by scaling proportionally to dataset size.
DEFAULT_TOTAL_CAPACITY = 78  # matches the original repo's 26-question default
DEFAULT_CAPACITIES = split_capacity(DEFAULT_TOTAL_CAPACITY)


def run_stage(store, directory, t, stage_name, question, prompt, source_label):
    """One agent stage's turn. Returns (content, outcome, match_info);
    outcome is one of:

        "fresh"          -- Module 2 found nothing close enough; paid a
                            full LLM call via store.add().
        "shared_cached"  -- Module 2 found a match AND Module 1 still had
                            it resident, a real cache hit.
        "shared_evicted" -- Module 2 found a match, but Module 1 had
                            already evicted it past COLD, so store.access()
                            took a miss and paid a full regeneration anyway.

    match_info is the dict the directory's lookup() returned (None on a
    fresh miss) -- carries whether a hit was exact or semantic-only, where
    supported.
    """
    provider, model = hooks.select_model(stage_name, question)           # MODULE 3
    max_tokens = hooks.select_token_budget(stage_name, question)         # MODULE 4

    match = directory.lookup(stage_name, question)
    if match["hit"]:
        misses_before = store.misses
        item = store.access(match["item_id"], t, max_tokens=max_tokens)
        evicted = store.misses > misses_before
        return item.content, ("shared_evicted" if evicted else "shared_cached"), match

    item = store.add(t, prompt, provider, model, max_tokens=max_tokens)
    directory.record(stage_name, question, item.id, source_label)
    return item.content, "fresh", None


def run_pipeline_once(store, directory, questions, t0, verbose=True):
    """Runs every pipeline stage, in order, over every question in
    `questions`. Returns (next free timestep, this pass's outcome counts,
    this pass's semantic-hit records)."""
    t = t0
    counts = dict(fresh=0, shared_cached=0, shared_evicted=0)
    semantic_hit_records = []
    TAGS = {"fresh": "LLM call", "shared_cached": "shared-hit",
            "shared_evicted": "REGEN(evicted)"}
    for i, question in enumerate(questions):
        context = {}
        for stage_name, prompt_fn, context_key in PIPELINE_STAGES:
            prompt = prompt_fn(snippet=question, **context)
            content, outcome, match = run_stage(
                store, directory, t, stage_name, question, prompt, source_label=f"q{i}",
            )
            context[context_key] = content
            counts[outcome] += 1
            if match is not None and not match.get("exact", True):
                semantic_hit_records.append(
                    (stage_name, question, match["matched_query"], match["similarity"])
                )
            if verbose:
                preview = " ".join(content.split())[:60]
                print(f"  t={t:<4} q{i:<3} {stage_name:<10} "
                      f"[{TAGS[outcome]:<14}] {preview}")
            t += 1
    return t, counts, semantic_hit_records


def run_demo(questions, passes=2, verbose=True, llm="mock", baseline=False, capacities=None,
            router_path="my_hybrid_trained.pkl", module4_budget_path="module4_budget.joblib",
            module4_small=None, module4_large=None):
    """Runs the full pipeline and returns a stats dict.

    `questions`: list of question/input strings to replay, `passes` times
        each (pass 2+ is where Module 1/2's savings actually show up --
        pass 1 is always all-fresh, since nothing has been asked yet).
    `baseline`: use the basic exact-match Module 2 directory instead of
        the semantic (TF-IDF + keyword) one.
    `capacities`: override DEFAULT_CAPACITIES (active/vector/cold dict from
        utils.split_capacity()) -- see the module docstring's warning about
        NOT scaling this proportionally to a much larger question set.

    `llm` selects which generation path hooks.select_model() + this run
    actually use -- three options:
        "mock"     -- fully offline, no Module 3/4 involved at all. Same
                     as the original Module 1+2-only wiring check.
        "ollama"   -- real local Ollama inference, still no Module 3/4 --
                     Module 1+2 only, same as before Module 3/4 existed.
        "module34" -- the real, full 4-module pipeline: Module 3's trained
                     router (needs the Longformer download the first time,
                     and `router_path`) decides small/large, Module 4
                     (needs `module4_budget_path`) predicts a token budget
                     and generates the answer. `module4_small`/
                     `module4_large` are module4 backend objects (default:
                     module4.MockBackend() for both -- safe, no GPU/API key
                     needed; pass real module4.HFBackend(...) /
                     module4.OpenAIChatBackend(...) instances to use real
                     models -- see MODULE3_TEMPLATE.md / MODULE4_TEMPLATE.md).
    """
    if llm == "mock":
        from memory_store import default_mock_call_llm as call_llm
        LOCAL_CLIENTS = {"ollama": None}
    elif llm == "ollama":
        from local_ollama_llm import call_llm, LOCAL_CLIENTS
    elif llm == "module34":
        import module4
        import llm_backends
        if not llm_backends.router_is_configured():
            llm_backends.configure_router(router_path)
        llm_backends.configure_module4(
            module4_budget_path,
            small=module4_small or module4.MockBackend("small", 60),
            large=module4_large or module4.MockBackend("large", 300),
        )
        call_llm = llm_backends.unified_call_llm
        LOCAL_CLIENTS = {"module4": None}
    else:
        raise ValueError(f"Unknown llm={llm!r} -- use 'mock', 'ollama', or 'module34'")

    caps = capacities or DEFAULT_CAPACITIES
    directory = ExactMatchDirectory() if baseline else SemanticSharedMemoryDirectory()
    per_pass = []
    all_semantic_hits = []

    with tempfile.TemporaryDirectory() as root:
        path = Path(root)
        store = RealCostAwareMemoryStore(
            LOCAL_CLIENTS, call_llm_fn=call_llm,
            active_capacity=caps["active_capacity"],
            vector_capacity=caps["vector_capacity"],
            cold_capacity=caps["cold_capacity"],
            active_backend=ActiveBackend(),
            vector_backend=VectorBackend(path=str(path / "chroma"), persistent=False),
            cold_backend=ColdBackend(db_path=str(path / "store.db")),
            deleted_backend=DeletedBackend(db_path=str(path / "store.db")),
        )

        t0 = 0
        wall0 = time.perf_counter()
        for p in range(1, passes + 1):
            if verbose:
                print(f"\n=== Pass {p}/{passes} ({len(questions)} questions x "
                      f"{len(PIPELINE_STAGES)} stages) ===")
            t0, counts, semantic_hits = run_pipeline_once(store, directory, questions, t0, verbose)
            all_semantic_hits.extend((p, *rec) for rec in semantic_hits)
            per_pass.append(dict(
                pass_num=p, fresh=counts["fresh"],
                shared_cached=counts["shared_cached"], shared_evicted=counts["shared_evicted"],
            ))
        wall_elapsed = time.perf_counter() - wall0

    dir_stats = directory.stats()
    store_stats = compute_row_stats(store)
    total_items_per_pass = len(questions) * len(PIPELINE_STAGES)
    total_requests = passes * total_items_per_pass

    return dict(
        n_questions=len(questions), passes=passes, total_requests=total_requests,
        capacities=caps, per_pass=per_pass, directory_stats=dir_stats,
        store_stats=store_stats, wall_elapsed=wall_elapsed, semantic_hits=all_semantic_hits,
    )


def print_report(result):
    dir_stats = result["directory_stats"]
    store_stats = result["store_stats"]

    print("\n" + "=" * 78)
    print("Module 2 -- Shared Memory directory")
    print("=" * 78)
    print(f"  Stage requests total:      {result['total_requests']}")
    print(f"  Capacities (active/vector/cold): {result['capacities']}")
    print(f"  Directory lookups:         {dir_stats['lookups']}")
    print(f"  Directory hits:            {dir_stats['hits']} ({dir_stats['hit_rate']:.1%}) "
          f"-- {dir_stats['exact_hits']} exact, {dir_stats['semantic_only_hits']} semantic-only")
    for p in result["per_pass"]:
        print(f"    Pass {p['pass_num']}: {p['fresh']} fresh LLM calls, "
              f"{p['shared_cached']} shared+cached (work actually saved), "
              f"{p['shared_evicted']} shared-but-EVICTED (paid regeneration anyway)")

    saved = sum(p["shared_cached"] for p in result["per_pass"])
    stale = sum(p["shared_evicted"] for p in result["per_pass"])
    if stale:
        print(f"\n  NOTE: Module 2's raw hit rate ({dir_stats['hit_rate']:.1%}) OVERSTATES the")
        print(f"  work saved -- of its {dir_stats['hits']} hits, {stale} pointed at items Module 1")
        print(f"  had already evicted. Effective saving is {saved}/{result['total_requests']} "
              f"requests ({saved / result['total_requests']:.1%}).")

    print("\n" + "=" * 78)
    print("Module 1 -- Cost-Aware Memory Store")
    print("=" * 78)
    print(f"  Active/Vector/Cold hits:   {store_stats['active']}/{store_stats['vector']}/{store_stats['cold']}")
    print(f"  Misses (had to regen):     {store_stats['misses']}")
    print(f"  Overall storage hit rate:  {store_stats['hit_rate']:.1%}")
    print(f"  Total tokens:              {store_stats['total_tokens']:,.0f}")
    print(f"  Wall-clock time:           {result['wall_elapsed']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Module 1 + 2 + 3 + 4 unified pipeline")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--llm", choices=["mock", "ollama", "module34"], default="mock",
                        help="'mock': fully offline Module 1+2 wiring check (default). "
                             "'ollama': real local Ollama inference, Module 1+2 only. "
                             "'module34': the full 4-module pipeline -- Module 3's real "
                             "trained router + Module 4's budgeting/generation (MockBackend "
                             "text generation by default -- see MODULE3_TEMPLATE.md / "
                             "MODULE4_TEMPLATE.md to plug in real models).")
    parser.add_argument("--baseline", action="store_true",
                        help="Use the basic exact-match Module 2 instead of semantic matching")
    parser.add_argument("--router-path", default="my_hybrid_trained.pkl",
                        help="Module 3's trained router (only used with --llm module34)")
    parser.add_argument("--module4-budget-path", default="module4_budget.joblib",
                        help="Module 4's trained budget model (only used with --llm module34; "
                             "run train_module4.py first if this doesn't exist yet)")
    args = parser.parse_args()

    # Replace this with your own question list, or import from a dataset
    # module -- this default set only exists so the pipeline runs out of
    # the box with no extra setup.
    questions = [
        "What is the capital of France?",
        "What is 12 multiplied by 8?",
        "Name a primary color.",
        "What is the boiling point of water in Celsius?",
    ]

    result = run_demo(questions, passes=args.passes, verbose=True,
                      llm=args.llm, baseline=args.baseline,
                      router_path=args.router_path,
                      module4_budget_path=args.module4_budget_path)
    print_report(result)


if __name__ == "__main__":
    main()
