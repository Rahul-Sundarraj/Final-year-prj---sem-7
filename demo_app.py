"""
demo_app.py

A small Flask UI for demoing the combined 4-module pipeline live.

Type a question in the browser, click Run, and see:
  - BASELINE: a plain orchestrator pipeline with no Module 1 (no
    eviction/caching) and no Module 2 (no shared memory) -- just the 4
    stages (Extractor -> Analyzer -> Writer -> Reviewer), each one a
    fresh call to ONE common LLM, every single time, including on the
    replayed pass (there is nothing to cache, by construction -- this
    is the "no memory system at all" starting point).
  - FULL: the same 4 stages through the real combined pipeline --
    Module 3's trained router picks "small"/"large" per stage (real
    Longformer embedding + MLPRegressor score), Module 4 predicts a
    token budget and generates the answer, and Module 1+2's cache
    means the replayed pass is a real cache hit (no LLM call at all).
  - a side-by-side token total + "% saved" for BASELINE vs FULL, for
    one live question -- the same comparison
    unified_pipeline_experiment.py runs at scale, just for one question
    typed into the browser.

This is the REAL pipeline, not a mock of the UI: module3.route() calls
the actual Longformer embedding + trained MLPRegressor, module4.run()
calls the actual trained budget model (MockBackend for text generation,
same as the rest of this repo's demos -- see README's "Connecting real
models" section to swap in a real backend).

RUN:
    python3 demo_app.py
    # then open http://127.0.0.1:5000
"""
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import hooks
import llm_backends
import module3
import module4
from example_agents import PIPELINE_STAGES
from memory_store import RealCostAwareMemoryStore
from semantic_directory import SemanticSharedMemoryDirectory
from storage_backends import ActiveBackend, ColdBackend, DeletedBackend, VectorBackend
from utils import split_capacity

app = Flask(__name__)

CAPACITIES = split_capacity(20)

# The ONE common LLM the plain-orchestrator baseline calls for every
# stage, every pass -- no routing, no budgeting, no caching. Uses
# module4's MockBackend purely as a stand-in text generator (same
# no-GPU-needed convention as the rest of this repo's demos), NOT as
# "Module 4" -- the baseline has none of Module 3/4's logic involved.
_COMMON_LLM = module4.MockBackend("common", 260)

# Loaded once at start-up so the first demo request isn't stuck
# downloading/loading the Longformer model -- see module3.configure().
print("Loading Module 3's router and Module 4's budget model (once)...")
llm_backends.configure_router("my_hybrid_trained.pkl")
llm_backends.configure_module4(
    "module4_budget.joblib",
    small=module4.MockBackend("small", 80),
    large=module4.MockBackend("large", 260),
)
print("Ready.")


def _tokens_paid_so_far(store):
    return store.generation_tokens_paid + store.regeneration_tokens_paid


def _run_baseline_pass(question, t0):
    """Plain orchestrator, ONE common LLM, no Module 1 (no caching/
    eviction) and no Module 2 (no shared memory) -- every stage is a
    fresh call, every single pass, by construction."""
    trace = []
    context = {}
    t = t0
    for stage_name, prompt_fn, context_key in PIPELINE_STAGES:
        prompt = prompt_fn(snippet=question, **context)
        r = _COMMON_LLM.generate(module4.build_system(None), prompt, module4.NORMAL_MAX)
        content = r["text"]
        context[context_key] = content
        trace.append(dict(
            stage=stage_name,
            cache_status="fresh",
            model="common",
            score=None,
            tokens=r["tokens"],
            content=content,
        ))
        t += 1
    return trace, t


def _run_full_pass(question, store, directory, t0):
    """The real combined pipeline: Module 3 routes, Module 2 checks the
    shared-memory directory, Module 1 serves the cache hit or triggers a
    real Module 4 regeneration on a miss."""
    trace = []
    context = {}
    t = t0
    for stage_name, prompt_fn, context_key in PIPELINE_STAGES:
        prompt = prompt_fn(snippet=question, **context)
        score = module3.route(question)["score"]
        provider, model = hooks.select_model(stage_name, question)

        before = _tokens_paid_so_far(store)
        match = directory.lookup(stage_name, question)
        if match["hit"]:
            misses_before = store.misses
            item = store.access(match["item_id"], t)
            cache_status = "evicted-refetch" if store.misses > misses_before else "cache-hit"
            content = item.content
        else:
            item = store.add(t, prompt, provider, model)
            directory.record(stage_name, question, item.id, f"q{question[:12]}")
            cache_status = "fresh"
            content = item.content
        tokens_this_stage = _tokens_paid_so_far(store) - before

        context[context_key] = content
        trace.append(dict(
            stage=stage_name,
            cache_status=cache_status,
            model=model,
            score=score,
            tokens=round(tokens_this_stage, 1),
            content=content,
        ))
        t += 1
    return trace, t


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    question = (request.json or {}).get("question", "").strip()
    if not question:
        return jsonify(error="Please enter a question."), 400

    # BASELINE: plain orchestrator + one common LLM, no Module 1, no Module 2
    baseline_pass1, t = _run_baseline_pass(question, 0)
    baseline_pass2, _ = _run_baseline_pass(question, t)
    baseline_tokens = sum(s["tokens"] or 0 for s in baseline_pass1 + baseline_pass2)

    # FULL: all 4 modules
    with tempfile.TemporaryDirectory() as root:
        path = Path(root)
        full_store = RealCostAwareMemoryStore(
            {"module4": None}, call_llm_fn=llm_backends.unified_call_llm,
            active_capacity=CAPACITIES["active_capacity"],
            vector_capacity=CAPACITIES["vector_capacity"],
            cold_capacity=CAPACITIES["cold_capacity"],
            active_backend=ActiveBackend(),
            vector_backend=VectorBackend(path=str(path / "chroma_full"), persistent=False),
            cold_backend=ColdBackend(db_path=str(path / "store_full.db")),
            deleted_backend=DeletedBackend(db_path=str(path / "store_full.db")),
        )
        full_dir = SemanticSharedMemoryDirectory()
        full_pass1, t = _run_full_pass(question, full_store, full_dir, 0)
        full_pass2, _ = _run_full_pass(question, full_store, full_dir, t)
        full_tokens = sum(s["tokens"] or 0 for s in full_pass1 + full_pass2)

    saved_pct = (100 * (baseline_tokens - full_tokens) / baseline_tokens
                 if baseline_tokens else 0)

    return jsonify(
        question=question,
        baseline=dict(pass1=baseline_pass1, pass2=baseline_pass2, total_tokens=baseline_tokens),
        full=dict(pass1=full_pass1, pass2=full_pass2, total_tokens=full_tokens),
        saved_pct=round(saved_pct, 1),
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
