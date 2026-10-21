# Multi-Agent LLM Framework — Module 1 + Module 2 (combined), template for Module 3 + Module 4

This repo is the **combined, working integration of Module 1 (cost-aware
memory eviction) and Module 2 (shared / transactive memory)**, plus a
ready-to-extend **template for Module 3 (LLM routing) and Module 4
(adaptive token budgeting)**. It's meant to be the shared base the whole
team builds the final pipeline on top of.

It is a trimmed, cleaned-up copy of the Module 1 research repo — only the
files Module 1 + Module 2 actually need to run are here, not the full
experiment/sweep/comparison harness that repo also contains.

## What's already working

Run this right now, no setup beyond `pip install -r requirements.txt`:

```bash
python3 pipeline.py --mock      # instant, offline, checks the wiring
```

Or, with real local LLM inference (needs Ollama — see **Setup** below):

```bash
python3 pipeline.py             # real Ollama inference, semantic Module 2
python3 pipeline.py --baseline  # real Ollama inference, basic exact-match Module 2
```

This replays a small question set through a 4-stage agent pipeline
(Extractor → Analyzer → Writer → Reviewer) twice, and prints a report
showing: how many LLM calls were actually needed, how many were served
from shared memory, and how many Module 2 *thought* it could serve from
memory but Module 1 had already evicted (a real regeneration, not a free
hit). Pass 2 is where the savings show up — pass 1 is always all-fresh
calls, since nothing has been asked yet.

## Architecture

```
            User Query / Multi-Agent Task
                        |
        ┌───────────────┼───────────────┐
    Module 3         Module 4        Agent Pipeline
  (LLM routing)   (token budget)   (Extractor/Analyzer/
        |               |           Writer/Reviewer)
        └───────┬───────┘                |
                |                        |
        Module 2 (shared memory) ──> Module 1 (cost-aware store)
      "has anyone answered this?"   "do I still have it cheaply?"
                                            |
                              ACTIVE → VECTOR → COLD → DELETED
                                            |
                              Real LLM call on a MISS only
```

Every stage call goes through exactly this sequence (see `pipeline.py`'s
`run_stage()`):

1. **Module 3's hook** picks which model to use for this stage/question.
2. **Module 4's hook** picks a token budget for this stage/question.
3. **Module 2** is checked: has an equivalent question already been
   answered? If yes, go to step 4 with that answer's id. If no, go
   straight to step 5.
4. **Module 1** is asked for that item. If it's still resident, that's a
   real hit — done, no LLM call. If Module 1 already evicted it, this
   silently falls through to a real regeneration (a "stale hit" — see
   `pipeline.py`'s `shared_evicted` outcome).
5. A fresh LLM call is made (using Module 3's model and Module 4's token
   budget), the result goes into Module 1's store, and Module 2 records it
   for future reuse.

## Files

| File | What it is |
|---|---|
| `models.py` | `Tier` enum and `RealMemoryItem` — the data shape every tier/backend stores |
| `storage_backends.py` | The 4 real tiers: `ActiveBackend` (dict), `VectorBackend` (ChromaDB), `ColdBackend`/`DeletedBackend` (SQLite) |
| `memory_store.py` | **Module 1** — `RealCostAwareMemoryStore`, the cost-aware tiered eviction engine |
| `shared_memory.py` | **Module 2 (basic)** — exact-match shared-memory directory |
| `semantic_directory.py` | **Module 2 (semantic)** — TF-IDF + keyword-similarity matching, with the false-positive divergence guard (see its own docstring for why that guard exists) |
| `exact_match_directory.py` | Adapter so the basic directory exposes the same interface as the semantic one |
| `local_ollama_llm.py` | Real local LLM adapter (Ollama) — also where Module 4's token-budget hook actually takes effect |
| `utils.py` | Two small helpers (`split_capacity`, `compute_row_stats`) |
| `example_agents.py` | A working example 4-stage agent definition — **replace with your team's real agents** |
| **`hooks.py`** | **The entire Module 3 / Module 4 integration surface — see below** |
| `pipeline.py` | The orchestration loop wiring all of the above together |
| `tests/` | The Module 1 + Module 2 test suite (64 tests) carried over from the research repo |

## How to integrate Module 3 and Module 4

**You only need to edit `hooks.py`.** Nothing else in this repo should
need to change. It defines exactly two functions:

```python
def select_model(stage_name: str, question: str) -> tuple[str, str]:
    """MODULE 3 HOOK. Return (provider, model) to actually use."""
    return DEFAULT_PROVIDER, DEFAULT_MODEL   # <- replace with your router

def select_token_budget(stage_name: str, question: str) -> "int | None":
    """MODULE 4 HOOK. Return a max-output-token ceiling, or None for no cap."""
    return None   # <- replace with your predict_budget()
```

`pipeline.py` calls both of these once per stage per question, before
checking Module 2 or Module 1. Full details, contracts, and worked
examples are in:

- **[`MODULE3_TEMPLATE.md`](MODULE3_TEMPLATE.md)** — LLM routing
- **[`MODULE4_TEMPLATE.md`](MODULE4_TEMPLATE.md)** — adaptive token budgeting

Both hooks have been **verified end-to-end against real Ollama inference**
already — `select_token_budget()` returning a number genuinely caps the
model's real output length (confirmed: an uncapped call produced 256
tokens, the same call with a budget of 15 produced exactly 15). You're
plugging a trained predictor into a socket that's already proven to work,
not an unverified stub.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For real LLM inference (optional — --mock works with none of this):
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
ollama serve   # usually auto-starts after install; check with `ollama list`
```

## Running the tests

```bash
pytest -q
```

All 64 tests should pass, offline, no Ollama required (they use a
deterministic fake embedding function and a mock LLM — see
`tests/conftest.py`).

## Using your own question set / agents

`pipeline.py`'s `main()` hardcodes 4 example questions purely so the
pipeline runs out of the box. Replace `example_agents.py` (or add your own
module with the same `(stage_name, prompt_fn, context_key)` shape and
import it in `pipeline.py` instead) with your team's real agent stages,
and pass your own question list into `run_demo(questions, ...)`.

## Known limitations (inherited from Module 1 + Module 2's own evaluation)

- **Capacity vs. pool size**: eviction capacity is deliberately held fixed
  (`DEFAULT_CAPACITIES` in `pipeline.py`) rather than scaled up with a
  larger question set. Scaling it up proportionally was found, during
  Module 1's own evaluation, to risk severe system memory pressure. If you
  need a bigger capacity, set it deliberately via `run_demo(capacities=...)`,
  don't scale it proportionally to dataset size.
- **Semantic matching's residual false positives**: even after the
  divergence guard in `semantic_directory.py`, 3 known patterns remain
  unresolved — short sentences that differ from an unrelated question by
  only a few words. This is a documented, inherent limit of surface-text
  similarity matching, not an oversight. See that file's own docstring for
  the full account.
