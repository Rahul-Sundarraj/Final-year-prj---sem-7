# Multi-Agent LLM Framework — all 4 modules combined

All four modules are combined here and **verified working together, end
to end**, not just plumbed:

| | What it does | Real artifact |
|---|---|---|
| **Module 1** | Cost-aware tiered memory eviction (ACTIVE → VECTOR → COLD → DELETED) | `memory_store.py` |
| **Module 2** | Shared / transactive memory — exact-match and semantic (TF-IDF+keyword) directories | `shared_memory.py`, `semantic_directory.py` |
| **Module 3** | Trained query-complexity router (Longformer embedding → MLPRegressor → small/large decision) | `module3.py` + `my_hybrid_trained.pkl` |
| **Module 4** | ML-predicted token budget + generation, with safe-mode fallback for cut-off answers | `module4.py` + `module4_budget.joblib` |

## Run it right now

```bash
pip install -r requirements.txt

python3 pipeline.py --llm mock       # instant, offline, Module 1+2 wiring only
python3 pipeline.py --llm ollama     # real Ollama inference, Module 1+2 only
python3 pipeline.py --llm module34   # the full 4-module pipeline
```

`--llm module34` is the real thing: Module 3's actual trained router
decides small/large per question (downloads the Longformer embedding
model from Hugging Face the first time, then it's cached), Module 4
predicts a token budget and generates the answer (using
`module4.MockBackend` for text generation by default — see
**"Connecting real models"** below to swap in a real Llama).

Verified output from an actual run — notice the router making different
small/large decisions per question, and pass 2 correctly replaying each
question's *original* routed answer from Module 2's cache rather than
re-routing it:
```
=== Pass 1/2 ===
  t=0  q0  extractor [LLM call] [small] answer to: Identify what type...
  t=4  q1  extractor [LLM call] [large] answer to: Identify what type...
=== Pass 2/2 ===
  t=16 q0  extractor [shared-hit] [small] answer to: Identify what type...
  t=20 q1  extractor [shared-hit] [large] answer to: Identify what type...

Module 2 -- Shared Memory directory
  Directory hits: 16 (50.0%) -- 16 exact, 0 semantic-only
    Pass 1: 16 fresh LLM calls, 0 shared+cached, 0 shared-but-EVICTED
    Pass 2: 0 fresh LLM calls, 16 shared+cached, 0 shared-but-EVICTED
```

## Architecture

```
            User Query / Multi-Agent Task
                        |
                 hooks.select_model()  <---- Module 3's trained router
                 hooks.select_token_budget()  (always None -- Module 4
                        |                      owns its own budget, see below)
        Module 2 (shared memory) ──> Module 1 (cost-aware store)
      "has anyone answered this?"   "do I still have it cheaply?"
                                            |
                              ACTIVE → VECTOR → COLD → DELETED
                                            |
                          On a MISS only: llm_backends.unified_call_llm()
                                            |
                          Module 4: predict budget -> generate -> safe
                          fallback if cut off (module4.run())
```

Per stage, per question (see `pipeline.py`'s `run_stage()`):

1. **Module 3** (`hooks.select_model`) decides `"small"` or `"large"` from
   the question's trained-router score.
2. **Module 2** checks: has an equivalent question already been answered?
   Hit → go to Module 1. Miss → go straight to step 4.
3. **Module 1** checks if that answer is still resident. Resident → real
   cache hit, done, no LLM call. Already evicted → falls through to a
   real regeneration anyway (`"shared_evicted"` outcome — Module 2 *knew*,
   Module 1 had already forgotten).
4. **Module 4** (`llm_backends.call_llm_module4`) predicts a token budget
   for Module 3's chosen model and generates the final answer in one
   call — including its own continuation/safe-fallback logic if the
   budgeted answer gets cut off. The result goes into Module 1's store and
   Module 2 records it for future reuse.

## Why Module 4 "owns" the actual LLM call

Module 4 wasn't built as "just a budget number" — `module4.run(query,
model_choice)` predicts the budget *and* generates the answer *and*
handles a cut-off answer safely (continue once, or fall back to an
uncapped answer), all in one tested, verified pipeline (18/18 of its own
wiring checks pass — see `module4_wiring_check.py`). That's more complete
than bolting a token cap onto Module 1's own Ollama call, so this repo
uses Module 4's real pipeline as the actual generation step once Module 3
+ Module 4 are configured (`--llm module34`), and keeps the simpler Ollama
path (`--llm ollama`) only as a no-router, no-API-key fallback for
testing Module 1 + Module 2 in isolation.

## Connecting real models

Right now `--llm module34` uses `module4.MockBackend` for both the small
and large model — safe, no GPU or API key needed, good for confirming the
whole pipeline's *logic* is correct. To connect real models:

```python
import module4
small = module4.HFBackend("meta-llama/Llama-3.1-8B-Instruct", hf_token=TOKEN)
large = module4.OpenAIChatBackend("<your 49B model>", "<base url>/v1", "<api key>")

import pipeline
result = pipeline.run_demo(questions, llm="module34",
                           module4_small=small, module4_large=large)
```

**Read this before connecting real models — the three teammates tested
against three different model pairs, and that was never reconciled:**
- Module 1+2's own demo (`--llm ollama`) used a local Ollama model
  (`llama3.2:1b`).
- Module 3's router was tested routing between **Gemini 2.5 Flash / Gemini
  3.6 Flash**.
- Module 4's backends were tested against **Llama 3.1 8B (local HF) /
  Llama 3.3 Nemotron 49B (hosted API)**.

Module 3's router still returns a valid `"small"`/`"large"` *decision*
regardless of which real models you plug into Module 4's backends — but
the team needs to **consciously pick one small/large pair** for the final
system (whatever `module4_small`/`module4_large` you pass in), not assume
either teammate's own testing pairing carries over automatically. See
`llm_backends.py`'s module docstring for the same note in code.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Module 3's router needs internet access the first time (to download
`allenai/longformer-base-4096`, ~500MB, then it's cached locally by
`transformers`). Module 4's budget model is already trained and committed
(`module4_budget.joblib`) — if you need to retrain it on new data:
```bash
python3 train_module4.py path/to/training_data.jsonl module4_budget.joblib
```

For `--llm ollama` (optional — not needed for `--mock` or `--llm module34`):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
ollama serve
```

## Running the tests

```bash
pytest -q                          # 71 tests, all offline, no GPU/internet needed
python3 module4_wiring_check.py module4_budget.joblib   # Module 4's own 18-check suite
```

The pytest suite's Module 3/4 integration tests (`tests/
test_module34_integration.py`) stub only the expensive external piece
(the Longformer forward pass) — Module 3's routing *decision logic* and
Module 4's *entire* budget+generation pipeline run for real against the
real trained `module4_budget.joblib`.

## Files

| File | What it is |
|---|---|
| `models.py`, `storage_backends.py`, `memory_store.py` | **Module 1** |
| `shared_memory.py`, `semantic_directory.py`, `exact_match_directory.py` | **Module 2** (semantic directory includes the false-positive divergence guard — see its own docstring) |
| `module3.py`, `my_hybrid_trained.pkl` | **Module 3** — the router, extracted and cleaned up from the team's own `router.ipynb` |
| `module4.py`, `module4_budget.joblib`, `train_module4.py`, `module4_wiring_check.py` | **Module 4**, exactly as delivered by its owner |
| `local_ollama_llm.py` | Module 1+2's standalone Ollama fallback path |
| `llm_backends.py` | **Where Module 3's decision and Module 4's generation plug into Module 1's `call_llm_fn`** |
| `hooks.py` | The 2-function integration surface between Module 1+2's orchestration loop and Module 3/4 |
| `pipeline.py` | The orchestration loop (Extractor → Analyzer → Writer → Reviewer) |
| `example_agents.py` | Swap this for your team's real agent prompts if you have different ones |
| `utils.py` | Two small helpers (`split_capacity`, `compute_row_stats`) |
| `tests/` | 71 tests covering all 4 modules' integration |

## Known limitations

- **Capacity vs. pool size** (Module 1): eviction capacity is deliberately
  fixed (`DEFAULT_CAPACITIES` in `pipeline.py`), not scaled up with a
  larger question set — scaling it proportionally was found, during
  Module 1's own evaluation, to risk severe system memory pressure.
- **Semantic matching's residual false positives** (Module 2): 3 known
  short-sentence collision patterns remain even after the divergence
  guard — a documented, inherent limit of surface-text similarity
  matching. See `semantic_directory.py`'s docstring.
- **Module 3's router accuracy** isn't independently re-validated here —
  this repo verifies the router is *wired in correctly* and produces a
  valid decision for any input, not that its small/large calls are always
  the "right" ones; that's Module 3's own evaluation to stand on.
- **Module 4's known limits** (from its own docs): budgets are reliable
  for the task types it was trained on; free-form questions well outside
  those can get a poor budget (the safe-mode fallback keeps the answer
  complete but saves less in that case). The 49B/large backend path was
  tested less than the small one during Module 4's own development.
- **Three different model pairs were tested across the three teammates'
  own demos** (see "Connecting real models" above) — not yet reconciled
  into one real pair for final deployment.
