# Module 4 integration template — Adaptive Token Budgeting

## Where you plug in

`hooks.py`, one function:

```python
def select_token_budget(stage_name: str, question: str) -> "int | None":
    return None
```

`pipeline.py` calls this once per pipeline stage per question, right
after `select_model()` (Module 3's hook) and before the Module 2 lookup.
Whatever you return is threaded all the way through to the real LLM call
— **this has already been verified to work**, not just plumbed:

```python
# Verified against real Ollama inference:
#   uncapped call  -> 256 output tokens
#   max_tokens=15  -> exactly 15 output tokens
```

## The contract

- **Input**: `stage_name` and `question`, same as Module 3's hook (see
  `MODULE3_TEMPLATE.md` if you need the full prompt text instead of just
  the question).
- **Output**: an `int` token ceiling, or `None` for "no explicit cap, use
  the model's own default" — this is the behavior before Module 4 exists,
  so returning `None` for everything is always a safe, inert default.
- **Only consulted when a real LLM call actually happens** — a Module 2
  cache hit never calls the LLM, so it never needs a token budget; you
  don't need to special-case that.
- **Model-specific**: your own findings show the small-model and
  large-model regressors predict *different* budgets for the same query.
  Since Module 3's `select_model()` is called first (same stage, same
  question), you can condition your budget prediction on which model
  Module 3 picked, by calling `hooks.select_model(stage_name, question)`
  again inside `select_token_budget()` (cheap — it's not an LLM call) or,
  cleaner, by refactoring `run_stage()` in `pipeline.py` to call
  `select_model()` once and pass the result into `select_token_budget()`.

## How it actually reaches the LLM call (for reference, you shouldn't need to touch this)

```
hooks.select_token_budget(stage, question)
        |
pipeline.py's run_stage()
        |
memory_store.py's store.add() / store.access(max_tokens=...)
        |
local_ollama_llm.py's call_llm(..., max_tokens=...)
        |
Ollama's options={"num_predict": max_tokens}
```

If your team swaps in a different LLM provider later (not Ollama), that
provider's own `call_llm`-shaped function needs to accept `max_tokens=None`
too and map it to whatever that API calls its max-output-tokens parameter
(e.g. OpenAI's `max_tokens`, Anthropic's `max_tokens`) — this is the one
place token budgeting is provider-specific.

## Worked example

Based on your own Module 4 findings (two regressors, by task type):

```python
import joblib

_small_model_budget = joblib.load("module4_small_regressor.joblib")
_large_model_budget = joblib.load("module4_large_regressor.joblib")

TASK_TYPES = ["arithmetic", "factual", "code", "open_ended", "multi_part"]

def _classify_task_type(question: str) -> str:
    # Your actual task-type classifier -- illustrative placeholder here.
    ...

def select_token_budget(stage_name: str, question: str) -> "int | None":
    provider, model = select_model(stage_name, question)  # ask Module 3 too
    task_type = _classify_task_type(question)
    regressor = _small_model_budget if model == DEFAULT_MODEL else _large_model_budget
    predicted = regressor.predict([[TASK_TYPES.index(task_type)]])[0]
    return max(1, round(predicted))   # Ollama's num_predict wants a positive int
```

## Testing your integration

1. Sanity-check `select_token_budget()` in isolation first, same as
   Module 3 — a handful of example questions, confirm the returned budgets
   look reasonable for their task type.
2. `python3 pipeline.py --mock` — the offline mock LLM (`memory_store.
   default_mock_call_llm`) **honors `max_tokens` too** (it caps its
   simulated output token count), so you can confirm your budgeting logic
   changes the reported token totals even without real inference.
3. `python3 pipeline.py` for real, and compare total tokens against a run
   with `select_token_budget()` temporarily reverted to always return
   `None` — that delta is your module's actual, measured saving, the same
   way your own evaluation measured 69-74% average savings across three
   scales.
4. Spot-check a few real answers for truncation/quality loss, the same way
   your own 4-query and multi-domain tests did (comparing quality scores,
   not just token counts) — a smaller budget that silently cuts off a
   correct answer mid-sentence is a regression even if the token count
   looks good.
