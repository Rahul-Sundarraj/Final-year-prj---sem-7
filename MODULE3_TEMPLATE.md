# Module 3 integration template — LLM Routing

## Where you plug in

`hooks.py`, one function:

```python
def select_model(stage_name: str, question: str) -> tuple[str, str]:
    return DEFAULT_PROVIDER, DEFAULT_MODEL
```

That's it. `pipeline.py` calls this once per pipeline stage per question,
*before* Module 2's shared-memory lookup, and uses whatever `(provider,
model)` you return if a fresh LLM call turns out to be necessary.

## The contract

- **Input**: `stage_name` (e.g. `"extractor"`, `"analyzer"`, `"writer"`,
  `"reviewer"` — whatever your `example_agents.py`/your own agents module
  defines in `PIPELINE_STAGES`) and `question` (the raw input text for
  this stage — note this is the *question*, not the full prompt built
  from it; if your router needs the full prompt text instead, see "If you
  need the full prompt" below).
- **Output**: a `(provider, model)` tuple. `provider` is a key into the
  `clients` dict passed to `RealCostAwareMemoryStore` (see
  `local_ollama_llm.LOCAL_CLIENTS` — currently just `{"ollama": ollama}`).
  `model` is whatever model string that provider's `call_llm` expects
  (e.g. `"llama3.2:1b"`, `"phi3:mini"`).
- **Called on every stage**, even when the result turns out not to be
  needed (a Module 2 cache hit means the model you picked is never
  actually invoked) — so keep this cheap. A trained classifier/regressor
  inference call is fine; don't do anything that itself calls an LLM.

## Worked example

Say your router is a trained classifier that predicts "easy" vs. "hard"
from a handful of cheap features (question length, presence of certain
keywords, etc.), and you've decided "easy" goes to a small model, "hard"
goes to a larger one:

```python
import joblib

_router = joblib.load("module3_router.joblib")   # your trained classifier
SMALL_MODEL = ("ollama", "llama3.2:1b")
LARGE_MODEL = ("ollama", "phi3:mini")

def _features(stage_name, question):
    # Whatever your actual router was trained on -- this is illustrative.
    return [[len(question), question.count("?"), stage_name == "writer"]]

def select_model(stage_name: str, question: str) -> tuple[str, str]:
    is_hard = _router.predict(_features(stage_name, question))[0]
    return LARGE_MODEL if is_hard else SMALL_MODEL
```

If you have more than two models/tiers, just return the right tuple for
whichever tier your router picked — `pipeline.py` doesn't assume there are
only two.

## If you need the full prompt, not just the question

Right now `select_model()` only receives `question` (the raw input text),
not the full stage prompt (which includes prior stages' outputs as
context — see `example_agents.py`'s `analyzer_prompt`/`writer_prompt`/
`reviewer_prompt`). If your router's features genuinely need the full
prompt text:

1. Change `select_model`'s signature to also accept `prompt: str`.
2. In `pipeline.py`'s `run_stage()`, the prompt is already built (the
   `prompt` parameter) *before* `select_model()` is called — just pass it
   through: `hooks.select_model(stage_name, question, prompt)`.

This is a one-line change in exactly one place (`run_stage()`); nothing
else depends on the hook's exact signature.

## Testing your integration

1. First, sanity-check your router in isolation — call `select_model()`
   directly with a few example `(stage_name, question)` pairs and confirm
   it returns what you expect, before wiring it into the pipeline.
2. Run `python3 pipeline.py --mock` — this exercises the full pipeline
   (including your router) with zero LLM cost, so you can confirm nothing
   crashes and the routing decisions look sane in the printed trace
   (`t=... q... stage_name [...] preview`).
3. Run `python3 pipeline.py` for real and compare total tokens/cost against
   a run with `select_model()` temporarily reverted to always return the
   large model — that delta is your routing module's actual, measured
   saving, the same way Module 3's own CrewAI evaluation measured a 31.6%
   cost reduction.
