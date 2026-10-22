# Module 3 — now integrated (this file used to be a template)

Module 3's real code arrived and is wired in: **`module3.py`** +
**`my_hybrid_trained.pkl`** (the trained router, extracted and cleaned up
from the team's `router.ipynb`). `hooks.select_model()` calls it for real
— see `README.md`'s architecture section and `hooks.py`'s own docstring
for exactly how.

## What it actually does

```python
import module3
module3.configure("my_hybrid_trained.pkl")   # once, downloads Longformer
                                              # from HF the first time
result = module3.route("some question")
# {"decision": "small" or "large", "score": float in [0, 1]}
```

A Longformer embedding (mean-pooled over `last_hidden_state`) of the query
goes into a trained `MLPRegressor`; score ≥ 0.5 → `"small"`, else
`"large"`. No keyword rules.

## If you want to change the router itself

- **Retrain it**: that's Module 3's own notebook (`router.ipynb`'s
  training cells, not included in this repo — ask Module 3's owner for
  the training pipeline if you need to retrain on new data). Drop the new
  `.pkl` in and point `configure()` at it — nothing else changes.
- **Change the decision threshold**: `module3.route(query, threshold=0.6)`
  — see `module3.py`.
- **Feed it the full prompt instead of just the question**: `route()`
  currently takes whatever text `hooks.select_model()` passes it (the raw
  `question`, not the full stage prompt with prior-stage context — see
  `example_agents.py`). If your router needs the fuller prompt text,
  change `hooks.select_model()`'s call to `module3.route(prompt)` and
  thread `prompt` through from `pipeline.py`'s `run_stage()` the same way
  `llm_backends.py`'s docstring describes for Module 4.

## Known limitation (noted in README too)

Module 3's own demo routed between Gemini 2.5 Flash / Gemini 3.6 Flash —
a different small/large pair than Module 4's own tested backends (Llama
3.1 8B / Llama 3.3 Nemotron 49B). The router's *decision* ("small" vs.
"large") is model-agnostic and works fine either way, but the team needs
to consciously choose one real small/large pair for final deployment —
see `llm_backends.py`'s module docstring.
