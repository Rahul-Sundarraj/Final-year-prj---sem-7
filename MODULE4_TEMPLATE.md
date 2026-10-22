# Module 4 — now integrated (this file used to be a template)

Module 4's real code arrived and is wired in: **`module4.py`** (the
module exactly as delivered, unmodified) + **`module4_budget.joblib`**
(trained for real on the project's `3.0 my_model_data_train.jsonl`, 6,443
good answers / 3,944 unique queries — confirmed via Module 4's own
`module4_wiring_check.py`, 18/18 PASS). `llm_backends.call_llm_module4()`
calls it for real — see `README.md`'s architecture section and
`llm_backends.py`'s own docstring for exactly how.

## What it actually does

```python
import module4
module4.configure("module4_budget.joblib",
                  small=module4.MockBackend("small", 60),   # swap for a real backend
                  large=module4.MockBackend("large", 300))  # when ready -- see below
result = module4.run(query, model_choice)   # model_choice: "small" or "large"
                                             # (this is Module 3's decision)
print(result["final_answer"])
```

One call predicts a token budget *and* generates the final answer *and*
handles a cut-off answer safely (continues once if the backend supports
it, otherwise falls back to an uncapped answer) — see `module4.py`'s own
docstring and `MODULE4_ORIGINAL_DOCS.md` for the full original
integration note, including the real-Llama behavior table (26-51% fewer
tokens on in-distribution questions; known weaker budgets on free-form
questions outside the 14 trained task types).

## Connecting real models (currently MockBackend for both)

```python
import module4
small = module4.HFBackend("meta-llama/Llama-3.1-8B-Instruct", hf_token=TOKEN)
large = module4.OpenAIChatBackend("<your 49B model>", "<base url>/v1", "<api key>")
```
Pass these as `module4_small=`/`module4_large=` to `pipeline.run_demo(...,
llm="module34", ...)`. Test the large/49B path specifically before
relying on it — per Module 4's own docs, it was **not** run against a
real model during Module 4's own testing, only `MockBackend`.

## Retraining the budget model

```bash
python3 train_module4.py path/to/training_data.jsonl module4_budget.joblib
```

## Known limitation (noted in README too)

Module 4's backends were tested against Llama 3.1 8B / Llama 3.3 Nemotron
49B — a different pair than Module 3's own Gemini-based routing demo. See
`MODULE3_TEMPLATE.md`'s matching note and `llm_backends.py`'s module
docstring.
