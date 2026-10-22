# Module 4 - integration note for Module 3

**Files:** `module4.py` (the code), `train_module4.py` (train once), `test_module4.py` (18 checks, no GPU), `module4_budget.joblib` (created by training).
**Needs:** python, numpy, pandas, scikit-learn, joblib. `torch` + `transformers` only for `HFBackend`; `requests` only for `OpenAIChatBackend`.

## Setup (once, at start-up)
```python
import module4
# python train_module4.py 3_0_my_model_data_train.jsonl module4_budget.joblib      (one time)
module4.configure("module4_budget.joblib",
                  small=module4.HFBackend("meta-llama/Llama-3.1-8B-Instruct", hf_token=TOKEN),            # Llama 3.1 8B
                  large=module4.OpenAIChatBackend("<49B model>", "<base url>/v1", "<api key>"))           # Llama 3.3 49B
```
## Check the wiring before connecting the real models (no GPU)
```
python test_module4.py module4_budget.joblib        # every line must say PASS (18 of 18)
```
## Call (per query)
```python
result = module4.run(query, model_choice)      # model_choice = "small" | "large"  (also accepts "llama-3.1-8b", "llama-3.3-49b")
final_answer = result["final_answer"]
```
## What comes back
| key | meaning |
|---|---|
| `final_answer` | the answer to show the user |
| `model_choice` | `small` or `large` |
| `predicted_budget` | ML prediction (tokens) |
| `budget_given` | budget actually given to the model (= prediction x 0.5, multiple of 32) |
| `tokens_used`, `seconds` | cost of this answer |
| `cut_off` | `True` only if even the fallback was cut off |
| `source` | `budget answer` or `normal answer (fallback: ...)` |
| `note` | set if a backend was missing and the other model answered |
| `quality_check` | only with `check_quality=True`: normal-answer tokens, tokens saved, saving %, coverage, judge, verdict |

`module4.format_report(result)` prints the answer with the report under it. `module4.run_from_module3({"query": ..., "model": ...})` accepts a dict.

## Rules the code follows
* The budget comes from a trained ML model (no keyword rules). Both models answer with the same prompt; the budget is stated as a ceiling.
* If the budget answer is cut off it is continued once (HF backend); if it is still cut off, the normal answer is returned (safe mode).
* `check_quality=True` generates an extra normal answer - use it for testing/demo, not in production.


## Behaviour seen in the real-Llama check (Kaggle, Llama 3.1 8B) - read before connecting
| Question type | Result |
|---|---|
| Normal factual / maths / multiple-choice | 26-51% fewer tokens, answers correct (Pride and Prejudice, clips problem, mammal question) |
| A very short answer expected (budget 32) | the budget version can use a few tokens MORE than a terse normal answer (22 vs 7 tokens for a capital-city question). Answer is still correct. |
| Open-ended "explain in detail" question outside the training tasks | 83% fewer tokens but a shorter, less detailed answer (quality check: REDUCED). If detail matters more than tokens, call `module4.run(query, model, ratio=1.0)` (budget = the full ML prediction) or `ratio=1.5`. |

`ratio` is the share of the ML prediction given to the model (default 0.5, the value tested on 46 real queries: 57% fewer tokens, no correct answer lost).

## Known limits (tested results: 57% fewer tokens, no correct answer lost on 36 checkable real queries)
* Budgets are reliable for query types like the 14 logged tasks and resume prompts; free-form questions outside them can get poor budgets (the fallback keeps answers complete but saves less).
* A similarity guard to detect "unfamiliar" questions was tried and does not separate them reliably, so it is not included.
* The `large` (49B) path was **not run** during testing; only Llama 3.1 8B generated answers. Test your 49B backend with `MockBackend` swapped for it on 3-5 queries first.
