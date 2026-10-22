"""No-GPU wiring test for Module 4 (uses fake models).   python test_module4.py module4_budget.joblib
Run it after training and before connecting the real Llama models. Every line must say PASS."""
import sys, math, types
import module4

path = sys.argv[1] if len(sys.argv) > 1 else "module4_budget.joblib"
module4.configure(path, small=module4.MockBackend("small", 60), large=module4.MockBackend("large", 300))
results = []
def check(name, cond):
    results.append(bool(cond)); print(("PASS  " if cond else "FAIL  ") + name)

r = module4.run("What is the capital of France?", "small")
check("run() returns a final answer", bool(r["final_answer"]))
check("result has every field Module 3 needs", all(k in r for k in ("final_answer", "model_choice", "predicted_budget", "budget_given", "tokens_used", "cut_off", "source", "seconds")))
check("budget is a multiple of 32 between 32 and 1024", 32 <= r["budget_given"] <= 1024 and r["budget_given"] % 32 == 0)
check("budget = ML prediction x 0.5, rounded up to 32",
      r["budget_given"] == min(1024, max(32, math.ceil(r["predicted_budget"] * module4.BUDGET_RATIO / 32) * 32)))
check("same query -> same budget (deterministic)", module4.predict_budget("What is the capital of France?", "small") == r["predicted_budget"])

for given, expected in [("small", "small"), ("LARGE", "large"), ("llama-3.1-8b-instruct", "small"), ("Llama-3.3-nemotron-49B", "large")]:
    check(f"model choice {given!r} understood as {expected}", module4.normalize_choice(given) == expected)
try: module4.normalize_choice("gpt-4"); check("unknown model name is rejected", False)
except ValueError: check("unknown model name is rejected", True)

r = module4.run_from_module3({"query": "What is 2+2?", "model": "large"})
check("run_from_module3(dict) works and uses the large model", r["model_choice"] == "large" and "[large]" in r["final_answer"])
r = module4.run_from_module3(("What is 2+2?", "small"))
check("run_from_module3(tuple) works and uses the small model", r["model_choice"] == "small")

r = module4.run("Explain deep learning in detail, covering neural networks, CNNs and transformers.", "large", safe=True)
check("safe mode: a cut-off budget answer is replaced by the normal answer", r["source"].startswith("normal") and not r["cut_off"])
r = module4.run("Explain deep learning in detail, covering neural networks, CNNs and transformers.", "large", safe=False)
check("safe=False returns the budget answer even if cut off", r["source"] == "budget answer")

r = module4.run("What is the capital of France?", "small", check_quality=True)
q = r["quality_check"]
check("check_quality=True adds normal tokens, tokens saved, coverage, judge and verdict",
      all(k in q for k in ("normal_tokens", "tokens_saved", "saving_pct", "content_coverage", "judge_score", "verdict")))
check("format_report() prints answer + report", "MODULE 4 REPORT" in module4.format_report(r) and "Quality check" in module4.format_report(r))

module4.configure(path, small=module4.MockBackend("small", 60), large=None)
r = module4.run("What is 2+2?", "large")
check("missing large backend: the small model answers and the result says so", r["note"] != "" and "[small]" in r["final_answer"])

calls = []
class _R:
    def __init__(s, j): s.j = j
    def raise_for_status(s): pass
    def json(s): return s.j
def _post(url, headers, timeout, json):
    calls.append(json["max_tokens"]); cut = json["max_tokens"] < 200
    return _R({"choices": [{"message": {"content": "api answer " + "w " * 50}, "finish_reason": "length" if cut else "stop"}], "usage": {"completion_tokens": 50}})
sys.modules["requests"] = types.SimpleNamespace(post=_post)
module4.configure(path, small=module4.OpenAIChatBackend("m", "http://x/v1", "k"), large=None)
r = module4.run("Explain deep learning in detail, covering neural networks, CNNs and transformers.", "small")
check("API backend: budget call first, then the normal call (1024) when cut off", calls[0] < 1024 and calls[-1] == 1024)

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
