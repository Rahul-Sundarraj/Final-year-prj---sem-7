"""
module4.py  -  MODULE 4: ML adaptive token budget
=================================================================================================
What it does
    Module 3 gives you:  query  +  model choice ("small" = Llama 3.1 8B, "large" = Llama 3.3 49B)
    Module 4 then:       1. predicts a token budget with a TRAINED ML model (no keyword rules)
                         2. gives that budget to the chosen Llama and returns the FINAL ANSWER
                         3. (optional) checks quality against the normal answer and reports the tokens saved

How Module 3 uses it (3 lines)
    import module4
    module4.configure("module4_budget.joblib", small=<small backend>, large=<large backend>)   # once, at start-up
    result = module4.run(query, model_choice)          # model_choice: "small" or "large"
    print(result["final_answer"])

Backends (what actually calls Llama)
    HFBackend(model_id, hf_token)                    local Hugging Face model  (this is what was tested on Kaggle, Llama 3.1 8B)
    OpenAIChatBackend(model, base_url, api_key)      any OpenAI-compatible chat API (for example a hosted Llama 3.3 49B;
                                                     use the model name / base url from your provider's documentation)
    MockBackend()                                    fake model, to test the integration without a GPU

Train the budget model once (needs the project jsonl):
    python train_module4.py 3_0_my_model_data_train.jsonl module4_budget.joblib
=================================================================================================
"""
from __future__ import annotations
__version__ = "1.0"
import re, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ settings (same as the tested notebook)
SMALL_NAME, LARGE_NAME = "llama-3.1-8b-instruct", "llama-3.3-nemotron-super-49b-v1"
MIN_BUDGET, MAX_BUDGET, STEP = 32, 1024, 32
NORMAL_MAX    = 1024      # ceiling of the normal (no budget) answer
SUCCESS       = 0.5       # a logged answer counts as "good" if performance >= this
COVERAGE      = 0.90      # the budget should cover about this share of unseen queries
N_BINS        = 8
BUDGET_RATIO  = 0.5       # budget given to Llama = ML prediction x this ratio
CONTINUE_FRAC = 0.5       # if cut off, continue by at most this share of the budget
MAX_RETRIES   = 1
SYSTEM_PROMPT = "You are a helpful assistant."


# ------------------------------------------------------------------ 1. the ML budget model
def structure_features(queries):
    q = pd.Series(list(queries)).astype(str)
    return np.c_[np.log1p(q.str.split().str.len()), np.log1p(q.str.len()), q.str.count(r"\d"), q.str.count(r"\?"),
                 q.str.count("\n"), q.str.count(r"(?m)^\s*(?:\d+[.)]|[-*])\s"), q.str.count(r"[{}();=<>]"), q.str.count(",")]

def to_dense(m):
    return m.toarray()

def make_pipeline():
    feats = ColumnTransformer([
        ("text", Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=1500, sublinear_tf=True)),
                           ("dense", FunctionTransformer(to_dense, validate=False))]), "query"),
        ("struct", Pipeline([("f", FunctionTransformer(structure_features, validate=False)), ("sc", StandardScaler())]), "query"),
        ("flag", "passthrough", ["is_large"]),
    ])
    reg = HistGradientBoostingRegressor(loss="squared_error", max_iter=200, learning_rate=0.1,
                                        max_depth=6, min_samples_leaf=10, random_state=42)
    return Pipeline([("features", feats), ("reg", reg)])

def fit_model(d):
    return make_pipeline().fit(d[["query", "is_large"]], d["y"])

def safety_margin(d, coverage=COVERAGE):
    """Out-of-fold errors (grouped by query) -> one log-margin per predicted-length band."""
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=4).split(d, groups=d["query"]):
        oof[te] = fit_model(d.iloc[tr]).predict(d.iloc[te][["query", "is_large"]])
    resid = d["y"].values - oof
    edges = np.quantile(oof, np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins = np.digitize(oof, edges)
    margins = np.array([np.quantile(resid[bins == b], coverage) if (bins == b).sum() >= 30 else np.quantile(resid, coverage)
                        for b in range(N_BINS)])
    return {"edges": edges, "margins": margins}

def to_budget(log_pred, margin):
    log_pred = np.asarray(log_pred, dtype=float)
    m = margin["margins"][np.digitize(log_pred, margin["edges"])]
    tok = np.ceil(np.expm1(log_pred + m) / STEP) * STEP          # + safety margin, rounded UP to a multiple of 32
    return np.clip(tok, MIN_BUDGET, MAX_BUDGET).astype(int)

def train(jsonl_path, out_path="module4_budget.joblib", holdout_test=False):
    """Train the budget model from the project jsonl and save it. Run once.
    holdout_test=False : train on ALL logged answers (use this for the real integration).
    holdout_test=True  : leave out the same 42 test queries as the evaluation notebook (3 per task, seed 7),
                         which reproduces the notebook's budgets exactly."""
    raw = pd.read_json(jsonl_path, lines=True)
    df = raw[raw.model_name.isin([SMALL_NAME, LARGE_NAME])].copy()
    df["is_large"] = (df.model_name == LARGE_NAME).astype(int)
    df = df.rename(columns={"output_tokens": "tokens"})
    if holdout_test:
        small = df[df.is_large == 0].drop_duplicates("query")
        held = small.groupby("task_name", group_keys=False).sample(n=3, random_state=7)
        df = df[~df["query"].isin(set(held["query"]))]
    d = df[(df.performance >= SUCCESS) & (df.tokens > 0)].drop_duplicates(["query", "is_large"]).reset_index(drop=True)
    d["y"] = np.log1p(d["tokens"])
    model, margin = fit_model(d), safety_margin(d)
    joblib.dump({"model": model, "margin": margin, "rows": len(d)}, out_path)
    print(f"Saved {out_path} | trained on {len(d)} good answers ({d['query'].nunique()} unique queries)")
    return out_path

_STATE = {"model": None, "margin": None}

def load_budget_model(path):
    b = joblib.load(path); _STATE.update(model=b["model"], margin=b["margin"])

def predict_budget(query, model_choice="small"):
    """ML prediction of how many tokens a good answer needs (no keyword rules)."""
    if _STATE["model"] is None: raise RuntimeError("Call module4.configure(...) first")
    row = pd.DataFrame({"query": [str(query)], "is_large": [int(normalize_choice(model_choice) == "large")]})
    return int(to_budget(_STATE["model"].predict(row), _STATE["margin"])[0])


# ------------------------------------------------------------------ 2. backends (what calls the Llama models)
class MockBackend:
    """Fake model for testing the integration without a GPU. Writes a long answer that is cut at max_new_tokens."""
    can_continue = True
    def __init__(self, name="mock", words=120): self.name, self.words = name, words
    def generate(self, system, user, max_new_tokens, prefix=""):
        full = f"[{self.name}] answer to: {user[:40]} " + " ".join(f"w{i}" for i in range(self.words))
        toks = full.split(" "); done = len(prefix.split(" ")) if prefix else 0
        part = toks[done:done + max_new_tokens]; finished = done + max_new_tokens >= len(toks)
        raw = (" " if prefix else "") + " ".join(part)
        return {"text": raw.strip(), "raw": raw, "tokens": len(part), "finished": finished}

class HFBackend:
    """Local Hugging Face Llama model (tested: meta-llama/Llama-3.1-8B-Instruct on a Kaggle T4)."""
    can_continue = True
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct", hf_token=None, load_4bit=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        kw = dict(token=hf_token, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw); self.model.eval()
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.tok.truncation_side = "left"      # if a prompt is ever longer than 3072 tokens, cut its beginning, not the generation prompt
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}

    def generate(self, system, user, max_new_tokens, prefix=""):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix
        enc = self.tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=3072, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        ids = out[0][enc["input_ids"].shape[1]:].tolist()
        n, finished = len(ids), False
        for k, t in enumerate(ids):
            if t in self.eos_ids: n, finished = k, True; break
        raw = self.tok.decode(ids[:n], skip_special_tokens=True)
        return {"text": raw.strip(), "raw": raw, "tokens": n, "finished": finished}

class OpenAIChatBackend:
    """Any OpenAI-compatible chat-completions API (for example a hosted Llama 3.3 49B).
    The API cannot continue a cut-off answer, so with safe mode a cut-off answer is replaced by the normal answer."""
    can_continue = False
    def __init__(self, model, base_url, api_key, timeout=300):
        self.model, self.url, self.key, self.timeout = model, base_url.rstrip("/") + "/chat/completions", api_key, timeout
    def generate(self, system, user, max_new_tokens, prefix=""):
        import requests
        r = requests.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout,
                          json={"model": self.model, "temperature": 0, "max_tokens": int(max_new_tokens),
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status(); j = r.json(); ch = j["choices"][0]
        text = ch["message"]["content"] or ""
        return {"text": text.strip(), "raw": text, "tokens": int(j.get("usage", {}).get("completion_tokens", len(text.split()))),
                "finished": ch.get("finish_reason") == "stop"}

_BACKENDS = {"small": None, "large": None}

def configure(budget_model_path, small=None, large=None):
    """Call once at start-up. small / large = backend objects. If one is missing, the other one answers (the result says so)."""
    load_budget_model(budget_model_path); _BACKENDS.update(small=small, large=large)

def normalize_choice(x):
    """Accepts 'small'/'large' or a model name such as 'llama-3.1-8b' / 'llama-3.3-49b' / 'nemotron'."""
    s = str(x).strip().lower()
    if s in ("large", "l") or any(k in s for k in ("49b", "3.3", "nemotron", "70b")): return "large"
    if s in ("small", "s") or any(k in s for k in ("8b", "3.1")): return "small"
    raise ValueError(f"Unknown model choice: {x!r} (use 'small' or 'large')")


# ------------------------------------------------------------------ 3. the pipeline: budget -> answer -> quality check
def build_system(budget=None):
    """The normal answer uses SYSTEM_PROMPT only. The adaptive answer also gets the budget as a ceiling."""
    if budget is None: return SYSTEM_PROMPT
    return (SYSTEM_PROMPT + f" Be as brief as the question allows. Never use more than about {int(budget * 0.75)} words, "
            "and always finish your last sentence.")

_STOP = set("a an the and or but if of to in on for with by at from as is are was were be been it this that these those "
            "you your i we they he she his her their our not no can will would should could may might do does did have has had".split())

def content_coverage(adaptive, normal):
    words = lambda s: {x for x in re.findall(r"[a-z0-9]+", s.lower()) if x not in _STOP}
    n = words(normal)
    return len(words(adaptive) & n) / len(n) if n else 1.0

def llm_judge(backend, query, answer, reference):
    """The model grades the budget answer against the normal answer (0-10)."""
    prompt = ("You are a strict grader. Compare ANSWER to REFERENCE for the QUESTION.\n"
              "10 = same key content and equally complete, 5 = missing important parts, 0 = wrong or empty.\n"
              "A shorter answer is fine if it keeps all key points. Reply with ONE number from 0 to 10 and nothing else.\n\n"
              f"QUESTION:\n{query[:1500]}\n\nREFERENCE:\n{reference[:2500]}\n\nANSWER:\n{answer[:2500]}\n\nScore:")
    m = re.search(r"\d+", backend.generate(SYSTEM_PROMPT, prompt, 4)["text"])
    return min(10, int(m.group())) if m else None

def _pick_backend(choice):
    b, note = _BACKENDS[choice], ""
    if b is None:
        other = "small" if choice == "large" else "large"
        b = _BACKENDS[other]
        if b is None: raise RuntimeError("No backend configured: call module4.configure(path, small=..., large=...)")
        note = f"no '{choice}' backend configured, so the '{other}' model answered"
    return b, note

def run(query, model_choice="small", check_quality=False, ratio=None, safe=True, prompt=None):
    """
    query         the user's question (this is what the budget model reads)
    model_choice  'small' or 'large'  (Module 3's decision)
    check_quality True = also generate the NORMAL answer to measure the real saving and quality (costs extra tokens: demo/testing)
    safe          True = if the budget answer is still cut off, return the normal answer instead
    prompt        optional: the text actually sent to Llama if it differs from `query` (for example a question plus its options)
    """
    choice = normalize_choice(model_choice); backend, note = _pick_backend(choice)
    ratio = BUDGET_RATIO if ratio is None else ratio
    text = prompt or query
    predicted = predict_budget(query, choice)
    budget = int(np.clip(np.ceil(predicted * ratio / STEP) * STEP, MIN_BUDGET, MAX_BUDGET))

    t0 = time.perf_counter()
    r = backend.generate(build_system(budget), text, budget)
    raw, spent, retries = r["raw"], r["tokens"], 0
    while (not r["finished"]) and backend.can_continue and retries < MAX_RETRIES and spent < MAX_BUDGET:
        r = backend.generate(build_system(budget), text, min(max(int(budget * CONTINUE_FRAC), MIN_BUDGET), MAX_BUDGET - spent), prefix=raw)
        raw, spent, retries = raw + r["raw"], spent + r["tokens"], retries + 1
    answer, truncated = raw.strip(), not r["finished"]
    out = {"final_answer": answer, "model_choice": choice, "predicted_budget": predicted, "budget_given": budget,
           "tokens_used": spent, "continued": retries, "cut_off": truncated, "source": "budget answer", "note": note}

    if safe and truncated:                                        # never return a cut-off answer
        n = backend.generate(build_system(None), text, NORMAL_MAX)
        out.update(final_answer=n["text"], tokens_used=spent + n["tokens"], cut_off=not n["finished"],
                   source="normal answer (fallback: the budget answer was cut off)")
        answer = n["text"]
    out["seconds"] = round(time.perf_counter() - t0, 2)

    if check_quality:                                             # the extra normal answer, only to show the saving + quality
        nm = backend.generate(build_system(None), text, NORMAL_MAX)
        cov = content_coverage(answer, nm["text"]); judge = llm_judge(backend, text, answer, nm["text"])
        ok = (not out["cut_off"]) and (cov >= 0.8 or (judge is not None and judge >= 8))
        saved = nm["tokens"] - out["tokens_used"]
        out["quality_check"] = {"normal_tokens": nm["tokens"], "tokens_saved": saved,
                                "saving_pct": round(100 * saved / max(nm["tokens"], 1), 1),
                                "content_coverage": round(cov, 2), "judge_score": judge,
                                "verdict": "PRESERVED" if ok else "REDUCED"}
    return out

def run_from_module3(payload):
    """Accepts what Module 3 hands over: a dict {'query':..., 'model':...} or a tuple (query, model)."""
    if isinstance(payload, dict): q, m = payload["query"], payload.get("model", payload.get("model_choice", "small"))
    else: q, m = payload
    return run(q, m)

def format_report(res):
    """The final answer with the Module 4 report under it."""
    lines = [res["final_answer"], "", "-" * 72, "MODULE 4 REPORT",
             f"Model used     : {res['model_choice']}" + (f"   ({res['note']})" if res["note"] else ""),
             f"ML budget      : predicted {res['predicted_budget']} tokens -> budget given to the model {res['budget_given']}",
             f"Tokens used    : {res['tokens_used']}   (continued {res['continued']}x, cut off: {res['cut_off']})   | {res['seconds']}s",
             f"Answer source  : {res['source']}"]
    q = res.get("quality_check")
    if q:
        lines += [f"Normal answer  : {q['normal_tokens']} tokens  ->  saved {q['tokens_saved']} tokens ({q['saving_pct']}%)",
                  f"Quality check  : content coverage {q['content_coverage']} | judge {q['judge_score']}/10 | {q['verdict']}"]
    else:
        lines.append("Quality check  : not run (run(..., check_quality=True) to measure saving and quality)")
    return "\n".join(lines)
