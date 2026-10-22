"""
llm_backends.py

Wires Module 3 (query routing) and Module 4 (adaptive token budgeting)
into Module 1's single `call_llm_fn` slot (see memory_store.py's
`RealCostAwareMemoryStore(clients, call_llm_fn=...)`).

THE ACTUAL INTEGRATION, IN ONE SENTENCE: hooks.select_model() asks
Module 3's trained router for a "small"/"large" decision; pipeline.py's
run_stage() passes that decision through store.add()/access() as `model`;
call_llm_module4() (below) uses it to ask Module 4 to both predict a
token budget AND generate the final answer, in one call.

Module 4 OWNS the LLM call here, not Ollama -- it was built and tested
with its own backends (a local Hugging-Face Llama 3.1 8B, or any
OpenAI-compatible API for a larger model), including its own
continuation/safe-fallback logic for a cut-off answer (see module4.py's
own docstring). That's a more complete, already-tested pipeline than
"Module 1 calls Ollama with a token cap," so this repo uses it as the
real generation path once Module 3 + Module 4 are configured, and keeps
the simpler Ollama path (local_ollama_llm.py) available as a
no-API-key, no-router fallback for quick Module 1 + Module 2-only testing
-- see pipeline.py's `--llm` flag.

IMPORTANT, read before connecting real models: Module 3's own demo was
tested against Gemini (gemini-2.5-flash / gemini-3.6-flash) as its
small/large pair; Module 4's own demo was tested against Llama 3.1 8B /
Llama 3.3 Nemotron 49B. The two were never tested against the SAME pair
of real models together. Module 3's router still returns a valid
"small"/"large" DECISION regardless of which actual models you plug into
Module 4's backends -- the team needs to consciously pick one small/large
model pair for the final integrated system (passed to
configure_module4()'s `small=`/`large=` backends), not assume either
teammate's own demo pairing carries over automatically.
"""

import module3
import module4

_ROUTER_CONFIGURED = False


def configure_router(router_path="my_hybrid_trained.pkl"):
    """Module 3 setup -- call once at start-up before using 'module4'
    as a provider in select_model()'s return value (see hooks.py)."""
    global _ROUTER_CONFIGURED
    module3.configure(router_path)
    _ROUTER_CONFIGURED = True


def router_is_configured():
    return _ROUTER_CONFIGURED


def configure_module4(budget_path="module4_budget.joblib", small=None, large=None):
    """Module 4 setup -- call once at start-up. `small`/`large` are
    module4 backend objects (module4.MockBackend() for no-GPU testing,
    module4.HFBackend(...) for a local Hugging Face model,
    module4.OpenAIChatBackend(...) for any OpenAI-compatible API) -- see
    MODULE4_TEMPLATE.md for how to swap these for your team's real models."""
    module4.configure(budget_path, small=small, large=large)


def call_llm_module4(clients, provider, model, prompt, max_tokens=None):
    """The `call_llm_fn` RealCostAwareMemoryStore actually calls when
    `provider == "module4"`. `model` here is Module 3's "small"/"large"
    decision (not a model name string) -- that's what module4.run()
    expects as its model_choice argument. `max_tokens` is accepted for
    interface compatibility with every other call_llm_fn in this repo
    (see memory_store.py's add()/access()) but is NOT used -- Module 4
    predicts and applies its own token budget internally; passing a
    second, independent budget on top would just double-constrain the
    same generation for no reason.

    Module 4's own `tokens_used` already covers both the budgeted
    generation and any safe-mode fallback/continuation cost, so it's
    reported here as `output_tokens` with `input_tokens=0` (rather than
    guessing a split) -- this keeps total-tokens accounting correct
    without inventing a number Module 4 doesn't actually report.
    """
    result = module4.run(prompt, model)
    return dict(
        content=result["final_answer"],
        input_tokens=0,
        output_tokens=result["tokens_used"],
        cost=0.0,
        latency_s=result["seconds"],
        provider=provider,
        model=f"module4:{model}",
    )


def unified_call_llm(clients, provider, model, prompt, max_tokens=None):
    """The single call_llm_fn passed to RealCostAwareMemoryStore when
    running with `--llm module34` (see pipeline.py) -- dispatches to
    whichever provider hooks.select_model() picked for this call, so a
    single pipeline run can mix Module 3+4-routed calls with a plain
    Ollama fallback if select_model() is ever customized to do so."""
    if provider == "module4":
        return call_llm_module4(clients, provider, model, prompt, max_tokens=max_tokens)
    if provider == "ollama":
        from local_ollama_llm import call_llm as ollama_call_llm
        return ollama_call_llm(clients, provider, model, prompt, max_tokens=max_tokens)
    raise ValueError(f"Unknown provider {provider!r} -- add a case in unified_call_llm()")
