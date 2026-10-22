"""
hooks.py

THE two integration points for Module 3 (LLM routing) and Module 4
(adaptive token budgeting). `pipeline.py` calls exactly these two
functions, once per pipeline stage per question, and nothing else --
that is the entire integration surface.

THIS IS NOW WIRED TO THE REAL MODULES, not stubs:

- `select_model()` calls Module 3's trained router (module3.route()) when
  it's been configured (see llm_backends.configure_router()), and returns
  ("module4", decision) -- "module4" tells pipeline.py's call_llm_fn
  (llm_backends.unified_call_llm) to hand this call to Module 4 for
  budgeting + generation, with `decision` ("small"/"large") as Module 4's
  model_choice.
- `select_token_budget()` deliberately still returns None always: Module 4
  predicts and applies its OWN token budget internally as part of
  generating the answer (see llm_backends.call_llm_module4()'s docstring
  for why a second, independent budget here would be redundant).

FALLBACK (if Module 3's router isn't configured -- e.g. running
`pipeline.py --llm ollama` for a quick Module 1+2-only check with no
GPU/API keys needed): select_model() returns a single fixed Ollama model
instead, exactly how Module 1+2 behaved before Module 3/4 existed. This
also means `pipeline.py --mock` and `--llm ollama` still work with zero
change to this file.
"""

import llm_backends

# Used only as the fallback below, when Module 3's router hasn't been
# configured (see llm_backends.configure_router()) -- e.g. a quick
# Module 1 + Module 2-only run via `pipeline.py --llm ollama`.
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3.2:1b"


def select_model(stage_name: str, question: str) -> tuple[str, str]:
    """
    MODULE 3 HOOK -- LLM routing.

    Called once per pipeline stage per question, BEFORE Module 2's shared-
    memory lookup. Returns the (provider, model) pair that stage/question
    should actually be generated with if a fresh LLM call turns out to be
    necessary (on a Module 2 miss, or a Module 1 miss on an evicted item).

    Returns ("module4", "small"|"large") -- Module 3's real routing
    decision -- once `llm_backends.configure_router(...)` has been called.
    Falls back to a single fixed Ollama model otherwise (no routing).
    """
    if llm_backends.router_is_configured():
        import module3
        decision = module3.route(question)["decision"]
        return "module4", decision
    return DEFAULT_PROVIDER, DEFAULT_MODEL


def select_token_budget(stage_name: str, question: str) -> "int | None":
    """
    MODULE 4 HOOK -- adaptive token budgeting.

    Always returns None: when select_model() routed this call to Module 4
    (provider == "module4"), Module 4's own module4.run() predicts and
    applies its own token budget as an integral part of generating the
    answer -- see llm_backends.call_llm_module4(). This hook exists
    separately only for the Ollama fallback path (provider == "ollama"),
    where a non-None return value WOULD be honored (see
    local_ollama_llm.call_llm()'s `max_tokens` -> Ollama's `num_predict`)
    -- it's left as None there too since the Ollama path is meant as a
    simple Module 1+2-only fallback, not a second budgeting mechanism.
    """
    return None
