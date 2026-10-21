"""
hooks.py

THE two integration points for Module 3 (LLM routing) and Module 4
(adaptive token budgeting). `pipeline.py` calls exactly these two
functions, once per pipeline stage per question, and nothing else --
that is the entire integration surface. Replace the bodies below with
your real logic; do not change the function names or the shape of what
they return, since `pipeline.py` depends on both.

DEFAULT BEHAVIOR (before Module 3 / Module 4 are wired in): every stage
uses one fixed model and no token cap -- i.e. exactly how Module 1 + 2
behaved on their own. Swap these two functions out one at a time and
nothing else in the codebase needs to change.
"""

# The model Module 3 should treat as "the large/capable option" and "the
# small/cheap option" -- adjust to whatever two models you're routing
# between (e.g. two different Ollama models, or a small local model vs. a
# hosted API model).
DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3.2:1b"


def select_model(stage_name: str, question: str) -> tuple[str, str]:
    """
    MODULE 3 HOOK -- LLM routing.

    Called once per pipeline stage per question, BEFORE Module 2's shared-
    memory lookup. Return the (provider, model) pair that stage/question
    should actually be generated with if a fresh LLM call turns out to be
    necessary (on a Module 2 miss, or a Module 1 miss on an evicted item).

    Args:
        stage_name: which pipeline stage is asking ("extractor",
            "analyzer", "writer", "reviewer", or whatever stage names your
            PIPELINE_STAGES defines -- see example_agents.py).
        question: the actual input text for this stage.

    Returns:
        (provider, model) -- provider is a key into the `clients` dict
        passed to RealCostAwareMemoryStore (see memory_store.py's
        `clients` parameter and local_ollama_llm.LOCAL_CLIENTS for the
        shape); model is whatever model string that provider's call_llm
        implementation expects (e.g. "llama3.2:1b", "phi3:mini").

    DEFAULT (no routing): always returns the same fixed model.
    Replace this with your trained router's prediction -- typically: build
    a feature vector from (stage_name, question) [e.g. question length,
    task-type classification, reasoning-step count], feed it to your
    complexity classifier/regressor, and return the small model's
    (provider, model) tuple below a threshold, the large model's above it.
    """
    return DEFAULT_PROVIDER, DEFAULT_MODEL


def select_token_budget(stage_name: str, question: str) -> "int | None":
    """
    MODULE 4 HOOK -- adaptive token budgeting.

    Called once per pipeline stage per question, at the same point as
    select_model() above. Return a maximum-output-token ceiling for this
    specific generation, or None for "no explicit cap" (the model's own
    default applies).

    This value is threaded all the way through to the real LLM call: see
    memory_store.py's `add()`/`access()` `max_tokens` parameter and
    local_ollama_llm.py's `call_llm()`, which maps it to Ollama's
    `num_predict` option. It is only ever CONSULTED on an actual LLM call
    (a Module 2 miss, or a Module 1 miss on an evicted item) -- a real
    cache hit never calls the LLM at all, so it has no token budget to
    set, by construction.

    Args:
        stage_name: which pipeline stage is asking.
        question: the actual input text for this stage.

    Returns:
        An int token ceiling, or None.

    DEFAULT (no budgeting): always returns None (no cap).
    Replace this with your trained predict_budget() -- typically: classify
    the query's task type (arithmetic, factual, code, open-ended
    explanation, multi-part analysis -- see your own Module 4 task
    categories), then return the matching regressor's predicted token
    ceiling for whichever model select_model() chose for this same call
    (the small-model and large-model regressors predict different budgets
    for the same query, per your own Module 4 findings).
    """
    return None
