"""
Offline tests for the Module 3 + Module 4 integration (hooks.py,
llm_backends.py). Module 3's real router needs a Longformer download +
embedding inference; these tests stub that one piece (embed_query) so the
suite stays fast and runs with no internet/GPU, while still exercising
the REAL trained module4_budget.joblib and the REAL module4.py pipeline
(predict + generate + continuation/safe-fallback) end-to-end.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import module3
import module4
import llm_backends
import hooks


class _FakeRouter:
    """Deterministic stand-in for the trained MLPRegressor -- routes any
    query containing 'hard' to "large" (score < 0.5), everything else to
    "small" (score >= 0.5), so tests can assert specific routing decisions
    without depending on the real model's actual judgment calls."""
    def predict(self, embedding):
        # embedding is whatever embed_query() returned, reshaped (1, -1) --
        # the fake embed_query below encodes "hardness" directly into it.
        return np.array([embedding[0, 0]])


@pytest.fixture(autouse=True)
def _reset_module3_and_module4_state():
    """Each test gets a clean module3/module4/llm_backends module-level
    state -- these modules use module-level globals (the same pattern
    module4.py's own test_module4.py already relies on)."""
    module3._STATE.update(router=None, tokenizer=None, embedder=None)
    module4._BACKENDS.update(small=None, large=None)
    module4._STATE.update(model=None, margin=None)
    llm_backends._ROUTER_CONFIGURED = False
    yield
    module3._STATE.update(router=None, tokenizer=None, embedder=None)
    module4._BACKENDS.update(small=None, large=None)
    module4._STATE.update(model=None, margin=None)
    llm_backends._ROUTER_CONFIGURED = False


def _stub_module3(monkeypatch):
    """Installs the fake router + a fake embed_query that encodes the
    literal word 'hard' into the embedding, so route() is fully
    deterministic and needs no network/GPU access."""
    def fake_embed_query(query):
        return np.array([[0.1 if "hard" in query.lower() else 0.9]])
    monkeypatch.setattr(module3, "embed_query", fake_embed_query)
    module3._STATE.update(router=_FakeRouter(), tokenizer=object(), embedder=object())


def test_module3_route_without_real_model(monkeypatch):
    _stub_module3(monkeypatch)
    assert module3.route("an easy question")["decision"] == "small"
    assert module3.route("a hard question")["decision"] == "large"


def test_hooks_select_model_falls_back_to_ollama_when_router_not_configured():
    llm_backends._ROUTER_CONFIGURED = False
    provider, model = hooks.select_model("extractor", "anything")
    assert (provider, model) == (hooks.DEFAULT_PROVIDER, hooks.DEFAULT_MODEL)


def test_hooks_select_model_uses_module3_once_configured(monkeypatch):
    _stub_module3(monkeypatch)
    llm_backends._ROUTER_CONFIGURED = True
    provider, model = hooks.select_model("extractor", "a hard question")
    assert provider == "module4"
    assert model == "large"


def test_hooks_select_token_budget_always_none():
    # Module 4 owns its own budget internally -- see hooks.py's docstring.
    assert hooks.select_token_budget("extractor", "anything") is None


def test_call_llm_module4_shape(monkeypatch):
    """call_llm_module4() must return the exact dict shape
    memory_store.py's add()/access() expect from any call_llm_fn."""
    budget_path = Path(__file__).resolve().parent.parent / "module4_budget.joblib"
    if not budget_path.exists():
        pytest.skip("module4_budget.joblib not present -- run train_module4.py first")
    llm_backends.configure_module4(
        str(budget_path),
        small=module4.MockBackend("small", 20),
        large=module4.MockBackend("large", 20),
    )
    result = llm_backends.call_llm_module4({}, "module4", "small", "What is 2+2?")
    assert set(result) == {"content", "input_tokens", "output_tokens", "cost",
                           "latency_s", "provider", "model"}
    assert result["input_tokens"] == 0
    assert result["output_tokens"] > 0
    assert result["model"] == "module4:small"
    assert isinstance(result["content"], str) and result["content"]


def test_unified_call_llm_dispatches_on_provider(monkeypatch):
    budget_path = Path(__file__).resolve().parent.parent / "module4_budget.joblib"
    if not budget_path.exists():
        pytest.skip("module4_budget.joblib not present -- run train_module4.py first")
    llm_backends.configure_module4(
        str(budget_path),
        small=module4.MockBackend("small", 20),
        large=module4.MockBackend("large", 20),
    )
    result = llm_backends.unified_call_llm({}, "module4", "large", "a question")
    assert result["model"] == "module4:large"

    with pytest.raises(ValueError):
        llm_backends.unified_call_llm({}, "nonexistent_provider", "x", "a question")


def test_full_pipeline_with_stubbed_module3_and_real_module4(monkeypatch):
    """End-to-end: the real orchestration loop (pipeline.run_demo), Module
    3's real routing LOGIC (embedding stubbed, decision rule real), and
    Module 4's real budget model + real run()/generate pipeline (text
    generation mocked via module4.MockBackend) -- everything except the
    two genuinely expensive externals (a real Longformer forward pass and
    a real Llama call) is exercised for real."""
    budget_path = Path(__file__).resolve().parent.parent / "module4_budget.joblib"
    if not budget_path.exists():
        pytest.skip("module4_budget.joblib not present -- run train_module4.py first")
    _stub_module3(monkeypatch)
    llm_backends._ROUTER_CONFIGURED = True

    import pipeline
    questions = ["an easy question", "a hard question"]
    result = pipeline.run_demo(
        questions, passes=2, verbose=False, llm="module34",
        module4_budget_path=str(budget_path),
        module4_small=module4.MockBackend("small", 20),
        module4_large=module4.MockBackend("large", 20),
    )
    assert result["total_requests"] == 2 * len(questions) * 4  # 2 passes x 2 questions x 4 stages
    assert result["per_pass"][0]["fresh"] == len(questions) * 4   # pass 1: all fresh
    assert result["per_pass"][1]["fresh"] == 0                    # pass 2: all served from memory
