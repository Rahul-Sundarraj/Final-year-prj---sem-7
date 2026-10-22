"""
module3.py  -  MODULE 3: query-complexity LLM router
=================================================================================================
What it does
    Takes a query (an agent's actual task text, e.g. the Researcher's or
    Writer's framing of a question -- see route() below) and decides
    whether it's simple enough for the SMALL model or needs the LARGE one.

    The decision comes from a trained MLPRegressor (my_hybrid_trained.pkl)
    scoring a mean-pooled Longformer embedding of the query -- no keyword
    rules. score >= THRESHOLD (0.5) -> "small", else -> "large".

How Module 1/2's pipeline uses it (see hooks.py's select_model())
    import module3
    module3.configure("my_hybrid_trained.pkl")        # once, at start-up
    decision = module3.route(query)["decision"]        # "small" or "large"

Extracted and cleaned up from the team's own router.ipynb (the "Query
based routing demo" section) -- same model artifact, same embedding
method (mean pooling over Longformer's last_hidden_state), same decision
rule. Only the demo-specific CrewAI/Gemini wiring was left out; that
belongs in whichever pipeline actually calls this module, not in the
router itself.
=================================================================================================
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

EMBEDDING_MODEL = "allenai/longformer-base-4096"
THRESHOLD = 0.5

_STATE = {"router": None, "tokenizer": None, "embedder": None}


def configure(router_path, embedding_model=EMBEDDING_MODEL):
    """Call once at start-up. Loads the trained router (a pickled
    MLPRegressor) and the Longformer embedding model (downloaded once from
    Hugging Face, then cached locally)."""
    import pickle
    with open(router_path, "rb") as f:
        router = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    embedder = AutoModel.from_pretrained(embedding_model)
    embedder.eval()
    _STATE.update(router=router, tokenizer=tokenizer, embedder=embedder)


def embed_query(query: str) -> np.ndarray:
    """Mean-pooled Longformer embedding -- MUST match the pooling method
    the router was trained on (mean over the sequence dimension of
    last_hidden_state)."""
    tokenizer, embedder = _STATE["tokenizer"], _STATE["embedder"]
    if tokenizer is None:
        raise RuntimeError("Call module3.configure(router_path) first")
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        outputs = embedder(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.cpu().numpy()


def route(query: str, threshold: float = THRESHOLD) -> dict:
    """Returns {"decision": "small"|"large", "score": float in [0,1]}.
    Higher score = the router is more confident the SMALL model suffices."""
    if _STATE["router"] is None:
        raise RuntimeError("Call module3.configure(router_path) first")
    embedding = embed_query(query).reshape(1, -1)
    raw_score = _STATE["router"].predict(embedding)[0]
    score = float(np.clip(raw_score, 0, 1))
    decision = "small" if score >= threshold else "large"
    return {"decision": decision, "score": score}
