"""
Local LLM wrapper via Ollama -- drop-in replacement for the notebook's
call_groq / call_gemini / call_openrouter, but talks to a model running
entirely on YOUR machine. No API key, no billing, no rate limits, no
internet needed once the model is pulled.

SETUP (run once, in a terminal, outside Python):
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2:1b        # ~1.3GB, fast, decent quality
    # or: ollama pull phi3:mini    # ~2.3GB, stronger reasoning
    # or: ollama pull qwen2.5:0.5b # ~400MB, tiny/fast, good for quick tests
    ollama serve                   # starts the local server on :11434
                                    # (often auto-starts after install --
                                    #  check with `ollama list` first)

Then:
    pip install ollama --break-system-packages

USAGE (matches call_llm's exact return shape, so it plugs straight into
RealCostAwareMemoryStore.add()/access() with zero changes to that code):

    from local_ollama_llm import call_llm, LOCAL_CLIENTS
    r = call_llm(LOCAL_CLIENTS, "ollama", "llama3.2:1b", "hello")
"""

import time
import ollama

# Local generation has no per-token $ price -- there's nothing to bill.
# Per the notebook's own Cell 13 guidance ("if you used a free-tier /
# free model, your $ cost will legitimately be 0 -- report token counts
# as the cost proxy instead"), cost is reported as 0.0 here and TOKEN
# COUNT is what you should actually report/compare in your writeup.
LOCAL_CLIENTS = {"ollama": ollama}


def call_llm(clients, provider, model, prompt, max_tokens=None):
    """
    Same signature/return shape as the notebook's call_llm(), so it can
    replace it directly in RealCostAwareMemoryStore without touching any
    store/eviction code.

    `max_tokens`: Module 4's hook (see hooks.py's select_token_budget()).
    None (the default) means no explicit cap -- Ollama's own default
    applies, which is this project's original behavior before Module 4
    existed. When set, it's passed as Ollama's `num_predict` option, the
    equivalent of a max-output-tokens ceiling for this one call.
    """
    t0 = time.time()
    options = {"num_predict": max_tokens} if max_tokens is not None else {}
    resp = clients[provider].chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=options,
    )
    latency = time.time() - t0

    # Ollama's response includes real token counts if the model supports it
    in_tok = resp.get("prompt_eval_count", 0)
    out_tok = resp.get("eval_count", 0)

    return dict(
        content=resp["message"]["content"],
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost=0.0,  # local inference -- no $ cost, use tokens as the proxy
        latency_s=latency,
        provider=provider,
        model=model,
    )


if __name__ == "__main__":
    # Quick smoke test -- run this file directly to confirm Ollama is
    # reachable and the model responds, before wiring it into the store.
    test_prompt = "In one sentence, what does a cost-aware memory eviction policy do?"
    result = call_llm(LOCAL_CLIENTS, "ollama", "llama3.2:1b", test_prompt)
    print(f"tokens in={result['input_tokens']} out={result['output_tokens']} "
          f"latency={result['latency_s']:.2f}s")
    print(f"-> {result['content']}")
