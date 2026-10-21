"""
memory_store.py

RealCostAwareMemoryStore, refactored to store tier data in real backends
instead of one big in-memory dict of dataclass objects:

    ACTIVE  -> ActiveBackend  (in-process dict -- the hot working set)
    VECTOR  -> VectorBackend  (real embedded ChromaDB collection)
    COLD    -> ColdBackend    (real SQLite table on disk)
    DELETED -> DeletedBackend (real SQLite table on disk -- durable history)

The scoring/tiering ALGORITHM is unchanged from the original notebook cell
(same recency/frequency/cost-weighted score, same capacity cuts). What
changed is where an item's data actually lives: on every rebalance, items
move between backends (delete from the old one, write to the new one)
instead of just having their `.tier` field flipped in a shared dict.

IMPORTANT -- migrating tests off direct dict mutation
-------------------------------------------------------
Old code (and old tests) did things like:

    store.items[item.id].regeneration_cost = 999.0

That mutated a live Python object sitting in one shared dict, so the
mutation was automatically "durable" for free. With real backends, `.items`
returns a freshly-assembled read view (see below) -- for VECTOR/COLD items
that view is reconstructed from Chroma/SQLite on every access, so mutating
the object you get back does NOT persist for those two tiers. Use:

    store.update_item(item.id, regeneration_cost=999.0)

which finds the item wherever it currently lives and writes the change back
to that backend. `store.get_item(item_id)` is the read-only equivalent of
`store.items[item_id]`.

NOT a universal guarantee -- ACTIVE is the one exception, and it's easy to
get burned by relying on it: `ActiveBackend.get()`/`.all()` return the
literal live objects sitting in its dict (unlike VectorBackend/ColdBackend,
which reconstruct fresh copies from Chroma/SQLite metadata on every read),
so a direct mutation on an item currently resident in ACTIVE silently DOES
persist -- and then silently stops working the moment that same item gets
evicted down to VECTOR/COLD, with no error either way. Don't rely on this;
always go through `update_item()` regardless of which tier an item happens
to be in right now.
"""

import math
import time
import uuid
import hashlib

from models import Tier, RealMemoryItem
from storage_backends import ActiveBackend, VectorBackend, ColdBackend, DeletedBackend


# ---------------------------------------------------------------------------
# Default mock LLM layer (swap via call_llm_fn=... for a real provider, see
# local_ollama_llm.py / local_hf_llm.py)
# ---------------------------------------------------------------------------

def default_mock_call_llm(clients, provider, model, prompt, max_tokens=None):
    """Deterministic, free, offline stand-in for a real call_llm(). Cost/
    tokens vary by prompt (via hashlib -- stable across processes, unlike
    Python's randomized built-in hash()) so cost-aware eviction still has
    real variance to work with.

    `max_tokens` (Module 4's hook, see hooks.py) is honored here too --
    capping the simulated output token count -- so Module 4's budget
    predictor can be exercised and tested offline, not only against real
    Ollama inference."""
    h = int(hashlib.md5(f"{provider}|{model}|{prompt}".encode()).hexdigest(), 16) % 1000
    in_tok = 20 + (h % 80)
    out_tok = 10 + (h % 150)
    if max_tokens is not None:
        out_tok = min(out_tok, max_tokens)
    cost = (in_tok / 1000) * 0.0003 + (out_tok / 1000) * 0.0025
    return dict(
        content=f"mock response for: {prompt[:30]}",
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost=cost,
        latency_s=0.0,
        provider=provider,
        model=model,
    )


class RealCostAwareMemoryStore:
    # See storage_backends.py / models.py for the fallback-cost rationale;
    # unchanged from the original notebook cell.
    TOKEN_COST_RATE = 1e-4

    def __init__(self, clients, active_capacity=4, vector_capacity=8, cold_capacity=12,
                 # recency_halflife=10. NOTE, because the history here is
                 # instructive: this was briefly changed to 5, on the
                 # strength of a 10-seed weight_sweep.py run showing hl=5 as
                 # the only config beating LRU on both hit rate and spend.
                 # That sweep ran against the UNIFORM workload regime, which
                 # was later shown to contain no exploitable signal at all
                 # (all policies within ~3pp, Random nominally best -- see
                 # print_comparison.WORKLOAD_REGIMES). The hl=5 result was
                 # therefore fitted noise. Re-running the identical sweep
                 # against the SKEWED (realistic) regime reversed it: hl=5
                 # scores 88.8% / -13.5% spend vs LRU, while hl=10 scores
                 # 90.9% / -18.4% -- i.e. hl=10 dominates hl=5 on BOTH axes.
                 # Reverted accordingly.
                 #
                 # Still workload-specific (800 steps, 5% new-item rate,
                 # capacities 3/4/5): re-run `python3 weight_sweep.py`
                 # before trusting these against a different workload shape
                 # or the real task bank. On the skewed regime the
                 # recency-heavy config (0.7/0.15/0.15) is the only one that
                 # beats LRU on both axes (+0.29pp, -9.1%), but it saves
                 # only half as much as this default does -- which is the
                 # wrong trade for a policy whose entire thesis is cost.
                 w_recency=0.4, w_frequency=0.3, w_cost=0.3, recency_halflife=10,
                 active_backend=None, vector_backend=None, cold_backend=None,
                 deleted_backend=None, call_llm_fn=None):
        self.clients = clients
        self.active_capacity = active_capacity
        self.vector_capacity = vector_capacity
        self.cold_capacity = cold_capacity
        self.w = (w_recency, w_frequency, w_cost)
        self.recency_halflife = recency_halflife

        # Real storage backends -- this replaces the old single `self.items`
        # dict. Each is independently swappable/injectable (tests inject
        # tmp-path SQLite files and a fake embedding function; production
        # code can just take the defaults).
        # NOTE: deliberately `is None` checks, not `x or Default()` -- every
        # backend implements __len__, so an empty-but-valid backend (0
        # resident items) would be falsy and get silently replaced by a
        # fresh default under `or`.
        self.active_backend = active_backend if active_backend is not None else ActiveBackend()
        self.vector_backend = vector_backend if vector_backend is not None else VectorBackend()
        self.cold_backend = cold_backend if cold_backend is not None else ColdBackend()
        self.deleted_backend = deleted_backend if deleted_backend is not None else DeletedBackend()

        self._call_llm = call_llm_fn if call_llm_fn is not None else default_mock_call_llm

        self.regeneration_events = 0
        self.regeneration_cost_paid = 0.0
        self.generation_cost_paid = 0.0
        self.regeneration_tokens_paid = 0
        self.generation_tokens_paid = 0
        self.active_hits = 0
        self.vector_hits = 0
        self.cold_hits = 0
        self.misses = 0
        self._cost_norm = 1e-6
        self._latency_norm = 1e-6  # seconds; used by LatencyAwareCostMemoryStore

        # Wall-clock latency samples, in SECONDS, for the real backend I/O
        # behind each kind of event -- what the person actually pays in
        # time when a query lands in each tier. Recorded around the real
        # backend call only (dict lookup / Chroma get / SQLite select /
        # LLM call), NOT around _rebalance(), since rebalance cost is
        # store-wide bookkeeping, not "how long did THIS query take".
        self.latencies = {
            Tier.ACTIVE: [],
            Tier.VECTOR: [],
            Tier.COLD: [],
        }
        self.miss_latencies = []        # full miss -> regeneration (real LLM call)
        self.generation_latencies = []  # add() -> first-time generation (real LLM call)

    # -- backend plumbing ---------------------------------------------------

    def _backend_for_tier(self, tier):
        return {
            Tier.ACTIVE: self.active_backend,
            Tier.VECTOR: self.vector_backend,
            Tier.COLD: self.cold_backend,
        }[tier]

    def _find_item(self, item_id):
        """Look for a resident item across ACTIVE -> VECTOR -> COLD.
        Returns (item, backend) or (None, None). Times each backend's real
        `.get()` call individually -- the successful one is recorded as
        that tier's hit latency (see self.latencies)."""
        for tier in (Tier.ACTIVE, Tier.VECTOR, Tier.COLD):
            backend = self._backend_for_tier(tier)
            t0 = time.perf_counter()
            item = backend.get(item_id)
            elapsed = time.perf_counter() - t0
            if item is not None:
                self.latencies[tier].append(elapsed)
                return item, backend
        return None, None

    def _all_items(self):
        return self.active_backend.all() + self.vector_backend.all() + self.cold_backend.all()

    def get_item(self, item_id):
        """Read-only lookup across all resident tiers."""
        item, _ = self._find_item(item_id)
        return item

    def update_item(self, item_id, **fields):
        """Mutate fields on an item and persist it back to whichever
        backend currently holds it. Required for tests/callers that need to
        edit an item in place -- see module docstring."""
        item, backend = self._find_item(item_id)
        if item is None:
            return None
        for k, v in fields.items():
            setattr(item, k, v)
        backend.put(item)
        return item

    @property
    def items(self):
        """Backward-compatible read view: {id: item} across all resident
        tiers. This is assembled fresh on every access (a real query against
        Chroma/SQLite for VECTOR/COLD items) -- it is NOT a live shared
        object graph. Mutating an item obtained this way will not persist;
        use update_item() instead."""
        return {item.id: item for item in self._all_items()}

    # -- cost helpers (unchanged from the original) --------------------------

    def _effective_cost(self, item):
        if item.regeneration_cost > 0:
            return item.regeneration_cost
        return item.regeneration_tokens * self.TOKEN_COST_RATE

    def _update_norms(self, item):
        """Called once, right after an item is generated/regenerated, with
        the fresh item (regeneration_cost/tokens/latency_s all populated).
        Base implementation tracks the cost normalizer used by the
        cost-aware score; subclasses (e.g. LatencyAwareCostMemoryStore)
        override/extend this to track their own normalizers too."""
        self._cost_norm = max(self._cost_norm, self._effective_cost(item))

    # -- core API -------------------------------------------------------------

    def add(self, t, prompt, provider, model, max_tokens=None):
        """`max_tokens`: Module 4's hook (see hooks.py's select_token_budget())
        -- an optional output-token ceiling for this specific generation,
        instead of the model's own flat default. None means "no explicit
        cap", which is the behavior before Module 4 existed."""
        t0 = time.perf_counter()
        result = self._call_llm(self.clients, provider, model, prompt, max_tokens=max_tokens)
        gen_latency = time.perf_counter() - t0
        self.generation_latencies.append(gen_latency)

        tokens = result.get("input_tokens", 0) + result.get("output_tokens", 0)
        item = RealMemoryItem(
            id=str(uuid.uuid4())[:8], prompt=prompt, content=result["content"],
            provider=provider, model=model, created_at=t, last_accessed_at=t,
            regeneration_cost=result["cost"], regeneration_tokens=tokens,
            regeneration_latency_s=gen_latency, tier=Tier.ACTIVE,
        )
        self.generation_cost_paid += result["cost"]
        self.generation_tokens_paid += tokens
        self._update_norms(item)
        self.active_backend.put(item)
        self._rebalance(t)
        return item

    def access(self, item_id, t, max_tokens=None):
        """`max_tokens`: Module 4's hook, applied only if this access turns
        out to be a miss and a real regeneration is required -- a hit never
        calls the LLM at all, so it has no token budget to set."""
        item, backend = self._find_item(item_id)

        if item is None:
            deleted_ref = self.deleted_backend.get(item_id)
            prompt = deleted_ref.prompt if deleted_ref else "unknown prompt"
            provider = deleted_ref.provider if deleted_ref else "groq"
            model = deleted_ref.model if deleted_ref else "llama-3.1-8b-instant"

            self.misses += 1
            self.regeneration_events += 1
            t0 = time.perf_counter()
            result = self._call_llm(self.clients, provider, model, prompt, max_tokens=max_tokens)
            miss_latency = time.perf_counter() - t0
            self.miss_latencies.append(miss_latency)

            tokens = result.get("input_tokens", 0) + result.get("output_tokens", 0)
            self.regeneration_cost_paid += result["cost"]
            self.regeneration_tokens_paid += tokens

            new_item = RealMemoryItem(
                id=item_id, prompt=prompt, content=result["content"],
                provider=provider, model=model, created_at=t, last_accessed_at=t,
                regeneration_cost=result["cost"], regeneration_tokens=tokens,
                regeneration_latency_s=miss_latency, tier=Tier.ACTIVE,
            )
            self._update_norms(new_item)
            self.active_backend.put(new_item)
            self._rebalance(t)
            return new_item

        if item.tier == Tier.ACTIVE:
            self.active_hits += 1
        elif item.tier == Tier.VECTOR:
            self.vector_hits += 1
            self.regeneration_cost_paid += 0.05 * item.regeneration_cost
            self.regeneration_tokens_paid += 0.05 * item.regeneration_tokens
        elif item.tier == Tier.COLD:
            self.cold_hits += 1
            self.regeneration_cost_paid += 0.25 * item.regeneration_cost
            self.regeneration_tokens_paid += 0.25 * item.regeneration_tokens

        item.touch(t)
        if backend is not self.active_backend:
            # apply_batch([], [item.id]) instead of backend.delete(item.id):
            # both VectorBackend.delete() and _SqliteItemTable.delete() do a
            # .get() pre-read before the actual delete, purely so they can
            # return the removed item -- a real extra round-trip this call
            # site never uses (we already have `item` from _find_item()
            # above). apply_batch's delete-only path skips that read
            # entirely: one write instead of one read + one write. (This
            # does NOT defer the delete into _rebalance()'s own batched
            # apply_batch call below -- that would require _all_items() to
            # not double-count this item while it's simultaneously present
            # in both the old backend and the active_backend.put() just
            # below, which needs more careful handling than this pass makes;
            # this fix only removes the redundant pre-read, not the
            # rebalance-adjacent round-trip itself.)
            backend.apply_batch([], [item.id])
        item.tier = Tier.ACTIVE
        self.active_backend.put(item)
        self._rebalance(t)
        return item

    def _rebalance(self, t):
        w_recency, w_frequency, w_cost = self.w
        items = self._all_items()

        for item in items:
            age = max(0, t - item.last_accessed_at)
            recency_component = math.exp(-math.log(2) * age / self.recency_halflife)
            frequency_component = 1 - math.exp(-item.access_count / 5.0)
            cost_component = 1 - math.exp(-self._effective_cost(item) / self._cost_norm)
            item.score = (w_recency * recency_component + w_frequency * frequency_component
                          + w_cost * cost_component)

        ranked = sorted(items, key=lambda x: x.score, reverse=True)
        self._apply_tier_assignment(ranked)

    def _apply_tier_assignment(self, ranked):
        """Shared by every eviction policy in this project (cost-aware here,
        plus LRU/FIFO/LFU/Random/Belady/GDSF/latency-aware in
        baseline_stores.py): given items already sorted best-to-worst by
        WHATEVER ranking that policy uses, cut them into
        ACTIVE/VECTOR/COLD/DELETED by capacity and move each item between
        the real backends accordingly. This is the one place that actually
        talks to the backends during a rebalance -- every policy subclass
        only needs to override how `ranked` is computed, never this.

        Moves are collected per-backend and flushed via each backend's
        `apply_batch()` at the end, instead of issuing one Chroma
        upsert/delete or one SQLite commit per item as items are visited.
        Profiling showed this loop calling out to real backends item-by-item
        was the dominant cost of every add()/access() (a Chroma round-trip
        and a SQLite commit are each tens of ms, and a rebalance can move
        many items) -- batching turns that into at most one round-trip per
        backend per rebalance, with the exact same net set of tier moves.
        """
        active_cut = self.active_capacity
        vector_cut = active_cut + self.vector_capacity
        cold_cut = vector_cut + self.cold_capacity

        deletes = {Tier.ACTIVE: [], Tier.VECTOR: [], Tier.COLD: []}
        puts = {Tier.ACTIVE: [], Tier.VECTOR: [], Tier.COLD: []}
        to_deleted_backend = []

        for i, item in enumerate(ranked):
            if i < active_cut:
                new_tier = Tier.ACTIVE
            elif i < vector_cut:
                new_tier = Tier.VECTOR
            elif i < cold_cut:
                new_tier = Tier.COLD
            else:
                new_tier = Tier.DELETED

            if new_tier == Tier.DELETED:
                deletes[item.tier].append(item.id)
                item.tier = Tier.DELETED
                to_deleted_backend.append(item)
            elif item.tier != new_tier:
                deletes[item.tier].append(item.id)
                puts[new_tier].append(item)  # apply_batch() sets item.tier
            # else: item stays in the same backend. `.score` is recomputed
            # from scratch on every rebalance regardless of what's stored,
            # so there's no need to round-trip a write to Chroma/SQLite
            # just to persist a value nothing durable depends on -- that
            # write-amplification is what makes real backends meaningfully
            # more expensive than a plain dict, and it's not buying
            # correctness here.

        for tier in (Tier.ACTIVE, Tier.VECTOR, Tier.COLD):
            self._backend_for_tier(tier).apply_batch(puts[tier], deletes[tier])
        if to_deleted_backend:
            self.deleted_backend.apply_batch(to_deleted_backend, [])

    def tier_counts(self):
        return {
            Tier.ACTIVE: len(self.active_backend),
            Tier.VECTOR: len(self.vector_backend),
            Tier.COLD: len(self.cold_backend),
        }

    # -- latency stats ---------------------------------------------------------

    @staticmethod
    def _summarize_latencies(samples_s):
        """samples_s: list of seconds. Returns count + min/avg/max in ms."""
        if not samples_s:
            return dict(count=0, min_ms=0.0, avg_ms=0.0, max_ms=0.0)
        ms = sorted(s * 1000 for s in samples_s)
        return dict(
            count=len(ms),
            min_ms=ms[0],
            avg_ms=sum(ms) / len(ms),
            max_ms=ms[-1],
        )

    def latency_stats(self):
        """Real, measured wall-clock time for each kind of event -- the
        thing you actually pay when a query lands in each tier:

            active_hit / vector_hit / cold_hit  -> real backend .get() time
            miss_regen                          -> real call_llm() time on a
                                                    full miss (item wasn't
                                                    resident anywhere)
            generation                          -> real call_llm() time for
                                                    a brand-new item via add()

        Each value is {count, min_ms, avg_ms, max_ms}. Empty categories
        return zeros rather than raising, so a partial run can still print
        a full table.
        """
        return {
            "active_hit": self._summarize_latencies(self.latencies[Tier.ACTIVE]),
            "vector_hit": self._summarize_latencies(self.latencies[Tier.VECTOR]),
            "cold_hit": self._summarize_latencies(self.latencies[Tier.COLD]),
            "miss_regen": self._summarize_latencies(self.miss_latencies),
            "generation": self._summarize_latencies(self.generation_latencies),
        }

    # -- deleted_log compatibility shim --------------------------------------

    @property
    def deleted_log(self):
        """Backward-compatible view of the durable DeletedBackend, in the
        same append-order a plain list would have had."""
        return self.deleted_backend.all_ordered()

    # -- resource cleanup -----------------------------------------------------

    def close(self):
        """Closes this store's SQLite connections (ColdBackend/
        DeletedBackend). Nothing calls this automatically -- a script that
        creates many short-lived stores in a loop (a comparison, a sweep)
        should call it once done with each store, or connections accumulate
        for the rest of the process (each is a real open file descriptor;
        a 20-config x 3-seed sweep creates 100+ of them). ActiveBackend/
        VectorBackend need no cleanup (a plain dict, and an ephemeral
        chromadb.Client() that isn't file-handle-backed the same way)."""
        for backend in (self.cold_backend, self.deleted_backend):
            close = getattr(backend, "close", None)
            if close is not None:
                close()
