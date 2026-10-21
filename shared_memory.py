"""
shared_memory.py

Basic Module 2 stub -- Shared / Transactive Memory ("who already knows
this?"), per the Zeroth review report:

    Module 2 -- Shared / Transactive Memory: an AI matcher checks a shared
    directory ("who already knows this?") before an agent re-derives
    information another agent has already computed, cutting duplicate
    LLM calls.

This directory sits ON TOP of Module 1's RealCostAwareMemoryStore
(memory_store.py), not instead of it. It answers a different question than
Module 1 does:

    Module 2 (this file):  "has ANY agent, on ANY resume/turn, already
                             computed the answer to this exact
                             (stage, input) before?"
    Module 1 (memory_store.py): "is that answer still cheap to fetch right
                             now, or has it aged out to VECTOR/COLD/
                             regeneration?"

Concretely: a hit here returns a memory_store item id, not the content
itself -- the actual fetch still goes through store.access(item_id, t), so
it's subject to Module 1's normal tier/hit-latency accounting (an item the
directory "knows about" may still cost a real regeneration call if Module 1
already evicted it all the way past COLD). This is exactly the two-step
"Per-Turn Decision Flow" the report describes: Module 2 directory check,
then Module 1 own-memory check.

Known limitation ("basic", as the report's Zeroth-review stage allows):
this is an EXACT-match directory, keyed on a normalized (stage, input)
signature -- not the "AI matcher" / semantic-similarity version the report
describes for the full Module 2. Two prompts that are semantically
identical but textually different (e.g. two resumes phrasing the same
skill differently) will NOT hit here. VectorBackend.query_similar()
(storage_backends.py) already gives Module 1's VECTOR tier real semantic
search -- swapping this directory's exact-match lookup() for a
query_similar() call above some similarity threshold is the natural
upgrade path from "basic" to the report's full Module 2, without touching
this file's public interface.
"""

import hashlib


def make_signature(stage, *parts):
    """Deterministic key for "this exact stage was asked to do this exact
    thing before". Normalizes whitespace/case so trivial formatting
    differences (extra spaces, capitalization) don't defeat an otherwise
    real duplicate-work match."""
    normalized = "|".join(" ".join(str(p).split()).lower() for p in parts)
    digest = hashlib.sha256(f"{stage}::{normalized}".encode()).hexdigest()[:16]
    return f"{stage}:{digest}"


class SharedMemoryDirectory:
    """"Who already knows this?" -- maps a (stage, input) signature to the
    memory_store item id that holds the answer, plus which agent/source
    first produced it (kept only for reporting -- lookup()/record() don't
    need it).
    """

    def __init__(self):
        self._index = {}   # signature -> item_id
        self._origin = {}  # signature -> (stage, source_label), for reporting
        self.lookups = 0
        self.hits = 0

    def lookup(self, signature):
        """Returns the memory_store item id that already holds the answer
        to this signature, or None if no agent has computed it yet."""
        self.lookups += 1
        item_id = self._index.get(signature)
        if item_id is not None:
            self.hits += 1
        return item_id

    def record(self, signature, item_id, stage, source_label):
        """Called after a real (non-shared) computation, so the NEXT agent
        that asks this exact question gets a hit instead of re-deriving it."""
        self._index[signature] = item_id
        self._origin[signature] = (stage, source_label)

    def origin_of(self, signature):
        return self._origin.get(signature)

    def stats(self):
        hit_rate = self.hits / self.lookups if self.lookups else 0.0
        return dict(lookups=self.lookups, hits=self.hits, hit_rate=hit_rate,
                    entries=len(self._index))
