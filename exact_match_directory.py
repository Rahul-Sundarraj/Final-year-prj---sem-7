"""
exact_match_directory.py

Adapter around shared_memory.py's basic exact-hash SharedMemoryDirectory so
it exposes the SAME lookup()/record()/stats() shape as
semantic_directory.SemanticSharedMemoryDirectory. This is what makes
qa_pipeline.py able to run the identical orchestration loop against either
Module 2 implementation and produce directly comparable stats -- this is
the "base module 1 and 2" (basic, resume-pipeline Module 2) side of the
baseline-vs-semantic comparison.

shared_memory.py itself is untouched; this file only wraps it.
"""

from shared_memory import SharedMemoryDirectory, make_signature


class ExactMatchDirectory:
    """Same public shape as SemanticSharedMemoryDirectory, backed by the
    basic exact-hash SharedMemoryDirectory. semantic_only_hits is always 0
    here -- every hit this class can produce is by definition exact."""

    def __init__(self):
        self._inner = SharedMemoryDirectory()

    def lookup(self, stage, query):
        signature = make_signature(stage, query)
        item_id = self._inner.lookup(signature)
        if item_id is None:
            return dict(hit=False, item_id=None, matched_query=None,
                       similarity=0.0, keyword_similarity=0.0, exact=False)
        return dict(hit=True, item_id=item_id, matched_query=query,
                   similarity=1.0, keyword_similarity=1.0, exact=True)

    def record(self, stage, query, item_id, source_label):
        signature = make_signature(stage, query)
        self._inner.record(signature, item_id, stage, source_label)

    def stats(self):
        s = self._inner.stats()
        return dict(
            lookups=s["lookups"], hits=s["hits"], hit_rate=s["hit_rate"],
            exact_hits=s["hits"], semantic_only_hits=0, guard_rejections=0,
        )
