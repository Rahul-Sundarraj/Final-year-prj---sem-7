"""
test_shared_memory.py

Unit tests for shared_memory.py's SharedMemoryDirectory / make_signature --
pure bookkeeping, no LLM/store/network involved, so these run instantly
and offline like the rest of this suite.

pipeline.py (agents.py + shared_memory.py + memory_store.py wired together,
real Ollama) is exercised as a live integration demo instead of here --
tying it to pytest would make the fast offline suite depend on a running
Ollama server, which the rest of this project deliberately avoids (see
conftest.py's FakeEmbeddingFunction for the same reasoning applied to
VectorBackend).
"""

from shared_memory import SharedMemoryDirectory, make_signature


def test_signature_deterministic_and_stage_scoped():
    # Same stage + same input -> same signature, every time.
    assert make_signature("extractor", "resume text") == make_signature("extractor", "resume text")
    # Same input, different stage -> different signature (an extractor
    # result must never be handed back to a reviewer request just because
    # the underlying text happens to match).
    assert make_signature("extractor", "resume text") != make_signature("reviewer", "resume text")


def test_signature_normalizes_whitespace_and_case():
    # Real resume snippets come from resumes with inconsistent whitespace;
    # a signature keyed on raw text would silently miss real duplicates.
    a = make_signature("extractor", "  Resume   TEXT  ")
    b = make_signature("extractor", "resume text")
    assert a == b


def test_signature_multi_part_is_order_sensitive_and_distinct():
    # Distinct combinations of parts must not collide.
    assert make_signature("analyzer", "resume A", "extraction 1") != \
           make_signature("analyzer", "resume B", "extraction 1")


def test_lookup_miss_then_hit_after_record():
    d = SharedMemoryDirectory()
    sig = make_signature("extractor", "resume text")

    assert d.lookup(sig) is None  # nobody has computed this yet
    d.record(sig, item_id="item-123", stage="extractor", source_label="resume-0")
    assert d.lookup(sig) == "item-123"  # now a later agent gets a hit


def test_stats_count_lookups_and_hits_separately():
    d = SharedMemoryDirectory()
    sig = make_signature("extractor", "resume text")

    d.lookup(sig)              # miss
    d.record(sig, "item-1", "extractor", "resume-0")
    d.lookup(sig)               # hit
    d.lookup(sig)               # hit

    stats = d.stats()
    assert stats["lookups"] == 3
    assert stats["hits"] == 2
    assert stats["entries"] == 1
    assert stats["hit_rate"] == 2 / 3


def test_record_overwrites_and_origin_reflects_latest_writer():
    """If two different resumes' pipelines somehow race to record the same
    signature (shouldn't happen in pipeline.py's sequential run, but the
    directory itself has no locking), the second record() wins -- document
    that behavior explicitly rather than leaving it implicit."""
    d = SharedMemoryDirectory()
    sig = make_signature("extractor", "resume text")

    d.record(sig, "item-1", "extractor", "resume-0")
    d.record(sig, "item-2", "extractor", "resume-1")

    assert d.lookup(sig) == "item-2"
    assert d.origin_of(sig) == ("extractor", "resume-1")


def test_independent_directories_do_not_share_state():
    d1 = SharedMemoryDirectory()
    d2 = SharedMemoryDirectory()
    sig = make_signature("extractor", "resume text")

    d1.record(sig, "item-1", "extractor", "resume-0")
    assert d2.lookup(sig) is None


# ---------------------------------------------------------------------------
# Storage isolation regression (see VectorBackend.__init__)
# ---------------------------------------------------------------------------

def test_ephemeral_vector_backends_are_isolated_by_default():
    """chromadb.Client() is a process-wide shared instance that ignores
    `path`, so two ephemeral VectorBackends sharing a collection name would
    silently see each other's items -- which once corrupted a capacity sweep
    in this project (two 'independent' pipeline stores in one process). They
    must be isolated unless a caller explicitly asks to share."""
    from storage_backends import VectorBackend
    from models import RealMemoryItem
    from conftest import FakeEmbeddingFunction

    def make():
        return VectorBackend(path="/tmp/unused-ephemeral",
                             embedding_function=FakeEmbeddingFunction(),
                             persistent=False)

    a, b = make(), make()
    a.put(RealMemoryItem(id="only-in-a", prompt="p", content="c",
                         provider="g", model="m", created_at=0, last_accessed_at=0))

    assert len(a) == 1
    assert len(b) == 0, "a second ephemeral VectorBackend must not see the first's items"
    assert b.get("only-in-a") is None


def _dummy_item(item_id):
    from models import RealMemoryItem
    return RealMemoryItem(id=item_id, prompt="p", content="c", provider="g", model="m",
                          created_at=0, last_accessed_at=0)


def test_explicit_collection_name_still_shares_when_asked():
    """The isolation default must not break deliberate sharing -- passing the
    same explicit collection_name is how a caller opts back in."""
    from storage_backends import VectorBackend
    from models import RealMemoryItem
    from conftest import FakeEmbeddingFunction

    def make():
        return VectorBackend(path="/tmp/unused-ephemeral",
                             collection_name="deliberately_shared",
                             embedding_function=FakeEmbeddingFunction(),
                             persistent=False)

    a, b = make(), make()
    a.put(RealMemoryItem(id="shared", prompt="p", content="c",
                         provider="g", model="m", created_at=0, last_accessed_at=0))
    assert b.get("shared") is not None


def test_deleted_backend_is_genuinely_append_only():
    """Regression: DeletedBackend.put() used to call the shared UPSERT
    (ON CONFLICT(id) DO UPDATE) path -- its own comment claimed
    "append-only... keep every record rather than overwriting" while doing
    exactly the opposite for a repeat deletion of the same id. An item
    evicted, regenerated under the same id (memory_store.access()'s miss
    path reuses item_id), then evicted again must produce a SECOND history
    row, not silently overwrite the first deletion's record."""
    import tempfile
    from storage_backends import DeletedBackend
    from models import RealMemoryItem

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    backend = DeletedBackend(db_path=db_path)

    first = RealMemoryItem(id="x", prompt="first-deletion", content="c", provider="g",
                           model="m", created_at=0, last_accessed_at=0, access_count=3)
    backend.put(first)
    assert len(backend) == 1

    second = RealMemoryItem(id="x", prompt="second-deletion", content="c", provider="g",
                            model="m", created_at=50, last_accessed_at=55, access_count=1)
    backend.put(second)
    assert len(backend) == 2, "second deletion of the same id must be a NEW row, not an overwrite"

    # get() -- used by memory_store.access()'s miss-regeneration lookup --
    # must return the MOST RECENT record, not silently fail or return stale data.
    assert backend.get("x").prompt == "second-deletion"

    # all_ordered() -- backing store.deleted_log -- must preserve BOTH
    # records in insertion order, matching its "most-recent-deletion-last"
    # documented contract.
    ordered = backend.all_ordered()
    assert [item.prompt for item in ordered] == ["first-deletion", "second-deletion"]


def test_deleted_backend_apply_batch_is_also_append_only():
    """Same regression, via the production write path (_apply_tier_assignment
    calls apply_batch(), never put() directly) -- the batched path shares
    the same upsert-vs-insert bug surface and needed the same fix."""
    import tempfile
    from storage_backends import DeletedBackend
    from models import RealMemoryItem

    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    backend = DeletedBackend(db_path=db_path)

    first = RealMemoryItem(id="x", prompt="first", content="c", provider="g", model="m",
                           created_at=0, last_accessed_at=0)
    backend.apply_batch([first], [])
    second = RealMemoryItem(id="x", prompt="second", content="c", provider="g", model="m",
                            created_at=10, last_accessed_at=10)
    backend.apply_batch([second], [])

    assert len(backend) == 2
    assert [item.prompt for item in backend.all_ordered()] == ["first", "second"]
