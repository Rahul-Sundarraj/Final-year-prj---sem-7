"""
storage_backends.py

Real storage backends for the three (+1) tiers of RealCostAwareMemoryStore.
Each backend implements the same tiny interface:

    put(item)           -> write/overwrite item, keyed by item.id
    get(item_id)         -> RealMemoryItem | None
    delete(item_id)       -> RealMemoryItem | None (the removed item, if present)
    all()                 -> list[RealMemoryItem]  (every item currently resident)
    __len__()             -> resident count

Nothing in here talks to the notebook's cost-aware scoring logic -- that
stays in memory_store.py. This module only knows how to durably store and
retrieve RealMemoryItem objects.

Backends used:
    ActiveBackend  -- plain in-process dict. ACTIVE is, by definition, the
                      "hot" tier the app reads on every turn; a network or
                      disk round-trip here would defeat the point.
    VectorBackend  -- a real embedded ChromaDB collection. Items are
                      embedded (by default with a local sentence-transformers
                      model) so VECTOR tier is an actual semantic store, not
                      just "the middle tier" -- you can query_similar() it.
    ColdBackend    -- a real SQLite table on disk. Cheap, durable, no
                      external service required.
    DeletedBackend -- a second SQLite table (or file) recording items that
                      fell out of all three tiers, so a miss/regeneration
                      lookup survives a process restart (unlike the old
                      pure-in-memory deleted_log).
"""

import json
import sqlite3
import threading
import uuid

from models import RealMemoryItem, Tier


# ---------------------------------------------------------------------------
# ACTIVE -- in-process dict
# ---------------------------------------------------------------------------

class ActiveBackend:
    """The hot tier. Deliberately just a dict -- this IS the active context
    window / working set, so it must be zero-latency."""

    def __init__(self):
        self._store = {}

    def put(self, item):
        item.tier = Tier.ACTIVE
        self._store[item.id] = item
        return item

    def get(self, item_id):
        return self._store.get(item_id)

    def delete(self, item_id):
        return self._store.pop(item_id, None)

    def all(self):
        return list(self._store.values())

    def __len__(self):
        return len(self._store)

    def apply_batch(self, put_items, delete_ids):
        """Batch entry point used by `_apply_tier_assignment` during a
        rebalance. Plain dict ops are already O(1)/zero-latency, so this is
        just the put/delete loop -- it exists purely so every backend shares
        one interface; the real payoff is in VectorBackend/_SqliteItemTable
        below, where batching turns N network/disk round-trips into 1."""
        for item_id in delete_ids:
            self._store.pop(item_id, None)
        for item in put_items:
            item.tier = Tier.ACTIVE
            self._store[item.id] = item


# ---------------------------------------------------------------------------
# VECTOR -- ChromaDB
# ---------------------------------------------------------------------------

# Chroma metadata values must be str/int/float/bool (no None, no enums).
_META_FIELDS = (
    "prompt", "content", "provider", "model", "created_at",
    "last_accessed_at", "access_count", "regeneration_cost",
    "regeneration_tokens", "regeneration_latency_s", "score",
)


def _item_to_metadata(item):
    return {
        "prompt": item.prompt,
        "content": item.content,
        "provider": item.provider,
        "model": item.model,
        "created_at": int(item.created_at),
        "last_accessed_at": int(item.last_accessed_at),
        "access_count": int(item.access_count),
        "regeneration_cost": _safe_float(item.regeneration_cost),
        "regeneration_tokens": int(item.regeneration_tokens),
        "regeneration_latency_s": _safe_float(item.regeneration_latency_s),
        "score": _safe_float(item.score),
    }


def _safe_float(x, sentinel=1e18):
    """ChromaDB metadata is serialized as JSON, which has no representation
    for +/-inf or NaN -- writing one of those silently drops the whole
    field on upsert (no error), which then blows up as a KeyError on the
    next read. Clamp to a large-but-finite sentinel instead so ranking
    behavior is preserved (a policy that wants "effectively infinite" can
    still get it) without corrupting the stored record."""
    if x != x:  # NaN
        return 0.0
    if x == float("inf"):
        return sentinel
    if x == float("-inf"):
        return -sentinel
    return float(x)


def _metadata_to_item(item_id, metadata):
    return RealMemoryItem(
        id=item_id,
        prompt=metadata["prompt"],
        content=metadata["content"],
        provider=metadata["provider"],
        model=metadata["model"],
        created_at=metadata["created_at"],
        last_accessed_at=metadata["last_accessed_at"],
        access_count=metadata["access_count"],
        regeneration_cost=metadata["regeneration_cost"],
        regeneration_tokens=metadata["regeneration_tokens"],
        # .get() with a default: tolerates reading an item written before
        # regeneration_latency_s existed (e.g. a Chroma dir from an older
        # run of this project), rather than raising KeyError.
        regeneration_latency_s=metadata.get("regeneration_latency_s", 0.0),
        score=metadata["score"],
        tier=Tier.VECTOR,
    )


def default_embedding_function(model_name="all-MiniLM-L6-v2"):
    """Real local embeddings via sentence-transformers, wrapped the way
    Chroma expects. Lazily imported so importing this module doesn't force
    a torch/sentence-transformers import (and the ~80MB model download) for
    callers who only need ActiveBackend/ColdBackend, or who are injecting
    their own embedding_function for tests."""
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


class VectorBackend:
    """VECTOR tier, backed by a real (embedded) ChromaDB collection. Items
    are embedded on their prompt text, so this tier can also answer real
    semantic-similarity queries via query_similar() -- something a plain
    dict/LRU tier fundamentally can't do.

    `embedding_function` is injectable so tests can swap in a cheap fake
    (see tests) instead of downloading real model weights.

    `persistent` controls whether the collection is backed by an on-disk
    ChromaDB directory (`PersistentClient`, the default -- what production
    code and the real-Ollama run want, so memory actually survives a
    process restart) or an in-memory-only collection (`Client()`, ephemeral
    -- `path` is ignored). Profiling measured ChromaDB's PersistentClient at
    ~80-90ms per upsert()/delete() call regardless of item count (the fsync
    cost of durability), versus ~1-2ms for the ephemeral client on the same
    call. Every unit test and print_comparison.py build a throwaway store
    in a temp directory that's discarded the moment the process/test ends
    -- they were paying that ~80ms/call durability tax for durability
    nothing ever reads back, which is most of why the test suite and
    comparison script were slow. Use `persistent=False` for exactly that
    kind of run; leave the default (True) for anything meant to persist.
    """

    def __init__(self, path="./chroma_memory", collection_name=None,
                 embedding_function=None, persistent=True):
        import chromadb  # local import: keep chromadb optional at module load

        if collection_name is None:
            # Persistent backends are namespaced by `path`, so a fixed
            # default name is safe and keeps a store's data findable across
            # restarts. Ephemeral ones are NOT: chromadb.Client() returns a
            # process-wide shared system instance and ignores `path`
            # entirely, so two VectorBackend(persistent=False) objects built
            # with the same collection name silently share one collection --
            # each sees the other's items, and "independent" stores in one
            # process corrupt each other's results. Defaulting ephemeral
            # backends to a unique name makes isolation the default instead
            # of something every caller has to remember. (Callers that pass
            # an explicit collection_name still get exactly what they ask
            # for, including deliberate sharing.)
            collection_name = ("vector_tier" if persistent
                               else f"vector_tier_{uuid.uuid4().hex[:8]}")

        self.client = chromadb.PersistentClient(path=path) if persistent else chromadb.Client()
        self.embedding_function = embedding_function or default_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function,
        )

    def put(self, item):
        item.tier = Tier.VECTOR
        self.collection.upsert(
            ids=[item.id],
            documents=[item.prompt],
            metadatas=[_item_to_metadata(item)],
        )
        return item

    def get(self, item_id):
        res = self.collection.get(ids=[item_id], include=["metadatas"])
        if not res["ids"]:
            return None
        return _metadata_to_item(res["ids"][0], res["metadatas"][0])

    def delete(self, item_id):
        item = self.get(item_id)
        if item is not None:
            self.collection.delete(ids=[item_id])
        return item

    def all(self):
        res = self.collection.get(include=["metadatas"])
        return [_metadata_to_item(i, m) for i, m in zip(res["ids"], res["metadatas"])]

    def query_similar(self, text, top_k=5):
        """Bonus real capability the old dict-only design never had:
        semantic search over everything currently resident in VECTOR tier."""
        if len(self) == 0:
            return []
        res = self.collection.query(query_texts=[text], n_results=min(top_k, len(self)))
        out = []
        for item_id, metadata, distance in zip(
            res["ids"][0], res["metadatas"][0], res["distances"][0]
        ):
            item = _metadata_to_item(item_id, metadata)
            out.append((item, distance))
        return out

    def __len__(self):
        return self.collection.count()

    def apply_batch(self, put_items, delete_ids):
        """Batch entry point used by `_apply_tier_assignment` during a
        rebalance: one `collection.delete(ids=[...])` call and one
        `collection.upsert(...)` call covering every item moving in/out of
        VECTOR this rebalance, instead of one Rust-binding round-trip per
        item. This is the single biggest cost in a rebalance (each
        individual upsert/delete was ~80ms in profiling)."""
        if delete_ids:
            self.collection.delete(ids=list(delete_ids))
        if put_items:
            for item in put_items:
                item.tier = Tier.VECTOR
            self.collection.upsert(
                ids=[it.id for it in put_items],
                documents=[it.prompt for it in put_items],
                metadatas=[_item_to_metadata(it) for it in put_items],
            )


# ---------------------------------------------------------------------------
# COLD + DELETED -- SQLite
# ---------------------------------------------------------------------------

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    id TEXT PRIMARY KEY,
    prompt TEXT NOT NULL,
    content TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL,
    access_count INTEGER NOT NULL,
    regeneration_cost REAL NOT NULL,
    regeneration_tokens INTEGER NOT NULL,
    regeneration_latency_s REAL NOT NULL DEFAULT 0.0,
    score REAL NOT NULL
)
"""

_COLUMNS = (
    "id, prompt, content, provider, model, created_at, last_accessed_at, "
    "access_count, regeneration_cost, regeneration_tokens, "
    "regeneration_latency_s, score"
)

# Append-only variant: `id` is NOT unique here (deliberately), so a second
# deletion of the same logical item (re-added under the same id, then
# evicted again) gets its own row instead of overwriting the first
# deletion's. `history_id` is the real primary key; `id` is just an
# indexed lookup column. Used by DeletedBackend -- see the append_only
# param on _SqliteItemTable below for why ColdBackend can't share this
# (COLD legitimately wants "one current row per id", upsert semantics).
_APPEND_ONLY_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    content TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_accessed_at INTEGER NOT NULL,
    access_count INTEGER NOT NULL,
    regeneration_cost REAL NOT NULL,
    regeneration_tokens INTEGER NOT NULL,
    regeneration_latency_s REAL NOT NULL DEFAULT 0.0,
    score REAL NOT NULL
)
"""


class _SqliteItemTable:
    """Shared implementation behind ColdBackend and DeletedBackend -- same
    row shape, two different id-uniqueness policies.

    `append_only=False` (ColdBackend): `id` is the primary key, `put()`/
    `apply_batch()` upsert -- exactly one current row per id, which is the
    right semantics for "where is this item's cold copy right now."

    `append_only=True` (DeletedBackend): `id` is NOT unique -- every put()
    is a genuine INSERT, so re-deleting the same logical item (evicted,
    regenerated under the same id, evicted again) gets its own history
    row instead of silently overwriting the first deletion. `get()`
    returns the most recent row for an id (what a miss-regeneration
    lookup wants); `all()`/`all_ordered()` return every row (the full
    history `deleted_log` is documented to be).
    """

    def __init__(self, db_path, table, tier, append_only=False):
        self._table = table
        self._tier = tier
        self._append_only = append_only
        # check_same_thread=False: fine here since each store owns its own
        # connection and this project is single-threaded; swap for a real
        # connection pool if you multithread this.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            # WAL + synchronous=NORMAL: the default rollback-journal mode
            # fsyncs the *whole* file on every commit() (~20-30ms each,
            # measured in profiling -- the single biggest cost in a
            # rebalance after batching). WAL only fsyncs the small
            # append-only log and checkpoints it back to the main file in
            # the background, and NORMAL only forces a sync at WAL
            # checkpoints rather than on every commit. This trades "durable
            # through the exact instant of an OS crash" for "durable through
            # a process crash, and through an OS crash except for a tiny
            # window" -- the right tradeoff for an eviction cache, not a
            # ledger. Set once per connection; SQLite persists WAL mode in
            # the db file itself.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            if append_only:
                self._conn.execute(_APPEND_ONLY_TABLE_SCHEMA.format(table=table))
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_id_idx ON {table}(id)"
                )
            else:
                self._conn.execute(_TABLE_SCHEMA.format(table=table))
                # Backfills the column for a database created before
                # regeneration_latency_s existed (older run of this
                # project) -- ALTER TABLE ADD COLUMN is a no-op error if
                # it's already there, which we just swallow. Only needed
                # for the id-keyed schema; an append-only table is always
                # created fresh with the current schema.
                try:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN regeneration_latency_s REAL NOT NULL DEFAULT 0.0"
                    )
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()

    def put(self, item):
        item.tier = self._tier
        with self._lock:
            if self._append_only:
                self._conn.execute(
                    f"""INSERT INTO {self._table}
                        (id, prompt, content, provider, model, created_at,
                         last_accessed_at, access_count, regeneration_cost,
                         regeneration_tokens, regeneration_latency_s, score)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (item.id, item.prompt, item.content, item.provider, item.model,
                     item.created_at, item.last_accessed_at, item.access_count,
                     item.regeneration_cost, item.regeneration_tokens,
                     item.regeneration_latency_s, item.score),
                )
            else:
                self._conn.execute(
                    f"""INSERT INTO {self._table}
                        (id, prompt, content, provider, model, created_at,
                         last_accessed_at, access_count, regeneration_cost,
                         regeneration_tokens, regeneration_latency_s, score)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(id) DO UPDATE SET
                            prompt=excluded.prompt, content=excluded.content,
                            provider=excluded.provider, model=excluded.model,
                            created_at=excluded.created_at,
                            last_accessed_at=excluded.last_accessed_at,
                            access_count=excluded.access_count,
                            regeneration_cost=excluded.regeneration_cost,
                            regeneration_tokens=excluded.regeneration_tokens,
                            regeneration_latency_s=excluded.regeneration_latency_s,
                            score=excluded.score
                    """,
                    (item.id, item.prompt, item.content, item.provider, item.model,
                     item.created_at, item.last_accessed_at, item.access_count,
                     item.regeneration_cost, item.regeneration_tokens,
                     item.regeneration_latency_s, item.score),
                )
            self._conn.commit()
        return item

    def _row_to_item(self, row):
        (id_, prompt, content, provider, model, created_at, last_accessed_at,
         access_count, regeneration_cost, regeneration_tokens,
         regeneration_latency_s, score) = row
        return RealMemoryItem(
            id=id_, prompt=prompt, content=content, provider=provider, model=model,
            created_at=created_at, last_accessed_at=last_accessed_at,
            access_count=access_count, regeneration_cost=regeneration_cost,
            regeneration_tokens=regeneration_tokens,
            regeneration_latency_s=regeneration_latency_s, score=score, tier=self._tier,
        )

    def get(self, item_id):
        # append_only: an id can have multiple rows now (deletion
        # history) -- ORDER BY history_id DESC LIMIT 1 gets the most
        # recent, which is what a miss-regeneration lookup wants (the
        # freshest known prompt/cost/tokens for that item).
        order = "ORDER BY history_id DESC LIMIT 1" if self._append_only else ""
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_COLUMNS} FROM {self._table} WHERE id = ? {order}",
                (item_id,),
            )
            row = cur.fetchone()
        return self._row_to_item(row) if row else None

    def delete(self, item_id):
        # Not called anywhere in production for DeletedBackend today (only
        # ColdBackend's delete() is exercised by _apply_tier_assignment),
        # kept correct for API consistency / direct test use: deletes just
        # the most recent history row, matching get()'s semantics, rather
        # than every historical row for that id.
        item = self.get(item_id)
        if item is not None:
            with self._lock:
                if self._append_only:
                    self._conn.execute(
                        f"DELETE FROM {self._table} WHERE history_id IN "
                        f"(SELECT history_id FROM {self._table} WHERE id = ? "
                        f"ORDER BY history_id DESC LIMIT 1)",
                        (item_id,),
                    )
                else:
                    self._conn.execute(f"DELETE FROM {self._table} WHERE id = ?", (item_id,))
                self._conn.commit()
        return item

    def all(self):
        order = "ORDER BY history_id ASC" if self._append_only else ""
        with self._lock:
            cur = self._conn.execute(f"SELECT {_COLUMNS} FROM {self._table} {order}")
            rows = cur.fetchall()
        return [self._row_to_item(r) for r in rows]

    def __len__(self):
        with self._lock:
            cur = self._conn.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]

    def close(self):
        self._conn.close()

    def apply_batch(self, put_items, delete_ids):
        """Batch entry point used by `_apply_tier_assignment` during a
        rebalance: every delete and every upsert for this table happens
        under `executemany()` + a single `commit()`, instead of one
        `commit()` per item. `sqlite3.Connection.commit()` is a synchronous
        disk fsync (~20-30ms each in profiling) -- doing it once per
        rebalance instead of once per moved item is the dominant win here."""
        if not put_items and not delete_ids:
            return
        with self._lock:
            if delete_ids:
                if self._append_only:
                    for item_id in delete_ids:  # see delete()'s docstring
                        self._conn.execute(
                            f"DELETE FROM {self._table} WHERE history_id IN "
                            f"(SELECT history_id FROM {self._table} WHERE id = ? "
                            f"ORDER BY history_id DESC LIMIT 1)",
                            (item_id,),
                        )
                else:
                    self._conn.executemany(
                        f"DELETE FROM {self._table} WHERE id = ?",
                        [(item_id,) for item_id in delete_ids],
                    )
            if put_items:
                for item in put_items:
                    item.tier = self._tier
                rows = [(item.id, item.prompt, item.content, item.provider, item.model,
                        item.created_at, item.last_accessed_at, item.access_count,
                        item.regeneration_cost, item.regeneration_tokens,
                        item.regeneration_latency_s, item.score) for item in put_items]
                if self._append_only:
                    self._conn.executemany(
                        f"""INSERT INTO {self._table}
                            (id, prompt, content, provider, model, created_at,
                             last_accessed_at, access_count, regeneration_cost,
                             regeneration_tokens, regeneration_latency_s, score)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        rows,
                    )
                else:
                    self._conn.executemany(
                        f"""INSERT INTO {self._table}
                            (id, prompt, content, provider, model, created_at,
                             last_accessed_at, access_count, regeneration_cost,
                             regeneration_tokens, regeneration_latency_s, score)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(id) DO UPDATE SET
                                prompt=excluded.prompt, content=excluded.content,
                                provider=excluded.provider, model=excluded.model,
                                created_at=excluded.created_at,
                                last_accessed_at=excluded.last_accessed_at,
                                access_count=excluded.access_count,
                                regeneration_cost=excluded.regeneration_cost,
                                regeneration_tokens=excluded.regeneration_tokens,
                                regeneration_latency_s=excluded.regeneration_latency_s,
                                score=excluded.score
                        """,
                        rows,
                    )
            self._conn.commit()


class ColdBackend(_SqliteItemTable):
    """COLD tier -- durable, cheap, on-disk SQLite table."""

    def __init__(self, db_path="./memory_store.db", table="cold_items"):
        super().__init__(db_path, table, Tier.COLD)


class DeletedBackend(_SqliteItemTable):
    """Durable replacement for the old in-memory-only `deleted_log` list --
    an item that falls out of ACTIVE/VECTOR/COLD is recorded here so a
    later access() miss can still regenerate it from its original prompt,
    even across process restarts.

    Genuinely append_only=True (see _SqliteItemTable): `id` is NOT unique
    in this table, so an item evicted, later regenerated under the same
    id, and evicted again gets a SECOND history row instead of the second
    eviction silently overwriting the first one's record in place. This
    used to be broken -- put() called the shared upsert path (`ON
    CONFLICT(id) DO UPDATE`), which the class's own docstring here
    self-contradicted ("append-only... but still upsert on the same id")
    -- confirmed losing data: test_stress_repeated_regeneration_of_same_evicted_item
    exercises exactly the repeated-eviction case this fixes, though it
    never asserted on deleted_log's length/order before, so the loss was
    silent.
    """

    def __init__(self, db_path="./memory_store.db", table="deleted_items"):
        super().__init__(db_path, table, Tier.DELETED, append_only=True)

    def all_ordered(self):
        """Most-recent-deletion-last, matching the old list's append order
        (used by tests that read deleted_log like a list)."""
        with self._lock:
            cur = self._conn.execute(
                f"SELECT {_COLUMNS} FROM {self._table} ORDER BY history_id ASC"
            )
            rows = cur.fetchall()
        return [self._row_to_item(r) for r in rows]
