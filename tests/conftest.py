"""
conftest.py

Shared pytest fixtures/helpers for building stores backed by REAL storage
(ActiveBackend dict, ChromaDB VectorBackend, SQLite ColdBackend/DeletedBackend)
without needing network access or a slow real embedding model in every test.

FakeEmbeddingFunction is a deterministic, offline, hashlib-based stand-in
that satisfies ChromaDB's EmbeddingFunction protocol -- it's injected into
VectorBackend for tests. Production code (see memory_store.py's default
VectorBackend()) uses a real local sentence-transformers model instead;
swap it back in for an end-to-end/integration run when you have internet
access to download the model weights once.
"""

import hashlib
import pathlib
import uuid

import pytest

from storage_backends import ActiveBackend, VectorBackend, ColdBackend, DeletedBackend


class FakeEmbeddingFunction:
    """Deterministic, offline, dependency-free embedding stand-in -- same
    prompt always maps to the same vector, different prompts (almost
    always) map to different vectors, which is all the store's own logic
    needs (it doesn't do semantic search internally). Keeps the test suite
    fast and fully offline."""

    def __init__(self, dim=16):
        self.dim = dim

    def __call__(self, input):
        vectors = []
        for text in input:
            h = hashlib.md5(text.encode()).digest()
            vectors.append([(h[i % len(h)] / 255.0) * 2 - 1 for i in range(self.dim)])
        return vectors

    def name(self):
        return "fake"

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def build_from_config(config):
        return FakeEmbeddingFunction()

    def get_config(self):
        return {}

    def is_legacy(self):
        return False


def make_backends(tmp_path):
    """Fresh, isolated ACTIVE/VECTOR/COLD/DELETED backends rooted under a
    pytest tmp_path -- every test gets its own ChromaDB directory and
    SQLite file, so tests can't see each other's data."""
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    unique = uuid.uuid4().hex[:8]
    active = ActiveBackend()
    vector = VectorBackend(
        path=str(tmp_path / "chroma"),
        collection_name=f"vector_{unique}",
        embedding_function=FakeEmbeddingFunction(),
        # Ephemeral (in-memory) Chroma client: this collection is thrown
        # away with tmp_path the moment the test ends, so paying ChromaDB's
        # PersistentClient fsync-per-call cost (~80ms/call, see
        # VectorBackend's docstring) for durability nothing reads back was
        # most of why this test suite was slow.
        persistent=False,
    )
    db_path = str(tmp_path / "store.db")
    cold = ColdBackend(db_path=db_path)
    deleted = DeletedBackend(db_path=db_path)
    return dict(active_backend=active, vector_backend=vector,
                cold_backend=cold, deleted_backend=deleted)


@pytest.fixture
def backends(tmp_path):
    return make_backends(tmp_path)
