"""Lexical baseline (ZOMAH's real search_knowledge) and brute-force semantic ranking."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from zomah.capabilities import SearchKnowledgeRequest, search_knowledge
from zomah.knowledge import KnowledgeIndex

from embedding_bench.corpus import Document, relative_path

# Qwen3-Embedding's documented query format: "Instruct: {task}\nQuery:{query}"
# (no space after "Query:"). Documents are embedded with no instruction.
QUERY_INSTRUCTION = (
    "Given a question about a local software project and its documentation, "
    "retrieve the document that contains the evidence needed to answer it."
)
WARMUP_QUERY = "warm-up query that is not part of the benchmark"


def format_query(query: str, instruction: str = QUERY_INSTRUCTION) -> str:
    return f"Instruct: {instruction}\nQuery:{query}"


@dataclass(frozen=True, slots=True)
class RankedDoc:
    path: str
    score: float


@dataclass(frozen=True, slots=True)
class QueryRun:
    ranking: tuple[RankedDoc, ...]
    total_ms: float
    embed_ms: float | None = None
    score_ms: float | None = None


# --- lexical ------------------------------------------------------------------------


def run_lexical(index: KnowledgeIndex, root: Path, query: str, *, limit: int) -> QueryRun:
    """One query through the unchanged model-facing search_knowledge capability.

    The capability refreshes the index incrementally before searching, exactly
    as it does for the model, so ``total_ms`` includes that (no-op) refresh.
    Documents with no matching term are absent from the ranking.
    """

    start = time.perf_counter()
    response = search_knowledge(SearchKnowledgeRequest(query=query, max_results=limit), index)
    elapsed = (time.perf_counter() - start) * 1000
    ranking = tuple(
        RankedDoc(path=relative_path(result.path, root), score=result.relevance)
        for result in response.results
    )
    return QueryRun(ranking=ranking, total_ms=elapsed)


# --- semantic -----------------------------------------------------------------------


class Embedder(Protocol):
    """Turns one text into one vector. Formatting (instructions) is the caller's job."""

    def embed(self, text: str) -> Sequence[float]: ...

    def token_count(self, text: str) -> int | None: ...

    @property
    def max_tokens(self) -> int | None: ...


def normalize(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(v) for v in vector)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("embedding must be a non-empty vector of finite numbers")
    norm = math.sqrt(math.fsum(v * v for v in values))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(v / norm for v in values)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def rank_by_similarity(
    query_vector: Sequence[float],
    documents: Sequence[tuple[str, Sequence[float]]],
) -> tuple[RankedDoc, ...]:
    """Rank every document by cosine similarity (dot of unit vectors).

    Ties break by path ascending, mirroring the lexical ``ORDER BY bm25, path``.
    """

    query = normalize(query_vector)
    scored = [RankedDoc(path=path, score=dot(query, normalize(vec))) for path, vec in documents]
    return tuple(sorted(scored, key=lambda doc: (-doc.score, doc.path)))


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    path: str
    vector: tuple[float, ...]
    embed_s: float
    tokens: int | None


class SemanticIndex:
    """An in-memory matrix of normalized document embeddings, searched brute force."""

    def __init__(self, embedder: Embedder, documents: Sequence[EmbeddedDocument]) -> None:
        self.embedder = embedder
        self.documents = tuple(documents)
        dimensions = {len(doc.vector) for doc in self.documents}
        if len(dimensions) != 1:
            raise ValueError(f"document embeddings disagree on dimension: {sorted(dimensions)}")
        self.dimension = dimensions.pop()
        self._matrix = tuple((doc.path, doc.vector) for doc in self.documents)

    @classmethod
    def build(cls, embedder: Embedder, documents: Sequence[Document]) -> "SemanticIndex":
        """Embed each document once, as-is (no instruction), one at a time."""

        embedded: list[EmbeddedDocument] = []
        for document in documents:
            start = time.perf_counter()
            vector = normalize(embedder.embed(document.text))
            elapsed = time.perf_counter() - start
            embedded.append(
                EmbeddedDocument(
                    path=document.path,
                    vector=vector,
                    embed_s=elapsed,
                    tokens=embedder.token_count(document.text),
                )
            )
        return cls(embedder, embedded)

    def search(self, query: str, instruction: str = QUERY_INSTRUCTION) -> QueryRun:
        start = time.perf_counter()
        query_vector = self.embedder.embed(format_query(query, instruction))
        embedded = time.perf_counter()
        if len(query_vector) != self.dimension:
            raise ValueError("query embedding dimension differs from the document embeddings")
        ranking = rank_by_similarity(query_vector, self._matrix)
        done = time.perf_counter()
        return QueryRun(
            ranking=ranking,
            total_ms=(done - start) * 1000,
            embed_ms=(embedded - start) * 1000,
            score_ms=(done - embedded) * 1000,
        )


class HashingEmbedder:
    """Deterministic bag-of-words stand-in for pipeline checks. Not a semantic model."""

    name = "fake-hashing"

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[^\W_]+", text.casefold()):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimension] += 1.0
        return vector

    def token_count(self, text: str) -> int | None:
        return None

    @property
    def max_tokens(self) -> int | None:
        return None

    def describe(self) -> dict[str, object]:
        return {
            "embedder": self.name,
            "note": "hashing bag-of-words pipeline check; results are NOT semantic",
            "dimension": self.dimension,
        }
