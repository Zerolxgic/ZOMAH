from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from zomah.knowledge import KnowledgeIndex


SearchQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]

DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 50


class KnowledgeResult(BaseModel):
    """One ranked document-level retrieval result."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    path: str
    title: str
    excerpt: str
    relevance: float
    size_bytes: int = Field(ge=0)


class SearchKnowledgeRequest(BaseModel):
    """Validated model-facing input for lexical knowledge retrieval."""

    model_config = ConfigDict(extra="forbid")

    query: SearchQuery
    max_results: int = Field(default=DEFAULT_MAX_RESULTS, ge=1, le=MAX_RESULTS)


class SearchKnowledgeResponse(BaseModel):
    """Ranked references into the approved knowledge index."""

    model_config = ConfigDict(extra="forbid")

    query: str
    results: list[KnowledgeResult]
    returned: int = Field(ge=0)


def search_knowledge(
    request: SearchKnowledgeRequest,
    index: KnowledgeIndex,
) -> SearchKnowledgeResponse:
    """Search approved documents without exposing index maintenance.

    The derived FTS index is refreshed immediately before every search. The
    refresh is incremental: unchanged files are fingerprinted and skipped,
    while new, changed, moved, or deleted documents are reconciled before the
    query runs. Elyria never needs a separate indexing tool.

    Returned results remain references/excerpts only; `read_file` is still the
    capability for bounded source content.
    """

    index.refresh()
    rows = index.search(request.query, limit=request.max_results)
    results = [
        KnowledgeResult(
            rank=rank,
            path=row["path"],
            title=row["title"],
            excerpt=row["excerpt"],
            relevance=row["relevance"],
            size_bytes=row["size_bytes"],
        )
        for rank, row in enumerate(rows, start=1)
    ]

    return SearchKnowledgeResponse(
        query=request.query,
        results=results,
        returned=len(results),
    )
