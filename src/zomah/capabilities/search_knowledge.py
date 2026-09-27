from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from zomah.knowledge import KnowledgeIndex


SearchQuery = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]

DEFAULT_MAX_RESULTS = 10
MAX_RESULTS = 50


class KnowledgeResult(BaseModel):
    """One ranked document, located at its strongest matching lines.

    ``excerpt`` is taken from lines ``start_line``..``end_line`` of the file,
    numbered like ``read_file``. To inspect the evidence, call ``read_file``
    with this ``path`` and ``start_line``; reading from line 1 is unnecessary.
    """

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    path: str
    title: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    excerpt: str
    relevance: float
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def check_line_range(self) -> "KnowledgeResult":
        if self.end_line < self.start_line:
            raise ValueError("end_line must not precede start_line")
        return self


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
            path=hit.path,
            title=hit.title,
            start_line=hit.start_line,
            end_line=hit.end_line,
            excerpt=hit.excerpt,
            relevance=hit.relevance,
            size_bytes=hit.size_bytes,
        )
        for rank, hit in enumerate(rows, start=1)
    ]

    return SearchKnowledgeResponse(
        query=request.query,
        results=results,
        returned=len(results),
    )
