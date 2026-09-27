"""T3e: fixed-size passage windows, bounded excerpts, and semantic passage ranking.

Standard library only. Everything here sees a query and a document's text,
never a benchmark label: the evaluation module reads labels afterwards.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

WINDOW_LINES = 3
STRIDE = 1
EXCERPT_BUDGETS = (80, 120, 160)
_MEANINGFUL = re.compile(r"[^\W_]")  # a window needs at least one letter or digit


@dataclass(frozen=True, slots=True)
class Passage:
    """Lines start_line..end_line (1-based, split on newline like read_file) of one document.

    ``text`` is exactly ``document[char_start:char_end]``.
    """

    start_line: int
    end_line: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class Excerpt:
    text: str  # what a judge would see, with "…" when cut
    char_start: int  # source span of the retained characters
    char_end: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class ScoredPassage:
    passage: Passage
    score: float


def line_starts(text: str) -> list[int]:
    starts, offset = [], 0
    for line in text.split("\n"):
        starts.append(offset)
        offset += len(line) + 1
    return starts


def lines_passage(text: str, start_line: int, end_line: int) -> Passage:
    """Lines start_line..end_line of ``text`` as a passage with its exact source span."""

    lines = text.split("\n")
    starts = line_starts(text)
    char_start = starts[start_line - 1]
    char_end = starts[end_line - 1] + len(lines[end_line - 1])
    return Passage(start_line, end_line, text[char_start:char_end], char_start, char_end)


def build_windows(text: str, window: int = WINDOW_LINES, stride: int = STRIDE) -> tuple[Passage, ...]:
    """Every contiguous ``window``-line slice, stepping by ``stride``; blank windows are skipped."""

    if window < 1 or stride < 1:
        raise ValueError("window and stride must be positive")
    count = len(text.split("\n"))
    first_lines = range(1, max(1, count - window + 1) + 1, stride)
    windows = []
    for start in first_lines:
        passage = lines_passage(text, start, min(count, start + window - 1))
        if _MEANINGFUL.search(passage.text):
            windows.append(passage)
    return tuple(windows)


def bounded_excerpt(passage: Passage, budget: int) -> Excerpt:
    """The passage's leading characters, at most ``budget`` of them (ellipsis not counted).

    Leading whitespace is skipped; a cut inside a word backs up to the previous
    whitespace when that keeps at least half the budget. Derived from the
    passage alone.
    """

    if budget < 1:
        raise ValueError("budget must be positive")
    body = passage.text
    lead = len(body) - len(body.lstrip())
    body = body[lead:]
    if len(body) <= budget:
        kept = body.rstrip()
        truncated = False
    else:
        cut = budget
        if not body[cut].isspace() and not body[cut - 1].isspace():
            boundary = max(body.rfind(" ", 0, cut), body.rfind("\n", 0, cut))
            if boundary >= budget // 2:
                cut = boundary
        kept = body[:cut].rstrip()
        truncated = True
    start = passage.char_start + lead
    return Excerpt(kept + ("…" if truncated else ""), start, start + len(kept), truncated)


class Embedder(Protocol):
    def embed(self, text: str) -> Sequence[float]: ...


def unit(vector: Sequence[float]) -> tuple[float, ...]:
    """L2-normalize, exactly as the T3a semantic ranking does."""

    values = tuple(float(v) for v in vector)
    norm = math.sqrt(math.fsum(v * v for v in values))
    if not values or norm == 0.0 or not math.isfinite(norm):
        raise ValueError("cannot normalize an empty, zero, or non-finite vector")
    return tuple(v / norm for v in values)


def similarity(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    path: str
    passages: tuple[Passage, ...]
    vectors: tuple[tuple[float, ...], ...]
    embed_s: float


def embed_document(embedder: Embedder, path: str, text: str) -> EmbeddedDocument:
    """Embed every window of one document, one window per call, as-is (no instruction)."""

    passages = build_windows(text)
    start = time.perf_counter()
    vectors = tuple(unit(embedder.embed(p.text)) for p in passages)
    return EmbeddedDocument(path, passages, vectors, time.perf_counter() - start)


def rank_passages(query_vector: Sequence[float], document: EmbeddedDocument) -> tuple[ScoredPassage, ...]:
    """All windows by cosine similarity; ties go to the earliest start line."""

    query = unit(query_vector)
    scored = [ScoredPassage(p, similarity(query, v)) for p, v in zip(document.passages, document.vectors)]
    return tuple(sorted(scored, key=lambda s: (-s.score, s.passage.start_line)))
