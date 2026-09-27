"""Localized lexical search results: grounded line ranges and excerpts."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from zomah.access import ReadScope
from zomah.capability_registry import default_capability_registry
from zomah.knowledge import MAX_EXCERPT_CHARS, KnowledgeIndex, _fts_query, localize
from zomah.model_runtime import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResultMessage,
)
from zomah.model_tools import build_session_tools
from zomah.tracing import TraceOutcome, TraceStore
from zomah import worker_session
from zomah.worker_session import WorkerSession


def grounded(body: str, start_line: int, end_line: int, excerpt: str) -> bool:
    """True if ``excerpt`` is cut from exactly lines start..end of ``body``."""

    lines = body.split("\n")
    region = "\n".join(lines[start_line - 1 : end_line])
    core = excerpt.removeprefix("…").removesuffix("…")
    offset = region.find(core)
    if offset < 0 or not core:
        return False
    first_line_len = len(lines[start_line - 1])
    last_line_start = len(region) - len(lines[end_line - 1])
    return offset <= first_line_len and offset + len(core) > last_line_start or (
        start_line == end_line
    )


def filler(count: int, word: str = "gardening") -> list[str]:
    return [f"Line about {word} and weather number {i}." for i in range(count)]


# --- localize(): the second stage -----------------------------------------------------


def test_early_match_reports_an_early_range() -> None:
    body = "\n".join(["# Budget", "The budget is small.", *filler(50)])
    region = localize(body, "budget")
    assert (region.start_line, region.end_line) == (1, 2)
    assert grounded(body, region.start_line, region.end_line, region.excerpt)


def test_deep_match_reports_the_deep_range_not_line_one() -> None:
    lines = ["# Notes", *filler(400)]
    lines[300] = "The per-turn tool-result budget defaults to 32768 characters."
    body = "\n".join(lines)
    region = localize(body, "per-turn tool-result budget")
    assert (region.start_line, region.end_line) == (301, 301)
    assert "32768" in region.excerpt
    assert grounded(body, region.start_line, region.end_line, region.excerpt)


def test_more_distinct_terms_beat_repeated_single_terms() -> None:
    lines = filler(40)
    lines[5] = "budget budget budget budget budget"
    lines[30] = "The tool result budget is per turn."
    region = localize("\n".join(lines), "tool result budget")
    assert region.start_line == 31


def test_more_occurrences_break_distinct_ties_then_earliest_wins() -> None:
    lines = filler(60)
    lines[10] = "budget once here"
    lines[40] = "budget and budget again"
    assert localize("\n".join(lines), "budget").start_line == 41

    lines[40] = "budget once there"
    assert localize("\n".join(lines), "budget").start_line == 11


def test_window_spans_adjacent_matching_lines_only() -> None:
    lines = filler(20)
    lines[7] = "## Tool budget"
    lines[8] = ""
    lines[9] = "The result budget is enforced per turn."
    region = localize("\n".join(lines), "tool result budget")
    assert (region.start_line, region.end_line) == (8, 10)


def test_case_punctuation_and_diacritics_do_not_destabilize() -> None:
    lines = filler(30)
    lines[12] = "TOOL-RESULT Budget: enforced (per-turn)!"
    lines[20] = "Café crème notes."
    body = "\n".join(lines)
    assert localize(body, "tool-result budget").start_line == 13
    assert localize(body, "Tool-Result, BUDGET?").start_line == 13
    assert localize(body, "cafe creme").start_line == 21


def test_hyphenated_term_needs_the_whole_phrase() -> None:
    lines = filler(30)
    lines[3] = "a tool on its own"
    lines[25] = "the tool result arrives"
    assert localize("\n".join(lines), "tool-result").start_line == 26


def test_long_region_excerpt_is_bounded_and_centred_on_the_match() -> None:
    long_line = "x " * 2000 + "the budget sentence " + "y " * 2000
    body = "\n".join(["intro", long_line, "outro"])
    region = localize(body, "budget")
    assert (region.start_line, region.end_line) == (2, 2)
    assert "budget" in region.excerpt
    assert region.excerpt.startswith("…") and region.excerpt.endswith("…")
    assert len(region.excerpt) <= MAX_EXCERPT_CHARS + 2
    assert grounded(body, 2, 2, region.excerpt)


def test_no_body_match_falls_back_to_first_non_blank_line() -> None:
    region = localize("\n\n  \nfirst real line\nsecond", "zeppelin")
    assert (region.start_line, region.end_line, region.excerpt) == (4, 4, "first real line")


def test_query_terms_match_what_fts_is_given() -> None:
    assert _fts_query("per-turn tool-result budget") == '"per-turn" OR "tool-result" OR "budget"'
    region = localize("per turn\ntool result\nbudget", "per-turn tool-result budget")
    assert (region.start_line, region.end_line) == (1, 3)


# --- through the index / capability ---------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    lines = ["# ZOMAH", "", *filler(180)]
    lines += [
        "## Per-turn tool-result budget",
        "",
        "The per-turn tool-result budget defaults to 32768 characters; it is "
        "a character bound, not a token estimate.",
        "",
        *filler(20, "weather"),
    ]
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "PROJECT.md").write_text(
        "# ZOMAH\n\n" + "\n".join(filler(40)) + "\n- tool-result normalization\n",
        encoding="utf-8",
    )
    return root


def tools_for(root: Path, tmp_path: Path) -> tuple[Any, Any, TraceStore]:
    scope = ReadScope.from_paths([root])
    trace_store = TraceStore(tmp_path / "trace.db")
    tools, executor = build_session_tools(
        default_capability_registry(),
        worker="elyria",
        trace_store=trace_store,
        allowed_ids=("read_file", "search_knowledge"),
        dependencies={
            "read_file": {"scope": scope},
            "search_knowledge": {"index": KnowledgeIndex(tmp_path / "knowledge.db", scope)},
        },
    )
    return tools, executor, trace_store


def call(executor: Any, name: str, **arguments: Any) -> dict[str, Any]:
    content = asyncio.run(executor.execute(ToolCall(id="c", name=name, arguments=arguments)))
    return json.loads(content)


def search(executor: Any, query: str, **kwargs: Any) -> list[dict[str, Any]]:
    return call(executor, "search_knowledge", query=query, **kwargs)["result"]["results"]


def test_every_result_has_a_grounded_line_range(root: Path, tmp_path: Path) -> None:
    _, executor, _ = tools_for(root, tmp_path)
    results = search(executor, "per-turn tool-result budget gardening", max_results=50)
    assert results
    for result in results:
        body = Path(result["path"]).read_text(encoding="utf-8")
        assert 1 <= result["start_line"] <= result["end_line"]
        assert grounded(body, result["start_line"], result["end_line"], result["excerpt"])
        assert len(result["excerpt"]) <= MAX_EXCERPT_CHARS + 2


def test_budget_section_is_located_deep_in_the_readme(root: Path, tmp_path: Path) -> None:
    _, executor, _ = tools_for(root, tmp_path)
    top = search(executor, "per-turn tool-result budget")[0]
    assert top["path"].endswith("README.md")
    assert (top["start_line"], top["end_line"]) == (183, 185)
    envelope = call(executor, "read_file", path=top["path"], start_line=top["start_line"], max_lines=5)
    assert "32768 characters" in envelope["result"]["content"]
    assert "not a token estimate" in envelope["result"]["content"]


def test_result_order_is_still_fts_bm25(root: Path, tmp_path: Path) -> None:
    _, executor, _ = tools_for(root, tmp_path)
    ranked = [r["path"] for r in search(executor, "tool-result budget gardening", max_results=50)]
    index = KnowledgeIndex(tmp_path / "knowledge.db", ReadScope.from_paths([root]))
    with index.connect() as conn:
        expected = [
            row["path"]
            for row in conn.execute(
                "SELECT d.path FROM knowledge_fts JOIN knowledge_documents AS d "
                "ON d.path = knowledge_fts.path WHERE knowledge_fts MATCH ? "
                "ORDER BY bm25(knowledge_fts, 0.0, 5.0, 1.0), d.path",
                (_fts_query("tool-result budget gardening"),),
            )
        ]
    assert ranked == expected


def test_moved_section_reports_its_new_range_after_refresh(root: Path, tmp_path: Path) -> None:
    _, executor, _ = tools_for(root, tmp_path)
    doc = root / "moving.md"
    doc.write_text("\n".join(["# Moving", "zeppelin arrives here", *filler(50)]), encoding="utf-8")
    assert search(executor, "zeppelin")[0]["start_line"] == 2

    doc.write_text("\n".join(["# Moving", *filler(90), "zeppelin arrives here"]), encoding="utf-8")
    os.utime(doc, ns=(1, 1))
    moved = search(executor, "zeppelin")[0]
    assert (moved["start_line"], moved["end_line"]) == (92, 92)

    doc.unlink()
    assert search(executor, "zeppelin") == []


def test_line_numbers_match_read_file_despite_unusual_line_breaks(
    root: Path, tmp_path: Path
) -> None:
    # \f and   are line breaks for str.splitlines() but not for read_file.
    doc = root / "breaks.md"
    doc.write_text("intro\fstill line one still\ncrlf line\r\nlone cr\rzeppelin here\n",
                   encoding="utf-8", newline="")
    _, executor, _ = tools_for(root, tmp_path)
    hit = search(executor, "zeppelin")[0]
    read = call(executor, "read_file", path=hit["path"], start_line=hit["start_line"], max_lines=1)
    assert "zeppelin here" in read["result"]["content"]
    assert hit["start_line"] == 4


def test_title_only_match_returns_a_valid_first_line_range(root: Path, tmp_path: Path) -> None:
    (root / "zeppelin.txt").write_text("\nnothing relevant inside\n", encoding="utf-8")
    _, executor, _ = tools_for(root, tmp_path)
    hit = search(executor, "zeppelin")[0]
    assert (hit["start_line"], hit["end_line"], hit["excerpt"]) == (2, 2, "nothing relevant inside")


def test_long_line_document_is_found_but_read_file_refuses_that_line(
    root: Path, tmp_path: Path
) -> None:
    """Known contract edge: the index admits lines read_file will not return."""

    doc = root / "longline.md"
    doc.write_text("short first line\n" + "x " * 9000 + "zeppelin " + "y " * 9000 + "\nlast\n",
                   encoding="utf-8")
    _, executor, _ = tools_for(root, tmp_path)
    hit = search(executor, "zeppelin")[0]
    assert (hit["start_line"], hit["end_line"]) == (2, 2)
    assert "zeppelin" in hit["excerpt"] and len(hit["excerpt"]) <= MAX_EXCERPT_CHARS + 2
    refused = call(executor, "read_file", path=hit["path"], start_line=2, max_lines=1)
    assert refused["error"]["code"] == "unsupported_text_file"
    elsewhere = call(executor, "read_file", path=hit["path"], start_line=3, max_lines=1)
    assert elsewhere["result"]["content"] == "last\n"


# --- model loop -----------------------------------------------------------------------


class ScriptedRuntime:
    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        step = self.steps.pop(0)
        return step(request) if callable(step) else step


def last_result(request: ModelRequest) -> dict[str, Any]:
    message = request.messages[-1]
    assert isinstance(message, ToolResultMessage)
    return json.loads(message.content)


def read_at_hint(request: ModelRequest) -> ModelResponse:
    top = last_result(request)["result"]["results"][0]
    arguments = {"path": top["path"], "start_line": top["start_line"], "max_lines": 10}
    return ModelResponse(
        message=AssistantMessage("", (ToolCall(id="read_1", name="read_file", arguments=arguments),))
    )


def answer(request: ModelRequest) -> ModelResponse:
    content = last_result(request)["result"]["content"]
    found = "32768 characters" in content and "not a token estimate" in content
    return ModelResponse(message=AssistantMessage("32768, characters" if found else "not found"))


def test_fake_model_searches_reads_at_the_hint_and_answers(root: Path, tmp_path: Path) -> None:
    tools, executor, trace_store = tools_for(root, tmp_path)
    search_call = ToolCall(id="search_1", name="search_knowledge",
                           arguments={"query": "per-turn tool-result budget"})
    runtime = ScriptedRuntime(
        ModelResponse(message=AssistantMessage("", (search_call,))), read_at_hint, answer
    )
    worker = WorkerSession(runtime, worker="elyria", model="m", tools=tools, tool_executor=executor)

    result = asyncio.run(worker.send("Where is the per-turn budget documented?"))

    assert result.assistant.text == "32768, characters"
    assert len(runtime.requests) == 3  # search, one targeted read, final answer
    read_call = runtime.requests[2].messages[-2].tool_calls[0]  # type: ignore[union-attr]
    assert read_call.name == "read_file" and read_call.arguments["start_line"] == 183
    # After the search only its own result went back: ZOMAH read nothing itself.
    after_search = [m for m in runtime.requests[1].messages if isinstance(m, ToolResultMessage)]
    assert [m.call_id for m in after_search] == ["search_1"]
    readme = str((root / "README.md").resolve())
    assert [(r.capability, r.outcome, r.target) for r in reversed(trace_store.recent())] == [
        ("search_knowledge", TraceOutcome.SUCCEEDED, "knowledge-search"),
        ("read_file", TraceOutcome.SUCCEEDED, readme),
    ]
    assert [m.role for m in worker.history] == [
        "user", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    # Both tool results together use a small share of the default budget.
    sizes = [len(m.content) for m in worker.history if isinstance(m, ToolResultMessage)]
    assert sum(sizes) < worker.tool_result_budget_chars // 8


def test_loop_limits_are_unchanged() -> None:
    assert (
        worker_session.MAX_TOOL_ROUNDS,
        worker_session.MAX_TOOL_CALLS_PER_RESPONSE,
        worker_session.MAX_TOOL_CALLS_PER_TURN,
        worker_session.DEFAULT_TOOL_RESULT_BUDGET_CHARS,
    ) == (4, 4, 8, 32768)
