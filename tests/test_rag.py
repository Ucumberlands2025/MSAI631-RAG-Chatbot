"""Offline tests deliberately use mocks, not downloaded model weights."""
from pathlib import Path

import numpy as np
import pytest

from rag import (Passage, RAGAssistant, VectorStore, build_store, fingerprint,
                 format_evidence, keyword_overlap, read_passages, split_passages)


class WordEncoder:
    """Small deterministic test-only embedding model; not a real sentence transformer."""
    def encode(self, texts):
        terms = ("sunday", "extension", "discussion", "weather")
        return np.asarray([[float(word in s.lower()) for word in terms] for s in texts], dtype=np.float32)


class TestGenerator:
    def generate(self, question, matches):
        return f"Test-only response from {matches[0].passage.source}."


def test_chunking_overlap():
    parts = split_passages("one two three four five six seven", size=4, overlap=2)
    assert parts == ["one two three four", "three four five six", "five six seven"]


def test_chunking_edge_cases():
    assert split_passages(" ") == []
    with pytest.raises(ValueError):
        split_passages("hello", size=4, overlap=4)


def test_local_text_files_only(tmp_path):
    (tmp_path / "a.txt").write_text("Some sample content", encoding="utf-8")
    (tmp_path / "private.pdf").write_text("Not indexed", encoding="utf-8")
    pieces = read_passages(tmp_path)
    assert len(pieces) == 1 and pieces[0].source == "a.txt"


def test_fingerprint_changes_with_content(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("alpha", encoding="utf-8")
    first = fingerprint(tmp_path)
    file.write_text("beta", encoding="utf-8")
    assert fingerprint(tmp_path) != first


def test_build_load_and_rank(tmp_path):
    data, idx = tmp_path / "data", tmp_path / "index"
    data.mkdir()
    (data / "due.txt").write_text("The assignment is due Sunday.", encoding="utf-8")
    (data / "policy.txt").write_text("Extension requests need approval.", encoding="utf-8")
    count = build_store(WordEncoder(), docs=data, destination=idx)
    assert count == 2
    loaded = VectorStore.load(docs=data, destination=idx)
    matches = loaded.search(WordEncoder().encode(["When is it due Sunday?"])[0], k=1)
    assert matches[0].passage.source == "due.txt"


def test_stale_index_requires_rebuild(tmp_path):
    data, idx = tmp_path / "data", tmp_path / "index"
    data.mkdir()
    file = data / "due.txt"
    file.write_text("due Sunday", encoding="utf-8")
    build_store(WordEncoder(), docs=data, destination=idx)
    file.write_text("due Monday", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        VectorStore.load(docs=data, destination=idx)


def assistant():
    passages = [Passage("schedule.txt", 1, "The assignment is due Sunday."),
                Passage("policy.txt", 1, "An extension request needs approval.")]
    return RAGAssistant(WordEncoder(), VectorStore(passages, np.array([[1, 0, 0, 0], [0, 1, 0, 0]])), TestGenerator())


def test_grounded_answer_and_sources():
    result = assistant().ask("What happens on Sunday?")
    assert result["status"] == "answered"
    assert result["sources"][0].passage.source == "schedule.txt"
    assert "schedule.txt" in format_evidence(result["sources"])


def test_out_of_scope_returns_no_sources():
    result = assistant().ask("Weather in Boston?")
    assert result["status"] == "not_found"
    assert not result["sources"]


def test_empty_and_long_inputs():
    assert assistant().ask(" ")["status"] == "empty"
    assert assistant().ask("x" * 501)["status"] == "too_long"


def test_keyword_guard():
    assert keyword_overlap("What day is Sunday?", "The work is due Sunday.")
    assert not keyword_overlap("What is the forecast?", "The assignment is due Sunday.")


def test_mismatched_vector_dimensions():
    store = VectorStore([Passage("x.txt", 1, "Sunday")], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="dimensions"):
        store.search(np.array([1.0, 0.0, 0.0]))
