"""
태스크 10 테스트: RAGEngine
- Property 8: RAG 청크 분할 내용 보존
- Property 9: RAG 검색 결과 수 상한
"""
import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.rag_engine import RAGEngine, EMBEDDING_DIM


# ── 단위 테스트 ────────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    """기본 청크 분할 동작 확인"""
    engine = RAGEngine(chunk_size=10, chunk_overlap=2)
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = engine.chunk_text(text)
    assert len(chunks) > 0
    # 모든 청크는 chunk_size 이하
    for chunk in chunks:
        assert len(chunk) <= 10


def test_chunk_text_empty():
    """빈 텍스트 입력 시 빈 리스트 반환"""
    engine = RAGEngine()
    assert engine.chunk_text("") == []


def test_chunk_text_short_text():
    """텍스트가 chunk_size보다 짧으면 청크 1개 반환"""
    engine = RAGEngine(chunk_size=500)
    text = "짧은 텍스트"
    chunks = engine.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_search_before_build_index_returns_empty():
    """build_index 호출 전 search는 빈 리스트 반환"""
    engine = RAGEngine()
    result = engine.search("테스트 쿼리")
    assert result == []


def test_get_context_prompt_before_build_index_returns_empty():
    """build_index 호출 전 get_context_prompt는 빈 문자열 반환"""
    engine = RAGEngine()
    result = engine.get_context_prompt("테스트 쿼리")
    assert result == ""


def make_mock_embedding(texts):
    """테스트용 임베딩 mock 응답 생성"""
    mock_response = MagicMock()
    mock_response.data = []
    for _ in texts:
        item = MagicMock()
        item.embedding = list(np.random.rand(EMBEDDING_DIM).astype(float))
        mock_response.data.append(item)
    return mock_response


@patch("modules.rag_engine.OpenAI")
def test_build_index_and_search(mock_openai):
    """build_index 후 search가 결과를 반환하는지 확인"""
    chunks = ["파이썬 기초", "머신러닝 개요", "딥러닝 소개"]

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda **kwargs: make_mock_embedding(
        kwargs["input"]
    )

    engine = RAGEngine()
    engine.build_index(chunks)
    results = engine.search("파이썬", top_k=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(r in chunks for r in results)


@patch("modules.rag_engine.OpenAI")
def test_get_context_prompt_format(mock_openai):
    """get_context_prompt가 시스템 프롬프트 형식으로 반환하는지 확인"""
    chunks = ["교안 내용 1", "교안 내용 2"]

    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda **kwargs: make_mock_embedding(
        kwargs["input"]
    )

    engine = RAGEngine()
    engine.build_index(chunks)
    prompt = engine.get_context_prompt("질문")

    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "참고" in prompt


def test_load_pdf_invalid_bytes_raises_runtime_error():
    """유효하지 않은 바이트 입력 시 RuntimeError 발생"""
    engine = RAGEngine()
    with pytest.raises(RuntimeError):
        engine.load_pdf(b"not a pdf")


# ── Property 8: RAG 청크 분할 내용 보존 ───────────────────────────────────────

# Feature: campusguard-enhancement, Property 8: RAG 청크 분할 내용 보존
@settings(max_examples=100)
@given(text=st.text(min_size=1))
def test_property8_chunk_content_preservation(text):
    """Property 8: RAG 청크 분할 내용 보존
    Validates: Requirements 3.3
    chunk_text로 분할된 청크들을 이어 붙이면 원본 텍스트의 모든 내용이 포함되어야 한다.
    (overlap 허용, 내용 손실 없음)
    """
    engine = RAGEngine(chunk_size=50, chunk_overlap=10)
    chunks = engine.chunk_text(text)

    # 청크가 있으면 원본 텍스트의 모든 부분이 어느 청크에든 포함되어야 한다
    if chunks:
        # 원본 텍스트의 각 위치가 최소 하나의 청크에 포함되는지 확인
        # 방법: 청크들을 순서대로 이어붙인 커버리지 확인
        # chunk_size 단위로 원본을 슬라이싱한 것과 동일한 내용이 청크에 있어야 함
        step = max(1, engine.chunk_size - engine.chunk_overlap)
        start = 0
        chunk_idx = 0
        while start < len(text) and chunk_idx < len(chunks):
            expected_chunk = text[start: start + engine.chunk_size]
            assert chunks[chunk_idx] == expected_chunk, (
                f"청크 {chunk_idx}의 내용이 원본과 다릅니다. "
                f"expected={expected_chunk!r}, got={chunks[chunk_idx]!r}"
            )
            chunk_idx += 1
            if start + engine.chunk_size >= len(text):
                break
            start += step


# ── Property 9: RAG 검색 결과 수 상한 ─────────────────────────────────────────

# Feature: campusguard-enhancement, Property 9: RAG 검색 결과 수 상한
@settings(max_examples=100)
@given(chunks=st.lists(st.text(min_size=1), min_size=1))
@patch("modules.rag_engine.OpenAI")
def test_property9_search_result_count_upper_bound(mock_openai, chunks):
    """Property 9: RAG 검색 결과 수 상한
    Validates: Requirements 3.4
    search(query, top_k=3)의 결과 수는 항상 min(3, len(chunks)) 이하여야 한다.
    """
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda **kwargs: make_mock_embedding(
        kwargs["input"]
    )

    engine = RAGEngine()
    engine.build_index(chunks)
    results = engine.search("테스트 쿼리", top_k=3)

    expected_max = min(3, len(chunks))
    assert len(results) <= expected_max, (
        f"결과 수 {len(results)}가 상한 {expected_max}을 초과했습니다. "
        f"chunks={chunks!r}"
    )
