"""
태스크 3 단위 테스트: 감성 분석
- OpenAI API를 mock으로 대체하여 테스트
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.sentiment import analyze_sentiment, is_high_risk


def make_mock_response(content: str):
    """OpenAI 응답 mock 생성 헬퍼"""
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@patch("modules.sentiment.OpenAI")
def test_negative_sentiment(mock_openai):
    """부정 텍스트 → 부정 감성 반환"""
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
        '{"sentiment": "부정", "score": 0.85, "signals": ["포기의사"], "summary": "포기 고려 중"}'
    )
    result = analyze_sentiment("이제 그냥 포기하고 싶어요")
    assert result["sentiment"] == "부정"
    assert result["score"] == 0.85
    assert "포기의사" in result["signals"]


@patch("modules.sentiment.OpenAI")
def test_positive_sentiment(mock_openai):
    """긍정 텍스트 → 긍정 감성 반환"""
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
        '{"sentiment": "긍정", "score": 0.0, "signals": ["없음"], "summary": "학습 의욕 높음"}'
    )
    result = analyze_sentiment("오늘 드디어 이해했어요!")
    assert result["sentiment"] == "긍정"
    assert result["score"] == 0.0


def test_empty_input_returns_neutral():
    """빈 입력 → API 호출 없이 중립 반환"""
    result = analyze_sentiment("   ")
    assert result["sentiment"] == "중립"
    assert result["score"] == 0.0


def test_is_high_risk_above_threshold():
    assert is_high_risk({"score": 0.7}) is True


def test_is_high_risk_below_threshold():
    assert is_high_risk({"score": 0.3}) is False


def test_is_high_risk_custom_threshold():
    assert is_high_risk({"score": 0.5}, threshold=0.4) is True
    assert is_high_risk({"score": 0.5}, threshold=0.6) is False


def test_no_api_key_returns_neutral(monkeypatch):
    """Property 1: API 키 미설정 시 graceful degradation"""
    # Feature: campusguard-deployment, Property 1: API 키 미설정 시 graceful degradation
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = analyze_sentiment("포기하고 싶어요")
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "score" in result
    assert result["score"] == 0.0


@patch("modules.sentiment.OpenAI")
def test_score_always_in_range(mock_openai, monkeypatch):
    """Property 5: 감성 점수 범위 보장 (0.0 ~ 1.0)"""
    # Feature: campusguard-deployment, Property 5: 감성 점수 범위 보장
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for score in [0.0, 0.5, 1.0]:
        mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
            f'{{"sentiment": "부정", "score": {score}, "signals": ["없음"], "summary": "테스트"}}'
        )
        result = analyze_sentiment("테스트 텍스트")
        assert 0.0 <= result["score"] <= 1.0
