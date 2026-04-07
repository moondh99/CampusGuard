"""
태스크 5 단위 테스트: 행정 서류 자동 생성
- OpenAI API를 mock으로 대체
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.admin_writer import generate_counseling_log, generate_reason_letter


def make_mock_response(content: str):
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@patch("modules.admin_writer.OpenAI")
def test_counseling_log_generated(mock_openai):
    """상담일지 초안이 생성됨"""
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
        "상담일지: [이름] 훈련생은 취업 고민으로 상담을 요청하였음."
    )
    result = generate_counseling_log(
        name="김훈련",
        keywords="취업 고민, 파이썬 기초 부족",
        risk_level="경고",
        absent_info="지각 2회, 조퇴 1회",
    )
    assert len(result) > 0
    assert "[이름]" in result  # PII 마스킹 확인


@patch("modules.admin_writer.OpenAI")
def test_reason_letter_generated(mock_openai):
    """사유서 초안이 생성됨"""
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
        "사유서: 2026-04-07 시스템 장애로 인해 출결 체크가 불가하였음."
    )
    result = generate_reason_letter("2026-04-07 시스템 장애로 출결 체크 불가")
    assert len(result) > 0


@patch("modules.admin_writer.OpenAI")
def test_counseling_log_uses_correct_model(mock_openai):
    """gpt-4o-mini 모델 사용 확인"""
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response("초안")
    generate_counseling_log("김훈련", "고민", "정상", "출석 정상")
    call_args = mock_openai.return_value.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "gpt-4o-mini"


def test_no_api_key_counseling_returns_warning(monkeypatch):
    """Property 1: API 키 미설정 시 graceful degradation - counseling log"""
    # Feature: campusguard-deployment, Property 1: API 키 미설정 시 graceful degradation
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_counseling_log("김훈련", "고민", "정상", "출석 정상")
    assert isinstance(result, str)
    assert "API" in result or "키" in result


def test_no_api_key_reason_letter_returns_warning(monkeypatch):
    """Property 1: API 키 미설정 시 graceful degradation - reason letter"""
    # Feature: campusguard-deployment, Property 1: API 키 미설정 시 graceful degradation
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = generate_reason_letter("시스템 장애")
    assert isinstance(result, str)
    assert "API" in result or "키" in result


@patch("modules.admin_writer.OpenAI")
def test_openai_exception_returns_korean_error(mock_openai, monkeypatch):
    """Property 2: OpenAI 예외 시 한국어 에러 반환"""
    # Feature: campusguard-deployment, Property 2: OpenAI 예외 시 한국어 에러 반환
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_openai.return_value.chat.completions.create.side_effect = Exception("연결 실패")
    result = generate_counseling_log("김훈련", "고민", "정상", "출석 정상")
    assert isinstance(result, str)
    assert "오류" in result or "❌" in result


@patch("modules.admin_writer.OpenAI")
def test_counseling_log_uses_placeholder(mock_openai, monkeypatch):
    """Property 6: 상담일지 이름 플레이스홀더 치환"""
    # Feature: campusguard-deployment, Property 6: 상담일지 이름 플레이스홀더 치환
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
        "상담일지: [이름] 훈련생은 취업 고민으로 상담을 요청하였음."
    )
    result = generate_counseling_log("김훈련", "취업 고민", "경고", "지각 2회")
    assert "[이름]" in result
    assert "김훈련" not in result
