"""
태스크 1 단위 테스트: AI 강의 비서
- OpenAI API를 mock으로 대체
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chat_assistant import ask_assistant


def make_mock_response(content: str):
    mock_msg = MagicMock()
    mock_msg.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


@patch("modules.chat_assistant.OpenAI")
def test_basic_question(mock_openai):
    """일반 질문에 답변 반환"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
        "NameError는 변수가 정의되지 않았을 때 발생합니다."
    )
    result = ask_assistant("NameError: name 'x' is not defined 에러가 뭔가요?")
    assert "NameError" in result


@patch("modules.chat_assistant.OpenAI")
def test_question_with_context(mock_openai):
    """교안 맥락 포함 시 system message 2개 전달"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response(
        "pandas DataFrame에서 발생하는 에러입니다."
    )
    result = ask_assistant("에러가 나요", context="pandas 기초 - DataFrame 생성")
    # context가 있을 때 create가 호출되었는지 확인
    call_args = mock_openai.return_value.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_messages = [m for m in messages if m["role"] == "system"]
    assert len(system_messages) == 2


def test_empty_input_returns_guide(monkeypatch):
    """빈 입력 시 API 호출 없이 안내 메시지 반환"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = ask_assistant("   ")
    assert "입력" in result


@patch("modules.chat_assistant.OpenAI")
def test_no_context_single_system_message(mock_openai):
    """context 없을 때 system message 1개"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response("답변")
    ask_assistant("질문")
    call_args = mock_openai.return_value.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    system_messages = [m for m in messages if m["role"] == "system"]
    assert len(system_messages) == 1


def test_no_api_key_returns_warning(monkeypatch):
    """Property 1: API 키 미설정 시 graceful degradation"""
    # Feature: campusguard-deployment, Property 1: API 키 미설정 시 graceful degradation
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = ask_assistant("테스트 질문")
    assert isinstance(result, str)
    assert "API" in result or "키" in result


@patch("modules.chat_assistant.OpenAI")
def test_openai_exception_returns_korean_error(mock_openai):
    """Property 2: OpenAI 예외 시 한국어 에러 반환"""
    # Feature: campusguard-deployment, Property 2: OpenAI 예외 시 한국어 에러 반환
    import os
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_openai.return_value.chat.completions.create.side_effect = Exception("API 연결 실패")
    result = ask_assistant("테스트 질문")
    assert isinstance(result, str)
    assert "오류" in result or "❌" in result


@patch("modules.chat_assistant.OpenAI")
def test_context_injected_in_messages(mock_openai):
    """Property 7: 컨텍스트 주입 보장"""
    # Feature: campusguard-deployment, Property 7: 컨텍스트 주입 보장
    import os
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response("답변")
    ask_assistant("질문", context="pandas 기초")
    call_args = mock_openai.return_value.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    all_content = " ".join(m["content"] for m in messages)
    assert "pandas 기초" in all_content
