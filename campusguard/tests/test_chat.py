"""
태스크 1 단위 테스트: AI 강의 비서
- OpenAI API를 mock으로 대체
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.chat_assistant import ask_assistant, detect_traceback, clear_history


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


# ── Property 7: 멀티턴 히스토리 포함 보장 ──────────────────────────────────────

# Feature: campusguard-enhancement, Property 7: 멀티턴 히스토리 포함 보장
@settings(max_examples=100)
@given(
    history=st.lists(
        st.fixed_dictionaries({
            "role": st.sampled_from(["user", "assistant"]),
            "content": st.text(min_size=1, max_size=100),
        }),
        min_size=0,
        max_size=10,
    ),
    user_message=st.text(min_size=1, max_size=50),
)
@patch("modules.chat_assistant.OpenAI")
def test_property7_multiturn_history_included(mock_openai, history, user_message):
    """Property 7: 멀티턴 히스토리 포함 보장
    Validates: Requirements 3.2
    ask_assistant가 OpenAI API에 전달하는 messages 배열은 히스토리의 모든 메시지를 포함해야 한다.
    """
    os.environ["OPENAI_API_KEY"] = "test-key"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "답변"
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    ask_assistant(user_message, history=history)

    call_args = mock_openai.return_value.chat.completions.create.call_args
    messages_sent = call_args.kwargs["messages"]

    # 히스토리의 모든 메시지가 전달된 messages 배열에 포함되어야 한다
    for hist_msg in history:
        assert hist_msg in messages_sent, (
            f"히스토리 메시지 {hist_msg}가 API 호출 messages에 포함되지 않았습니다."
        )


# ── Property 10: Traceback 감지 정확성 ────────────────────────────────────────

# Feature: campusguard-enhancement, Property 10: Traceback 감지 정확성
@settings(max_examples=100)
@given(text=st.text())
def test_property10_traceback_detection_accuracy(text):
    """Property 10: Traceback 감지 정확성
    Validates: Requirements 3.5
    detect_traceback은 "Traceback (most recent call last):" 패턴을 포함하는 문자열에서만
    비-None 값을 반환해야 한다.
    """
    pattern = "Traceback (most recent call last):"
    result = detect_traceback(text)

    if pattern in text:
        assert result is not None, "패턴이 있는데 None을 반환했습니다."
        assert pattern in result, "반환값에 패턴이 포함되어야 합니다."
    else:
        assert result is None, f"패턴이 없는데 None이 아닌 값을 반환했습니다: {result!r}"


# ── 추가 단위 테스트: detect_traceback, clear_history ─────────────────────────

def test_detect_traceback_with_pattern():
    """traceback 패턴이 있을 때 비-None 반환"""
    text = "some output\nTraceback (most recent call last):\n  File 'x.py', line 1\nValueError"
    result = detect_traceback(text)
    assert result is not None
    assert "Traceback (most recent call last):" in result


def test_detect_traceback_without_pattern():
    """traceback 패턴이 없을 때 None 반환"""
    assert detect_traceback("일반 텍스트입니다.") is None
    assert detect_traceback("") is None


def test_clear_history_returns_empty_list():
    """clear_history는 빈 리스트를 반환해야 한다"""
    result = clear_history()
    assert result == []
    assert isinstance(result, list)


@patch("modules.chat_assistant.OpenAI")
def test_ask_assistant_with_history(mock_openai):
    """history 파라미터가 messages에 포함되는지 확인"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response("답변")

    history = [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답변"},
    ]
    ask_assistant("새 질문", history=history)

    call_args = mock_openai.return_value.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert {"role": "user", "content": "이전 질문"} in messages
    assert {"role": "assistant", "content": "이전 답변"} in messages


@patch("modules.chat_assistant.OpenAI")
def test_ask_assistant_no_history_backward_compat(mock_openai):
    """history 없이 호출해도 기존 동작 유지 (하위 호환성)"""
    os.environ["OPENAI_API_KEY"] = "test-key"
    mock_openai.return_value.chat.completions.create.return_value = make_mock_response("답변")
    result = ask_assistant("질문")
    assert isinstance(result, str)
