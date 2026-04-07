"""
KakaoParser 단위 테스트 및 Property-Based 테스트
"""
import sys
import os
import pytest
from datetime import datetime
from hypothesis import given, settings, assume
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.kakao_parser import (
    KakaoMessage,
    parse_kakao_export,
    group_by_sender,
    detect_duplicate_senders,
)


# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------

def _make_kakao_line(year, month, day, ampm, hour, minute, sender, content):
    """카카오톡 형식 문자열 생성 헬퍼."""
    return f"{year}년 {month}월 {day}일 {ampm} {hour}:{minute:02d}, {sender} : {content}"


class TestParseKakaoExport:
    def test_empty_string_returns_empty_list(self):
        assert parse_kakao_export("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert parse_kakao_export("   \n  ") == []

    def test_single_valid_line_parsed(self):
        line = "2024년 3월 5일 오전 9:30, 홍길동 : 안녕하세요"
        messages = parse_kakao_export(line)
        assert len(messages) == 1
        assert messages[0].sender == "홍길동"
        assert messages[0].content == "안녕하세요"
        assert messages[0].timestamp == datetime(2024, 3, 5, 9, 30)

    def test_afternoon_time_conversion(self):
        line = "2024년 3월 5일 오후 3:00, 김철수 : 테스트"
        messages = parse_kakao_export(line)
        assert messages[0].timestamp.hour == 15

    def test_noon_pm_stays_12(self):
        line = "2024년 3월 5일 오후 12:00, 이영희 : 점심"
        messages = parse_kakao_export(line)
        assert messages[0].timestamp.hour == 12

    def test_midnight_am_becomes_0(self):
        line = "2024년 3월 5일 오전 12:00, 박민준 : 자정"
        messages = parse_kakao_export(line)
        assert messages[0].timestamp.hour == 0

    def test_multiple_valid_lines(self):
        text = "\n".join([
            "2024년 1월 1일 오전 9:00, 홍길동 : 새해 복 많이 받으세요",
            "2024년 1월 1일 오전 9:05, 김철수 : 감사합니다",
            "2024년 1월 1일 오전 9:10, 홍길동 : 올해도 잘 부탁드립니다",
        ])
        messages = parse_kakao_export(text)
        assert len(messages) == 3

    def test_invalid_format_raises_value_error(self):
        with pytest.raises(ValueError, match="카카오톡 내보내기 형식이 아닙니다"):
            parse_kakao_export("이것은 카카오톡 형식이 아닙니다")

    def test_mixed_valid_invalid_lines_parses_valid_only(self):
        # 유효한 줄과 빈 줄이 섞인 경우 — 빈 줄은 무시
        text = "\n".join([
            "2024년 1월 1일 오전 9:00, 홍길동 : 메시지",
            "",
            "2024년 1월 1일 오전 9:05, 김철수 : 또 다른 메시지",
        ])
        messages = parse_kakao_export(text)
        assert len(messages) == 2


class TestGroupBySender:
    def test_empty_list_returns_empty_dict(self):
        assert group_by_sender([]) == {}

    def test_single_sender(self):
        msgs = [
            KakaoMessage("홍길동", datetime(2024, 1, 1, 9, 0), "안녕"),
            KakaoMessage("홍길동", datetime(2024, 1, 1, 9, 5), "반가워"),
        ]
        result = group_by_sender(msgs)
        assert set(result.keys()) == {"홍길동"}
        assert len(result["홍길동"]) == 2

    def test_multiple_senders(self):
        msgs = [
            KakaoMessage("홍길동", datetime(2024, 1, 1, 9, 0), "안녕"),
            KakaoMessage("김철수", datetime(2024, 1, 1, 9, 1), "반가워"),
            KakaoMessage("홍길동", datetime(2024, 1, 1, 9, 2), "잘 지내?"),
        ]
        result = group_by_sender(msgs)
        assert set(result.keys()) == {"홍길동", "김철수"}
        assert len(result["홍길동"]) == 2
        assert len(result["김철수"]) == 1


# ---------------------------------------------------------------------------
# Property 3: 카카오톡 파싱 Round-Trip
# Feature: campusguard-enhancement, Property 3: 카카오톡 파싱 Round-Trip
# Validates: Requirements 2.1, 8.4
# ---------------------------------------------------------------------------

# 발신자명과 내용에 사용할 안전한 텍스트 전략 (줄바꿈, 콜론, 제어문자 등 제외)
safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Zs"),
        whitelist_characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789가나다라마바사아자차카타파하",
    ),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() == s and " : " not in s and len(s) > 0)

content_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Lt", "Lm", "Lo", "Nd", "Zs"),
        whitelist_characters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789가나다라마바사아자차카타파하 .,!?",
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip() == s and len(s) > 0)


@given(
    sender=safe_text,
    content=content_text,
    dt=st.datetimes(
        min_value=datetime(2000, 1, 1, 0, 0),
        max_value=datetime(2099, 12, 31, 23, 59),
    ),
)
@settings(max_examples=100)
def test_property3_kakao_parse_round_trip(sender, content, dt):
    """Property 3: 카카오톡 파싱 Round-Trip
    Validates: Requirements 2.1, 8.4
    """
    # 오전/오후 결정
    hour = dt.hour
    if hour == 0:
        ampm = "오전"
        display_hour = 12
    elif hour < 12:
        ampm = "오전"
        display_hour = hour
    elif hour == 12:
        ampm = "오후"
        display_hour = 12
    else:
        ampm = "오후"
        display_hour = hour - 12

    line = f"{dt.year}년 {dt.month}월 {dt.day}일 {ampm} {display_hour}:{dt.minute:02d}, {sender} : {content}"

    messages = parse_kakao_export(line)

    assert len(messages) == 1
    msg = messages[0]
    assert msg.sender == sender
    assert msg.content == content
    assert msg.timestamp.year == dt.year
    assert msg.timestamp.month == dt.month
    assert msg.timestamp.day == dt.day
    assert msg.timestamp.hour == dt.hour
    assert msg.timestamp.minute == dt.minute


# ---------------------------------------------------------------------------
# Property 4: 잘못된 형식 파싱 거부
# Feature: campusguard-enhancement, Property 4: 잘못된 형식 파싱 거부
# Validates: Requirements 2.2, 8.4
# ---------------------------------------------------------------------------

# 카카오 패턴을 포함하지 않는 문자열 전략
invalid_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=100,
).filter(
    lambda s: (
        s.strip()
        and not any(
            line.strip() and
            bool(__import__("re").match(
                r"^\d{4}년 \d{1,2}월 \d{1,2}일 (오전|오후) \d{1,2}:\d{2}, .+ : .+$",
                line.strip()
            ))
            for line in s.splitlines()
        )
    )
)


@given(text=invalid_text)
@settings(max_examples=100)
def test_property4_invalid_format_rejected(text):
    """Property 4: 잘못된 형식 파싱 거부
    Validates: Requirements 2.2, 8.4
    """
    try:
        result = parse_kakao_export(text)
        # ValueError가 발생하지 않으면 빈 리스트여야 함
        assert result == [], f"유효하지 않은 형식에서 메시지가 파싱됨: {result}"
    except ValueError:
        pass  # ValueError 발생은 정상


# ---------------------------------------------------------------------------
# Property 5: 발신자 그룹화 완전성
# Feature: campusguard-enhancement, Property 5: 발신자 그룹화 완전성
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

kakao_message_strategy = st.builds(
    KakaoMessage,
    sender=safe_text,
    timestamp=st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2099, 12, 31),
    ),
    content=content_text,
)


@given(messages=st.lists(kakao_message_strategy, min_size=0, max_size=50))
@settings(max_examples=100)
def test_property5_sender_grouping_completeness(messages):
    """Property 5: 발신자 그룹화 완전성
    Validates: Requirements 2.3
    """
    result = group_by_sender(messages)

    expected_senders = {msg.sender for msg in messages}
    actual_senders = set(result.keys())

    assert actual_senders == expected_senders, (
        f"그룹화 키 집합 {actual_senders} != 원본 발신자 집합 {expected_senders}"
    )

    # 각 그룹의 메시지 수 합계 == 전체 메시지 수
    total_grouped = sum(len(msgs) for msgs in result.values())
    assert total_grouped == len(messages)
