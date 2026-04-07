"""
NotificationService 단위 테스트 + Property-Based 테스트
- Property 14: 알림 메시지 필수 필드 포함 (Requirements 6.3)
- Property 15: 알림 발송 조건 정확성 (Requirements 6.1)
"""
import sys
import os
import logging
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.notifier import (
    NotificationConfig,
    load_config_from_env,
    should_notify,
    format_alert_message,
    send_email,
    send_slack,
    send_alert,
    generate_weekly_report,
)
from modules.risk_predictor import RiskResult

import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def make_risk(name="홍길동", level="위험", score=0.8, sent=0.7, att="위험", rec="즉시 상담 필요"):
    return RiskResult(
        name=name,
        final_score=score,
        final_level=level,
        attendance_level=att,
        sentiment_score=sent,
        recommendation=rec,
    )


def make_config(enabled=True, email=True, slack=True):
    return NotificationConfig(
        enabled=enabled,
        email_enabled=email,
        email_host="smtp.example.com",
        email_port=587,
        email_user="test@example.com",
        email_password="password",
        email_to="admin@example.com",
        slack_enabled=slack,
        slack_webhook_url="https://hooks.slack.com/test",
    )


# ---------------------------------------------------------------------------
# 단위 테스트 — NotificationConfig & load_config_from_env
# ---------------------------------------------------------------------------

class TestLoadConfigFromEnv:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("NOTIFICATION_ENABLED", raising=False)
        config = load_config_from_env()
        assert config.enabled is False

    def test_disabled_when_false(self, monkeypatch):
        monkeypatch.setenv("NOTIFICATION_ENABLED", "false")
        config = load_config_from_env()
        assert config.enabled is False

    def test_enabled_when_true(self, monkeypatch):
        monkeypatch.setenv("NOTIFICATION_ENABLED", "true")
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        monkeypatch.setenv("EMAIL_HOST", "smtp.gmail.com")
        monkeypatch.setenv("EMAIL_PORT", "465")
        monkeypatch.setenv("EMAIL_USER", "user@gmail.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setenv("EMAIL_TO", "admin@gmail.com")
        monkeypatch.setenv("SLACK_ENABLED", "true")
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/xxx")
        config = load_config_from_env()
        assert config.enabled is True
        assert config.email_enabled is True
        assert config.email_host == "smtp.gmail.com"
        assert config.email_port == 465
        assert config.slack_enabled is True
        assert config.slack_webhook_url == "https://hooks.slack.com/xxx"

    def test_disabled_returns_minimal_config(self, monkeypatch):
        monkeypatch.setenv("NOTIFICATION_ENABLED", "false")
        config = load_config_from_env()
        assert config.enabled is False
        assert config.email_enabled is False
        assert config.slack_enabled is False


# ---------------------------------------------------------------------------
# 단위 테스트 — should_notify
# ---------------------------------------------------------------------------

class TestShouldNotify:
    def test_empty_list(self):
        assert should_notify([]) is False

    def test_all_normal(self):
        results = [make_risk(level="정상"), make_risk(level="경고")]
        assert should_notify(results) is False

    def test_one_danger(self):
        results = [make_risk(level="정상"), make_risk(level="위험")]
        assert should_notify(results) is True

    def test_all_danger(self):
        results = [make_risk(level="위험"), make_risk(level="위험")]
        assert should_notify(results) is True


# ---------------------------------------------------------------------------
# 단위 테스트 — format_alert_message
# ---------------------------------------------------------------------------

class TestFormatAlertMessage:
    def test_no_danger(self):
        results = [make_risk(level="정상"), make_risk(level="경고")]
        msg = format_alert_message(results)
        assert "위험 등급 훈련생이 없습니다" in msg

    def test_contains_required_fields(self):
        r = make_risk(name="김훈련", level="위험", score=0.85, rec="즉시 상담 필요")
        msg = format_alert_message([r])
        assert "김훈련" in msg
        assert "위험" in msg
        assert "0.85" in msg
        assert "즉시 상담 필요" in msg

    def test_multiple_danger_trainees(self):
        results = [
            make_risk(name="A", level="위험", score=0.9, rec="상담 필요"),
            make_risk(name="B", level="위험", score=0.75, rec="모니터링"),
            make_risk(name="C", level="경고"),
        ]
        msg = format_alert_message(results)
        assert "A" in msg
        assert "B" in msg
        assert "C" not in msg or msg.count("C") == 0 or "경고" not in msg.split("B")[1]


# ---------------------------------------------------------------------------
# 단위 테스트 — send_email
# ---------------------------------------------------------------------------

class TestSendEmail:
    def test_disabled_config_skips(self):
        config = make_config(enabled=False)
        with patch("smtplib.SMTP") as mock_smtp:
            send_email(config, "제목", "내용")
            mock_smtp.assert_not_called()

    def test_email_disabled_skips(self):
        config = make_config(enabled=True, email=False)
        with patch("smtplib.SMTP") as mock_smtp:
            send_email(config, "제목", "내용")
            mock_smtp.assert_not_called()

    def test_sends_email_when_enabled(self):
        config = make_config(enabled=True, email=True)
        mock_server = MagicMock()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
            send_email(config, "테스트 제목", "테스트 내용")
            mock_smtp.assert_called_once_with(config.email_host, config.email_port)

    def test_error_logged_not_raised(self, caplog):
        config = make_config(enabled=True, email=True)
        with patch("smtplib.SMTP", side_effect=Exception("연결 실패")):
            with caplog.at_level(logging.ERROR):
                send_email(config, "제목", "내용")  # 예외 전파 없음
        assert "이메일 발송 실패" in caplog.text


# ---------------------------------------------------------------------------
# 단위 테스트 — send_slack
# ---------------------------------------------------------------------------

class TestSendSlack:
    def test_disabled_config_skips(self):
        config = make_config(enabled=False)
        with patch("requests.post") as mock_post:
            send_slack(config, "메시지")
            mock_post.assert_not_called()

    def test_slack_disabled_skips(self):
        config = make_config(enabled=True, slack=False)
        with patch("requests.post") as mock_post:
            send_slack(config, "메시지")
            mock_post.assert_not_called()

    def test_sends_slack_when_enabled(self):
        config = make_config(enabled=True, slack=True)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_response) as mock_post:
            send_slack(config, "테스트 메시지")
            mock_post.assert_called_once_with(
                config.slack_webhook_url,
                json={"text": "테스트 메시지"},
                timeout=10,
            )

    def test_error_logged_not_raised(self, caplog):
        config = make_config(enabled=True, slack=True)
        with patch("requests.post", side_effect=Exception("웹훅 실패")):
            with caplog.at_level(logging.ERROR):
                send_slack(config, "메시지")  # 예외 전파 없음
        assert "슬랙 발송 실패" in caplog.text


# ---------------------------------------------------------------------------
# 단위 테스트 — send_alert
# ---------------------------------------------------------------------------

class TestSendAlert:
    def test_no_danger_no_send(self):
        config = make_config()
        results = [make_risk(level="정상")]
        with patch("modules.notifier.send_email") as mock_email, \
             patch("modules.notifier.send_slack") as mock_slack:
            send_alert(results, config)
            mock_email.assert_not_called()
            mock_slack.assert_not_called()

    def test_danger_sends_both(self):
        config = make_config(enabled=True, email=True, slack=True)
        results = [make_risk(level="위험")]
        with patch("modules.notifier.send_email") as mock_email, \
             patch("modules.notifier.send_slack") as mock_slack:
            send_alert(results, config)
            mock_email.assert_called_once()
            mock_slack.assert_called_once()

    def test_disabled_config_skips_all(self):
        config = make_config(enabled=False)
        results = [make_risk(level="위험")]
        with patch("modules.notifier.send_email") as mock_email, \
             patch("modules.notifier.send_slack") as mock_slack:
            send_alert(results, config)
            mock_email.assert_not_called()
            mock_slack.assert_not_called()


# ---------------------------------------------------------------------------
# 단위 테스트 — generate_weekly_report
# ---------------------------------------------------------------------------

class TestGenerateWeeklyReport:
    def test_empty_inputs(self):
        report = generate_weekly_report(pd.DataFrame(), [], {})
        assert "주간 리포트" in report
        assert "출결 데이터가 없습니다" in report
        assert "위험도 분석 결과가 없습니다" in report
        assert "감성 분석 데이터가 없습니다" in report

    def test_with_data(self):
        df = pd.DataFrame([
            {"name": "김훈련", "date": "2026-04-01", "status": "출석"},
            {"name": "이수강", "date": "2026-04-01", "status": "결석"},
        ])
        results = [
            make_risk(name="김훈련", level="정상", score=0.1),
            make_risk(name="이수강", level="위험", score=0.9),
        ]
        sentiment_map = {"김훈련": 0.2, "이수강": 0.8}
        report = generate_weekly_report(df, results, sentiment_map)
        assert "출결 요약" in report
        assert "위험도 분포" in report
        assert "감성 분석 요약" in report
        assert "이수강" in report
        assert "위험" in report

    def test_report_contains_level_distribution(self):
        results = [
            make_risk(level="정상"),
            make_risk(level="경고"),
            make_risk(level="위험"),
        ]
        report = generate_weekly_report(pd.DataFrame(), results, {})
        assert "정상" in report
        assert "경고" in report
        assert "위험" in report


# ---------------------------------------------------------------------------
# Property 14: 알림 메시지 필수 필드 포함
# Feature: campusguard-enhancement, Property 14: 알림 메시지 필수 필드 포함
# ---------------------------------------------------------------------------

# RiskResult 생성 전략
valid_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo", "Nd")),
    min_size=1,
    max_size=10,
)
valid_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_level_st = st.sampled_from(["정상", "경고", "위험"])
valid_rec_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo", "Nd", "Zs", "Po")),
    min_size=1,
    max_size=30,
)

risk_result_st = st.builds(
    RiskResult,
    name=valid_name_st,
    final_score=valid_score_st,
    final_level=st.just("위험"),  # 위험 등급으로 고정
    attendance_level=valid_level_st,
    sentiment_score=valid_score_st,
    recommendation=valid_rec_st,
)


@given(risk_result=risk_result_st)
@settings(max_examples=100)
def test_property_14_alert_message_required_fields(risk_result: RiskResult):
    # Feature: campusguard-enhancement, Property 14: 알림 메시지 필수 필드 포함
    """Validates: Requirements 6.3
    위험 등급 RiskResult에 대해 format_alert_message는
    훈련생 이름, RiskLevel, 위험 점수, 권고 사항을 모두 포함해야 한다.
    """
    msg = format_alert_message([risk_result])

    # 훈련생 이름 포함
    assert risk_result.name in msg, f"이름 누락: {risk_result.name!r} not in message"
    # RiskLevel 포함
    assert risk_result.final_level in msg, f"RiskLevel 누락: {risk_result.final_level!r} not in message"
    # 위험 점수 포함 (소수점 2자리 포맷)
    score_str = f"{risk_result.final_score:.2f}"
    assert score_str in msg, f"위험 점수 누락: {score_str!r} not in message"
    # 권고 사항 포함
    assert risk_result.recommendation in msg, f"권고 사항 누락: {risk_result.recommendation!r} not in message"


# ---------------------------------------------------------------------------
# Property 15: 알림 발송 조건 정확성
# Feature: campusguard-enhancement, Property 15: 알림 발송 조건 정확성
# ---------------------------------------------------------------------------

any_risk_result_st = st.builds(
    RiskResult,
    name=valid_name_st,
    final_score=valid_score_st,
    final_level=valid_level_st,
    attendance_level=valid_level_st,
    sentiment_score=valid_score_st,
    recommendation=valid_rec_st,
)


@given(risk_results=st.lists(any_risk_result_st, max_size=20))
@settings(max_examples=100)
def test_property_15_should_notify_condition(risk_results: list[RiskResult]):
    # Feature: campusguard-enhancement, Property 15: 알림 발송 조건 정확성
    """Validates: Requirements 6.1
    should_notify는 위험 등급 훈련생이 1명 이상 존재할 때만 True를 반환해야 한다.
    """
    result = should_notify(risk_results)
    has_danger = any(r.final_level == "위험" for r in risk_results)

    assert result == has_danger, (
        f"should_notify={result} but has_danger={has_danger}, "
        f"levels={[r.final_level for r in risk_results]}"
    )
