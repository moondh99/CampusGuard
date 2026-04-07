"""
NotificationService — 이메일/슬랙 알림 및 주간 리포트 생성
- smtplib 이메일 발송, Slack Incoming Webhook POST 요청
- .env 기반 설정 로드
- 발송 실패 시 logging.error 기록, 예외 전파 없음
"""
from __future__ import annotations

import logging
import os
import smtplib
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import requests

from modules.risk_predictor import RiskResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NotificationConfig
# ---------------------------------------------------------------------------

@dataclass
class NotificationConfig:
    enabled: bool = False
    email_enabled: bool = False
    email_host: str = ""
    email_port: int = 587
    email_user: str = ""
    email_password: str = ""
    email_to: str = ""
    slack_enabled: bool = False
    slack_webhook_url: str = ""


def load_config_from_env() -> NotificationConfig:
    """환경 변수(.env)에서 NotificationConfig를 로드한다."""
    enabled_str = os.environ.get("NOTIFICATION_ENABLED", "false").strip().lower()
    enabled = enabled_str not in ("false", "0", "no", "")

    if not enabled:
        return NotificationConfig(enabled=False)

    return NotificationConfig(
        enabled=True,
        email_enabled=os.environ.get("EMAIL_ENABLED", "false").strip().lower()
            not in ("false", "0", "no", ""),
        email_host=os.environ.get("EMAIL_HOST", ""),
        email_port=int(os.environ.get("EMAIL_PORT", "587")),
        email_user=os.environ.get("EMAIL_USER", ""),
        email_password=os.environ.get("EMAIL_PASSWORD", ""),
        email_to=os.environ.get("EMAIL_TO", ""),
        slack_enabled=os.environ.get("SLACK_ENABLED", "false").strip().lower()
            not in ("false", "0", "no", ""),
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
    )


# ---------------------------------------------------------------------------
# 알림 조건 및 메시지 포맷팅
# ---------------------------------------------------------------------------

def should_notify(risk_results: list[RiskResult]) -> bool:
    """위험 등급 훈련생이 1명 이상 존재하면 True를 반환한다."""
    return any(r.final_level == "위험" for r in risk_results)


def format_alert_message(risk_results: list[RiskResult]) -> str:
    """위험 등급 훈련생에 대한 알림 메시지를 포맷팅한다.

    메시지에 훈련생 이름, RiskLevel, 위험 점수, 권고 사항을 포함한다.
    """
    danger_results = [r for r in risk_results if r.final_level == "위험"]

    if not danger_results:
        return "위험 등급 훈련생이 없습니다."

    lines = ["[CampusGuard 위험 알림] 위험 등급 훈련생이 감지되었습니다.\n"]
    for r in danger_results:
        lines.append(f"- 이름: {r.name}")
        lines.append(f"  위험 등급(RiskLevel): {r.final_level}")
        lines.append(f"  위험 점수: {r.final_score:.2f}")
        lines.append(f"  권고 사항: {r.recommendation}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 발송 함수
# ---------------------------------------------------------------------------

def send_email(config: NotificationConfig, subject: str, body: str) -> None:
    """smtplib를 사용하여 이메일을 발송한다.

    발송 실패 시 logging.error로 기록하고 예외를 전파하지 않는다.
    NOTIFICATION_ENABLED=false 또는 email_enabled=false 시 즉시 반환한다.
    """
    if not config.enabled or not config.email_enabled:
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = config.email_user
        msg["To"] = config.email_to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(config.email_host, config.email_port) as server:
            server.starttls()
            server.login(config.email_user, config.email_password)
            server.sendmail(config.email_user, config.email_to, msg.as_string())
    except Exception as exc:
        logger.error("이메일 발송 실패: %s", exc)


def send_slack(config: NotificationConfig, message: str) -> None:
    """Slack Incoming Webhook으로 메시지를 발송한다.

    발송 실패 시 logging.error로 기록하고 예외를 전파하지 않는다.
    NOTIFICATION_ENABLED=false 또는 slack_enabled=false 시 즉시 반환한다.
    """
    if not config.enabled or not config.slack_enabled:
        return

    try:
        response = requests.post(
            config.slack_webhook_url,
            json={"text": message},
            timeout=10,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.error("슬랙 발송 실패: %s", exc)


def send_alert(risk_results: list[RiskResult], config: NotificationConfig) -> None:
    """위험 훈련생 알림을 발송한다. 오류 시 로그 기록 후 계속 진행한다."""
    if not config.enabled:
        return

    if not should_notify(risk_results):
        return

    message = format_alert_message(risk_results)
    subject = "[CampusGuard] 위험 등급 훈련생 알림"

    send_email(config, subject, message)
    send_slack(config, message)


# ---------------------------------------------------------------------------
# 주간 리포트
# ---------------------------------------------------------------------------

def generate_weekly_report(
    attendance_df: pd.DataFrame,
    risk_results: list[RiskResult],
    sentiment_map: dict[str, float],
) -> str:
    """주간 리포트 텍스트를 생성한다.

    전체 훈련생 출결 요약, 위험도 분포, 감성 분석 요약을 포함한다.
    """
    lines = ["=" * 50, "CampusGuard 주간 리포트", "=" * 50, ""]

    # 1. 출결 요약
    lines.append("## 출결 요약")
    if attendance_df.empty:
        lines.append("출결 데이터가 없습니다.")
    else:
        total_students = attendance_df["name"].nunique() if "name" in attendance_df.columns else 0
        lines.append(f"- 전체 훈련생 수: {total_students}명")

        if "status" in attendance_df.columns:
            status_counts = attendance_df["status"].value_counts()
            for status, count in status_counts.items():
                lines.append(f"  - {status}: {count}건")
    lines.append("")

    # 2. 위험도 분포
    lines.append("## 위험도 분포")
    if not risk_results:
        lines.append("위험도 분석 결과가 없습니다.")
    else:
        level_counts: dict[str, int] = {"정상": 0, "경고": 0, "위험": 0}
        for r in risk_results:
            level_counts[r.final_level] = level_counts.get(r.final_level, 0) + 1
        total = len(risk_results)
        for level, count in level_counts.items():
            pct = count / total * 100 if total > 0 else 0
            lines.append(f"  - {level}: {count}명 ({pct:.1f}%)")

        danger_list = [r.name for r in risk_results if r.final_level == "위험"]
        if danger_list:
            lines.append(f"  ※ 위험 훈련생: {', '.join(danger_list)}")
    lines.append("")

    # 3. 감성 분석 요약
    lines.append("## 감성 분석 요약")
    if not sentiment_map:
        lines.append("감성 분석 데이터가 없습니다.")
    else:
        scores = list(sentiment_map.values())
        avg_score = sum(scores) / len(scores)
        high_risk = [name for name, score in sentiment_map.items() if score >= 0.6]
        lines.append(f"- 평균 부정 감성 점수: {avg_score:.2f}")
        lines.append(f"- 고위험 감성 훈련생 수: {len(high_risk)}명")
        if high_risk:
            lines.append(f"  ※ 해당 훈련생: {', '.join(high_risk)}")
    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)
