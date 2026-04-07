"""
태스크 4: 탈락 위험 예측
- 출결 분석 결과 + 감성 분석 결과를 결합하여 최종 위험도 산출
- 규칙 기반(Rule-based)으로 구현 — 외부 ML 모델 불필요
"""
from dataclasses import dataclass
from modules.attendance import AttendanceResult


# 위험도 가중치
RISK_WEIGHTS = {
    "attendance": 0.6,   # 출결이 더 객관적 지표
    "sentiment": 0.4,
}

LEVEL_SCORE = {
    "위험": 1.0,
    "경고": 0.5,
    "정상": 0.0,
}


@dataclass
class RiskResult:
    name: str
    final_score: float      # 0.0 ~ 1.0
    final_level: str        # "정상" / "경고" / "위험"
    attendance_level: str
    sentiment_score: float
    recommendation: str


def predict_risk(
    attendance: AttendanceResult,
    sentiment_score: float = 0.0,
) -> RiskResult:
    """
    출결 + 감성 점수를 결합하여 최종 탈락 위험도 산출.

    Args:
        attendance: AttendanceResult 객체
        sentiment_score: 감성 분석의 부정 점수 (0.0~1.0)

    Returns:
        RiskResult
    """
    att_score = LEVEL_SCORE.get(attendance.risk_level, 0.0)
    final = (att_score * RISK_WEIGHTS["attendance"]
             + sentiment_score * RISK_WEIGHTS["sentiment"])

    if final >= 0.6:
        level = "위험"
        rec = "즉시 1:1 상담 진행 및 상담일지 작성 필요"
    elif final >= 0.3:
        level = "경고"
        rec = "이번 주 내 상담 일정 잡기 권장"
    else:
        level = "정상"
        rec = "정기 모니터링 유지"

    return RiskResult(
        name=attendance.name,
        final_score=round(final, 2),
        final_level=level,
        attendance_level=attendance.risk_level,
        sentiment_score=sentiment_score,
        recommendation=rec,
    )


def predict_all(
    attendance_results: list[AttendanceResult],
    sentiment_map: dict[str, float] = None,
) -> list[RiskResult]:
    """
    전체 훈련생 위험도 예측.

    Args:
        attendance_results: analyze_all() 결과
        sentiment_map: {이름: 부정점수} 딕셔너리 (없으면 0.0 사용)

    Returns:
        list[RiskResult]
    """
    sentiment_map = sentiment_map or {}
    return [
        predict_risk(att, sentiment_map.get(att.name, 0.0))
        for att in attendance_results
    ]
