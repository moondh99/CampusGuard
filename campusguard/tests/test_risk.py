"""
태스크 4 단위 테스트: 탈락 위험 예측
- API 호출 없음, 순수 로직 테스트
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.attendance import AttendanceResult
from modules.risk_predictor import predict_risk, predict_all


def make_attendance(name: str, level: str) -> AttendanceResult:
    return AttendanceResult(
        name=name,
        late_count=0,
        early_leave_count=0,
        absent_count=0,
        converted_absent=0.0,
        risk_level=level,
        message="",
    )


def test_danger_attendance_high_sentiment():
    """출결 위험 + 감성 고위험 → 최종 위험"""
    att = make_attendance("김위험", "위험")
    result = predict_risk(att, sentiment_score=0.9)
    assert result.final_level == "위험"
    assert result.final_score >= 0.6


def test_normal_attendance_low_sentiment():
    """출결 정상 + 감성 낮음 → 최종 정상"""
    att = make_attendance("이정상", "정상")
    result = predict_risk(att, sentiment_score=0.1)
    assert result.final_level == "정상"
    assert result.final_score < 0.3


def test_warning_attendance_medium_sentiment():
    """출결 경고 + 감성 중간 → 경고 또는 위험"""
    att = make_attendance("박경고", "경고")
    result = predict_risk(att, sentiment_score=0.5)
    assert result.final_level in ("경고", "위험")


def test_default_sentiment_zero():
    """감성 점수 미입력 시 0.0으로 처리"""
    att = make_attendance("최기본", "정상")
    result = predict_risk(att)
    assert result.sentiment_score == 0.0


def test_predict_all():
    """전체 예측 시 모든 훈련생 결과 반환"""
    attendances = [
        make_attendance("A", "위험"),
        make_attendance("B", "정상"),
    ]
    sentiment_map = {"A": 0.8, "B": 0.1}
    results = predict_all(attendances, sentiment_map)
    assert len(results) == 2
    names = [r.name for r in results]
    assert "A" in names and "B" in names


def test_predict_all_no_sentiment_map():
    """sentiment_map 없이도 동작"""
    attendances = [make_attendance("A", "경고")]
    results = predict_all(attendances)
    assert results[0].sentiment_score == 0.0


def test_recommendation_content():
    """위험 등급에 상담 권고 문구 포함"""
    att = make_attendance("김위험", "위험")
    result = predict_risk(att, sentiment_score=1.0)
    assert "상담" in result.recommendation


# ── Property-Based Tests ──────────────────────────────────────────────────────

from hypothesis import given, settings
import hypothesis.strategies as st_hyp


@given(
    level=st_hyp.sampled_from(["정상", "경고", "위험"]),
    sentiment=st_hyp.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_risk_score_formula(level, sentiment):
    """Property 4: 위험도 점수 계산 정확성"""
    # Feature: campusguard-deployment, Property 4: 위험도 점수 계산 정확성
    # Validates: Requirements 5.2
    from modules.risk_predictor import LEVEL_SCORE, RISK_WEIGHTS
    att = make_attendance("테스트", level)
    result = predict_risk(att, sentiment_score=sentiment)
    att_score = LEVEL_SCORE[level]
    expected = round(att_score * RISK_WEIGHTS["attendance"] + sentiment * RISK_WEIGHTS["sentiment"], 2)
    assert result.final_score == expected
    # 레벨 임계값 검증
    if expected >= 0.6:
        assert result.final_level == "위험"
    elif expected >= 0.3:
        assert result.final_level == "경고"
    else:
        assert result.final_level == "정상"
