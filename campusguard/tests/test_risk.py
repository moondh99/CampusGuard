"""
태스크 4 / 7 단위 테스트: 탈락 위험 예측
- API 호출 없음, 순수 로직 테스트
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.attendance import AttendanceResult
from modules.risk_predictor import predict_risk, predict_all, MLRiskPredictor


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
    raw = att_score * RISK_WEIGHTS["attendance"] + sentiment * RISK_WEIGHTS["sentiment"]
    expected = round(raw, 2)
    assert result.final_score == expected
    # 레벨 임계값 검증 — 구현과 동일하게 반올림 전 값으로 비교
    if raw >= 0.6:
        assert result.final_level == "위험"
    elif raw >= 0.3:
        assert result.final_level == "경고"
    else:
        assert result.final_level == "정상"


# ── Property 1: RiskPredictor 피처 추출 완전성 ────────────────────────────────

import pandas as pd

_EXPECTED_FEATURE_KEYS = {
    "recent_14d_absent_rate",
    "total_absent_rate",
    "late_change_rate",
    "sentiment_7d_ma",
    "converted_absent",
}


@given(
    statuses=st_hyp.lists(
        st_hyp.sampled_from(["출석", "지각", "조퇴", "결석"]),
        min_size=0,
        max_size=30,
    ),
    sentiment_history=st_hyp.lists(
        st_hyp.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=0,
        max_size=30,
    ),
)
@settings(max_examples=100)
def test_feature_extraction_completeness(statuses, sentiment_history):
    # Feature: campusguard-enhancement, Property 1: RiskPredictor 피처 추출 완전성
    # Validates: Requirements 1.2, 1.3, 1.4
    predictor = MLRiskPredictor()
    name = "테스트"

    if statuses:
        dates = pd.date_range("2024-01-01", periods=len(statuses), freq="D")
        df = pd.DataFrame({
            "name": [name] * len(statuses),
            "date": dates,
            "status": statuses,
        })
    else:
        df = pd.DataFrame(columns=["name", "date", "status"])

    features = predictor.extract_features(df, name, sentiment_history)

    assert isinstance(features, dict), "피처 추출 결과는 dict여야 한다"
    assert _EXPECTED_FEATURE_KEYS == set(features.keys()), (
        f"피처 키 불일치: 기대={_EXPECTED_FEATURE_KEYS}, 실제={set(features.keys())}"
    )


# ── Property 2: RiskResult 유효성 불변식 ─────────────────────────────────────

_VALID_LEVELS = {"정상", "경고", "위험"}


@given(
    attendance=st_hyp.builds(
        AttendanceResult,
        name=st_hyp.just("테스트"),
        late_count=st_hyp.integers(min_value=0, max_value=30),
        early_leave_count=st_hyp.integers(min_value=0, max_value=30),
        absent_count=st_hyp.integers(min_value=0, max_value=30),
        converted_absent=st_hyp.floats(min_value=0.0, max_value=30.0, allow_nan=False),
        risk_level=st_hyp.sampled_from(["정상", "경고", "위험"]),
        message=st_hyp.just(""),
    ),
    sentiment=st_hyp.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_risk_result_validity_invariant(attendance, sentiment):
    # Feature: campusguard-enhancement, Property 2: RiskResult 유효성 불변식
    # Validates: Requirements 1.7
    result = predict_risk(attendance, sentiment_score=sentiment)

    assert result.final_level in _VALID_LEVELS, (
        f"final_level '{result.final_level}'은 {_VALID_LEVELS} 중 하나여야 한다"
    )
    assert 0.0 <= result.final_score <= 1.0, (
        f"final_score {result.final_score}는 [0, 1] 범위여야 한다"
    )


# ── ML 단위 테스트 ─────────────────────────────────────────────────────────────

def test_ml_predictor_not_trained_initially():
    """새로 생성된 MLRiskPredictor는 is_trained() == False"""
    predictor = MLRiskPredictor()
    assert predictor.is_trained() is False


def test_ml_predictor_fallback_when_not_trained():
    """ML 모델 미학습 시 predict_risk는 규칙 기반 폴백으로 동작"""
    from modules.risk_predictor import _ml_predictor
    # 싱글턴이 학습되지 않은 상태에서 규칙 기반 결과와 동일해야 함
    att = make_attendance("테스트", "위험")
    result = predict_risk(att, sentiment_score=0.9)
    # 규칙 기반: 1.0 * 0.6 + 0.9 * 0.4 = 0.96 → 위험
    assert result.final_level == "위험"
    assert isinstance(result.final_score, float)


def test_ml_predictor_train_and_predict():
    """충분한 학습 데이터로 학습 후 is_trained() == True, predict() 동작"""
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        pytest.skip("scikit-learn 미설치")

    predictor = MLRiskPredictor()
    assert predictor.is_trained() is False

    # 10개 이상 샘플로 학습
    X = np.array([
        [0.0, 0.0, 0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0, 0.2, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.4, 1.0, 0.7, 5.0],
        [0.8, 0.7, 2.0, 0.9, 8.0],
        [0.0, 0.0, 0.0, 0.1, 0.5],
        [0.2, 0.2, 0.5, 0.4, 2.0],
        [0.9, 0.8, 3.0, 0.95, 10.0],
        [0.0, 0.0, 0.0, 0.05, 0.0],
        [0.3, 0.3, 0.8, 0.6, 3.0],
    ])
    y = ["정상", "정상", "정상", "경고", "위험", "정상", "경고", "위험", "정상", "경고"]

    predictor.train(X, y)
    assert predictor.is_trained() is True

    # predict() 호출 시 유효한 (level, score) 반환
    features = {
        "recent_14d_absent_rate": 0.8,
        "total_absent_rate": 0.7,
        "late_change_rate": 2.0,
        "sentiment_7d_ma": 0.9,
        "converted_absent": 8.0,
    }
    level, score = predictor.predict(features)
    assert level in {"정상", "경고", "위험"}
    assert 0.0 <= score <= 1.0


def test_ml_predictor_insufficient_data_stays_untrained():
    """학습 데이터 부족(< 10개) 시 is_trained() == False 유지"""
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy 미설치")

    predictor = MLRiskPredictor()
    X = np.array([[0.0, 0.0, 0.0, 0.1, 0.0]] * 5)
    y = ["정상"] * 5
    predictor.train(X, y)
    assert predictor.is_trained() is False
