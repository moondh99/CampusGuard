"""
태스크 4 / 7: 탈락 위험 예측
- 출결 분석 결과 + 감성 분석 결과를 결합하여 최종 위험도 산출
- 규칙 기반(Rule-based) + ML 레이어 (scikit-learn 선택적 의존성)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from modules.attendance import AttendanceResult

logger = logging.getLogger(__name__)

# ── scikit-learn 선택적 임포트 ────────────────────────────────────────────────
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    np = None  # type: ignore

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

_FEATURE_KEYS = [
    "recent_14d_absent_rate",
    "total_absent_rate",
    "late_change_rate",
    "sentiment_7d_ma",
    "converted_absent",
]


@dataclass
class RiskResult:
    name: str
    final_score: float      # 0.0 ~ 1.0
    final_level: str        # "정상" / "경고" / "위험"
    attendance_level: str
    sentiment_score: float
    recommendation: str


# ── MLRiskPredictor ───────────────────────────────────────────────────────────

class MLRiskPredictor:
    """
    ML 기반 위험도 예측기.
    scikit-learn 미설치 또는 학습 데이터 부족 시 is_trained() == False.
    """

    _MIN_SAMPLES = 10

    def __init__(self) -> None:
        self._model = None
        self._label_encoder = None
        self._trained = False

    # ── 피처 추출 ─────────────────────────────────────────────────────────────

    def extract_features(
        self,
        df: pd.DataFrame,
        name: str,
        sentiment_history: list[float],
    ) -> dict:
        """
        5개 피처 딕셔너리 반환.

        Args:
            df: 전체 출결 DataFrame (컬럼: name, date, status)
            name: 훈련생 이름
            sentiment_history: 감성 점수 시계열 (최신 순 또는 날짜 순)

        Returns:
            dict with keys: recent_14d_absent_rate, total_absent_rate,
                            late_change_rate, sentiment_7d_ma, converted_absent
        """
        student_df = df[df["name"] == name].copy() if not df.empty else pd.DataFrame()

        # ── recent_14d_absent_rate ────────────────────────────────────────────
        if not student_df.empty and "date" in student_df.columns:
            try:
                student_df["date"] = pd.to_datetime(student_df["date"])
                latest = student_df["date"].max()
                cutoff = latest - pd.Timedelta(days=14)
                recent_df = student_df[student_df["date"] >= cutoff]
                recent_14d_absent_rate = (
                    (recent_df["status"] == "결석").sum() / len(recent_df)
                    if len(recent_df) > 0 else 0.0
                )
            except Exception:
                recent_14d_absent_rate = 0.0
        else:
            recent_14d_absent_rate = 0.0

        # ── total_absent_rate ─────────────────────────────────────────────────
        if not student_df.empty and "status" in student_df.columns:
            total_absent_rate = (
                (student_df["status"] == "결석").sum() / len(student_df)
                if len(student_df) > 0 else 0.0
            )
        else:
            total_absent_rate = 0.0

        # ── late_change_rate ──────────────────────────────────────────────────
        if not student_df.empty and "date" in student_df.columns and "status" in student_df.columns:
            try:
                student_df["date"] = pd.to_datetime(student_df["date"])
                latest = student_df["date"].max()
                cutoff_7d = latest - pd.Timedelta(days=7)
                recent_7d_df = student_df[student_df["date"] >= cutoff_7d]
                recent_7d_late = (recent_7d_df["status"] == "지각").sum()
                total_late = (student_df["status"] == "지각").sum()
                total_days = len(student_df)
                avg_late = total_late / total_days if total_days > 0 else 0.0
                late_change_rate = (
                    float(recent_7d_late) / avg_late if avg_late > 0 else 0.0
                )
            except Exception:
                late_change_rate = 0.0
        else:
            late_change_rate = 0.0

        # ── sentiment_7d_ma ───────────────────────────────────────────────────
        if sentiment_history:
            window = sentiment_history[-7:]
            sentiment_7d_ma = sum(window) / len(window)
        else:
            sentiment_7d_ma = 0.0

        # ── converted_absent (AttendanceResult 기반) ──────────────────────────
        # df에서 직접 계산
        if not student_df.empty and "status" in student_df.columns:
            late_count = int((student_df["status"] == "지각").sum())
            early_leave_count = int((student_df["status"] == "조퇴").sum())
            absent_count = int((student_df["status"] == "결석").sum())
            converted_absent = absent_count + (late_count + early_leave_count) / 3.0
        else:
            converted_absent = 0.0

        return {
            "recent_14d_absent_rate": float(recent_14d_absent_rate),
            "total_absent_rate": float(total_absent_rate),
            "late_change_rate": float(late_change_rate),
            "sentiment_7d_ma": float(sentiment_7d_ma),
            "converted_absent": float(converted_absent),
        }

    # ── 학습 ──────────────────────────────────────────────────────────────────

    def train(self, X, y) -> None:
        """
        모델 학습.

        Args:
            X: 피처 배열 (n_samples, 5)
            y: 레이블 배열 ("정상"/"경고"/"위험")
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("scikit-learn 미설치 — ML 학습 불가")
            return

        n_samples = len(X) if hasattr(X, "__len__") else 0
        if n_samples < self._MIN_SAMPLES:
            logger.warning("학습 데이터 부족 (%d < %d) — ML 학습 불가", n_samples, self._MIN_SAMPLES)
            return

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self._model = RandomForestClassifier(n_estimators=50, random_state=42)
        self._model.fit(X, y_encoded)
        self._trained = True

    # ── 예측 ──────────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> tuple[str, float]:
        """
        ML 예측.

        Args:
            features: extract_features() 반환 딕셔너리

        Returns:
            (level, score) — level: "정상"/"경고"/"위험", score: 0.0~1.0
        """
        if not self._trained or self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다.")

        x = np.array([[features[k] for k in _FEATURE_KEYS]])
        proba = self._model.predict_proba(x)[0]
        pred_idx = int(proba.argmax())
        level = self._label_encoder.inverse_transform([pred_idx])[0]

        # 위험 클래스 확률을 점수로 사용
        classes = list(self._label_encoder.classes_)
        if "위험" in classes:
            danger_idx = classes.index("위험")
            score = float(proba[danger_idx])
        else:
            score = float(proba.max())

        # 점수 범위 보정
        score = max(0.0, min(1.0, score))
        return level, score

    # ── 상태 확인 ─────────────────────────────────────────────────────────────

    def is_trained(self) -> bool:
        """ML 모델이 학습 완료 상태인지 반환."""
        return self._trained and _SKLEARN_AVAILABLE


# ── 모듈 레벨 MLRiskPredictor 싱글턴 ─────────────────────────────────────────
_ml_predictor = MLRiskPredictor()


# ── 기존 인터페이스 (변경 없음) ───────────────────────────────────────────────

def predict_risk(
    attendance: AttendanceResult,
    sentiment_score: float = 0.0,
) -> RiskResult:
    """
    출결 + 감성 점수를 결합하여 최종 탈락 위험도 산출.
    ML 모델이 학습된 경우 ML 예측, 아니면 규칙 기반 폴백.

    Args:
        attendance: AttendanceResult 객체
        sentiment_score: 감성 분석의 부정 점수 (0.0~1.0)

    Returns:
        RiskResult
    """
    if _ml_predictor.is_trained():
        # ML 예측 경로
        features = {
            "recent_14d_absent_rate": 0.0,
            "total_absent_rate": 0.0,
            "late_change_rate": 0.0,
            "sentiment_7d_ma": sentiment_score,
            "converted_absent": attendance.converted_absent,
        }
        try:
            level, score = _ml_predictor.predict(features)
            score = max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning("ML 예측 실패, 규칙 기반 폴백: %s", e)
            return _rule_based_predict(attendance, sentiment_score)
    else:
        # 규칙 기반 폴백
        return _rule_based_predict(attendance, sentiment_score)

    rec = _make_recommendation(level)
    return RiskResult(
        name=attendance.name,
        final_score=round(score, 2),
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


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _rule_based_predict(
    attendance: AttendanceResult,
    sentiment_score: float,
) -> RiskResult:
    """규칙 기반 가중합 예측 (폴백)."""
    att_score = LEVEL_SCORE.get(attendance.risk_level, 0.0)
    final = (att_score * RISK_WEIGHTS["attendance"]
             + sentiment_score * RISK_WEIGHTS["sentiment"])

    if final >= 0.6:
        level = "위험"
    elif final >= 0.3:
        level = "경고"
    else:
        level = "정상"

    rec = _make_recommendation(level)
    return RiskResult(
        name=attendance.name,
        final_score=round(final, 2),
        final_level=level,
        attendance_level=attendance.risk_level,
        sentiment_score=sentiment_score,
        recommendation=rec,
    )


def _make_recommendation(level: str) -> str:
    if level == "위험":
        return "즉시 1:1 상담 진행 및 상담일지 작성 필요"
    elif level == "경고":
        return "이번 주 내 상담 일정 잡기 권장"
    return "정기 모니터링 유지"
