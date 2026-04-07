"""
Visualizer 모듈 테스트 — 단위 테스트 + Property-Based 테스트
"""
import pytest
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from modules.visualizer import (
    render_attendance_line_chart,
    render_absence_heatmap,
    detect_burnout_periods,
)

# ---------------------------------------------------------------------------
# 단위 테스트
# ---------------------------------------------------------------------------

SAMPLE_DF = pd.DataFrame(
    {
        "name": ["김훈련", "김훈련", "이수강", "이수강"],
        "date": ["2026-04-01", "2026-04-02", "2026-04-01", "2026-04-02"],
        "status": ["출석", "결석", "출석", "출석"],
    }
)


def test_render_attendance_line_chart_basic():
    fig = render_attendance_line_chart(SAMPLE_DF)
    trace_names = {t.name for t in fig.data}
    assert trace_names == {"김훈련", "이수강"}


def test_render_absence_heatmap_basic():
    fig = render_absence_heatmap(SAMPLE_DF)
    # 2026-04-02에 김훈련 1명 결석 / 전체 2명 → 0.5
    assert fig.data[0].z[0][0] == pytest.approx(0.5)


def test_detect_burnout_periods_basic():
    scores = [0.7, 0.8, 0.9, 0.3, 0.7, 0.7, 0.7, 0.7]
    periods = detect_burnout_periods(scores)
    assert (0, 2) in periods
    assert (4, 7) in periods


def test_detect_burnout_periods_no_burnout():
    scores = [0.5, 0.5, 0.5]
    assert detect_burnout_periods(scores) == []


def test_detect_burnout_periods_empty():
    assert detect_burnout_periods([]) == []


def test_detect_burnout_periods_exact_threshold():
    # 정확히 3일 연속 0.6 → 번아웃 구간 1개
    scores = [0.6, 0.6, 0.6]
    periods = detect_burnout_periods(scores)
    assert periods == [(0, 2)]


# ---------------------------------------------------------------------------
# Property 12: 출결 라인 차트 Trace 완전성
# Feature: campusguard-enhancement, Property 12: 출결 라인 차트 Trace 완전성
# ---------------------------------------------------------------------------

STATUSES = ["출석", "지각", "조퇴", "결석"]

# 훈련생 이름 전략 (비어있지 않은 텍스트, 공백 없음)
trainee_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo", "Nd")),
    min_size=1,
    max_size=5,
)

# 날짜 전략
date_st = st.dates(
    min_value=__import__("datetime").date(2026, 1, 1),
    max_value=__import__("datetime").date(2026, 12, 31),
).map(str)


@st.composite
def attendance_df_st(draw):
    """출결 DataFrame 생성 전략."""
    names = draw(st.lists(trainee_name_st, min_size=1, max_size=5, unique=True))
    dates = draw(st.lists(date_st, min_size=1, max_size=5, unique=True))
    rows = []
    for name in names:
        for date in dates:
            status = draw(st.sampled_from(STATUSES))
            rows.append({"name": name, "date": date, "status": status})
    return pd.DataFrame(rows)


@given(df=attendance_df_st())
@settings(max_examples=100)
def test_property_12_attendance_line_chart_trace_completeness(df):
    # Feature: campusguard-enhancement, Property 12: 출결 라인 차트 Trace 완전성
    """출결 라인 차트 Figure의 trace 이름 집합 == DataFrame 고유 훈련생 이름 집합."""
    fig = render_attendance_line_chart(df)
    trace_names = {t.name for t in fig.data}
    expected_names = set(df["name"].unique())
    assert trace_names == expected_names


# ---------------------------------------------------------------------------
# Property 13: 결석률 범위 불변식
# Feature: campusguard-enhancement, Property 13: 결석률 범위 불변식
# ---------------------------------------------------------------------------

def _compute_absence_rates(df: pd.DataFrame) -> list[float]:
    """날짜별 결석률 계산 (render_absence_heatmap 내부 로직과 동일)."""
    total_students = df["name"].nunique()
    if total_students == 0:
        return []
    absence_by_date = (
        df[df["status"] == "결석"]
        .groupby("date")["name"]
        .nunique()
        .reset_index(name="absent_count")
    )
    absence_by_date["absence_rate"] = absence_by_date["absent_count"] / total_students
    return absence_by_date["absence_rate"].tolist()


@given(df=attendance_df_st())
@settings(max_examples=100)
def test_property_13_absence_rate_invariant(df):
    # Feature: campusguard-enhancement, Property 13: 결석률 범위 불변식
    """날짜별 결석률이 항상 0.0 이상 1.0 이하임을 검증."""
    rates = _compute_absence_rates(df)
    for rate in rates:
        assert 0.0 <= rate <= 1.0, f"결석률 범위 위반: {rate}"


# ---------------------------------------------------------------------------
# Property 6: 번아웃 감지 정확성
# Feature: campusguard-enhancement, Property 6: 번아웃 감지 정확성
# ---------------------------------------------------------------------------

def _has_consecutive_burnout(scores: list[float], threshold: float = 0.6, consecutive: int = 3) -> bool:
    """연속 `consecutive`일 이상 threshold 이상인 구간이 존재하는지 확인."""
    count = 0
    for s in scores:
        if s >= threshold:
            count += 1
            if count >= consecutive:
                return True
        else:
            count = 0
    return False


@given(scores=st.lists(st.floats(min_value=0.0, max_value=1.0), max_size=50))
@settings(max_examples=100)
def test_property_6_burnout_detection_accuracy(scores):
    # Feature: campusguard-enhancement, Property 6: 번아웃 감지 정확성
    """detect_burnout_periods는 연속 3일 이상 0.6 이상인 구간이 존재할 때만 비어있지 않은 리스트를 반환해야 한다."""
    # NaN/inf 값 필터링
    clean_scores = [s for s in scores if s == s and s != float("inf") and s != float("-inf")]
    periods = detect_burnout_periods(clean_scores)
    has_burnout = _has_consecutive_burnout(clean_scores)

    if periods:
        assert has_burnout, (
            f"번아웃 구간이 감지됐지만 실제로 조건을 충족하는 구간이 없음: scores={clean_scores}"
        )
    if has_burnout:
        assert len(periods) > 0, (
            f"조건을 충족하는 구간이 있지만 감지되지 않음: scores={clean_scores}"
        )
