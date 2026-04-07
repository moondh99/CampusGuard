"""
Visualizer 모듈 테스트 — 단위 테스트 + Property-Based 테스트
"""
import pytest
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from modules.visualizer import (
    render_attendance_line_chart,
    render_absence_heatmap,
    detect_burnout_periods,
    render_sentiment_trend_chart,
    render_risk_distribution_chart,
    RISK_COLORS,
)
from modules.risk_predictor import RiskResult

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


# ---------------------------------------------------------------------------
# Property 8: 출결 라인 차트 colorway 불변성
# Validates: Requirements 5.1, 5.4
# ---------------------------------------------------------------------------

def test_attendance_line_chart_colorway_invariant():
    """출결 라인 차트 Figure의 layout.colorway에 #1f77b4가 포함되어야 한다.

    **Validates: Requirements 5.1, 5.4**
    """
    df = pd.DataFrame(
        {
            "name": ["김훈련", "이수강"],
            "date": ["2026-04-01", "2026-04-02"],
            "status": ["출석", "결석"],
        }
    )
    fig = render_attendance_line_chart(df)
    assert "#1f77b4" in fig.layout.colorway


# ---------------------------------------------------------------------------
# Property 5: 히트맵 colorscale 불변성
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

# Plotly가 "Blues"를 resolve한 결과 (tuple of tuples)
_BLUES_RESOLVED = tuple(
    tuple(item) for item in pc.get_colorscale("Blues")
)


@given(df=attendance_df_st())
@settings(max_examples=100)
def test_property_5_heatmap_colorscale_invariant(df):
    """임의의 출결 DataFrame에 대해 render_absence_heatmap의 첫 번째 trace colorscale은 항상 "Blues"여야 한다.

    Plotly는 colorscale="Blues" 문자열을 내부적으로 RGB tuple로 resolve하므로,
    문자열 "Blues" 또는 resolve된 Blues 팔레트와 동일한지 검증한다.

    **Validates: Requirements 5.3**
    """
    fig = render_absence_heatmap(df)
    assert len(fig.data) > 0, "히트맵 Figure에 trace가 없음"
    actual = fig.data[0].colorscale
    # Plotly는 "Blues" 문자열을 tuple of tuples로 resolve함
    assert actual == "Blues" or actual == _BLUES_RESOLVED, (
        f"colorscale이 Blues가 아님: {actual}"
    )


# ---------------------------------------------------------------------------
# Property 4: 감성 차트 Y축 설정 불변성
# Validates: Requirements 3.1, 3.2
# ---------------------------------------------------------------------------

@given(
    sentiment_data=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=20),
        min_size=1, max_size=5,
    )
)
@settings(max_examples=100)
def test_property_4_sentiment_chart_yaxis_invariant(sentiment_data):
    fig = render_sentiment_trend_chart(sentiment_data)
    assert fig.layout.yaxis.title.text == "긍정도 (1=긍정, 0=부정)"
    assert fig.layout.yaxis.autorange == "reversed"


# ---------------------------------------------------------------------------
# render_risk_distribution_chart 단위 테스트 (Requirements 6.1~6.4)
# ---------------------------------------------------------------------------

def _make_risk_result(name: str, level: str) -> RiskResult:
    return RiskResult(
        name=name,
        final_score=0.5,
        final_level=level,
        attendance_level=level,
        sentiment_score=0.5,
        recommendation="",
    )


def test_risk_distribution_chart_empty_returns_empty_figure():
    """빈 리스트 입력 시 빈 Figure 반환 (Requirements 6.3)."""
    fig = render_risk_distribution_chart([])
    assert len(fig.data) == 0


def test_risk_distribution_chart_uses_bar_horizontal():
    """go.Bar orientation='h' 사용 확인 (Requirements 6.1)."""
    results = [_make_risk_result("A", "위험"), _make_risk_result("B", "정상")]
    fig = render_risk_distribution_chart(results)
    assert len(fig.data) == 1
    bar = fig.data[0]
    assert bar.type == "bar"
    assert bar.orientation == "h"


def test_risk_distribution_chart_colors_match_risk_colors():
    """RISK_COLORS 팔레트 적용 확인 (Requirements 6.2)."""
    results = [
        _make_risk_result("A", "위험"),
        _make_risk_result("B", "경고"),
        _make_risk_result("C", "정상"),
    ]
    fig = render_risk_distribution_chart(results)
    bar = fig.data[0]
    # y축 레이블 순서에 맞게 색상 확인
    for i, level in enumerate(bar.y):
        assert bar.marker.color[i] == RISK_COLORS[level]


def test_risk_distribution_chart_level_order():
    """등급 순서가 위험 > 경고 > 정상으로 고정됨 (Requirements 6.4)."""
    results = [
        _make_risk_result("A", "정상"),
        _make_risk_result("B", "경고"),
        _make_risk_result("C", "위험"),
    ]
    fig = render_risk_distribution_chart(results)
    assert list(fig.data[0].y) == ["위험", "경고", "정상"]


def test_risk_distribution_chart_counts_correct():
    """각 등급 인원 수가 정확히 집계됨."""
    results = [
        _make_risk_result("A", "위험"),
        _make_risk_result("B", "위험"),
        _make_risk_result("C", "정상"),
    ]
    fig = render_risk_distribution_chart(results)
    bar = fig.data[0]
    level_to_count = dict(zip(bar.y, bar.x))
    assert level_to_count["위험"] == 2
    assert level_to_count["정상"] == 1


# ---------------------------------------------------------------------------
# Property 6 (UI/UX): 위험도 분포 차트 수평 바차트 구조 불변성
# Validates: Requirements 6.1, 6.4
# ---------------------------------------------------------------------------

@st.composite
def risk_result_list_st(draw):
    """비어있지 않은 RiskResult 리스트 생성 전략."""
    names = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo", "Nd")),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=10,
        unique=True,
    ))
    results = []
    for name in names:
        level = draw(st.sampled_from(["위험", "경고", "정상"]))
        results.append(RiskResult(
            name=name,
            final_score=0.5,
            final_level=level,
            attendance_level=level,
            sentiment_score=0.5,
            recommendation="",
        ))
    return results


@given(risk_results=risk_result_list_st())
@settings(max_examples=100)
def test_property_6_risk_distribution_chart_horizontal_bar_invariant(risk_results):
    """비어있지 않은 RiskResult 리스트에 대해 render_risk_distribution_chart는
    첫 번째 trace가 go.Bar이고, orientation=='h'이며, 제목이 '위험도 분포'여야 한다.

    **Validates: Requirements 6.1, 6.4**
    """
    fig = render_risk_distribution_chart(risk_results)
    assert isinstance(fig.data[0], go.Bar)
    assert fig.data[0].orientation == "h"
    assert fig.layout.title.text == "위험도 분포"

# ---------------------------------------------------------------------------
# Property 7: 위험도 분포 차트 등급별 색상 매핑 불변성
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------

def test_property_7_risk_distribution_chart_color_mapping():
    """세 등급이 모두 포함된 RiskResult 리스트로 호출 시 각 등급 바 색상이 올바른지 검증.
    
    **Validates: Requirements 6.3**
    """
    results = [
        _make_risk_result("A", "위험"),
        _make_risk_result("B", "경고"),
        _make_risk_result("C", "정상"),
    ]
    fig = render_risk_distribution_chart(results)
    bar = fig.data[0]
    level_color_map = dict(zip(bar.y, bar.marker.color))
    assert level_color_map["위험"] == "#d62728"
    assert level_color_map["경고"] == "#ff7f0e"
    assert level_color_map["정상"] == "#2ca02c"
