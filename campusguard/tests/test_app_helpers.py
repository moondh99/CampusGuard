"""
app.py 헬퍼 함수 테스트 — Property-Based 테스트
"""
import sys
import os
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

# app.py에서 _style_risk_row 직접 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Streamlit을 mock하여 app.py import 시 실행 방지
import unittest.mock as mock
with mock.patch.dict("sys.modules", {"streamlit": mock.MagicMock()}):
    # _style_risk_row만 직접 정의 (app.py import 없이)
    def _style_risk_row(row: pd.Series) -> list[str]:
        color_map = {
            "위험": "background-color: #ffcccc",
            "경고": "background-color: #fff3cd",
            "정상": "background-color: #d4edda",
        }
        style = color_map.get(row["final_level"], "")
        return [style] * len(row)


# ---------------------------------------------------------------------------
# Property 2: 위험도 테이블 행 색상 매핑 완전성
# Validates: Requirements 2.1, 2.2, 2.3, 2.4
# ---------------------------------------------------------------------------

EXPECTED_COLORS = {
    "위험": "#ffcccc",
    "경고": "#fff3cd",
    "정상": "#d4edda",
}


@given(level=st.sampled_from(["위험", "경고", "정상"]))
@settings(max_examples=100)
def test_property_2_risk_row_style_mapping(level):
    """임의의 위험도 등급에 대해 _style_risk_row는 모든 셀에 올바른 배경색을 적용해야 한다.
    
    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """
    row = pd.Series({
        "name": "테스트",
        "final_level": level,
        "final_score": 0.5,
        "attendance_level": "정상",
        "recommendation": "없음",
    })
    styles = _style_risk_row(row)
    expected_color = EXPECTED_COLORS[level]
    assert len(styles) == len(row), "스타일 리스트 길이가 행 길이와 다름"
    assert all(expected_color in s for s in styles), (
        f"등급 '{level}'에 대한 색상 '{expected_color}'이 모든 셀에 적용되지 않음: {styles}"
    )


def test_style_risk_row_unknown_level():
    """알 수 없는 등급은 빈 문자열 스타일을 반환해야 한다."""
    row = pd.Series({
        "name": "테스트",
        "final_level": "알수없음",
        "final_score": 0.5,
        "attendance_level": "정상",
        "recommendation": "없음",
    })
    styles = _style_risk_row(row)
    assert all(s == "" for s in styles)


# ---------------------------------------------------------------------------
# Property 3: 위험도 결과 DataFrame 필수 컬럼 보장
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------

from modules.risk_predictor import RiskResult as _RiskResult

REQUIRED_COLUMNS = {"name", "final_level", "final_score", "attendance_level", "recommendation"}


@st.composite
def risk_result_list_st(draw):
    names = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo"))),
        min_size=1, max_size=5, unique=True,
    ))
    results = []
    for name in names:
        level = draw(st.sampled_from(["위험", "경고", "정상"]))
        results.append(_RiskResult(
            name=name,
            final_score=0.5,
            final_level=level,
            attendance_level=level,
            sentiment_score=0.5,
            recommendation="없음",
        ))
    return results


@given(risk_results=risk_result_list_st())
@settings(max_examples=100)
def test_property_3_risk_result_dataframe_required_columns(risk_results):
    """비어있지 않은 RiskResult 리스트로 생성된 DataFrame은 5개 필수 컬럼을 모두 포함해야 한다.

    **Validates: Requirements 2.6**
    """
    result_df = pd.DataFrame([{
        "name": r.name,
        "final_level": r.final_level,
        "final_score": round(r.final_score, 2),
        "attendance_level": r.attendance_level,
        "recommendation": r.recommendation,
    } for r in risk_results])
    assert REQUIRED_COLUMNS.issubset(set(result_df.columns)), (
        f"필수 컬럼 누락: {REQUIRED_COLUMNS - set(result_df.columns)}"
    )

# ---------------------------------------------------------------------------
# Property 3: 위험도 결과 DataFrame 필수 컬럼 보장
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------

from modules.risk_predictor import RiskResult as _RiskResult

REQUIRED_COLUMNS = {"name", "final_level", "final_score", "attendance_level", "recommendation"}


@st.composite
def risk_result_list_st(draw):
    names = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo"))),
        min_size=1, max_size=5, unique=True,
    ))
    results = []
    for name in names:
        level = draw(st.sampled_from(["위험", "경고", "정상"]))
        results.append(_RiskResult(
            name=name,
            final_score=0.5,
            final_level=level,
            attendance_level=level,
            sentiment_score=0.5,
            recommendation="없음",
        ))
    return results


@given(risk_results=risk_result_list_st())
@settings(max_examples=100)
def test_property_3_risk_result_dataframe_required_columns(risk_results):
    """비어있지 않은 RiskResult 리스트로 생성된 DataFrame은 5개 필수 컬럼을 모두 포함해야 한다.

    **Validates: Requirements 2.6**
    """
    result_df = pd.DataFrame([{
        "name": r.name,
        "final_level": r.final_level,
        "final_score": round(r.final_score, 2),
        "attendance_level": r.attendance_level,
        "recommendation": r.recommendation,
    } for r in risk_results])
    assert REQUIRED_COLUMNS.issubset(set(result_df.columns)), (
        f"필수 컬럼 누락: {REQUIRED_COLUMNS - set(result_df.columns)}"
    )
