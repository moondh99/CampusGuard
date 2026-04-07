"""
태스크 2 단위 테스트: 출결 분석
- API 호출 없음, 순수 로직 테스트
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.attendance import analyze_student, analyze_all, load_csv, AttendanceResult


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"name": "김위험", "date": "2026-04-01", "status": "지각"},
        {"name": "김위험", "date": "2026-04-02", "status": "지각"},
        {"name": "김위험", "date": "2026-04-03", "status": "지각"},
        {"name": "김위험", "date": "2026-04-04", "status": "결석"},
        {"name": "김위험", "date": "2026-04-07", "status": "결석"},
        {"name": "이정상", "date": "2026-04-01", "status": "출석"},
        {"name": "이정상", "date": "2026-04-02", "status": "출석"},
        {"name": "박경고", "date": "2026-04-01", "status": "지각"},
        {"name": "박경고", "date": "2026-04-02", "status": "지각"},
        {"name": "박경고", "date": "2026-04-03", "status": "조퇴"},
        {"name": "박경고", "date": "2026-04-04", "status": "조퇴"},
    ])


def test_danger_student(sample_df):
    """지각 3회 + 결석 2일 = 환산 결석 3.0 → 경고 (3일 이상 5일 미만)"""
    result = analyze_student("김위험", sample_df)
    assert result.risk_level == "경고"
    assert result.late_count == 3
    assert result.absent_count == 2
    assert result.converted_absent == 3.0


def test_normal_student(sample_df):
    """출석만 있으면 정상"""
    result = analyze_student("이정상", sample_df)
    assert result.risk_level == "정상"
    assert result.converted_absent == 0.0


def test_warning_student(sample_df):
    """지각 2회 + 조퇴 2회 = 환산 결석 1.33 → 정상 (3일 미만)"""
    result = analyze_student("박경고", sample_df)
    # 4회 / 3 = 1.33 → 정상
    assert result.converted_absent == round(4 / 3, 2)
    assert result.risk_level == "정상"


def test_unknown_student_raises(sample_df):
    """없는 훈련생 조회 시 ValueError"""
    with pytest.raises(ValueError, match="데이터가 없습니다"):
        analyze_student("없는사람", sample_df)


def test_analyze_all_returns_all_students(sample_df):
    """전체 분석 시 모든 훈련생 포함"""
    results = analyze_all(sample_df)
    names = [r.name for r in results]
    assert "김위험" in names
    assert "이정상" in names
    assert "박경고" in names


def test_load_csv(tmp_path):
    """CSV 로드 및 컬럼 검증"""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("name,date,status\n홍길동,2026-04-01,출석\n")
    df = load_csv(str(csv_file))
    assert list(df.columns) == ["name", "date", "status"]
    assert len(df) == 1


def test_load_csv_missing_column(tmp_path):
    """필수 컬럼 누락 시 ValueError"""
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("name,date\n홍길동,2026-04-01\n")
    with pytest.raises(ValueError, match="필수 컬럼"):
        load_csv(str(csv_file))


# ── Property-Based Tests ──────────────────────────────────────────────────────

from hypothesis import given, settings
import hypothesis.strategies as st_hyp


@given(
    late=st_hyp.integers(min_value=0, max_value=100),
    early_leave=st_hyp.integers(min_value=0, max_value=100),
    absent=st_hyp.integers(min_value=0, max_value=100),
)
@settings(max_examples=100)
def test_converted_absent_formula(late, early_leave, absent):
    """Property 3: 출결 환산 결석일 계산 정확성"""
    # Feature: campusguard-deployment, Property 3: 출결 환산 결석일 계산 정확성
    # Validates: Requirements 5.1
    rows = []
    for i in range(late):
        rows.append({"name": "테스트", "date": f"2026-04-{i+1:02d}", "status": "지각"})
    for i in range(early_leave):
        rows.append({"name": "테스트", "date": f"2026-05-{i+1:02d}", "status": "조퇴"})
    for i in range(absent):
        rows.append({"name": "테스트", "date": f"2026-06-{i+1:02d}", "status": "결석"})
    if not rows:
        rows.append({"name": "테스트", "date": "2026-04-01", "status": "출석"})
    df = pd.DataFrame(rows)
    result = analyze_student("테스트", df)
    expected = round(absent + (late + early_leave) / 3, 2)
    assert result.converted_absent == expected
