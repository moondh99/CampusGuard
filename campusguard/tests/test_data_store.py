"""
DataStore 단위 테스트 + Property-Based 테스트
- Property 16: DataStore Round-Trip (Requirements 7.1, 7.2, 7.3)
- Property 17: 중복 삽입 Idempotence (Requirements 7.4)
"""
import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.data_store import DataStore
from modules.risk_predictor import RiskResult

from hypothesis import given, settings
from hypothesis import strategies as st


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def make_store() -> DataStore:
    return DataStore(":memory:")


def make_risk_result(name="홍길동", level="경고", score=0.5, sent=0.3, att="경고"):
    return RiskResult(
        name=name,
        final_score=score,
        final_level=level,
        attendance_level=att,
        sentiment_score=sent,
        recommendation="테스트",
    )


# ── 단위 테스트 ───────────────────────────────────────────────────────────────

class TestDataStoreInit:
    def test_initialize_creates_tables(self):
        ds = make_store()
        conn = ds._get_conn()
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"attendance_records", "trainee_profiles", "risk_history"}.issubset(tables)


class TestAttendanceRecords:
    def test_save_and_get(self):
        ds = make_store()
        df = pd.DataFrame([
            {"name": "김철수", "date": "2026-01-01", "status": "출석"},
            {"name": "이영희", "date": "2026-01-01", "status": "결석"},
        ])
        count = ds.save_attendance_records(df, course="AI과정", cohort="1기")
        assert count == 2
        records = ds.get_attendance_records(name="김철수")
        assert len(records) == 1
        assert records[0]["status"] == "출석"

    def test_get_by_course(self):
        ds = make_store()
        df = pd.DataFrame([{"name": "박민준", "date": "2026-01-02", "status": "지각"}])
        ds.save_attendance_records(df, course="웹개발", cohort="2기")
        records = ds.get_attendance_records(course="웹개발")
        assert len(records) == 1

    def test_duplicate_ignored(self):
        ds = make_store()
        df = pd.DataFrame([{"name": "김철수", "date": "2026-01-01", "status": "출석"}])
        ds.save_attendance_records(df, course="AI과정", cohort="1기")
        count2 = ds.save_attendance_records(df, course="AI과정", cohort="1기")
        assert count2 == 0
        assert len(ds.get_attendance_records(name="김철수")) == 1


class TestTraineeProfile:
    def test_upsert_and_get(self):
        ds = make_store()
        profile = {"name": "홍길동", "cohort": "1기", "course": "AI과정", "instructor": "김강사"}
        ds.upsert_trainee(profile)
        result = ds.get_trainee("홍길동")
        assert result is not None
        assert result["cohort"] == "1기"

    def test_upsert_updates_existing(self):
        ds = make_store()
        ds.upsert_trainee({"name": "홍길동", "cohort": "1기", "course": "AI과정"})
        ds.upsert_trainee({"name": "홍길동", "cohort": "2기", "course": "AI과정"})
        assert ds.get_trainee("홍길동")["cohort"] == "2기"

    def test_delete_trainee(self):
        ds = make_store()
        ds.upsert_trainee({"name": "홍길동", "cohort": "1기", "course": "AI과정"})
        assert ds.delete_trainee("홍길동") is True
        assert ds.get_trainee("홍길동") is None

    def test_delete_nonexistent(self):
        ds = make_store()
        assert ds.delete_trainee("없는사람") is False

    def test_list_trainees(self):
        ds = make_store()
        for name in ["나", "가", "다"]:
            ds.upsert_trainee({"name": name, "cohort": "1기", "course": "AI"})
        trainees = ds.list_trainees()
        assert len(trainees) == 3
        assert [t["name"] for t in trainees] == ["가", "나", "다"]


class TestRiskHistory:
    def test_save_and_get(self):
        ds = make_store()
        result = make_risk_result()
        ds.save_risk_result(result)
        history = ds.get_risk_history(name="홍길동")
        assert len(history) == 1
        assert history[0]["risk_level"] == "경고"
        assert history[0]["final_score"] == pytest.approx(0.5)

    def test_get_all(self):
        ds = make_store()
        ds.save_risk_result(make_risk_result("A"))
        ds.save_risk_result(make_risk_result("B"))
        assert len(ds.get_risk_history()) == 2

    def test_date_filter(self):
        ds = make_store()
        ds.save_risk_result(make_risk_result())
        history = ds.get_risk_history(start_date="2000-01-01", end_date="2099-12-31")
        assert len(history) == 1
        empty = ds.get_risk_history(start_date="2099-01-01", end_date="2099-12-31")
        assert len(empty) == 0


# ── Property-Based 테스트 ─────────────────────────────────────────────────────

# 유효한 이름 전략 (빈 문자열 제외)
valid_name = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Lo")),
    min_size=1,
    max_size=10,
)

valid_status = st.sampled_from(["출석", "지각", "조퇴", "결석"])
valid_level = st.sampled_from(["정상", "경고", "위험"])
valid_score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# Feature: campusguard-enhancement, Property 16: DataStore Round-Trip
@settings(max_examples=100)
@given(
    name=valid_name,
    cohort=st.text(min_size=1, max_size=5, alphabet="가나다라마바사아자차카타파하1234567890기"),
    course=st.text(min_size=1, max_size=10, alphabet="가나다라마바사아자차카타파하ABCDEFabcdef"),
    instructor=st.text(max_size=5, alphabet="가나다라마바사아자차카타파하"),
    risk_level=valid_level,
    final_score=valid_score,
    sentiment_score=valid_score,
    att_level=valid_level,
)
def test_property_16_datastore_round_trip(
    name, cohort, course, instructor, risk_level, final_score, sentiment_score, att_level
):
    """Property 16: 저장 후 조회 시 동일한 값이 반환되어야 한다."""
    ds = make_store()

    # TraineeProfile round-trip
    profile = {"name": name, "cohort": cohort, "course": course, "instructor": instructor, "enrolled_at": ""}
    ds.upsert_trainee(profile)
    retrieved = ds.get_trainee(name)
    assert retrieved is not None
    assert retrieved["name"] == name
    assert retrieved["cohort"] == cohort
    assert retrieved["course"] == course

    # RiskHistory round-trip
    result = RiskResult(
        name=name,
        final_score=round(final_score, 4),
        final_level=risk_level,
        attendance_level=att_level,
        sentiment_score=round(sentiment_score, 4),
        recommendation="테스트",
    )
    ds.save_risk_result(result)
    history = ds.get_risk_history(name=name)
    assert len(history) >= 1
    latest = history[0]
    assert latest["name"] == name
    assert latest["risk_level"] == risk_level
    assert latest["final_score"] == pytest.approx(round(final_score, 4), abs=1e-6)


# Feature: campusguard-enhancement, Property 17: 중복 삽입 Idempotence
@settings(max_examples=100)
@given(
    name=valid_name,
    date=st.dates().map(str),
    status=valid_status,
    n=st.integers(min_value=2, max_value=10),
)
def test_property_17_duplicate_insert_idempotence(name, date, status, n):
    """Property 17: 동일 레코드 N회 삽입 후 DB에 정확히 1개만 존재해야 한다."""
    ds = make_store()
    df = pd.DataFrame([{"name": name, "date": date, "status": status}])
    total_inserted = 0
    for _ in range(n):
        total_inserted += ds.save_attendance_records(df, course="테스트과정", cohort="1기")

    # 첫 번째 삽입만 성공, 나머지는 무시
    assert total_inserted == 1
    records = ds.get_attendance_records(name=name)
    assert len(records) == 1
