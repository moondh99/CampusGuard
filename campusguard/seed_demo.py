"""
데모용 예제 데이터 시드 스크립트
- seed_if_empty(ds): DataStore가 비어있을 때만 시드 실행 (앱 자동 호출용)
- 직접 실행: python seed_demo.py
"""
import os
import random
from datetime import date, timedelta

COURSE = "K-디지털 트레이닝 (AI·빅데이터)"
COHORT = "3기"
INSTRUCTOR = "김강사"
ENROLLED_AT = "2026-03-03"

TRAINEES = [
    {"name": "김민준", "pattern": "normal"},
    {"name": "이서연", "pattern": "normal"},
    {"name": "박지호", "pattern": "warning"},
    {"name": "최수아", "pattern": "normal"},
    {"name": "정우진", "pattern": "danger"},
    {"name": "강하은", "pattern": "normal"},
    {"name": "윤도현", "pattern": "warning"},
    {"name": "임지유", "pattern": "normal"},
    {"name": "한승민", "pattern": "danger"},
    {"name": "오채원", "pattern": "late_heavy"},
]

RISK_HISTORY = [
    ("정우진", "2026-03-14 18:00:00", "경고", 0.52, 0.45, "경고"),
    ("정우진", "2026-03-21 18:00:00", "경고", 0.61, 0.40, "경고"),
    ("정우진", "2026-03-28 18:00:00", "위험", 0.78, 0.35, "위험"),
    ("정우진", "2026-04-04 18:00:00", "위험", 0.85, 0.30, "위험"),
    ("한승민", "2026-03-14 18:00:00", "정상", 0.30, 0.70, "정상"),
    ("한승민", "2026-03-21 18:00:00", "경고", 0.55, 0.50, "경고"),
    ("한승민", "2026-03-28 18:00:00", "위험", 0.72, 0.38, "위험"),
    ("한승민", "2026-04-04 18:00:00", "위험", 0.88, 0.25, "위험"),
    ("박지호", "2026-03-14 18:00:00", "정상", 0.28, 0.72, "정상"),
    ("박지호", "2026-03-21 18:00:00", "경고", 0.48, 0.55, "경고"),
    ("박지호", "2026-03-28 18:00:00", "경고", 0.53, 0.50, "경고"),
    ("박지호", "2026-04-04 18:00:00", "경고", 0.58, 0.48, "경고"),
    ("윤도현", "2026-03-14 18:00:00", "정상", 0.22, 0.78, "정상"),
    ("윤도현", "2026-03-21 18:00:00", "정상", 0.31, 0.68, "정상"),
    ("윤도현", "2026-03-28 18:00:00", "경고", 0.47, 0.55, "경고"),
    ("윤도현", "2026-04-04 18:00:00", "경고", 0.51, 0.52, "경고"),
    ("오채원", "2026-03-14 18:00:00", "정상", 0.25, 0.75, "정상"),
    ("오채원", "2026-03-21 18:00:00", "경고", 0.44, 0.60, "경고"),
    ("오채원", "2026-03-28 18:00:00", "경고", 0.50, 0.55, "경고"),
    ("오채원", "2026-04-04 18:00:00", "경고", 0.56, 0.50, "경고"),
    ("김민준", "2026-04-04 18:00:00", "정상", 0.12, 0.88, "정상"),
    ("이서연", "2026-04-04 18:00:00", "정상", 0.10, 0.90, "정상"),
    ("최수아", "2026-04-04 18:00:00", "정상", 0.15, 0.85, "정상"),
    ("강하은", "2026-04-04 18:00:00", "정상", 0.11, 0.89, "정상"),
    ("임지유", "2026-04-04 18:00:00", "정상", 0.13, 0.87, "정상"),
]


def _get_class_days() -> list[date]:
    days, cur = [], date(2026, 3, 3)
    holidays = {date(2026, 3, 1)}
    while cur <= date(2026, 4, 7):
        if cur.weekday() < 5 and cur not in holidays:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _gen_status(pattern: str, day_idx: int, total: int) -> str:
    random.seed(42 + day_idx)  # 재현 가능한 결과
    r = random.random()
    progress = day_idx / total
    if pattern == "normal":
        return "출석" if r < 0.92 else ("지각" if r < 0.96 else "조퇴")
    elif pattern == "warning":
        if progress > 0.3 and r < 0.12: return "결석"
        if r < 0.25: return "지각"
        if r < 0.35: return "조퇴"
        return "출석"
    elif pattern == "danger":
        if progress > 0.2 and r < 0.22: return "결석"
        if r < 0.30: return "지각"
        if r < 0.38: return "조퇴"
        return "출석"
    elif pattern == "late_heavy":
        if r < 0.35: return "지각"
        if r < 0.42: return "조퇴"
        if r < 0.46: return "결석"
        return "출석"
    return "출석"


def seed_if_empty(ds) -> bool:
    """
    DB의 attendance_records가 비어있을 때만 데모 데이터를 삽입.
    Returns: True면 시드 실행됨, False면 이미 데이터 있음
    """
    existing = ds.get_attendance_records()
    if existing:
        return False

    class_days = _get_class_days()
    total = len(class_days)

    # 1. 훈련생 프로필
    for t in TRAINEES:
        ds.upsert_trainee({
            "name": t["name"],
            "cohort": COHORT,
            "course": COURSE,
            "instructor": INSTRUCTOR,
            "enrolled_at": ENROLLED_AT,
        })

    # 2. 출결 레코드
    import pandas as pd
    rows = []
    for t in TRAINEES:
        for idx, day in enumerate(class_days):
            rows.append({
                "name": t["name"],
                "date": str(day),
                "status": _gen_status(t["pattern"], idx, total),
            })
    df = pd.DataFrame(rows)
    ds.save_attendance_records(df, COURSE, COHORT)

    # 3. 위험도 히스토리
    for row in RISK_HISTORY:
        ds.insert_risk_history(*row)

    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from modules.data_store import DataStore
    db_path = os.path.join(os.path.dirname(__file__), "campusguard.db")
    ds = DataStore(db_path)
    ran = seed_if_empty(ds)
    print("🎉 시드 완료!" if ran else "ℹ️ 이미 데이터가 존재합니다. 시드를 건너뜁니다.")
