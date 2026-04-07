"""
데모용 예제 데이터 시드 스크립트
실행: python seed_demo.py  (campusguard/ 디렉터리 안에서)
"""
import sqlite3
import os
import sys
from datetime import date, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "campusguard.db")
COURSE = "K-디지털 트레이닝 (AI·빅데이터)"
COHORT = "3기"
INSTRUCTOR = "김강사"
ENROLLED_AT = "2026-03-03"

# 훈련생 프로필 (이름, 특성)
TRAINEES = [
    {"name": "김민준", "pattern": "normal"},       # 정상 출석
    {"name": "이서연", "pattern": "normal"},       # 정상 출석
    {"name": "박지호", "pattern": "warning"},      # 경고 수준
    {"name": "최수아", "pattern": "normal"},       # 정상 출석
    {"name": "정우진", "pattern": "danger"},       # 위험 수준
    {"name": "강하은", "pattern": "normal"},       # 정상 출석
    {"name": "윤도현", "pattern": "warning"},      # 경고 수준
    {"name": "임지유", "pattern": "normal"},       # 정상 출석
    {"name": "한승민", "pattern": "danger"},       # 위험 수준 (중도탈락 위기)
    {"name": "오채원", "pattern": "late_heavy"},   # 지각 집중형
]

# 수업일 생성 (월~금, 3/3 ~ 4/7, 공휴일 제외)
def get_class_days(start: date, end: date) -> list[date]:
    days = []
    cur = start
    # 간단히 공휴일 하드코딩 (삼일절)
    holidays = {date(2026, 3, 1)}
    while cur <= end:
        if cur.weekday() < 5 and cur not in holidays:
            days.append(cur)
        cur += timedelta(days=1)
    return days

CLASS_DAYS = get_class_days(date(2026, 3, 3), date(2026, 4, 7))

# 패턴별 출결 상태 생성
import random
random.seed(42)

def gen_status(pattern: str, day_idx: int, total: int) -> str:
    if pattern == "normal":
        # 95% 출석, 가끔 지각/조퇴
        r = random.random()
        if r < 0.92:
            return "출석"
        elif r < 0.96:
            return "지각"
        else:
            return "조퇴"

    elif pattern == "warning":
        # 결석 3~4일, 지각/조퇴 다수
        r = random.random()
        progress = day_idx / total
        if progress > 0.3 and r < 0.12:
            return "결석"
        elif r < 0.25:
            return "지각"
        elif r < 0.35:
            return "조퇴"
        else:
            return "출석"

    elif pattern == "danger":
        # 결석 6일 이상, 지각/조퇴 빈번
        r = random.random()
        progress = day_idx / total
        if progress > 0.2 and r < 0.22:
            return "결석"
        elif r < 0.30:
            return "지각"
        elif r < 0.38:
            return "조퇴"
        else:
            return "출석"

    elif pattern == "late_heavy":
        # 지각 집중 (지각 8회 이상)
        r = random.random()
        if r < 0.35:
            return "지각"
        elif r < 0.42:
            return "조퇴"
        elif r < 0.46:
            return "결석"
        else:
            return "출석"

    return "출석"

# 위험도 히스토리 데이터 (과거 분석 이력)
RISK_HISTORY = [
    # (name, analyzed_at, risk_level, final_score, sentiment_score, attendance_level)
    ("정우진",  "2026-03-14 18:00:00", "경고", 0.52, 0.45, "경고"),
    ("정우진",  "2026-03-21 18:00:00", "경고", 0.61, 0.40, "경고"),
    ("정우진",  "2026-03-28 18:00:00", "위험", 0.78, 0.35, "위험"),
    ("정우진",  "2026-04-04 18:00:00", "위험", 0.85, 0.30, "위험"),
    ("한승민",  "2026-03-14 18:00:00", "정상", 0.30, 0.70, "정상"),
    ("한승민",  "2026-03-21 18:00:00", "경고", 0.55, 0.50, "경고"),
    ("한승민",  "2026-03-28 18:00:00", "위험", 0.72, 0.38, "위험"),
    ("한승민",  "2026-04-04 18:00:00", "위험", 0.88, 0.25, "위험"),
    ("박지호",  "2026-03-14 18:00:00", "정상", 0.28, 0.72, "정상"),
    ("박지호",  "2026-03-21 18:00:00", "경고", 0.48, 0.55, "경고"),
    ("박지호",  "2026-03-28 18:00:00", "경고", 0.53, 0.50, "경고"),
    ("박지호",  "2026-04-04 18:00:00", "경고", 0.58, 0.48, "경고"),
    ("윤도현",  "2026-03-14 18:00:00", "정상", 0.22, 0.78, "정상"),
    ("윤도현",  "2026-03-21 18:00:00", "정상", 0.31, 0.68, "정상"),
    ("윤도현",  "2026-03-28 18:00:00", "경고", 0.47, 0.55, "경고"),
    ("윤도현",  "2026-04-04 18:00:00", "경고", 0.51, 0.52, "경고"),
    ("오채원",  "2026-03-14 18:00:00", "정상", 0.25, 0.75, "정상"),
    ("오채원",  "2026-03-21 18:00:00", "경고", 0.44, 0.60, "경고"),
    ("오채원",  "2026-03-28 18:00:00", "경고", 0.50, 0.55, "경고"),
    ("오채원",  "2026-04-04 18:00:00", "경고", 0.56, 0.50, "경고"),
    ("김민준",  "2026-04-04 18:00:00", "정상", 0.12, 0.88, "정상"),
    ("이서연",  "2026-04-04 18:00:00", "정상", 0.10, 0.90, "정상"),
    ("최수아",  "2026-04-04 18:00:00", "정상", 0.15, 0.85, "정상"),
    ("강하은",  "2026-04-04 18:00:00", "정상", 0.11, 0.89, "정상"),
    ("임지유",  "2026-04-04 18:00:00", "정상", 0.13, 0.87, "정상"),
]


def seed():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 스키마 보장
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            course TEXT NOT NULL,
            cohort TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(name, date, course)
        );
        CREATE TABLE IF NOT EXISTS trainee_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            cohort TEXT NOT NULL,
            course TEXT NOT NULL,
            instructor TEXT DEFAULT '',
            enrolled_at TEXT DEFAULT '',
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            analyzed_at TEXT DEFAULT (datetime('now')),
            risk_level TEXT NOT NULL,
            final_score REAL NOT NULL,
            sentiment_score REAL NOT NULL,
            attendance_level TEXT NOT NULL
        );
    """)

    # 1. 훈련생 프로필
    for t in TRAINEES:
        conn.execute(
            """INSERT INTO trainee_profiles (name, cohort, course, instructor, enrolled_at, updated_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(name) DO UPDATE SET
                   cohort=excluded.cohort, course=excluded.course,
                   instructor=excluded.instructor, enrolled_at=excluded.enrolled_at,
                   updated_at=datetime('now')""",
            (t["name"], COHORT, COURSE, INSTRUCTOR, ENROLLED_AT),
        )
    print(f"✅ 훈련생 프로필 {len(TRAINEES)}명 저장")

    # 2. 출결 레코드
    att_count = 0
    total_days = len(CLASS_DAYS)
    for t in TRAINEES:
        for idx, day in enumerate(CLASS_DAYS):
            status = gen_status(t["pattern"], idx, total_days)
            cur = conn.execute(
                """INSERT OR IGNORE INTO attendance_records (name, date, status, course, cohort)
                   VALUES (?, ?, ?, ?, ?)""",
                (t["name"], str(day), status, COURSE, COHORT),
            )
            att_count += cur.rowcount
    print(f"✅ 출결 레코드 {att_count}건 저장 ({total_days}일 × {len(TRAINEES)}명)")

    # 3. 위험도 히스토리
    hist_count = 0
    for row in RISK_HISTORY:
        cur = conn.execute(
            """INSERT INTO risk_history
               (name, analyzed_at, risk_level, final_score, sentiment_score, attendance_level)
               VALUES (?, ?, ?, ?, ?, ?)""",
            row,
        )
        hist_count += cur.rowcount
    print(f"✅ 위험도 히스토리 {hist_count}건 저장")

    conn.commit()
    conn.close()

    # 요약 출력
    print("\n── 훈련생별 출결 요약 ──────────────────────────────")
    conn2 = sqlite3.connect(DB_PATH)
    for t in TRAINEES:
        rows = conn2.execute(
            "SELECT status, COUNT(*) as cnt FROM attendance_records WHERE name=? GROUP BY status",
            (t["name"],),
        ).fetchall()
        summary = {r[0]: r[1] for r in rows}
        print(
            f"  {t['name']:6s} | 출석 {summary.get('출석',0):2d} "
            f"지각 {summary.get('지각',0):2d} "
            f"조퇴 {summary.get('조퇴',0):2d} "
            f"결석 {summary.get('결석',0):2d}  [{t['pattern']}]"
        )
    conn2.close()
    print("\n🎉 데모 데이터 시드 완료!")


if __name__ == "__main__":
    seed()
