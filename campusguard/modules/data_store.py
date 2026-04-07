"""
DataStore: SQLite 기반 데이터 영속성 레이어
- attendance_records, trainee_profiles, risk_history 테이블 관리
- 기존 모듈 인터페이스 변경 없이 독립적으로 추가
"""
import sqlite3
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, db_path: str = "campusguard.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.initialize()

    def _get_conn(self) -> "sqlite3.Connection":
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialize(self) -> None:
        """스키마 생성 (테이블 없을 시 자동 생성)."""
        conn = self._get_conn()
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
        conn.commit()

    # ── AttendanceRecord ──────────────────────────────────────────────────────

    def save_attendance_records(self, df, course: str, cohort: str) -> int:
        """
        DataFrame의 출결 레코드를 저장. 중복(name+date+course)은 무시.
        Returns: 삽입된 신규 레코드 수
        """
        conn = self._get_conn()
        inserted = 0
        for _, row in df.iterrows():
            cur = conn.execute(
                """INSERT OR IGNORE INTO attendance_records
                   (name, date, status, course, cohort)
                   VALUES (?, ?, ?, ?, ?)""",
                (row["name"], str(row["date"]), row["status"], course, cohort),
            )
            inserted += cur.rowcount
        conn.commit()
        return inserted

    def get_attendance_records(
        self, name: str = None, course: str = None
    ) -> list[dict]:
        """name 또는 course 필터로 출결 레코드 조회."""
        conn = self._get_conn()
        query = "SELECT * FROM attendance_records WHERE 1=1"
        params: list = []
        if name:
            query += " AND name = ?"
            params.append(name)
        if course:
            query += " AND course = ?"
            params.append(course)
        query += " ORDER BY date"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def insert_attendance_record(
        self, name: str, date: str, status: str, course: str, cohort: str
    ) -> bool:
        """단건 출결 레코드 삽입. 중복 시 False 반환."""
        conn = self._get_conn()
        cur = conn.execute(
            """INSERT OR IGNORE INTO attendance_records (name, date, status, course, cohort)
               VALUES (?, ?, ?, ?, ?)""",
            (name, date, status, course, cohort),
        )
        conn.commit()
        return cur.rowcount > 0

    def update_attendance_record(self, record_id: int, status: str) -> bool:
        """출결 레코드 상태 수정. 성공 시 True."""
        conn = self._get_conn()
        cur = conn.execute(
            "UPDATE attendance_records SET status = ? WHERE id = ?",
            (status, record_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def delete_attendance_record(self, record_id: int) -> bool:
        """출결 레코드 삭제. 성공 시 True."""
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM attendance_records WHERE id = ?", (record_id,)
        )
        conn.commit()
        return cur.rowcount > 0

    # ── TraineeProfile ────────────────────────────────────────────────────────

    def upsert_trainee(self, profile: dict) -> None:
        """훈련생 프로필 삽입 또는 업데이트."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO trainee_profiles (name, cohort, course, instructor, enrolled_at, updated_at)
               VALUES (:name, :cohort, :course, :instructor, :enrolled_at, datetime('now'))
               ON CONFLICT(name) DO UPDATE SET
                   cohort = excluded.cohort,
                   course = excluded.course,
                   instructor = excluded.instructor,
                   enrolled_at = excluded.enrolled_at,
                   updated_at = datetime('now')""",
            {
                "name": profile.get("name", ""),
                "cohort": profile.get("cohort", ""),
                "course": profile.get("course", ""),
                "instructor": profile.get("instructor", ""),
                "enrolled_at": profile.get("enrolled_at", ""),
            },
        )
        conn.commit()

    def get_trainee(self, name: str) -> Optional[dict]:
        """훈련생 프로필 단건 조회. 없으면 None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM trainee_profiles WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def delete_trainee(self, name: str) -> bool:
        """훈련생 프로필 삭제. 삭제 성공 시 True."""
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM trainee_profiles WHERE name = ?", (name,)
        )
        conn.commit()
        return cur.rowcount > 0

    def list_trainees(self) -> list[dict]:
        """전체 훈련생 프로필 목록 반환."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trainee_profiles ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── RiskHistory ───────────────────────────────────────────────────────────

    def save_risk_result(self, result) -> None:
        """RiskResult 객체를 risk_history 테이블에 저장."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO risk_history
               (name, analyzed_at, risk_level, final_score, sentiment_score, attendance_level)
               VALUES (?, datetime('now'), ?, ?, ?, ?)""",
            (
                result.name,
                result.final_level,
                result.final_score,
                result.sentiment_score,
                result.attendance_level,
            ),
        )
        conn.commit()

    def get_risk_history(
        self,
        name: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> list[dict]:
        """위험도 분석 히스토리 조회. 날짜 범위 및 이름 필터 지원."""
        conn = self._get_conn()
        query = "SELECT * FROM risk_history WHERE 1=1"
        params: list = []
        if name:
            query += " AND name = ?"
            params.append(name)
        if start_date:
            query += " AND analyzed_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND analyzed_at <= ?"
            params.append(end_date)
        query += " ORDER BY analyzed_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        """DB 연결 종료."""
        if self._conn:
            self._conn.close()
            self._conn = None
