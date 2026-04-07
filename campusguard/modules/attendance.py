"""
태스크 2: 출결 분석
- HRD-Net 규정: 지각/조퇴 3회 = 결석 1일
- 결석 누적 기준으로 위험 등급 산출
- CSV 파일 또는 DataFrame 입력 지원
"""
import pandas as pd
from dataclasses import dataclass


# HRD-Net 규정 상수
LATE_LEAVE_PER_ABSENT = 3   # 지각/조퇴 3회 = 결석 1일
DANGER_ABSENT_DAYS = 5      # 위험: 환산 결석 5일 이상
WARNING_ABSENT_DAYS = 3     # 경고: 환산 결석 3일 이상


@dataclass
class AttendanceResult:
    name: str
    late_count: int         # 지각 횟수
    early_leave_count: int  # 조퇴 횟수
    absent_count: int       # 실제 결석 횟수
    converted_absent: float # 환산 결석일 (지각+조퇴 포함)
    risk_level: str         # "정상" / "경고" / "위험"
    message: str


def load_csv(filepath: str) -> pd.DataFrame:
    """CSV 파일 로드. 컬럼: name, date, status"""
    df = pd.read_csv(filepath)
    required = {"name", "date", "status"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {required}")
    return df


def analyze_student(name: str, df: pd.DataFrame) -> AttendanceResult:
    """
    특정 훈련생의 출결 분석.

    Args:
        name: 훈련생 이름
        df: 전체 출결 DataFrame

    Returns:
        AttendanceResult
    """
    student_df = df[df["name"] == name]
    if student_df.empty:
        raise ValueError(f"'{name}' 훈련생 데이터가 없습니다.")

    late = int((student_df["status"] == "지각").sum())
    early_leave = int((student_df["status"] == "조퇴").sum())
    absent = int((student_df["status"] == "결석").sum())

    # 지각 + 조퇴 합산 후 3회당 결석 1일로 환산
    converted = absent + (late + early_leave) / LATE_LEAVE_PER_ABSENT

    if converted >= DANGER_ABSENT_DAYS:
        level = "위험"
        msg = f"환산 결석 {converted:.1f}일 — 즉시 상담 필요"
    elif converted >= WARNING_ABSENT_DAYS:
        level = "경고"
        msg = f"환산 결석 {converted:.1f}일 — 주의 관찰 필요"
    else:
        level = "정상"
        msg = f"환산 결석 {converted:.1f}일 — 정상 범위"

    return AttendanceResult(
        name=name,
        late_count=late,
        early_leave_count=early_leave,
        absent_count=absent,
        converted_absent=round(converted, 2),
        risk_level=level,
        message=msg,
    )


def analyze_all(df: pd.DataFrame) -> list[AttendanceResult]:
    """전체 훈련생 출결 분석 결과 반환."""
    return [analyze_student(name, df) for name in df["name"].unique()]
