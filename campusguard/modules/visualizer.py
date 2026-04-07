"""
Visualizer 모듈 — Plotly 기반 대시보드 차트 함수 모음.
Requirements: 5.1, 5.2, 5.3, 2.5
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from modules.risk_predictor import RiskResult

# 출결 상태 → 숫자 매핑 (라인 차트 Y축)
STATUS_MAP = {
    "출석": 1,
    "지각": 2,
    "조퇴": 3,
    "결석": 4,
}


def render_attendance_line_chart(df: pd.DataFrame) -> go.Figure:
    """훈련생별 날짜별 출결 상태 라인 차트.

    Args:
        df: 출결 DataFrame. 컬럼: name, date, status

    Returns:
        go.Figure — 훈련생별 색상 구분 라인 차트
    """
    fig = go.Figure()

    for name in df["name"].unique():
        student_df = df[df["name"] == name].sort_values("date")
        y_vals = student_df["status"].map(STATUS_MAP).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=student_df["date"],
                y=y_vals,
                mode="lines+markers",
                name=name,
            )
        )

    fig.update_layout(
        title="훈련생별 출결 추이",
        xaxis_title="날짜",
        yaxis=dict(
            title="출결 상태",
            tickvals=list(STATUS_MAP.values()),
            ticktext=list(STATUS_MAP.keys()),
        ),
    )
    return fig


def render_risk_distribution_chart(risk_results: list[RiskResult]) -> go.Figure:
    """RiskLevel 분포 파이/바 차트.

    Args:
        risk_results: RiskResult 리스트

    Returns:
        go.Figure — RiskLevel 파이 차트
    """
    from collections import Counter

    counts = Counter(r.final_level for r in risk_results)
    labels = list(counts.keys())
    values = list(counts.values())

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
        )
    )
    fig.update_layout(title="위험도 분포")
    return fig


def render_absence_heatmap(df: pd.DataFrame) -> go.Figure:
    """날짜별 결석률 히트맵.

    결석률 = 해당 날짜 결석 훈련생 수 / 전체 훈련생 수

    Args:
        df: 출결 DataFrame. 컬럼: name, date, status

    Returns:
        go.Figure — 날짜별 결석률 히트맵
    """
    total_students = df["name"].nunique()
    if total_students == 0:
        return go.Figure()

    absence_by_date = (
        df[df["status"] == "결석"]
        .groupby("date")["name"]
        .nunique()
        .reset_index(name="absent_count")
    )
    absence_by_date["absence_rate"] = absence_by_date["absent_count"] / total_students

    dates = absence_by_date["date"].tolist()
    rates = absence_by_date["absence_rate"].tolist()

    fig = go.Figure(
        go.Heatmap(
            x=dates,
            y=["결석률"],
            z=[rates],
            colorscale="Reds",
            zmin=0.0,
            zmax=1.0,
        )
    )
    fig.update_layout(title="날짜별 결석률 히트맵")
    return fig


def render_sentiment_trend_chart(sentiment_data: dict[str, list]) -> go.Figure:
    """훈련생별 감성 점수 추이 라인 차트.

    Args:
        sentiment_data: {훈련생명: [감성점수, ...]} 딕셔너리

    Returns:
        go.Figure — 감성 점수 추이 라인 차트
    """
    fig = go.Figure()

    for name, scores in sentiment_data.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores))),
                y=scores,
                mode="lines+markers",
                name=name,
            )
        )

    fig.update_layout(
        title="훈련생별 감성 점수 추이",
        xaxis_title="일차",
        yaxis_title="감성 점수 (0=긍정, 1=부정)",
        yaxis=dict(range=[0.0, 1.0]),
    )
    return fig


def detect_burnout_periods(
    sentiment_series: list[float],
    threshold: float = 0.6,
    consecutive_days: int = 3,
) -> list[tuple[int, int]]:
    """번아웃 구간 (시작 인덱스, 종료 인덱스) 리스트 반환.

    연속 `consecutive_days`일 이상 SentimentScore >= `threshold`인 구간을 탐지한다.

    Args:
        sentiment_series: 날짜 순서대로 정렬된 감성 점수 리스트 (0.0~1.0)
        threshold: 번아웃 판단 임계값 (기본 0.6)
        consecutive_days: 연속 일수 기준 (기본 3)

    Returns:
        list of (start_index, end_index) tuples — 번아웃 구간 목록
    """
    periods: list[tuple[int, int]] = []
    n = len(sentiment_series)
    if n == 0:
        return periods

    i = 0
    while i < n:
        if sentiment_series[i] >= threshold:
            start = i
            while i < n and sentiment_series[i] >= threshold:
                i += 1
            end = i - 1
            if (end - start + 1) >= consecutive_days:
                periods.append((start, end))
        else:
            i += 1

    return periods
