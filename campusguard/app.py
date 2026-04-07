"""
CampusGuard - 국비지원 부트캠프 AI 케어 & 행정 자동화 플랫폼
실행: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from modules.attendance import load_csv, analyze_all
from modules.sentiment import analyze_sentiment, is_high_risk
from modules.risk_predictor import predict_all
from modules.chat_assistant import ask_assistant
from modules.admin_writer import generate_counseling_log, generate_reason_letter

st.set_page_config(page_title="CampusGuard", page_icon="🎓", layout="wide")

# 전역 API 키 체크
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다. AI 기능(강의 비서, 감성 분석, 서류 생성)이 비활성화됩니다. .env 파일에 OPENAI_API_KEY를 설정하거나 Streamlit Cloud Secrets에 등록하세요.")

st.title("🎓 CampusGuard")
st.caption("국비지원 부트캠프 AI 학습 케어 & 행정 자동화 플랫폼")

tab1, tab2, tab3 = st.tabs(["📊 위험도 대시보드", "💬 AI 강의 비서", "📝 행정 서류 자동 생성"])

# ── 탭 1: 위험도 대시보드 ──────────────────────────────────────
with tab1:
    st.subheader("훈련생 탈락 위험도 분석")

    uploaded = st.file_uploader("출결 CSV 업로드 (name, date, status)", type="csv")
    use_sample = st.checkbox("샘플 데이터 사용", value=True)

    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required_cols = {"name", "date", "status"}
            if not required_cols.issubset(df.columns):
                st.error(f"❌ CSV에 필수 컬럼이 없습니다. 필요한 컬럼: {required_cols}")
                df = None
        except Exception as e:
            st.error(f"❌ CSV 파일을 읽는 중 오류가 발생했습니다: {str(e)}")
            df = None
    elif use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_attendance.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
        else:
            st.info("ℹ️ 샘플 데이터 파일을 찾을 수 없습니다. CSV 파일을 직접 업로드해주세요.")

    if df is not None:
        st.dataframe(df, use_container_width=True)

        st.markdown("#### 감성 분석 입력 (선택)")
        st.caption("훈련생별 최근 메시지를 입력하면 감성 점수가 반영됩니다.")

        names = df["name"].unique().tolist()
        sentiment_map = {}

        with st.expander("감성 분석 입력 열기"):
            for name in names:
                msg = st.text_input(f"{name} 최근 메시지", key=f"sent_{name}")
                if msg:
                    with st.spinner(f"{name} 감성 분석 중..."):
                        result = analyze_sentiment(msg)
                        sentiment_map[name] = result["score"]
                        st.caption(f"→ {result['summary']} (점수: {result['score']})")

        if st.button("위험도 분석 실행"):
            from modules.attendance import analyze_all as _analyze_all
            att_results = _analyze_all(df)
            risk_results = predict_all(att_results, sentiment_map)

            rows = []
            for r in risk_results:
                rows.append({
                    "이름": r.name,
                    "최종 위험도": r.final_level,
                    "위험 점수": r.final_score,
                    "출결 등급": r.attendance_level,
                    "감성 점수": r.sentiment_score,
                    "권고 사항": r.recommendation,
                })
            result_df = pd.DataFrame(rows)

            # 색상 강조
            def highlight_risk(val):
                if val == "위험":
                    return "background-color: #ffcccc"
                elif val == "경고":
                    return "background-color: #fff3cc"
                return ""

            st.dataframe(
                result_df.style.applymap(highlight_risk, subset=["최종 위험도"]),
                use_container_width=True,
            )

# ── 탭 2: AI 강의 비서 ────────────────────────────────────────
with tab2:
    st.subheader("💬 AI 강의 비서")
    st.caption("에러 메시지나 코드를 붙여넣으면 해결책을 알려드립니다.")

    context = st.text_area("현재 학습 중인 교안 맥락 (선택)", height=80,
                           placeholder="예: pandas 기초 - DataFrame 인덱싱")
    question = st.text_area("질문 또는 에러 메시지", height=150,
                            placeholder="예: KeyError: 'age' 에러가 발생했어요")

    if st.button("답변 받기"):
        if question.strip():
            with st.spinner("AI가 답변을 생성 중입니다..."):
                answer = ask_assistant(question, context)
            st.markdown("#### 답변")
            st.write(answer)
        else:
            st.warning("질문을 입력해주세요.")

# ── 탭 3: 행정 서류 자동 생성 ─────────────────────────────────
with tab3:
    st.subheader("📝 행정 서류 자동 생성")

    doc_type = st.radio("서류 종류", ["상담일지", "사유서"])

    if doc_type == "상담일지":
        col1, col2 = st.columns(2)
        with col1:
            keywords = st.text_input("상담 키워드", placeholder="예: 취업 고민, 파이썬 기초 부족")
            risk_level = st.selectbox("위험 등급", ["정상", "경고", "위험"])
        with col2:
            absent_info = st.text_input("출결 현황", placeholder="예: 지각 2회, 조퇴 1회")

        if st.button("상담일지 초안 생성"):
            if keywords:
                with st.spinner("생성 중..."):
                    draft = generate_counseling_log(
                        name="[이름]",
                        keywords=keywords,
                        risk_level=risk_level,
                        absent_info=absent_info or "정상",
                    )
                st.text_area("상담일지 초안", value=draft, height=200)
            else:
                st.warning("상담 키워드를 입력해주세요.")

    else:
        situation = st.text_area("상황 설명", height=100,
                                 placeholder="예: 2026-04-07 오전 9시 HRD-Net 시스템 장애로 출결 체크 불가")
        if st.button("사유서 초안 생성"):
            if situation.strip():
                with st.spinner("생성 중..."):
                    draft = generate_reason_letter(situation)
                st.text_area("사유서 초안", value=draft, height=200)
            else:
                st.warning("상황을 입력해주세요.")
