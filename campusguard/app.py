"""
CampusGuard - 국비지원 부트캠프 AI 케어 & 행정 자동화 플랫폼
실행: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from modules.attendance import load_csv, analyze_all
from modules.sentiment import analyze_sentiment, is_high_risk
from modules.risk_predictor import predict_all
from modules.chat_assistant import ask_assistant, detect_traceback, clear_history
from modules.admin_writer import (
    generate_counseling_log,
    generate_reason_letter,
    generate_counseling_docx,
)
from modules.data_store import DataStore
from modules.visualizer import (
    render_attendance_line_chart,
    render_risk_distribution_chart,
    render_absence_heatmap,
    render_sentiment_trend_chart,
)
from modules.notifier import load_config_from_env, send_alert, should_notify
from modules.kakao_parser import parse_kakao_export, group_by_sender, detect_duplicate_senders
from modules.rag_engine import RAGEngine

st.set_page_config(page_title="CampusGuard", page_icon="🎓", layout="wide")

# ── 앱 시작 시 DataStore 초기화 ────────────────────────────────
@st.cache_resource
def get_data_store() -> DataStore:
    db_path = os.path.join(os.path.dirname(__file__), "campusguard.db")
    return DataStore(db_path)

ds = get_data_store()

# ── 전역 API 키 체크 ───────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning(
        "⚠️ OpenAI API 키가 설정되지 않았습니다. AI 기능(강의 비서, 감성 분석, 서류 생성)이 비활성화됩니다. "
        ".env 파일에 OPENAI_API_KEY를 설정하거나 Streamlit Cloud Secrets에 등록하세요."
    )

st.title("🎓 CampusGuard")
st.caption("국비지원 부트캠프 AI 학습 케어 & 행정 자동화 플랫폼")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 위험도 대시보드",
    "💬 AI 강의 비서",
    "📝 행정 서류 자동 생성",
    "💬 카카오톡 감성 분석",
    "📈 분석 히스토리",
])

# ── 탭 1: 위험도 대시보드 ──────────────────────────────────────
with tab1:
    st.subheader("훈련생 탈락 위험도 분석")

    col_upload, col_meta = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("출결 CSV 업로드 (name, date, status)", type="csv")
        use_sample = st.checkbox("샘플 데이터 사용", value=True)
    with col_meta:
        course_name_input = st.text_input("과정명", value="K-디지털 트레이닝")
        cohort_input = st.text_input("기수", value="1기")

    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            required_cols = {"name", "date", "status"}
            if not required_cols.issubset(df.columns):
                st.error(f"❌ CSV에 필수 컬럼이 없습니다. 필요한 컬럼: {required_cols}")
                df = None
            else:
                inserted = ds.save_attendance_records(df, course_name_input, cohort_input)
                if inserted > 0:
                    st.success(f"✅ {inserted}건의 출결 레코드가 저장되었습니다.")
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

        # Plotly 차트 섹션
        st.markdown("#### 📊 시각화 대시보드")
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(render_attendance_line_chart(df), use_container_width=True)
        with chart_col2:
            st.plotly_chart(render_absence_heatmap(df), use_container_width=True)

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
            att_results = analyze_all(df)
            risk_results = predict_all(att_results, sentiment_map)

            # 위험도 분포 차트
            st.plotly_chart(render_risk_distribution_chart(risk_results), use_container_width=True)

            # 위험도 결과 저장
            for r in risk_results:
                ds.save_risk_result(r)

            # 알림 연동
            notif_config = load_config_from_env()
            if notif_config.enabled:
                try:
                    send_alert(risk_results, notif_config)
                    if should_notify(risk_results):
                        st.info("📨 위험 등급 훈련생 알림이 발송되었습니다.")
                except Exception as e:
                    st.error(f"❌ 알림 발송 중 오류: {str(e)}")
            else:
                st.caption("ℹ️ 알림 기능이 비활성화 상태입니다. (.env에서 NOTIFICATION_ENABLED=true로 설정)")

            # 결과 테이블 + 상담일지 버튼
            st.markdown("#### 위험도 분석 결과")
            for r in risk_results:
                level_color = {"위험": "🔴", "경고": "🟡", "정상": "🟢"}.get(r.final_level, "")
                cols = st.columns([2, 1, 1, 1, 3, 2])
                cols[0].write(r.name)
                cols[1].write(f"{level_color} {r.final_level}")
                cols[2].write(f"{r.final_score:.2f}")
                cols[3].write(r.attendance_level)
                cols[4].write(r.recommendation)

                if r.final_level in ("경고", "위험"):
                    if cols[5].button("상담일지 생성", key=f"docx_{r.name}"):
                        today_str = date.today().strftime("%Y-%m-%d")
                        content = generate_counseling_log(
                            name=r.name,
                            keywords=f"위험도: {r.final_level}, 출결등급: {r.attendance_level}",
                            risk_level=r.final_level,
                            absent_info=f"위험점수 {r.final_score:.2f}",
                        )
                        docx_bytes = generate_counseling_docx(
                            trainee_name=r.name,
                            course_name=course_name_input,
                            counseling_date=today_str,
                            content=content,
                            action=r.recommendation,
                        )
                        st.download_button(
                            label=f"📥 {r.name} 상담일지 다운로드",
                            data=docx_bytes,
                            file_name=f"상담일지_{r.name}_{today_str}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"dl_{r.name}",
                        )

# ── 탭 2: AI 강의 비서 ────────────────────────────────────────
with tab2:
    st.subheader("💬 AI 강의 비서")
    st.caption("에러 메시지나 코드를 붙여넣으면 해결책을 알려드립니다.")

    # RAG: PDF 업로드
    pdf_file = st.file_uploader("교안 PDF 업로드 (선택 — RAG 기반 답변)", type="pdf", key="rag_pdf")

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if pdf_file is not None:
        with st.spinner("PDF 인덱싱 중..."):
            try:
                engine = RAGEngine()
                engine.load_pdf(pdf_file.read())
                st.session_state.rag_engine = engine
                st.success("✅ PDF 인덱싱 완료. 교안 기반 답변이 활성화되었습니다.")
            except RuntimeError as e:
                st.error(f"❌ PDF 처리 실패: {e}")

    # 멀티턴 히스토리 관리
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 대화 히스토리 표시
    for msg in st.session_state.chat_history:
        role_label = "🧑 훈련생" if msg["role"] == "user" else "🤖 AI 튜터"
        st.markdown(f"**{role_label}:** {msg['content']}")

    context_input = st.text_area(
        "현재 학습 중인 교안 맥락 (선택)",
        height=60,
        placeholder="예: pandas 기초 - DataFrame 인덱싱",
        key="chat_context",
    )
    question = st.text_area(
        "질문 또는 에러 메시지",
        height=120,
        placeholder="예: KeyError: 'age' 에러가 발생했어요",
        key="chat_question",
    )

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_btn = st.button("답변 받기")
    with col_clear:
        if st.button("히스토리 초기화"):
            st.session_state.chat_history = clear_history()
            st.rerun()

    if ask_btn:
        if question.strip():
            # traceback 자동 감지
            tb = detect_traceback(question)
            effective_context = context_input
            if tb:
                effective_context = f"[에러 컨텍스트 자동 감지]\n{tb}\n\n{context_input}".strip()

            # RAG 컨텍스트 주입
            if st.session_state.rag_engine is not None:
                rag_ctx = st.session_state.rag_engine.get_context_prompt(question)
                if rag_ctx:
                    effective_context = f"{rag_ctx}\n\n{effective_context}".strip()

            with st.spinner("AI가 답변을 생성 중입니다..."):
                answer = ask_assistant(
                    question,
                    context=effective_context,
                    history=st.session_state.chat_history,
                )

            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
        else:
            st.warning("질문을 입력해주세요.")

# ── 탭 3: 행정 서류 자동 생성 ─────────────────────────────────
with tab3:
    st.subheader("📝 행정 서류 자동 생성")

    doc_type = st.radio("서류 종류", ["상담일지", "사유서"])

    if doc_type == "상담일지":
        col1, col2 = st.columns(2)
        with col1:
            adm_name = st.text_input("훈련생 이름", placeholder="예: 홍길동")
            adm_course = st.text_input("과정명", placeholder="예: K-디지털 트레이닝")
            keywords = st.text_input("상담 키워드", placeholder="예: 취업 고민, 파이썬 기초 부족")
            risk_level = st.selectbox("위험 등급", ["정상", "경고", "위험"])
        with col2:
            absent_info = st.text_input("출결 현황", placeholder="예: 지각 2회, 조퇴 1회")
            adm_date = st.date_input("상담일시", value=date.today())
            adm_action = st.text_input("조치사항", placeholder="예: 개인 면담 실시, 멘토링 연결")

        if st.button("상담일지 초안 생성"):
            if keywords:
                with st.spinner("생성 중..."):
                    draft = generate_counseling_log(
                        name=adm_name or "[이름]",
                        keywords=keywords,
                        risk_level=risk_level,
                        absent_info=absent_info or "정상",
                    )
                st.text_area("상담일지 초안", value=draft, height=200)

                # .docx 다운로드
                docx_bytes = generate_counseling_docx(
                    trainee_name=adm_name or "[이름]",
                    course_name=adm_course or "[과정명]",
                    counseling_date=str(adm_date),
                    content=draft,
                    action=adm_action or "[조치사항]",
                )
                st.download_button(
                    label="📥 상담일지 .docx 다운로드",
                    data=docx_bytes,
                    file_name=f"상담일지_{adm_name}_{adm_date}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.warning("상담 키워드를 입력해주세요.")

    else:
        situation = st.text_area(
            "상황 설명",
            height=100,
            placeholder="예: 2026-04-07 오전 9시 HRD-Net 시스템 장애로 출결 체크 불가",
        )
        if st.button("사유서 초안 생성"):
            if situation.strip():
                with st.spinner("생성 중..."):
                    draft = generate_reason_letter(situation)
                st.text_area("사유서 초안", value=draft, height=200)
                docx_bytes = generate_counseling_docx(
                    trainee_name="[이름]",
                    course_name="[과정명]",
                    counseling_date=str(date.today()),
                    content=draft,
                    action="사유서 제출",
                )
                st.download_button(
                    label="📥 사유서 .docx 다운로드",
                    data=docx_bytes,
                    file_name=f"사유서_{date.today()}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.warning("상황을 입력해주세요.")

# ── 탭 4: 카카오톡 감성 분석 ──────────────────────────────────
with tab4:
    st.subheader("💬 카카오톡 감성 분석")
    st.caption("카카오톡 단톡방 내보내기 .txt 파일을 업로드하면 훈련생별 감성 분석을 실행합니다.")

    kakao_file = st.file_uploader("카카오톡 내보내기 .txt 업로드", type="txt", key="kakao_txt")

    if kakao_file is not None:
        try:
            text_content = kakao_file.read().decode("utf-8", errors="replace")
            messages = parse_kakao_export(text_content)

            if not messages:
                st.warning("파싱된 메시지가 없습니다. 카카오톡 내보내기 형식인지 확인해주세요.")
            else:
                st.success(f"✅ {len(messages)}개 메시지 파싱 완료")

                grouped = group_by_sender(messages)
                senders = list(grouped.keys())

                # 동명이인 감지 및 사용자 확인
                duplicates = detect_duplicate_senders(messages)
                if duplicates:
                    st.warning(
                        f"⚠️ 다음 발신자명이 동명이인일 수 있습니다: {', '.join(duplicates)}\n"
                        "동일 이름이 서로 다른 훈련생인 경우 직접 확인 후 분석을 진행하세요."
                    )
                    confirmed = st.checkbox("동명이인 확인 완료, 분석을 계속합니다.")
                    if not confirmed:
                        st.stop()

                st.markdown(f"**발신자 목록:** {', '.join(senders)}")

                # 배치 감성 분석
                if st.button("배치 감성 분석 실행"):
                    sentiment_trend: dict[str, list[float]] = {}
                    progress = st.progress(0)
                    for i, (sender, msgs) in enumerate(grouped.items()):
                        scores = []
                        for m in msgs[:10]:  # 최대 10개 메시지만 분석 (API 비용 절감)
                            try:
                                result = analyze_sentiment(m.content)
                                scores.append(result["score"])
                            except Exception:
                                scores.append(0.0)
                        sentiment_trend[sender] = scores
                        progress.progress((i + 1) / len(grouped))

                    st.plotly_chart(
                        render_sentiment_trend_chart(sentiment_trend),
                        use_container_width=True,
                    )

                    # 번아웃 경고
                    from modules.visualizer import detect_burnout_periods
                    burnout_alerts = []
                    for sender, scores in sentiment_trend.items():
                        periods = detect_burnout_periods(scores)
                        if periods:
                            burnout_alerts.append(sender)

                    if burnout_alerts:
                        st.warning(
                            f"🔥 번아웃 패턴 감지 (연속 3일 이상 부정 감성 ≥ 0.6): "
                            f"{', '.join(burnout_alerts)}"
                        )

        except ValueError as e:
            st.error(f"❌ 파싱 실패: {e}")
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")

# ── 탭 5: 분석 히스토리 ───────────────────────────────────────
with tab5:
    st.subheader("📈 분석 히스토리")
    st.caption("저장된 위험도 분석 결과를 날짜 범위로 조회합니다.")

    hist_col1, hist_col2, hist_col3 = st.columns(3)
    with hist_col1:
        hist_name = st.text_input("훈련생 이름 (선택)", placeholder="전체 조회 시 비워두세요")
    with hist_col2:
        hist_start = st.date_input("시작일", value=date.today() - timedelta(days=30))
    with hist_col3:
        hist_end = st.date_input("종료일", value=date.today())

    if st.button("히스토리 조회"):
        history = ds.get_risk_history(
            name=hist_name or None,
            start_date=str(hist_start),
            end_date=str(hist_end) + " 23:59:59",
        )
        if history:
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, use_container_width=True)
            st.caption(f"총 {len(history)}건 조회됨")
        else:
            st.info("조회된 분석 히스토리가 없습니다.")
