"""
태스크 1: AI 강의 비서
- 훈련생이 에러 메시지 + 코드를 입력하면 해결책 제시
- OpenAI API 사용, 교안 맥락을 system prompt에 주입
"""
import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
당신은 국비지원 IT 부트캠프의 AI 강의 보조 튜터입니다.
훈련생이 겪는 파이썬, 머신러닝, 딥러닝, 데이터 엔지니어링 관련 에러와 질문에 답합니다.
- 에러 메시지가 있으면 원인과 해결책을 단계별로 설명하세요.
- 코드가 있으면 문제 있는 줄을 짚어주세요.
- 비전공자 기준으로 쉽게 설명하세요.
- 답변은 한국어로 작성하세요.
"""


def ask_assistant(
    user_message: str,
    context: str = "",
    history: Optional[list] = None,
) -> str:
    """
    훈련생 질문에 AI 답변 반환.

    Args:
        user_message: 훈련생의 질문 또는 에러 메시지
        context: 추가 교안 맥락 (선택)
        history: 이전 대화 히스토리 (선택). 각 항목은 {"role": ..., "content": ...} 형식.

    Returns:
        AI 답변 문자열
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OpenAI API 키가 설정되지 않아 AI 기능을 사용할 수 없습니다."

    if not user_message.strip():
        return "질문 내용을 입력해주세요."

    try:
        client = OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append({
                "role": "system",
                "content": f"현재 학습 중인 교안 맥락:\n{context}"
            })

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI 답변 생성 중 오류가 발생했습니다: {str(e)}"


def detect_traceback(text: str) -> Optional[str]:
    """Python traceback 패턴 감지 및 추출.

    Args:
        text: 검사할 문자열

    Returns:
        traceback 패턴이 포함된 경우 해당 텍스트, 없으면 None
    """
    pattern = "Traceback (most recent call last):"
    if pattern in text:
        idx = text.index(pattern)
        return text[idx:]
    return None


def clear_history() -> list:
    """빈 히스토리 반환 (session_state 초기화용).

    Returns:
        빈 리스트
    """
    return []
