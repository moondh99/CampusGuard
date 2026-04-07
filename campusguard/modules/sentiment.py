"""
태스크 3: 감성 분석
- 훈련생 텍스트(단톡방 메시지, 게시판 글 등)에서 부정 감정 감지
- LLM 기반으로 포기 의사, 강사 불만, 번아웃 신호 탐지
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SENTIMENT_PROMPT = """
아래 텍스트를 분석하여 JSON으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

분석 항목:
- sentiment: "긍정" / "중립" / "부정"
- score: 부정 강도 0.0(없음) ~ 1.0(매우 강함)
- signals: 감지된 위험 신호 목록 (포기의사, 강사불만, 번아웃, 학습포기, 없음 중 해당하는 것)
- summary: 한 줄 요약

응답 형식 예시:
{"sentiment": "부정", "score": 0.8, "signals": ["포기의사", "번아웃"], "summary": "학습 포기를 고려 중인 상태"}
"""


def analyze_sentiment(text: str) -> dict:
    """
    텍스트 감성 분석.

    Args:
        text: 분석할 텍스트

    Returns:
        dict: sentiment, score, signals, summary
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"sentiment": "중립", "score": 0.0, "signals": ["없음"], "summary": "API 키 미설정"}

    if not text.strip():
        return {"sentiment": "중립", "score": 0.0, "signals": ["없음"], "summary": "입력 없음"}

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SENTIMENT_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        return {"sentiment": "중립", "score": 0.0, "signals": ["없음"], "summary": f"오류: {str(e)}"}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"sentiment": "중립", "score": 0.0, "signals": ["없음"], "summary": "응답 파싱 실패"}


def is_high_risk(result: dict, threshold: float = 0.6) -> bool:
    """부정 점수가 임계값 이상이면 고위험으로 판단."""
    return result.get("score", 0.0) >= threshold
