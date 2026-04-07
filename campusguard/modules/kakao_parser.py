"""
KakaoParser: 카카오톡 단톡방 내보내기 .txt 파일 파싱 모듈
"""
import re
from dataclasses import dataclass
from datetime import datetime


# 카카오톡 내보내기 형식 정규식
# 예: 2024년 3월 5일 오전 9:30, 홍길동 : 안녕하세요
KAKAO_LINE_PATTERN = re.compile(
    r"^(\d{4})년 (\d{1,2})월 (\d{1,2})일 (오전|오후) (\d{1,2}):(\d{2}), (.+) : (.+)$"
)


@dataclass
class KakaoMessage:
    sender: str
    timestamp: datetime
    content: str


def parse_kakao_export(text: str) -> list[KakaoMessage]:
    """카카오톡 내보내기 .txt 파싱.

    Args:
        text: 카카오톡 내보내기 파일 전체 텍스트

    Returns:
        파싱된 KakaoMessage 리스트. 빈 파일이면 빈 리스트 반환.

    Raises:
        ValueError: 카카오톡 내보내기 형식이 아닌 경우
    """
    if not text or not text.strip():
        return []

    messages: list[KakaoMessage] = []
    has_valid_line = False
    has_invalid_line = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = KAKAO_LINE_PATTERN.match(line)
        if match:
            has_valid_line = True
            year, month, day, ampm, hour, minute, sender, content = match.groups()
            year, month, day, hour, minute = int(year), int(month), int(day), int(hour), int(minute)

            # 오전/오후 변환
            if ampm == "오후" and hour != 12:
                hour += 12
            elif ampm == "오전" and hour == 12:
                hour = 0

            timestamp = datetime(year, month, day, hour, minute)
            messages.append(KakaoMessage(sender=sender, timestamp=timestamp, content=content))
        else:
            has_invalid_line = True

    # 유효한 줄이 하나도 없고 유효하지 않은 줄이 있으면 ValueError
    if not has_valid_line and has_invalid_line:
        raise ValueError("카카오톡 내보내기 형식이 아닙니다")

    return messages


def group_by_sender(messages: list[KakaoMessage]) -> dict[str, list[KakaoMessage]]:
    """발신자별 메시지 그룹화.

    Args:
        messages: KakaoMessage 리스트

    Returns:
        발신자명 → 메시지 리스트 딕셔너리
    """
    result: dict[str, list[KakaoMessage]] = {}
    for msg in messages:
        result.setdefault(msg.sender, []).append(msg)
    return result


def detect_duplicate_senders(messages: list[KakaoMessage]) -> list[str]:
    """동명이인 후보 반환.

    발신자명이 동일하지만 메시지 시간대 패턴이 크게 다른 경우 동명이인 후보로 반환한다.
    MVP 구현: 동일 발신자명이 존재하는 모든 발신자를 후보로 반환한다.
    (실제 서비스에서는 사용자 확인 요청 대상)

    Args:
        messages: KakaoMessage 리스트

    Returns:
        동명이인 후보 발신자명 리스트
    """
    from collections import Counter
    sender_counts = Counter(msg.sender for msg in messages)
    # 메시지가 있는 모든 발신자를 동명이인 확인 후보로 반환
    # (동일 이름이 실제로 다른 사람일 수 있으므로 UI에서 확인 요청)
    return [sender for sender, count in sender_counts.items() if count > 0]
