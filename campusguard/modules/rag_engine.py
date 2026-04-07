"""
태스크 10: RAGEngine — PDF 기반 RAG (Retrieval-Augmented Generation)
- PDF 업로드 → 청크 분할 → 벡터 임베딩 → FAISS 인덱스 저장
- 질문 입력 → 유사도 검색 → 상위 3개 청크를 시스템 프롬프트에 주입
"""
import io
import os
from typing import Optional

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class RAGEngine:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu 패키지가 설치되지 않아 RAG 기능을 사용할 수 없습니다.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[str] = []
        self._index = None
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def load_pdf(self, pdf_bytes: bytes) -> None:
        """PDF 바이트에서 텍스트를 추출하고 청크로 분할한 뒤 인덱스를 빌드한다.

        Args:
            pdf_bytes: PDF 파일의 바이트 데이터

        Raises:
            RuntimeError: PDF 파싱 실패 시
        """
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            full_text = "\n".join(text_parts)
        except Exception as e:
            raise RuntimeError(f"PDF 파싱 실패: {e}") from e

        chunks = self.chunk_text(full_text)
        self._chunks = chunks
        if chunks:
            self.build_index(chunks)

    def chunk_text(self, text: str) -> list[str]:
        """텍스트를 chunk_size 크기로 chunk_overlap 만큼 겹치게 분할한다.

        Args:
            text: 분할할 원본 텍스트

        Returns:
            청크 문자열 리스트
        """
        if not text:
            return []

        chunks = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start += step

        return chunks

    def _embed(self, texts: list[str]) -> np.ndarray:
        """OpenAI text-embedding-3-small 모델로 텍스트 임베딩 생성.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            shape (len(texts), EMBEDDING_DIM) float32 배열
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def build_index(self, chunks: list[str]) -> None:
        """청크 리스트를 임베딩하여 FAISS IndexFlatL2에 저장한다.

        Args:
            chunks: 인덱싱할 텍스트 청크 리스트
        """
        if not chunks:
            return

        self._chunks = chunks
        vectors = self._embed(chunks)
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(vectors)
        self._index = index

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """쿼리와 유사한 청크를 벡터 검색으로 반환한다.

        Args:
            query: 검색 쿼리 문자열
            top_k: 반환할 최대 청크 수 (기본값 3)

        Returns:
            유사도 상위 top_k개 청크 리스트. 인덱스 미초기화 시 빈 리스트.
        """
        if self._index is None or not self._chunks:
            return []

        k = min(top_k, len(self._chunks))
        query_vec = self._embed([query])
        _, indices = self._index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._chunks):
                results.append(self._chunks[idx])
        return results

    def get_context_prompt(self, query: str) -> str:
        """쿼리 관련 상위 3개 청크를 시스템 프롬프트 형식으로 반환한다.

        Args:
            query: 검색 쿼리 문자열

        Returns:
            시스템 프롬프트용 컨텍스트 문자열
        """
        chunks = self.search(query, top_k=3)
        if not chunks:
            return ""

        context_parts = ["다음은 교안에서 관련된 내용입니다:\n"]
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[참고 {i}]\n{chunk}")

        return "\n\n".join(context_parts)
