"""
ingest.py — 로컬 문서를 청킹하여 ChromaDB에 저장하는 스크립트

사용법: python ingest.py
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from hybrid_search import build_bm25_index, save_bm25_index

load_dotenv()

DOCS_DIR = "sample_docs"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 400  # 청크당 최대 글자 수
CHUNK_OVERLAP = 50  # 청크 간 겹치는 글자 수

# Recursive 분할 구분자 우선순위: 문단 → 줄바꿈 → 문장 → 단어
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Recursive Character 방식 청킹.
    구분자 우선순위(\n\n → \n → '. ' → ' ' → '')로 재귀 분할하여
    의미 단위를 최대한 보존하면서 chunk_size 이하로 나눔.
    """

    def _split(text: str, separators: list[str]) -> list[str]:
        sep = separators[0]
        next_seps = separators[1:]

        # 현재 구분자로 분리
        parts = text.split(sep) if sep else list(text)
        parts = [p for p in parts if p.strip()]

        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # part 자체가 chunk_size 초과 → 더 작은 구분자로 재귀 분할
                if len(part) > chunk_size and next_seps:
                    sub_chunks = _split(part, next_seps)
                    # 오버랩 적용: 이전 청크 끝부분을 다음 청크 앞에 붙임
                    if chunks and overlap > 0:
                        tail = chunks[-1][-overlap:]
                        sub_chunks[0] = tail + sep + sub_chunks[0]
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1] if sub_chunks else ""
                else:
                    current = part

        if current:
            chunks.append(current)

        # 오버랩 적용
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                tail = overlapped[-1][-overlap:]
                overlapped.append(tail + sep + chunks[i] if tail else chunks[i])
            return overlapped

        return chunks

    return _split(text, SEPARATORS)


def ingest_documents():
    if not os.path.exists(DOCS_DIR):
        print(f"오류: '{DOCS_DIR}' 폴더가 없습니다.")
        return

    # ChromaDB 클라이언트 및 임베딩 함수 초기화
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="nlpai-lab/KURE-v1"
    )

    # 기존 컬렉션 초기화 (재인제스트 시 중복 방지)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 컬렉션 '{COLLECTION_NAME}' 삭제 완료")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_documents = []
    all_ids = []
    all_metadatas = []
    chunk_counter = 0

    supported_exts = (".txt", ".md")
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(supported_exts)]

    if not files:
        print(f"'{DOCS_DIR}' 폴더에 .txt 또는 .md 파일이 없습니다.")
        return

    print(f"\n총 {len(files)}개 파일 처리 중...\n")

    for filename in files:
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"  [{filename}] → {len(chunks)}개 청크 생성")

        for i, chunk in enumerate(chunks):
            all_documents.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metadatas.append({"source": filename, "chunk_index": i})
            chunk_counter += 1

    # ChromaDB에 일괄 저장
    print(f"\n임베딩 생성 및 저장 중... (총 {len(all_documents)}개 청크)")
    collection.add(
        documents=all_documents,
        ids=all_ids,
        metadatas=all_metadatas,
    )

    # BM25 인덱스 빌드 및 저장
    print("\nBM25 인덱스 생성 중...")
    bm25_data = build_bm25_index(all_documents, all_metadatas, all_ids)
    save_bm25_index(bm25_data)

    print(f"\n완료! {len(all_documents)}개 청크가 '{CHROMA_PATH}'에 저장되었습니다.")
    print("이제 query.py로 질문할 수 있습니다.\n")


if __name__ == "__main__":
    ingest_documents()
