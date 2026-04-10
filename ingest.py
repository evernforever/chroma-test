"""
ingest.py — 로컬 문서를 청킹하여 ChromaDB에 저장하는 스크립트

사용법: python ingest.py
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "sample_docs"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 400  # 청크당 최대 글자 수
CHUNK_OVERLAP = 50  # 청크 간 겹치는 글자 수


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """텍스트를 문단 기준으로 청킹. 청크가 너무 길면 추가 분할."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            # 문단 자체가 chunk_size보다 길면 강제 분할
            if len(para) > chunk_size:
                for start in range(0, len(para), chunk_size - overlap):
                    piece = para[start : start + chunk_size]
                    chunks.append(piece)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


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

    print(f"\n완료! {len(all_documents)}개 청크가 '{CHROMA_PATH}'에 저장되었습니다.")
    print("이제 query.py로 질문할 수 있습니다.\n")


if __name__ == "__main__":
    ingest_documents()
