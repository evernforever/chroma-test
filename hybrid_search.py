"""
hybrid_search.py — BM25 인덱스 관리 및 하이브리드 검색

벡터 검색 70% + BM25 키워드 검색 30% 결합
"""

import pickle
import numpy as np
from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = "./bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    """음절 n-gram(2~3글자) 토크나이저.

    형태소 분석 없이 글자 단위로 쪼개므로
    '대표자' ↔ '대표이사' 처럼 표현이 달라도 공통 n-gram('대표')으로 매칭된다.
    """
    tokens = []
    for word in text.split():
        # 2-gram
        for i in range(len(word) - 1):
            tokens.append(word[i:i + 2])
        # 3-gram
        for i in range(len(word) - 2):
            tokens.append(word[i:i + 3])
        # 2글자 이하 단어는 통째로 보존
        if len(word) <= 2:
            tokens.append(word)
    return tokens


def build_bm25_index(
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> dict:
    """문서 목록으로 BM25 인덱스를 빌드하고 인덱스 데이터 딕셔너리를 반환."""
    print("  BM25 인덱스 빌드 중 (형태소 분석)...")
    tokenized = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    return {
        "bm25": bm25,
        "texts": documents,
        "metadatas": metadatas,
        "ids": ids,
    }


def save_bm25_index(index_data: dict, path: str = BM25_INDEX_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(index_data, f)
    print(f"  BM25 인덱스 저장 완료 → {path}")


def load_bm25_index(path: str = BM25_INDEX_PATH) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def _normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


def hybrid_search(
    query: str,
    collection,
    bm25_data: dict,
    top_k: int = 5,
    vector_weight: float = 0.7,
) -> tuple[list[str], list[dict], list[float]]:
    """
    벡터 검색(70%)과 BM25 키워드 검색(30%)을 결합한 하이브리드 검색.

    Returns: (texts, metadatas, combined_scores)  — scores 범위: 0~1
    """
    bm25_weight = 1.0 - vector_weight
    total_docs = len(bm25_data["texts"])
    candidate_k = min(top_k * 4, total_docs)

    # ── 벡터 검색 ──────────────────────────────────────────────
    vec_results = collection.query(
        query_texts=[query],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )
    vec_docs = vec_results["documents"][0]
    vec_metas = vec_results["metadatas"][0]
    # cosine distance(0~2) → similarity(0~1)
    vec_scores = np.array([1.0 - d for d in vec_results["distances"][0]])
    vec_norm = _normalize(vec_scores)

    # ── BM25 검색 ──────────────────────────────────────────────
    query_tokens = tokenize(query)
    bm25_all = np.array(bm25_data["bm25"].get_scores(query_tokens))
    top_bm25_idx = np.argsort(bm25_all)[::-1][:candidate_k]
    bm25_norm = _normalize(bm25_all[top_bm25_idx])

    # ── 후보 합산 (청크 ID 기준 중복 제거) ─────────────────────
    def chunk_id(meta: dict, fallback: int) -> str:
        return f"{meta.get('source', '')}#{meta.get('chunk_index', fallback)}"

    candidates: dict[str, dict] = {}

    for i, (doc, meta, vs) in enumerate(zip(vec_docs, vec_metas, vec_norm)):
        cid = chunk_id(meta, i)
        candidates[cid] = {"text": doc, "meta": meta, "vec": float(vs), "bm25": 0.0}

    for rank, idx in enumerate(top_bm25_idx):
        meta = bm25_data["metadatas"][idx]
        cid = chunk_id(meta, int(idx))
        bs = float(bm25_norm[rank])
        if cid in candidates:
            candidates[cid]["bm25"] = bs
        else:
            candidates[cid] = {
                "text": bm25_data["texts"][idx],
                "meta": meta,
                "vec": 0.0,
                "bm25": bs,
            }

    # ── 최종 점수 계산 및 정렬 ─────────────────────────────────
    ranked = sorted(
        candidates.values(),
        key=lambda c: vector_weight * c["vec"] + bm25_weight * c["bm25"],
        reverse=True,
    )[:top_k]

    texts = [c["text"] for c in ranked]
    metas = [c["meta"] for c in ranked]
    scores = [vector_weight * c["vec"] + bm25_weight * c["bm25"] for c in ranked]
    return texts, metas, scores
