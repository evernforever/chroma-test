"""
query.py — ChromaDB에서 관련 문서를 검색하고 Claude API로 답변 생성

사용법:
  python query.py "질문 내용"
  python query.py  (인수 없이 실행하면 대화형 모드)
"""

import sys
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Windows 터미널 UTF-8 인코딩 강제 설정
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 3  # 검색할 유사 청크 수
CLAUDE_MODEL = "claude-sonnet-4-6"


def load_collection():
    """임베딩 모델과 ChromaDB 컬렉션을 로드."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="nlpai-lab/KURE-v1"
    )
    try:
        return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception:
        print("오류: ChromaDB 컬렉션을 찾을 수 없습니다.")
        print("먼저 'python ingest.py'를 실행해 문서를 저장하세요.")
        sys.exit(1)


def search_documents(collection, query: str, top_k: int = TOP_K):
    """ChromaDB에서 질문과 가장 유사한 청크를 검색."""
    return collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def build_context(chunks: list[str], metadatas: list[dict]) -> str:
    """검색된 청크를 출처 정보와 함께 컨텍스트 문자열로 조합."""
    sections = []
    for chunk, meta in zip(chunks, metadatas):
        source = meta.get("source", "알 수 없음")
        sections.append(f"[출처: {source}]\n{chunk}")
    return "\n\n---\n\n".join(sections)


def ask_claude_stream(question: str, context: str):
    """Claude API 스트리밍으로 토큰 단위 출력."""
    client = anthropic.Anthropic()

    prompt = f"""아래는 사내 문서에서 검색된 관련 내용입니다.

{context}

---

위 내용을 참고하여 다음 질문에 답해주세요.
문서에 명시되지 않은 내용은 "해당 정보는 문서에서 확인할 수 없습니다"라고 솔직하게 답하세요.

질문: {question}"""

    full_answer = []
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_answer.append(text)

    print()  # 스트리밍 끝나면 줄바꿈
    return "".join(full_answer)


def run_query(collection, question: str):
    import time

    print(f"\n질문: {question}")
    print("\n검색 중...")

    t0 = time.time()
    results = search_documents(collection, question)
    search_elapsed = time.time() - t0

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\n관련 문서 {len(chunks)}개 발견: (ChromaDB 검색 소요시간: {search_elapsed:.3f}s)")
    for i, (meta, dist) in enumerate(zip(metadatas, distances), 1):
        similarity = 1 - dist  # cosine distance → similarity
        print(f"  {i}. {meta['source']} (청크 #{meta['chunk_index']}, 유사도: {similarity:.3f})")

    context = build_context(chunks, metadatas)

    prompt = f"""아래는 사내 문서에서 검색된 관련 내용입니다.

{context}

---

위 내용을 참고하여 다음 질문에 답해주세요.
문서에 명시되지 않은 내용은 "해당 정보는 문서에서 확인할 수 없습니다"라고 솔직하게 답하세요.

질문: {question}"""

    print("\n" + "=" * 60)
    print("\n")
    print(">" * 60)
    print("Claude에게 보내는 프롬프트 전문")
    print(">" * 60)
    print("\n")
    print(prompt)
    print("\n")
    print(">" * 60)
    print("Claude에게 보내는 프롬프트 전문 끝")
    print(">" * 60)
    print("\n")
    print("\n")
    print("=" * 60)
    print("답변")
    print("=" * 60)
    t1 = time.time()
    ask_claude_stream(question, context)
    claude_elapsed = time.time() - t1

    print("=" * 60)
    print(f"Claude 소요시간: {claude_elapsed:.3f}s")
    print("=" * 60)


def interactive_mode(collection):
    print("준비 완료. 질문을 입력하세요. (종료: 'exit' 또는 Ctrl+C)\n")
    while True:
        try:
            question = input("질문> ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "종료"):
                print("종료합니다.")
                break
            run_query(collection, question)
            print()
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break


if __name__ == "__main__":
    import time
    print("모델 로딩 중...", end=" ", flush=True)
    t = time.time()
    collection = load_collection()
    print(f"완료 ({time.time() - t:.1f}s)\n")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        run_query(collection, question)
    else:
        interactive_mode(collection)
