"""
app.py — Streamlit 기반 RAG 챗 UI
실행: streamlit run app.py
"""

import streamlit as st
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from ingest import CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"
TOP_K = 5
CLAUDE_MODEL = "claude-sonnet-4-6"

st.set_page_config(page_title="TechStar 문서 검색 챗봇", page_icon="🤖", layout="centered")
st.title("🤖 TechStar 문서 검색 챗봇")
st.caption("사내 문서(회사소개, 제품카탈로그, 직원복지규정)를 기반으로 답변합니다.")

st.markdown("""
<style>
[data-testid="stSidebarCollapsedControl"]::after,
[data-testid="collapsedControl"]::after,
button[kind="headerNoPadding"]::after {
    content: "문서현황";
    font-size: 1.0rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin-top: 0px;
    line-height: 1;
    color: inherit;
    position: relative;
    top: +2px;
}
[data-testid="stChatMessage"] {
    align-items: flex-start !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="nlpai-lab/KURE-v1"
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def search_documents(query: str) -> tuple[list[str], list[dict], list[float]]:
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    return results["documents"][0], results["metadatas"][0], results["distances"][0]


def ask_claude(question: str, chunks: list[str], metadatas: list[dict]):
    context_sections = []
    for chunk, meta in zip(chunks, metadatas):
        context_sections.append(f"[출처: {meta.get('source', '알 수 없음')}]\n{chunk}")
    context = "\n\n---\n\n".join(context_sections)

    prompt = f"""아래는 사내 문서에서 검색된 관련 내용입니다.

{context}

---

위 내용을 참고하여 다음 질문에 답해주세요.
문서에 명시되지 않은 내용은 "해당 정보는 문서에서 확인할 수 없습니다"라고 솔직하게 답하세요.

질문: {question}"""

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


# 앱 시작 시 모델 로딩 (최초 1회)
with st.spinner("모델 로딩 중..."):
    get_collection()
st.success("임베딩 모델 로딩 완료 (nlpai-lab/KURE-v1)", icon="✅")

# 사이드바 — 문서 청크 현황
with st.sidebar:
    st.header("문서 현황")
    collection = get_collection()
    result = collection.get(include=["metadatas", "documents"])
    from collections import defaultdict
    chunk_map = defaultdict(list)
    for doc, meta in zip(result["documents"], result["metadatas"]):
        chunk_map[meta["source"]].append((meta["chunk_index"], doc))
    counter = {src: len(chunks) for src, chunks in chunk_map.items()}
    total = sum(counter.values())
    st.metric("총 청크 수", total)
    st.caption("청킹 알고리즘: 고정길이 Recursive")
    st.caption(f"청크 사이즈: {CHUNK_SIZE}자 / 오버랩: {CHUNK_OVERLAP}자")
    st.divider()
    for source in sorted(chunk_map.keys()):
        count = counter[source]
        with st.expander(f"📄 {source} ({count}개 청크)"):
            chunks_sorted = sorted(chunk_map[source], key=lambda x: x[0])
            for idx, content in chunks_sorted:
                st.markdown(f"**— 청크 #{idx}**")
                st.text(content)
                st.divider()

# 채팅 히스토리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("참고한 문서 청크 보기"):
                for src in msg["sources"]:
                    with st.expander(f"📄 {src['source']} (청크 #{src['chunk_index']}, 유사도: {src['similarity']:.3f})"):
                        st.text(src["content"])

# 입력창
if question := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 검색 + 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("문서 검색 중..."):
            try:
                chunks, metadatas, distances = search_documents(question)
                sources = [
                    {
                        "source": m["source"],
                        "chunk_index": m["chunk_index"],
                        "similarity": 1 - d,
                        "content": chunk,
                    }
                    for chunk, m, d in zip(chunks, metadatas, distances)
                ]
            except Exception as e:
                st.error(f"ChromaDB 오류: {e}\n먼저 ingest.py를 실행하세요.")
                st.stop()

        answer = st.write_stream(ask_claude(question, chunks, metadatas))

        with st.expander("참고한 문서 청크 보기"):
            for src in sources:
                with st.expander(f"📄 {src['source']} (청크 #{src['chunk_index']}, 유사도: {src['similarity']:.3f})"):
                    st.text(src["content"])

    # 히스토리 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
