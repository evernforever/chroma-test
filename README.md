# ChromaDB RAG Demo

로컬 문서를 ChromaDB에 저장하고 Claude API로 답변하는 한국어 RAG 시연 프로젝트입니다.

## 기술 스택

- **Vector DB**: ChromaDB
- **Embedding 모델**: nlpai-lab/KURE-v1 (한국어 특화)
- **LLM**: Claude API (claude-sonnet-4-6)
- **Web UI**: Streamlit
- **키워드 검색**: BM25 (rank-bm25, 음절 n-gram 토크나이저)

## 시작하기

### 1. 가상환경 생성 및 패키지 설치

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. API 키 설정

프로젝트 루트에 `.env` 파일을 생성하고 Anthropic API 키를 입력합니다.

```
ANTHROPIC_API_KEY=sk-ant-여기에_실제_키를_입력하세요
```

> Anthropic API 키는 [console.anthropic.com](https://console.anthropic.com)에서 발급받을 수 있습니다.

### 3. 문서 인제스트

`sample_docs/` 폴더의 문서를 청킹하여 ChromaDB에 저장하고, BM25 인덱스(`bm25_index.pkl`)를 생성합니다.

```bash
python ingest.py
```

### 4. 실행

**Streamlit 웹 UI:**
```bash
streamlit run app.py
```
브라우저에서 `http://localhost:8501` 접속

**터미널에서 직접 질문:**
```bash
# 벡터 검색 100% (기본)
python query.py "대표이사가 누구야?"
python query.py                      # 대화형 모드

# 하이브리드 검색 (벡터 70% + BM25 30%)
python query.py --hybrid "대표자?"
python query.py --hybrid             # 대화형 모드
```

## 프로젝트 구조

```
chroma-test/
├── sample_docs/          # 검색 대상 문서
│   ├── 회사소개.txt
│   ├── 제품카탈로그.txt
│   └── 직원복지규정.txt
├── ingest.py             # 문서 → ChromaDB 저장 + BM25 인덱스 생성
├── hybrid_search.py      # BM25 인덱스 관리 및 하이브리드 검색 로직
├── query.py              # 터미널용 질의응답
├── app.py                # Streamlit 웹 UI
├── chroma_db/            # ChromaDB 벡터 저장소 (자동 생성)
├── bm25_index.pkl        # BM25 인덱스 (ingest.py 실행 후 생성)
├── requirements.txt
├── .env                  # API 키 (git 제외)
└── .gitignore
```

## 동작 흐름

```
로컬 문서 → 텍스트 청킹 → KURE-v1 임베딩 → ChromaDB 저장
                       └→ BM25 인덱스 생성 → bm25_index.pkl 저장

질문 입력
  ├─ [벡터 모드]     → ChromaDB 유사도 검색
  └─ [하이브리드 모드] → 벡터 검색(70%) + BM25 키워드 검색(30%) → 점수 합산
                                    ↓
                    Claude API (상위 청크를 컨텍스트로 전달) → 답변
```

## 하이브리드 검색

순수 벡터 검색은 짧은 키워드 쿼리("대표?", "김민준" 등 고유명사)에서 정확도가 낮아질 수 있습니다. BM25 키워드 검색을 결합하면 이런 경우를 보완할 수 있습니다.

| 쿼리 유형 | 벡터 검색 | BM25 검색 |
|---|---|---|
| "복지 혜택이 뭐가 있어?" (자연어) | 강함 | 보통 |
| "대표자?" (단독 키워드) | 약함 | 강함 |
| "김민준" (고유명사) | 약함 | 강함 |

BM25 토크나이저는 **음절 n-gram(2~3글자)** 방식을 사용합니다. 형태소 분석 없이도 "대표자" ↔ "대표이사"처럼 표현이 다른 단어 간 공통 n-gram("대표")으로 매칭됩니다.

웹 UI에서는 사이드바의 **"하이브리드 검색"** 토글로 모드를 전환할 수 있습니다.
