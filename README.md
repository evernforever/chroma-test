# ChromaDB RAG Demo

로컬 문서를 ChromaDB에 저장하고 Claude API로 답변하는 한국어 RAG 시연 프로젝트입니다.

## 기술 스택

- **Vector DB**: ChromaDB
- **Embedding 모델**: nlpai-lab/KURE-v1 (한국어 특화)
- **LLM**: Claude API (claude-sonnet-4-6)
- **Web UI**: Streamlit

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

`sample_docs/` 폴더의 문서를 청킹하여 ChromaDB에 저장합니다.

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
python query.py "대표이사가 누구야?"
python query.py  # 대화형 모드
```

## 프로젝트 구조

```
chroma-test/
├── sample_docs/          # 검색 대상 문서
│   ├── 회사소개.txt
│   ├── 제품카탈로그.txt
│   └── 직원복지규정.txt
├── ingest.py             # 문서 → ChromaDB 저장
├── query.py              # 터미널용 질의응답
├── app.py                # Streamlit 웹 UI
├── requirements.txt
├── .env                  # API 키 (git 제외)
└── .gitignore
```

## 동작 흐름

```
로컬 문서 → 텍스트 청킹 → KURE-v1 임베딩 → ChromaDB 저장
                                                    ↓
질문 입력 → 유사 청크 검색 → Claude API (컨텍스트 포함) → 답변
```
