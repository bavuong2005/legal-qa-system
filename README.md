# RoadLawQA - Vietnamese Law QA System

## 🎬 Demo

![Demo](https://raw.githubusercontent.com/bavuong2005/legal-qa-system/refs/heads/main/demo.gif)

A question-answering system for Vietnamese road traffic law using RAG (Retrieval-Augmented Generation).

## 📋 Overview

This system combines:
- **BM25 + pyvi**: Keyword search with Vietnamese tokenization
- **Dense Retrieval**: Semantic search with Alibaba-NLP/gte-multilingual-base
- **Hybrid Search**: Dynamic alpha tuning based on query pattern
- **Reranking**: BAAI/bge-reranker-v2-m3 cross-encoder
- **Generation**: Gemini 2.5 Flash for Vietnamese answers

## 🏗️ Architecture

```
          ┌─────────────┐
          │   User      │
          │  Question   │
          └──────┬──────┘
                 │
                 v
┌──────────────────────────────────────┐
│          Hybrid Retrieval            │
│  ┌──────────┐       ┌──────────┐     │
│  │  BM25    │       │  Dense   │     │
│  │ (pyvi)   │       │ (gte-m)  │     │
│  └────┬─────┘       └────┬─────┘     │
│       └─────────┬────────┘           │
│                 v                    │
│         ┌───────────────┐            │
│         │ Alpha Tuning  │            │
│         └───────┬───────┘            │
│                 v                    │
│         ┌───────────────┐            │
│         │   Reranker    │            │
│         │  (bge-m3)     │            │
│         └───────┬───────┘            │
└─────────────────┼────────────────────┘
                  │
                  v
          ┌───────────────┐
          │    Context    │
          │   (Top 5)     │
          └───────┬───────┘
                  │
                  v
          ┌───────────────┐
          │    Gemini     │
          │  2.5 Flash    │
          └───────┬───────┘
                  │
                  v
          ┌───────────────┐
          │    Answer     │
          │  + Sources    │
          └───────────────┘
```

## 🚀 Setup

### 1. Requirements

```bash
pip install -r requirements.txt
````

### 2. Environment Variables

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start Weaviate

```bash
docker-compose up -d
```

### 4. Prepare data

Place legal text files into `data/raw/`:

* `nghidinhso-168-2024-NĐ-CP.txt`
* `luatso-36-2024-QH15.txt`
* `luatso-35-2024-QH15.txt`

### 5. Process data & Build Index

```bash
cd backend

# Step 1: Chunk legal documents
python clean_and_split.py

# Step 2: Build Weaviate index
python build_index.py
```

### 6. Run the application

```bash
# Frontend (Recommended) - Streamlit Web UI
cd frontend
streamlit run app.py

# Or CLI - Command line interface
cd backend
python rag_qa.py
```

## 💬 Usage

### Run Frontend (Web UI - Streamlit) - Recommended ⭐

```bash
cd frontend
streamlit run app.py
```

**Features:**

* ✨ Welcome screen with introduction and 3 illustrative images
* 💬 Modern chat interface with streaming text
* 📚 Source attribution (view referenced legal sources)
* ⏱️ Performance metrics (retrieval + generation time)
* 💡 Sample questions to get started quickly
* 🎚️ Settings sidebar: adjust number of chunks, clear history
* 📊 Display metrics: processing time, number of chunks used

### Run CLI (Command Line)

```bash
cd backend
python rag_qa.py
```

Or in Python:

```python
from retriever_custom import retrieve
from generator import generate_answer

question = "Kết cấu hạ tầng đường bộ bao gồm những gì?"

# Step 1: Retrieval
context, sources = retrieve(question, k=5)

# Step 2: Generation
answer, sources = generate_answer(question, context, sources)

print(answer)
```

Or simpler:

```python
from rag_qa import ask_law

question = "Kết cấu hạ tầng đường bộ bao gồm những gì?"
answer, sources = ask_law(question, k=5)

print(answer)
```

### Test Retriever

```bash
cd backend
python test_retriever.py
```

## 📊 Detailed Pipeline

### 1. Data Processing (`clean_and_split.py`)

* **Input**: Raw text files (.txt)
* **Output**: Structured JSON chunks (`data/processed/`)
* **Process**:

  * Hierarchical chunking: Điều → Khoản → Điểm → Bullet
  * Full context enrichment with tags `[CHAPTER]`, `[ARTICLE]`, `[CLAUSE]`, `[POINT]`
  * Sliding window for long chunks (max 1500 tokens)

### 2. Indexing (`build_index.py`)

* **Vector Database**: Weaviate
* **Embedding Model**: Alibaba-NLP/gte-multilingual-base
* **Schema**:

  * BM25 fields: `article_no`, `article_title`, `clause_no`, `point`, `clause_head`, `text`
  * Vector field: from `enriched_text`
  * Display fields: `display_citation`, `header`, `path_text`

### 3. Retrieval (`retriever_custom.py`)

#### BM25 Retrieval

* Tokenization: pyvi (Vietnamese word segmentation)
* Fields: article_no + article_title + clause_no + point + clause_head + text

#### Dense Retrieval

* Model: Alibaba-NLP/gte-multilingual-base
* Input: `enriched_text` (full context with tags)

#### Hybrid Search

* Formula: `score = α × dense_score + (1-α) × bm25_score`
* Dynamic alpha tuning:

  * Query with "Điều X", "Khoản Y" → α = 0.30 (favor BM25)
  * Query with numbers (km/h, triệu đồng) → α = 0.40
  * Semantic query → α = 0.75 (favor dense)

#### Reranking

* Model: BAAI/bge-reranker-v2-m3
* Top 20 candidates → Top 5 final results

### 4. Generation (`generator.py`)

* **Model**: Gemini 2.5 Flash
* **Prompt Strategy**:

  * System instruction: rules for accurate answers, no hallucination
  * Context: Top 5 chunks with citation `[Căn cứ: ...]`
  * Output format: Answer + "Legal basis: ..."

## 🎯 Features

* **Hierarchical Chunking**: Article → Clause → Point structure
* **Context Enrichment**: Full context with tags for embedding
* **Vietnamese Tokenization**: pyvi for BM25
* **Dynamic Alpha Tuning**: Automatically adjusts based on query pattern
* **Cross-Encoder Reranking**: High accuracy
* **Citation Tracking**: Clearly shows legal basis
* **Gemini Integration**: Vietnamese legal answer generation
* **Modern Web UI**: Streamlit chat interface with streaming
* **Source Attribution**: View references in Expander
* **Performance Metrics**: Display retrieval/generation time
* **Sample Questions**: Suggested queries
* **Adjustable Settings**: Control number of chunks, clear history

## 📂 Project Structure

```
legal-qa-system/
├── frontend/                   # Streamlit Web UI
│   ├── app.py                 # Main chat interface
│   ├── assets/                # Illustration images
│   │   ├── law.png
│   │   ├── traffic.png
│   │   └── legal.png
│   └── bm25_index.pkl         # BM25 index cache
│
├── backend/                   # RAG Pipeline
│   ├── clean_and_split.py     # Data processing & chunking
│   ├── build_index.py         # Weaviate indexing
│   ├── retriever_custom.py    # Hybrid retrieval + reranking
│   ├── generator.py           # Gemini answer generation
│   ├── rag_qa.py              # Main QA pipeline
│   └── test_retriever.py      # Test retrieval
│
├── docker/                    # Docker setup
│   └── docker-compose.yml     # Weaviate container
│
├── data/
│   ├── raw/                   # Raw law text files
│   └── processed/             # Processed JSON chunks
│
├── requirements.txt           # Python dependencies
├── .env                       # API keys (create this)
├── demo.gif                   # Quick review
└── README.md
```

## 🔧 Configuration

### Models

```python
# retriever_custom.py
EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# generator.py
LLM_MODEL = "gemini-2.5-flash"
```

### Retrieval Parameters

```python
INITIAL_K = 20    # Candidates before rerank
FINAL_K = 5       # Top results after rerank
BASE_ALPHA = 0.55 # Default hybrid alpha
```

### Chunking Parameters

```python
MAX_TOKENS_LEAF = 1500
WIN_TOK = 900
OVERLAP_TOK = 300
```

## 📝 Legal Documents

The system supports:

* Nghị định số 168/2024/NĐ-CP (Traffic violation penalties)
* Luật số 36/2024/QH15 (Road traffic order and safety)
* Luật số 35/2024/QH15 (Roads)

## 🧪 Testing

```bash
# Test retrieval
python test_retriever.py

# Test full pipeline
python rag_qa.py
```

## 🐛 Troubleshooting

### Cannot connect to Weaviate

```bash
docker-compose down
docker-compose up -d
docker ps  # Check container running
```

### BM25 index error

```bash
rm bm25_index.pkl
python clean_and_split.py  # Rebuild
```

### Out of memory

* Reduce `INITIAL_K` in retriever
* Reduce batch size in build_index.py

## 📚 References

* [https://weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate)
* [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
* [https://huggingface.co/Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
* [https://huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## 📄 License

MIT License

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
