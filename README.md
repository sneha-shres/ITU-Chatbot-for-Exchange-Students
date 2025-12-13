# ITU Chatbot (RAG + Ollama)

Answer ITU student questions using a Retrieval-Augmented Generation (RAG) pipeline that combines:
- SQL course database lookups (SQLite)
- Vector search over scraped ITU content (FAISS + SentenceTransformers)
- Optional LLM generation via Ollama with fallback

## Project Structure

```
├── src/
│   ├── core/
│   │   ├── app.py               # Flask server & API
│   │   └── rag_pipeline.py      # Query classification + retrieval + generation
│   ├── database/
│   │   ├── course_db.py         # Courses SQLite access
│   │   ├── vector_db.py         # FAISS index + embeddings
│   │   └── sql_store.py         # Optional page/chunk storage
│   └── utils/
│       └── scraper.py           # ITU website scraper utilities
│
├── data/
│   ├── courses/
│   │   └── courses.db           # SQLite DB (144 rows)
│   └── vectors/
│       ├── itu_vector_index.faiss
│       ├── itu_metadata.pkl
│       └── itu_scraped_data.json
│
├── tools/
│   ├── run_scraper.py           # Scrape + build vectors
│   └── create_courses_db.py     # Convert CSV → SQLite
│
├── config/
│   └── requirements.txt         # Python dependencies
│
├── tests/
│   └── test_rag_sql.py          # Simple RAG checks
│
├── templates/index.html         # Web UI
└── static/{styles.css,script.js}
```

## Prerequisites

- Python 3.10+
- pip
- Optional: virtualenv (`python3 -m venv .venv && source .venv/bin/activate`)
- Optional: Ollama (for LLM responses): https://ollama.ai

## Quick Start

1) Install Python dependencies

```bash
pip install -r config/requirements.txt
```

2) Set module path (required for direct script execution)

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

3) (Optional) Enable LLM via Ollama

```bash
ollama serve                     # start server on http://localhost:11434
ollama pull gpt-oss:20b-cloud    # pull a chat-capable model
```

Minimal `.env` (optional; if absent, the app will gracefully use template fallback):

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b-cloud
```

4) Run the web app (Option 2: PYTHONPATH + direct script)

```bash
python src/core/app.py
```

Open http://localhost:5000

Alternative (without exporting PYTHONPATH each time): run as a module:

```bash
python -m src.core.app
```

If you see `ModuleNotFoundError: No module named 'database'` it means `src` was not on `PYTHONPATH`. Export it (Option 2) or use the module form above.

## Testing

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python tests/test_rag_sql.py
```

## Vector Data: Build or Rebuild (optional)

The repo already expects vector files in `data/vectors/`. To rebuild from scratch:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python tools/run_scraper.py
```

This will:
- Scrape ITU content (student/progamme pages; news excluded)
- Save JSON to `data/vectors/itu_scraped_data.json`
- Create embeddings and FAISS index in `data/vectors/`

## Course Database

`data/courses/courses.db` is expected. If starting from CSV, create the DB:

```bash
python tools/create_courses_db.py
```

Inputs/outputs:
- Input CSV: `data/courses/courses.csv`
- Output DB: `data/courses/courses.db`

## Troubleshooting

- No LLM responses: ensure Ollama is running or omit `.env` to use template responses.
- Import errors / `No module named 'database'`:
     - Cause: Running `python src/core/app.py` without `PYTHONPATH` pointing to `src`.
     - Fix Option 2 (preferred here): `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
     - Alternative: use module run `python -m src.core.app`.
- Missing vectors: run `python tools/run_scraper.py` to recreate FAISS files.
- Missing courses DB: ensure `data/courses/courses.db` exists or run the CSV→SQLite step.



## LLM (Ollama) Quick Reference

Pull and serve model (optional):
```bash
ollama serve &               # start server
ollama pull gpt-oss:20b-cloud
```
Environment variables (via `.env`):
```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b-cloud
```
If Ollama is unavailable the pipeline falls back to template formatting (still functional).

## Example Questions

Try asking the chatbot:

- "What computer science programs does ITU offer?"
- "How do I apply for admission?"
- "What research opportunities are available?"
- "Tell me about student life at ITU"
- "What are the admission requirements?"
