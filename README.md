# Docs Assistant: local RAG over your documents

A retrieval-augmented question-answering app that runs entirely on your machine: no API key, no data leaves the
laptop. Drop documents in a folder, ask questions in a chat UI, get answers with the passages they came from.

```
docs/ (.md .txt .pdf) ─► chunk ─► embed (Ollama) ─► PostgreSQL + pgvector (HNSW, cosine)
                                                            │
question ─► embed ─► top-k similar chunks ─────────────────┘─► prompt ─► local LLM (Ollama) ─► answer + sources
```

**Stack**: Python, Streamlit, PostgreSQL + pgvector, Ollama (`llama3.2:1b` for answers, `all-minilm` for
embeddings), Docker Compose. The whole pipeline is about 130 lines in [`rag.py`](rag.py), with no framework.

## Run it

```bash
docker compose up -d --build
```

Open http://localhost:8501. The first start downloads the two models (about 1.4 GB) and indexes `docs/`.

To use your own documents, put `.md`, `.txt` or `.pdf` files in `docs/` and click **Re-index documents**.
To use a larger model: `LLM_MODEL=llama3.2:3b docker compose up -d`.

Command line:

```bash
docker compose exec app python rag.py ask "How many days of annual leave do I get?"
```

## Evaluation

[`eval/questions.json`](eval/questions.json) holds 18 questions about the sample corpus (a handbook for a fictional
company, so the model cannot answer from memory), each with its source document and the fact the answer must contain.

```bash
docker compose exec app python evaluate.py --generate
```

Result on the sample corpus (4 documents, 8 chunks, CPU only):

| Metric | Result |
|---|---|
| Retrieval hit@1 (right document ranked first) | 18/18 |
| Retrieval MRR | 1.00 |
| Answers containing the expected fact (`llama3.2:1b`) | 18/18 |

Read these numbers for what they are: the corpus is tiny, and the answer check is a keyword match, not a judgement
of the full answer. The evaluation was still useful: the first prompt version scored 12/18 because the 1B model
echoed the context instead of answering; moving the instruction after the question fixed it.

## Design notes

- **Chunking**: paragraphs are packed into chunks of about 800 characters with a one-paragraph overlap; oversized
  paragraphs are hard-split. 800 characters stays under the embedding model's 256-token window.
- **Search**: cosine distance in pgvector with an HNSW index. Re-indexing rebuilds the table, so it is idempotent.
- **Grounding**: the prompt restricts the model to the retrieved passages, asks for citations, and tells it to say
  when the answer is missing. Retrieved text is declared as data, not instructions.
- **Limits**: each question is answered on its own (no chat-history query rewriting), there is no reranker, and
  scanned PDFs need OCR first.
