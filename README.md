# Build a RAG Chatbot from Scratch with Local LLMs

A hands-on, phase-by-phase guide to building a Retrieval-Augmented Generation (RAG) chatbot using TypeScript, Ollama, and local LLMs. Each phase introduces one concept, builds on the previous one, and includes real discoveries from building it — what worked, what failed, and why.

## What You'll Build

A chatbot that reads your documents, understands them, and answers questions with source citations — running entirely on your machine (with optional cloud for bigger models).

## Learning Path

| Phase | Concept | What You Build | Guide |
|-------|---------|---------------|-------|
| 1 | LLM Basics | CLI chatbot with streaming + conversation memory | [Guide](docs/phase1-foundations.md) |
| 2 | File Reading | Read .txt, .json, .csv, .md, .pdf, .docx and feed to LLM | [Guide](docs/phase2-file-reading.md) |
| 3 | Chunking | 5 strategies to break documents into pieces | [Guide](docs/phase3-chunking.md) |
| 4 | Embeddings | Vector store with semantic search from scratch | [Guide](docs/phase4-embeddings.md) |
| 5 | RAG Pipeline | Hybrid search, query expansion, source citations | [Guide](docs/phase5-rag-pipeline.md) |
| 6 | Conversation Memory | Sliding window, summarization, persistent chat | [Guide](docs/phase6-memory.md) |
| 7-10 | Coming soon | Agents, multi-agent, web UI, production | [Roadmap](docs/learning-plan.md) |

Each guide includes: concepts, architecture diagrams, key code, what we discovered (real failures and fixes), industry context, and limitations.

---

## Prerequisites

- **Node.js** v18+ ([download](https://nodejs.org))
- **Ollama** ([download](https://ollama.com)) — runs LLMs locally

### Pull the required models

```bash
# Chat models (pick one or more)
ollama pull llama3.2          # 2GB, fast, good starting point
ollama pull deepseek-r1:7b    # 4.7GB, reasoning model (slower, outputs <think> blocks)

# Embedding model (needed from Phase 4 onwards)
ollama pull nomic-embed-text  # 274MB, produces 768-dim vectors
```

### Optional: Ollama Cloud (bigger models, better answers)

Create an account at [ollama.com](https://ollama.com), get an API key, and add to `.env`:
```bash
OLLAMA_API_KEY=your_key_here
```
This gives access to 27B-671B parameter models (gemma3:27b, deepseek-v3.1:671b, etc.) while embeddings stay local.

---

## Quick Start

```bash
git clone <your-repo-url>
cd chatbot
npm install

# Run any phase
npm run phase1   # Basic chat
npm run phase2   # File reader chat
npm run phase3   # Chunking strategies
npm run phase4   # Embeddings & vector store
npm run phase5   # Full RAG pipeline
npm run phase6   # Conversation memory
```

---

## Project Structure

```
chatbot/
├── src/
│   ├── phase1-chat.ts          # Phase 1: Basic Ollama chat
│   ├── phase2-file-reader.ts   # Phase 2: Read files + chat about them
│   ├── chunker.ts              # Phase 3: Reusable chunking module (5 strategies)
│   ├── phase3-chunking.ts      # Phase 3: Chunking demo (3 modes)
│   ├── vector-store.ts         # Phase 4: VectorStore class + cosine similarity
│   ├── phase4-vectors.ts       # Phase 4: Embeddings demo (4 modes)
│   ├── rag-pipeline.ts         # Phase 5: Hybrid search, RAG orchestration
│   ├── phase5-rag.ts           # Phase 5: RAG demo (4 modes)
│   ├── conversation.ts         # Phase 6: ConversationMemory class
│   └── phase6-memory.ts        # Phase 6: Memory demo (3 modes)
├── sample-docs/                # Test documents
│   ├── PDF/                    # Chapter PDFs (13 files)
│   ├── TEXT/                   # Chapter text files (13 files)
│   ├── sections/               # Chapter JSON metadata (12 files)
│   ├── important-dates.md      # Markdown sample
│   ├── important-people.txt    # Text sample
│   └── provinces-capitals.csv  # CSV sample
├── data/                       # Vector stores + saved conversations (runtime)
├── docs/                       # Learning guides
│   ├── learning-plan.md        # Full 10-phase roadmap
│   ├── phase1-foundations.md   # Phase 1 deep dive
│   ├── phase2-file-reading.md  # Phase 2 deep dive
│   ├── phase3-chunking.md      # Phase 3 deep dive
│   ├── phase4-embeddings.md    # Phase 4 deep dive
│   ├── phase5-rag-pipeline.md  # Phase 5 deep dive
│   └── phase6-memory.md        # Phase 6 deep dive
├── .env                        # Ollama Cloud API key (optional, gitignored)
├── package.json
└── tsconfig.json
```

**Architecture pattern**: Each phase has a **reusable module** (chunker.ts, vector-store.ts, etc.) and a **demo script** (phase3-chunking.ts, etc.). Modules are imported by later phases. Demo scripts are self-contained with interactive modes.

---

## Key Discoveries

Things we learned the hard way while building this:

| Discovery | Phase | What happened |
|-----------|-------|---------------|
| **PDF text extraction loses layout** | 2 | Tables and glossaries become garbled text. Claude reads PDFs as images — we extract text. |
| **187 seconds for one answer** | 2 | Stuffing a 25-page PDF into the prompt with deepseek-r1:7b. Chunking (Phase 3) fixed this. |
| **Keyword search can't understand meaning** | 3 | "When did Canada become a country?" doesn't match "Confederation 1867" — exact words only. |
| **Embedding search fails on abbreviations** | 4 | "RCMP" produces a weak vector. The model doesn't know what it stands for. |
| **Neither search alone is enough** | 4-5 | Keywords find exact matches, embeddings find meaning. You need both (hybrid search). |
| **Query expansion can be wrong** | 5 | LLM expanded "RCMP" as "Remote Control Media Player" — wrong domain entirely. |
| **Small models return empty answers** | 5 | llama3.2 (3B) and deepseek-r1:7b struggled with messy PDF text. gemma3:27b (cloud) worked. |
| **`<think>` blocks cause invisible output** | 5 | deepseek-r1 outputs reasoning in `<think>` tags — must strip them before displaying. |
| **Simpler prompts work better** | 5 | Complex multi-rule prompts confused small models. One-line instruction worked. |
| **LLM understands "he" but search doesn't** | 6 | LLM resolves pronouns from history, but the search query "What party did he lead?" finds nothing useful. |
| **LLM-based query rewriting > regex** | 6 | Asking the LLM to rewrite follow-ups as standalone queries is much smarter than checking for pronouns or references. |

---

## Key Concepts

### RAG (Retrieval-Augmented Generation)
Instead of asking the LLM to answer from its training data, you **retrieve** relevant chunks from your documents, **augment** the prompt with that context, then let the LLM **generate** a grounded answer.

```
Question → Retrieve relevant chunks → Add to prompt → LLM generates answer with citations
```

### Keyword Search vs Embedding Search vs Hybrid

| | Keyword | Embedding | Hybrid |
|---|---------|-----------|--------|
| "When did Canada become a country?" matches "Confederation 1867" | No | Yes | Yes |
| "RCMP" matches "RCMP" | Yes | Weak | Yes |
| Understands meaning | No | Yes | Yes |
| Finds exact terms | Yes | Not always | Yes |

### Why "Prompt Engineering Is Dead" Is Wrong (for RAG)

Articles about prompt engineering dying refer to manual tweaking for general chat. RAG prompts are **system design** — "answer ONLY from context", "cite sources", "say I don't know". LLMs can't guess your requirements. This is architecture, not tricks.

---

## Tech Stack

| Component | Tool | Phase |
|-----------|------|-------|
| Language | TypeScript | All |
| Runtime | Node.js + tsx | All |
| LLM (local) | Ollama (llama3.2 / deepseek-r1:7b) | All |
| LLM (cloud) | Ollama Cloud (gemma3:27b+) | 5+ |
| Embeddings | Ollama (nomic-embed-text, 768d) | 4+ |
| PDF Parsing | pdfjs-dist | 2+ |
| DOCX Parsing | mammoth | 5+ |
| CSV Parsing | csv-parse | 2+ |
| Env Config | dotenv | 5+ |
| Vector Store | Custom (built from scratch) | 4+ |
| Chat Persistence | JSON files | 6+ |

---

## Full Roadmap

See [`docs/learning-plan.md`](docs/learning-plan.md) for the complete 10-phase plan:

| Phase | Topic | Status |
|-------|-------|--------|
| 1-6 | Foundations → Memory | Done |
| 7 | Agents & Tool Use | Planned |
| 8 | Multi-Agent System | Planned |
| 9 | Web UI & API | Planned |
| 10 | Production Hardening | Planned |

Phase 5 is the **core milestone** — a fully working RAG chatbot. Everything after adds capabilities on top.

---

## License

ISC
