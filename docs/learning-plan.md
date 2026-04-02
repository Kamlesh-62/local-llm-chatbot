# Local LLM Chatbot - Full Learning Plan

## Context
Build a file-reading chatbot using TypeScript + Ollama (local LLM). The project progresses from basics to a full-scale RAG app with agents. Each phase builds on the previous one. Existing project has `ollama` and `tsx` already installed.

---

## Phase 1: Foundations - Talk to Ollama
**Goal:** Understand how to send prompts and get responses from a local LLM.

### Topics
- Install and run Ollama locally (pull models like `llama3`, `mistral`)
- Connect to Ollama from TypeScript using the `ollama` npm package
- Send a simple prompt, receive a response
- Streaming vs non-streaming responses
- Understand roles: `system`, `user`, `assistant`
- Build a basic CLI chat loop (readline + ollama)
- Manage conversation history (message array)

### Deliverable
A CLI app where you type a question, Ollama answers, and it remembers the conversation.

---

## Phase 2: File Reading & Text Processing
**Goal:** Read files and feed their content to the LLM.

### Topics
- Read files with Node.js `fs` (text, PDF, markdown, JSON, CSV)
- Parse different file formats:
  - Plain text / Markdown - direct read
  - PDF - use `pdf-parse` or `pdfjs-dist`
  - CSV - use `csv-parse`
  - JSON - native parsing
- Handle large files (streams vs full read)
- Feed file content as context in the system/user prompt
- Understand token limits - why you can't just dump a whole book into a prompt

### Deliverable
A CLI app that reads a file and answers questions about its content (small files only).

---

## Phase 3: Chunking Strategies
**Goal:** Break large documents into meaningful pieces.

### Topics
- Why chunking matters (token limits, relevance, accuracy)
- Chunking strategies:
  - **Fixed-size chunks** - split by character/token count with overlap
  - **Sentence-based** - split on sentence boundaries
  - **Paragraph-based** - split on double newlines
  - **Recursive character splitting** - try large splits first, fall back to smaller
  - **Semantic chunking** - split when topic changes (advanced)
- Chunk overlap - why and how much
- Metadata per chunk (source file, page number, chunk index)
- Experiment: same question, different chunk sizes - see how answers change

### Deliverable
A module that takes any document and returns an array of chunks with metadata.

---

## Phase 4: Embeddings & Vector Store
**Goal:** Convert text to vectors and store them for similarity search.

### Topics
- What are embeddings? (text -> number array representing meaning)
- Generate embeddings using Ollama (`ollama.embeddings()` with `nomic-embed-text` or `mxbai-embed-large`)
- Cosine similarity - how to compare two vectors
- Build a simple in-memory vector store from scratch (array + cosine search)
- Graduate to a proper vector DB:
  - **ChromaDB** (easy, runs locally, has JS client)
  - OR **LanceDB** (embedded, no server needed, works with TS)
- Store chunks with their embeddings
- Query: embed the question -> find top-K similar chunks -> return them

### Deliverable
A vector store that you can add documents to and query with natural language.

---

## Phase 5: RAG Pipeline (Retrieval-Augmented Generation)
**Goal:** Wire everything together into a working RAG system.

### Topics
- The RAG flow: Question -> Embed -> Retrieve -> Augment Prompt -> Generate
- Prompt engineering for RAG:
  - System prompt: "Answer based on the provided context only"
  - Inject retrieved chunks into the prompt
  - Handle "I don't know" when context is insufficient
- Top-K selection - how many chunks to include
- Re-ranking retrieved results (simple score threshold)
- Source attribution - tell the user which chunk/file the answer came from
- Evaluate answer quality (manual testing at this stage)

### Deliverable
A working RAG chatbot: ingest files -> ask questions -> get grounded answers with sources.

---

## Phase 6: Conversation Memory & History
**Goal:** Make the chatbot remember previous exchanges within a session.

### Topics
- Conversation history management (sliding window of messages)
- Context window budgeting: history + retrieved chunks + system prompt must fit
- Summarize old conversation turns to save tokens
- Persistent chat history (save/load from file or SQLite)
- Multi-turn RAG: use conversation context to refine retrieval queries

### Deliverable
A chatbot that maintains coherent multi-turn conversations while doing RAG.

---

## Phase 7: Agents & Tool Use
**Goal:** Give the LLM the ability to take actions, not just answer.

### Topics
- What is an agent? (LLM + reasoning + tools)
- The ReAct pattern: Thought -> Action -> Observation -> loop
- Define tools as functions the LLM can call:
  - `searchDocuments(query)` - search the vector store
  - `readFile(path)` - read a specific file
  - `listFiles(directory)` - browse available files
  - `calculator(expression)` - do math
  - `webSearch(query)` - search the web (optional)
- Tool calling format: structured JSON output from the LLM
- Parse LLM output to detect tool calls
- Execute tools and feed results back
- Handle multi-step reasoning (agent loops)
- Error handling when tools fail
- Max iterations / loop protection

### Deliverable
An agent that can decide which tools to use, chain multiple steps, and answer complex questions.

---

## Phase 8: Multi-Agent System
**Goal:** Multiple specialized agents that collaborate.

### Topics
- Agent specialization:
  - **Router Agent** - decides which specialist to call
  - **Research Agent** - searches and retrieves information
  - **Summarizer Agent** - condenses long content
  - **Code Agent** - analyzes/generates code from docs
  - **QA Agent** - validates answers against sources
- Agent communication patterns:
  - Sequential (pipeline)
  - Parallel (fan-out, fan-in)
  - Supervisor pattern (one agent orchestrates others)
- Shared state / blackboard between agents
- Agent handoff and delegation

### Deliverable
A multi-agent system where a router dispatches to specialized agents that collaborate on complex queries.

---

## Phase 9: Web UI & API Layer
**Goal:** Move from CLI to a proper web interface.

### Topics
- Build a REST API (Express or Fastify):
  - `POST /chat` - send message, get response
  - `POST /ingest` - upload and ingest documents
  - `GET /documents` - list ingested documents
  - `DELETE /documents/:id` - remove a document
  - Server-Sent Events (SSE) for streaming responses
- Build a frontend (React or plain HTML+JS):
  - Chat interface with message bubbles
  - File upload for document ingestion
  - Show sources/citations inline
  - Streaming response display
  - Conversation history sidebar
- WebSocket option for real-time chat

### Deliverable
A full web app with file upload, chat UI, streaming responses, and source citations.

---

## Phase 10: Production Hardening
**Goal:** Make it reliable, observable, and maintainable.

### Topics
- Error handling and graceful degradation
- Logging (structured logs with pino or winston)
- Rate limiting and request queuing
- Document processing queue (bull or similar)
- Caching frequent queries and embeddings
- Health checks and monitoring
- Configuration management (environment variables, config files)
- Testing:
  - Unit tests for chunking, embedding, retrieval
  - Integration tests for the RAG pipeline
  - Evaluation metrics (relevance, faithfulness, answer quality)
- Docker containerization (app + Ollama + vector DB)

### Deliverable
A production-ready, containerized RAG application with tests, logging, and monitoring.

---

## Tech Stack Summary

| Component | Tool |
|---|---|
| Runtime | Node.js + TypeScript |
| LLM | Ollama (llama3 / mistral) |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | ChromaDB or LanceDB |
| File Parsing | pdf-parse, csv-parse, fs |
| API | Express or Fastify |
| Frontend | React or vanilla HTML |
| Testing | Vitest |
| Containerization | Docker + docker-compose |

---

## Recommended Learning Order

```
Phase 1 (Foundations)
  |
Phase 2 (File Reading)
  |
Phase 3 (Chunking)
  |
Phase 4 (Vectors)
  |
Phase 5 (RAG) <-- This is the core milestone
  |
Phase 6 (Memory)
  |
Phase 7 (Agents)
  |
Phase 8 (Multi-Agent)
  |
Phase 9 (Web UI)
  |
Phase 10 (Production)
```

Each phase is self-contained. You can stop after Phase 5 and have a fully working RAG chatbot. Phases 7-8 add agent capabilities. Phases 9-10 make it production-ready.
