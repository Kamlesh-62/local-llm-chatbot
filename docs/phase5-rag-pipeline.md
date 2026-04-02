# Phase 5: RAG Pipeline — Wire Everything Together

## Goal

Combine chunking, embeddings, and search into a complete Retrieval-Augmented Generation system. This phase takes everything from Phases 1-4 and wires it into a pipeline that actually answers questions from your documents.

**Key source files:**
- `src/rag-pipeline.ts` — reusable RAG pipeline class
- `src/phase5-rag.ts` — interactive demo with 4 modes

---

## What You'll Learn

- How the full RAG flow works end to end
- Why hybrid search fixes the problems from Phase 4
- How query expansion helps (and where it fails)
- Why prompt engineering is still critical for RAG
- How to handle the `<think>` block problem with reasoning models
- Why model size matters for RAG quality
- How Ollama Cloud gives you access to larger models
- When to use page-based vs character-based chunking

---

## The Full RAG Flow

```
  User Question
       |
       v
 +-----------------+
 | Query Expansion |  Ask LLM to expand abbreviations
 +-----------------+
       |
       v
 +----------+
 |  Embed   |  Convert query to 768-dim vector
 +----------+
       |
       v
 +---------------+
 | Hybrid Search |  Combine embedding + keyword results
 +---------------+
       |
       v
 +----------+
 | Re-rank  |  Score and sort combined results
 +----------+
       |
       v
 +--------------+
 | Build Prompt |  System instructions + context chunks + question
 +--------------+
       |
       v
 +------------------+
 | Generate Answer  |  LLM reads context, produces answer
 +------------------+
       |
       v
 +---------------+
 | Cite Sources  |  Include which documents the answer came from
 +---------------+
```

Each step exists because we discovered a real problem that required it. This isn't over-engineering — every box fixes a specific failure mode.

---

## Hybrid Search — The Core Fix

Phase 4 showed that embedding search misses abbreviations and keyword search misses paraphrases. Hybrid search combines both.

### The Formula

```
hybridScore = (0.7 * embeddingScore) + (0.3 * keywordScore)
```

### Why These Weights?

- **0.7 for embeddings**: semantic understanding is the primary value of RAG. Most questions are natural language, and embeddings handle those well.
- **0.3 for keywords**: keyword search is the safety net. When someone searches "RCMP" and the embedding model produces a useless vector, the keyword match saves us.

### How It Works in Practice

```typescript
// Embedding search: finds chunks by meaning
const embeddingResults = vectorStore.search(queryEmbedding, { topK: 10 });

// Keyword search: finds chunks by exact text match
const keywordResults = keywordSearch(query, chunks, { topK: 10 });

// Merge: combine scores, deduplicate, re-sort
const hybridResults = mergeResults(embeddingResults, keywordResults, {
  embeddingWeight: 0.7,
  keywordWeight: 0.3,
});
```

For the query "RCMP":
- Embedding search: RCMP chunk at rank #5 (weak vector)
- Keyword search: RCMP chunk at rank #1 (exact match)
- Hybrid search: RCMP chunk at rank **#1** (keyword score pulls it up)

For the query "When did Canada become a country?":
- Embedding search: "Confederation 1867" at rank #1 (semantic match)
- Keyword search: nothing relevant (no word overlap with "Confederation")
- Hybrid search: "Confederation 1867" at rank **#1** (embedding score carries it)

Together they cover each other's weaknesses.

---

## Query Expansion

Before searching, we ask the LLM to expand abbreviations and add context to the query. This helps the embedding model produce a better vector.

### How It Works

```typescript
const expandedQuery = await llm.chat({
  messages: [
    {
      role: "system",
      content: "Expand any abbreviations in this search query. Return ONLY the expanded query.",
    },
    {
      role: "user",
      content: "What is the RCMP?",
    },
  ],
});
// Hoped-for result: "What is the Royal Canadian Mounted Police (RCMP)?"
```

### The Problem We Discovered

General-purpose LLMs don't know **domain-specific** abbreviations. When we asked the model to expand "BNA Act", it returned:

> "What is the BNA Act (Biological Nitrogen Accumulation)?"

That's plausible — but not what our documents are about. Our BNA Act is the "British North America Act" that created Confederation.

### The Fix

1. **Pass document source names** as context so the LLM has domain hints
2. **Always include the original query** alongside the expanded version
3. **Search with both** — the original catches keyword matches, the expanded improves embedding quality

```typescript
// Search with original AND expanded query
const results1 = search(originalQuery);
const results2 = search(expandedQuery);
const merged = mergeResults(results1, results2);
```

Query expansion helps more often than it hurts, but you cannot blindly trust it.

---

## Prompt Engineering for RAG — Why It's Not Dead

You'll see articles claiming "prompt engineering is dead" now that models are smarter. They're talking about the manual tweaking people do for general chat — adding "think step by step" or "you are an expert."

**RAG prompts are different.** They are system design, not clever tricks:

```typescript
const systemPrompt = `You are a helpful assistant that answers questions based ONLY on the
provided context. Follow these rules strictly:

1. ONLY use information from the context below to answer
2. If the context doesn't contain the answer, say "I don't have enough information to answer that"
3. Cite your sources by referencing the document name and page number
4. Do not make up information or use knowledge from your training data
5. If the context is unclear or garbled, say so rather than guessing

CONTEXT:
${relevantChunks.map((c) => `[${c.source}, p.${c.page}]: ${c.text}`).join("\n\n")}
`;
```

These instructions will **never** be "dead" because:
- The LLM cannot guess that you want source citations
- The LLM doesn't know you want it to refuse when context is insufficient
- The LLM will happily use training data if you don't tell it not to
- Every RAG application has different requirements for format, tone, and behavior

Prompt engineering for RAG is closer to writing an API contract than writing a magic incantation.

---

## The `<think>` Block Problem

The `deepseek-r1` model outputs its internal chain-of-thought reasoning inside `<think>...</think>` tags before producing the actual answer:

```
<think>
The user is asking about the RCMP. Let me look at the context...
I can see in the document that RCMP stands for Royal Canadian Mounted Police...
</think>

Based on the provided context, the RCMP is the Royal Canadian Mounted Police...
```

### What Went Wrong

When streaming the response token by token, our initial code displayed everything — including the think block. Worse, some responses were **entirely** think blocks with no visible answer, making it look like the chatbot returned nothing.

### The Fix

Collect the full response first, then strip think blocks before displaying:

```typescript
// After collecting the complete response
const cleaned = fullResponse.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

if (cleaned.length === 0) {
  console.log("(Model produced only internal reasoning — no answer generated)");
} else {
  console.log(cleaned);
}
```

If you're streaming for UX (showing tokens as they arrive), you need a state machine that tracks whether you're inside a think block and suppresses output until the closing tag.

---

## Model Size Matters

We tested the same RAG pipeline with different models. The results were dramatic:

| Model | Size | Result |
|-------|------|--------|
| `llama3.2` | 3B | Returned **empty answers** — couldn't follow the complex system prompt with messy PDF context |
| `deepseek-r1` | 7B | Returned **"e"** — a single character. The model was overwhelmed |
| `gemma3` (Ollama Cloud) | 27B | **Worked properly** — followed instructions, cited sources, refused when context was insufficient |

### Why Small Models Fail at RAG

RAG prompts are demanding. The model must simultaneously:
1. Parse a long system prompt with multiple rules
2. Read several chunks of potentially garbled PDF text
3. Determine which chunks are relevant
4. Synthesize an answer following the specified format
5. Add citations
6. Decide when to refuse

3B and 7B models simply don't have the capacity for all of this at once. **For RAG, use 13B+ models** — or use a cloud API for the chat model while keeping embeddings local.

---

## Ollama Cloud

Ollama Cloud lets you access larger models (27B to 671B) through the same API you use locally. The only changes are the host URL and an API key.

```typescript
import { Ollama } from "ollama";

// Local (default)
const local = new Ollama();

// Cloud — same API, bigger models
const cloud = new Ollama({
  host: "https://ollama.com",
  headers: { Authorization: `Bearer ${process.env.OLLAMA_API_KEY}` },
});
```

### The Hybrid Approach

A practical setup for development:

- **Embeddings**: run locally with `nomic-embed-text` (fast, small model, no API costs)
- **Chat/generation**: use Ollama Cloud with `gemma3:27b` or larger (smart enough for RAG)

This gives you the best of both worlds — fast local embeddings and high-quality cloud generation.

---

## PDF Chunking: Page-Based vs Character-Based

Phase 3 introduced recursive character splitting. For PDFs, we discovered a better approach.

### Page-Based Chunking (for PDFs)

```typescript
// One chunk per page — keeps related content together
const chunks = pdfPages.map((page, i) => ({
  text: page.text,
  metadata: { source: filename, page: i + 1 },
}));
```

**Why**: PDF pages are natural boundaries. A section about "Confederation and the Founding of Canada" usually lives on one page. Splitting mid-page breaks context that belongs together.

### Character-Based Chunking (for text files)

```typescript
// Recursive splitting at ~500 chars with 50 char overlap
const chunks = recursiveSplit(text, {
  chunkSize: 500,
  overlap: 50,
  separators: ["\n\n", "\n", ". ", " "],
});
```

**Why**: Plain text files have no page structure. Character splitting with overlap ensures no sentence gets cut without the neighboring chunk having the context to complete it.

**Rule of thumb**: if your source has natural boundaries (pages, sections, headings), use those. If not, use character splitting with overlap.

---

## The 4 Demo Modes

Run `npm run phase5` and choose a mode:

### Mode 1: Ingest

Processes your documents (PDFs and text files), chunks them, generates embeddings, and builds the vector store. Run this once, or whenever your documents change.

### Mode 2: RAG Chat

Interactive chat where you ask questions and get answers grounded in your documents. The full pipeline runs on every question: expand, embed, search, rank, prompt, generate, cite.

### Mode 3: Search Comparison

Runs a query through keyword search, embedding search, and hybrid search side by side. Shows scores and rankings for each approach. Use this to see exactly why hybrid search wins.

### Mode 4: Pipeline Breakdown

Shows each pipeline step in detail for a single query: what the expanded query looks like, which chunks were retrieved, what the final prompt contains, and what the model returns. Essential for debugging.

---

## Run It

```bash
npm run phase5
```

Make sure you have the required models:

```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1      # or your preferred chat model
```

For better results with a cloud model:

```bash
export OLLAMA_API_KEY=your_key_here
npm run phase5 -- --cloud
```

---

## Limitations

- Query expansion can produce incorrect expansions for domain-specific terms
- PDF text extraction is still garbled in places — garbage in, garbage out
- No conversation memory — each question is independent, follow-ups don't work
- Hybrid search weights (0.7/0.3) are hardcoded, not tuned per query
- No re-ranking model — just score merging

---

## What's Next

**Phase 6** adds conversation memory so follow-up questions work. "When was Confederation?" followed by "Who was the first Prime Minister?" then "What party did he lead?" — the chatbot will remember that "he" refers to the first PM.
