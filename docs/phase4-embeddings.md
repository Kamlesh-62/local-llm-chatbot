# Phase 4: Embeddings & Vector Store — Semantic Search

## Goal

Convert text into numerical vectors and find similar content by **meaning**, not keywords. This is the foundation that makes RAG actually work.

**Key source files:**
- `src/vector-store.ts` — reusable VectorStore class
- `src/phase4-vectors.ts` — interactive demo with 4 modes

---

## What You'll Learn

- What embeddings are and why they matter for search
- How cosine similarity works (we build it from scratch)
- How to store, persist, and search a vector store
- Why embedding search alone is **not enough** (a real discovery from building this)

---

## What Are Embeddings?

An embedding is a way to turn text into a fixed-length array of numbers — specifically **768 numbers** when using the `nomic-embed-text` model via Ollama.

```
"When did Canada become a country?"  → [0.0231, -0.0512, 0.1893, ..., 0.0044]  (768 floats)
"Confederation 1867"                 → [0.0198, -0.0487, 0.1921, ..., 0.0051]  (768 floats)
```

The magic: **similar meaning produces similar vectors**, even when the words are completely different. The two phrases above share almost no words, yet their vectors point in nearly the same direction in 768-dimensional space.

This is what lets us move beyond keyword matching. A user can ask "When did Canada become a country?" and we can find the chunk about "Confederation 1867" — because the model understands they mean the same thing.

**How to generate embeddings with Ollama:**

```typescript
import ollama from "ollama";

const response = await ollama.embed({
  model: "nomic-embed-text",
  input: "When did Canada become a country?",
});

const vector: number[] = response.embeddings[0]; // 768 numbers
```

---

## Cosine Similarity — Built From Scratch

We need a way to measure how similar two vectors are. Cosine similarity compares the **direction** two vectors point, ignoring their length.

### The Three Building Blocks

```typescript
// 1. Dot product: multiply corresponding elements, sum them up
function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

// 2. Magnitude: the "length" of a vector (Pythagorean theorem in N dimensions)
function magnitude(v: number[]): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}

// 3. Cosine similarity: dot product divided by the product of magnitudes
function cosineSimilarity(a: number[], b: number[]): number {
  const dot = dotProduct(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);
  if (magA === 0 || magB === 0) return 0;
  return dot / (magA * magB);
}
```

### The Formula

```
                    A . B
cos(theta) = ─────────────────
              ||A|| * ||B||
```

- Result of **1.0** = identical direction (same meaning)
- Result of **0.0** = perpendicular (unrelated)
- Result of **-1.0** = opposite direction (opposite meaning)

### Why Cosine Over Euclidean Distance?

Euclidean distance measures the straight-line gap between two points. The problem: a long document and a short document about the same topic will have vectors of different magnitudes, making Euclidean distance misleadingly large.

Cosine similarity only cares about **direction**, not length. Two vectors pointing the same way get a high score regardless of magnitude. For text similarity, direction is what captures meaning.

---

## VectorStore Class

The `VectorStore` class in `src/vector-store.ts` handles three jobs:

### 1. Add Chunks With Embeddings

```typescript
const store = new VectorStore();

// Each chunk gets stored with its text, metadata, and embedding vector
store.add({
  text: "Confederation. The Dominion of Canada was created in 1867...",
  metadata: { source: "chapter-4.pdf", page: 2 },
  embedding: [0.0198, -0.0487, ...], // 768 numbers from Ollama
});
```

### 2. Search by Similarity

```typescript
// Embed the query, then find the most similar chunks
const queryEmbedding = await getEmbedding("When did Canada become a country?");
const results = store.search(queryEmbedding, { topK: 5 });

// results: [{ text, metadata, score: 0.6451 }, ...]
```

Internally, `search` computes cosine similarity between the query vector and every stored vector, then returns the top K results sorted by score.

### 3. Save/Load to JSON for Persistence

```typescript
// Save — avoid re-embedding every run (embedding is slow!)
store.save("./data/vector-store.json");

// Load — instant startup
const loaded = VectorStore.load("./data/vector-store.json");
```

Embedding hundreds of chunks takes time. Persisting to JSON means you only embed once, then load instantly on subsequent runs. This is the simple version of what production systems do with databases like Pinecone or pgvector.

---

## The 4 Demo Modes

Run `npm run phase4` and choose a mode:

### Mode 1: Explore Embeddings (Similarity Matrix)

Embeds a set of sample phrases and prints a matrix showing similarity scores between every pair. Great for building intuition about what the model considers "similar."

### Mode 2: Build Store

Loads your document chunks, embeds each one via `nomic-embed-text`, and saves the resulting vector store to disk. This is the slow step you only need to do once.

### Mode 3: Search

Type natural language queries and see ranked results with similarity scores. Try paraphrasing the same question different ways to see how robust embedding search is.

### Mode 4: Keyword vs Embedding Showdown

Runs the same queries through both keyword search (exact string matching) and embedding search, side by side. This is where the strengths and weaknesses of each approach become obvious.

---

## Real Test Results

These are actual similarity scores from running `nomic-embed-text` on our documents:

| Query | Chunk | Cosine Similarity |
|-------|-------|:-----------------:|
| "Who founded Quebec?" | "Samuel de Champlain established Quebec" | **0.6451** |
| "Who founded Quebec?" | "provinces and territories" | 0.3651 |
| "When did Canada become a country?" | "Confederation 1867 Dominion of Canada" | **0.6102** |
| "When did Canada become a country?" | "Canadian geography and climate" | 0.2874 |

A score around **0.6+** means the model sees a strong semantic relationship. Below **0.4** means unrelated. The gap between related and unrelated content is clear enough to be useful for retrieval.

---

## Why Embedding Search Alone Isn't Enough

**This is the most important section in this guide.** We discovered these limitations by actually building and testing the system — they aren't obvious from tutorials.

### Problem 1: Abbreviations

Our documents use abbreviations like "RCMP" (Royal Canadian Mounted Police) and "BNA Act" (British North America Act). When you search for "RCMP", the embedding model produces a **weak, generic vector** because it has no idea what that abbreviation means. The relevant chunk ranks low or doesn't appear at all.

### Problem 2: Short Queries

A single word like "RCMP" or "PM" gives the embedding model almost **no signal** to work with. There's not enough context to infer meaning. Longer, more descriptive queries produce dramatically better results.

### Problem 3: Noisy Text

PDF extraction often produces garbled text — broken words, merged lines, stray characters. These errors propagate into the embeddings, producing **noisy vectors** that don't accurately represent the content.

### Problem 4: Top-K Cutoff

In our tests, the correct RCMP chunk appeared at **rank #5** — but we only retrieved the top 3 results. A user would never see the right answer. Increasing K helps but also increases noise.

### The Takeaway

**Real RAG systems NEVER use embeddings alone.** Production systems combine:

1. **Keyword search** — catches exact matches like "RCMP"
2. **Embedding search** — catches meaning like "When did Canada become a country?" = "Confederation 1867"
3. **Re-ranking** — a second model re-scores the combined results

This is called **hybrid search**, and it's what we build in Phase 5.

---

## Keyword vs Embedding Comparison

| | Keyword Search | Embedding Search |
|---|---|---|
| **Exact terms** (RCMP, BNA Act) | Finds them instantly | Often misses — weak vectors |
| **Paraphrases** ("become a country" vs "Confederation") | Misses completely | Finds them reliably |
| **Short queries** (1-2 words) | Works fine if word appears | Poor — not enough signal |
| **Noisy/garbled text** | Still matches substrings | Noisy vectors degrade results |
| **Synonyms** ("police" vs "RCMP") | Misses | Usually finds |
| **Speed** | Very fast (string ops) | Slower (vector math on every chunk) |

Neither approach is sufficient on its own. They have complementary strengths.

---

## Run It

```bash
npm run phase4
```

Make sure you have `nomic-embed-text` pulled in Ollama:

```bash
ollama pull nomic-embed-text
```

---

## Limitations

- Embedding search fails on abbreviations and acronyms the model hasn't seen
- Short queries produce weak vectors with poor recall
- Garbled PDF text creates noisy embeddings
- Top-K cutoff can exclude the correct result
- No hybrid search yet — embedding only

---

## What's Next

**Phase 5** combines keyword search and embedding search into a **hybrid search** pipeline, with query expansion and re-ranking — fixing the problems we discovered here.
