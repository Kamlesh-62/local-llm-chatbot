# Phase 3: Chunking Strategies

> **Key sources:** `src/chunker.ts` (reusable module), `src/phase3-chunking.ts` (demo)

## Goal

Break large documents into meaningful pieces so we can later retrieve only the parts that are relevant to a question.

## What You'll Learn

- Why chunking is necessary for RAG
- Five different chunking strategies and their trade-offs
- How chunk overlap prevents information loss at boundaries
- How chunk size affects answer quality
- Naive keyword search and its limitations

## Why Chunk?

Phase 2 stuffed entire documents into the prompt. That hits three walls:

1. **Token limits** -- Models have a finite context window. A 100-page document won't fit.
2. **Relevance** -- Less noise means better answers. If the LLM only sees the relevant paragraph instead of 25 pages, it focuses better.
3. **Speed** -- Processing 500 tokens is much faster than processing 12,000. Every token costs time.

Chunking solves all three: break the document into pieces, find the right pieces, and send only those.

## The 5 Strategies

### 1. Fixed-Size

Split the text into chunks of exactly N characters, sliding a window across the document.

```
Text:    "The quick brown fox jumps over the lazy dog near the river"
Size:    20 characters
Overlap: 5 characters

Chunk 1: "The quick brown fox "
Chunk 2: " fox jumps over the "
Chunk 3: " the lazy dog near t"
Chunk 4: "ear the river"
```

**Pros:** Dead simple to implement. Predictable chunk sizes.
**Cons:** Cuts mid-word and mid-sentence. "fox " and "jumps" end up in different chunks, breaking meaning.

### 2. Paragraph

Split on double newlines (`\n\n`), treating each paragraph as a chunk.

```
Text:
  "First paragraph about RAG.

   Second paragraph about embeddings.

   Third paragraph about chunking."

Chunk 1: "First paragraph about RAG."
Chunk 2: "Second paragraph about embeddings."
Chunk 3: "Third paragraph about chunking."
```

**Pros:** Respects the document's natural structure. Authors already grouped related ideas into paragraphs.
**Cons:** Paragraph sizes vary wildly -- one might be 20 words, another 500. Inconsistent chunk sizes cause problems downstream.

### 3. Sentence

Split on sentence-ending punctuation: `.` `!` `?`

```
Text:    "RAG stands for Retrieval-Augmented Generation. It combines
          search with LLMs. This improves accuracy!"

Chunk 1: "RAG stands for Retrieval-Augmented Generation."
Chunk 2: "It combines search with LLMs."
Chunk 3: "This improves accuracy!"
```

**Pros:** Each chunk preserves a complete thought.
**Cons:** Simple regex breaks on real-world text. "Dr. Smith" splits after "Dr." and the number "3.14" splits after "3." -- the regex sees a period and assumes end-of-sentence.

### 4. Recursive

Try multiple separators in order, falling back to smaller ones when chunks are still too large:

```
Try splitting on: "\n\n" --> "\n" --> ". " --> " " --> ""

Step 1: Split on "\n\n" (paragraphs)
  --> If a chunk is under the size limit, keep it
  --> If a chunk is still too large, split IT on "\n"
      --> Still too large? Split on ". "
          --> Still too large? Split on " "
              --> Last resort: split on "" (character by character)
```

This is the **LangChain approach** and the best general-purpose strategy. It preserves the largest natural boundaries it can while guaranteeing no chunk exceeds the size limit.

**Pros:** Adapts to document structure. Preserves meaning at the highest level possible.
**Cons:** More complex to implement. Behavior depends on document formatting.

### 5. Semantic

Split on meaningful structural markers: headings, numbered sections, or ALL CAPS lines.

```
Text:
  "# Introduction
   RAG is a technique for grounding LLMs.

   # How It Works
   Documents are split into chunks.

   # Benefits
   Better accuracy and less hallucination."

Chunk 1: "# Introduction\nRAG is a technique for grounding LLMs."
Chunk 2: "# How It Works\nDocuments are split into chunks."
Chunk 3: "# Benefits\nBetter accuracy and less hallucination."
```

Detects headings by pattern: lines starting with `#`, numbered patterns like `1.` or `Section 2`, and `ALL CAPS LINES`. Falls back to paragraph splitting if no structural markers are found.

**Pros:** Each chunk maps to a logical section of the document.
**Cons:** Only works when the document has clear structural markers. Falls back to paragraph splitting otherwise.

## Chunk Overlap

When you split text at a boundary, information that **straddles** two chunks gets lost.

```
Without overlap:
  Chunk 1: "...Confederation was established in"
  Chunk 2: "1867 with four founding provinces..."

  Question: "When was Confederation?"
  --> Neither chunk has the full answer.

With overlap (last 30 chars repeated):
  Chunk 1: "...Confederation was established in"
  Chunk 2: "was established in 1867 with four founding provinces..."

  --> Chunk 2 now contains the complete answer.
```

Overlap duplicates some text at the edges of each chunk. This costs a bit of extra storage but prevents the "split right in the middle of the answer" problem.

## Chunk Size Matters

| Size | Problem |
|------|---------|
| **Too small** (100 chars) | Chunks lack context. "four founding provinces" without knowing what event is being described. The LLM can't make sense of a sentence fragment. |
| **Too big** (2000+ chars) | Too much noise. You're back to the Phase 2 problem -- sending irrelevant text that dilutes the answer. |
| **Sweet spot** (500-1000 chars) | Enough context to be meaningful, small enough to be focused. This is the range most RAG systems use. |

The right size depends on your documents. Dense technical writing might need larger chunks; simple Q&A content works with smaller ones.

## The 3 Demo Modes

The Phase 3 demo (`npm run phase3`) offers three interactive modes:

1. **Chunk & Chat** -- Load a file, chunk it with a chosen strategy, then ask questions. The system finds relevant chunks by keyword matching and sends only those to the LLM.

2. **Compare Strategies** -- Run all 5 strategies on the same document side by side. See how each one splits the text differently: chunk count, average size, and the actual boundaries.

3. **Explore Chunks** -- Inspect individual chunks from a strategy. Browse through them, see their content and metadata, and understand exactly where the splits happened.

## Naive Keyword Search

Once we have chunks, we need to find the relevant ones. Phase 3 uses the simplest possible approach: **count matching words**.

```
Question: "When did Canada become a country?"
Keywords: ["when", "did", "canada", "become", "a", "country"]

Chunk 1: "Confederation. The Dominion of Canada was officially created..."
  Matches: "canada" --> score: 1

Chunk 2: "The maple leaf flag was adopted in 1965..."
  Matches: (none) --> score: 0

Chunk 3: "Canada has ten provinces and three territories..."
  Matches: "canada" --> score: 1
```

The problem is obvious: **"When did Canada become a country?"** won't match a chunk about **"Confederation 1867"** unless it happens to contain the exact word "Canada". The search has no understanding of meaning -- it's purely lexical.

- "become a country" and "Confederation" are related concepts, but keyword search doesn't know that
- "BNA Act" is an abbreviation for "British North America Act", but it won't match either
- Synonyms, paraphrases, and abbreviations are all invisible

## Run It

```bash
npm run phase3
```

Choose a demo mode, select a file, pick a chunking strategy, and experiment. Try the same question across different strategies to see how chunk boundaries affect the answers.

## Limitations

- **Keyword search doesn't understand meaning** -- synonyms, paraphrases, and related concepts are invisible
- **Exact word matching only** -- "Confederation" won't be found by searching "When did Canada become a country?"
- **No ranking by relevance** -- a chunk that mentions "Canada" once scores the same whether it's about Confederation or geography

## What's Next

Phase 4 converts text into vectors (embeddings) so we can search by **meaning** instead of exact words. "When did Canada become a country?" will match "Confederation 1867" because they point in the same direction in vector space.
