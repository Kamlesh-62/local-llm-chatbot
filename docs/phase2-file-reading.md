# Phase 2: File Reading -- Feed Documents to the LLM

> **Key source:** `src/phase2-file-reader.ts`

## Goal

Read different file formats, extract their text content, and use it as context so the LLM can answer questions about your documents.

## What You'll Learn

- How to parse multiple document formats in TypeScript
- Context stuffing: the simplest form of RAG
- Why PDF text extraction is unreliable
- Token limits and their practical impact on performance

## Supported Formats

| Format | Library / Method | Notes |
|--------|-----------------|-------|
| `.txt` | `fs.readFileSync` | Plain text, no parsing needed |
| `.md` | `fs.readFileSync` | Markdown read as raw text |
| `.json` | `JSON.parse` | Stringified back for the prompt |
| `.csv` | `csv-parse` | Parsed into rows, converted to readable text |
| `.pdf` | `pdfjs-dist` | Text extraction from PDF pages |
| `.docx` | `mammoth` | Extracts raw text from Word documents |

## How It Works

```
  Pick a file (or provide a path)
        |
        v
  +----------------------+
  | Detect format        |  (.txt? .pdf? .csv? ...)
  +----------------------+
        |
        v
  +----------------------+
  | Parse with the right |
  | library              |  (pdfjs-dist, mammoth, etc.)
  +----------------------+
        |
        v
  +----------------------+
  | Inject extracted     |
  | text into system     |  "Use this document to answer
  | prompt               |   questions: {content}"
  +----------------------+
        |
        v
  +----------------------+
  | Chat with Ollama     |  (same streaming loop as Phase 1)
  +----------------------+
```

## Key Concept: Context Stuffing

Context stuffing is the **simplest form of RAG** -- you take the entire document, paste it into the system prompt, and ask the LLM to answer from it.

```
System prompt:
  "You are a helpful assistant. Use the following document
   to answer the user's questions:

   [ENTIRE DOCUMENT TEXT HERE]"
```

**Why it works:** The LLM sees the document content as part of its instructions and can reference it when answering.

**Why it breaks:** The system prompt has a size limit (the model's context window). A large document fills it up, leaving no room for conversation -- or exceeding the limit entirely.

## PDF Text Extraction -- The Hard Part

`pdfjs-dist` extracts text from PDFs by reading the underlying text layer. This works well for simple documents, but **it loses all layout information**.

Consider a glossary table in a PDF:

**What the PDF looks like (visual layout):**

```
Term          Definition
-----------   ------------------------------------------
RAG           Retrieval-Augmented Generation
Embedding     A vector representation of text
Chunk         A segment of a larger document
```

**What `pdfjs-dist` extracts:**

```
Term Definition RAG Retrieval-Augmented Generation Embedding A vector
representation of text Chunk A segment of a larger document
```

The table structure is gone. Columns merge into a single stream of text. The LLM has to guess what goes with what.

**Compare with how Claude reads PDFs:** Claude uses a vision-based approach -- it "sees" the rendered page as an image, preserving tables, columns, and layout. Our text extraction approach is fundamentally different: it pulls raw text and loses structure.

This is a known limitation of text-based PDF parsing. Better alternatives include OCR-based extraction or vision models, but they come with their own trade-offs (cost, speed, complexity).

## Token Limits

Every character you inject into the prompt consumes tokens. Here is what that means in practice:

- A **chapter PDF from the Discover Canada guide** produces roughly **12,000 tokens** of extracted text
- With `deepseek-r1:7b`, processing that much context took **187 seconds** for a single response
- The model has to read and attend to **ALL** the text, even if only one paragraph is relevant to the question

This is the core problem context stuffing creates: **you pay the cost of the entire document on every question**, regardless of relevance.

## Run It

```bash
npm run phase2
```

Select a file when prompted. The program extracts text, injects it into the system prompt, and starts an interactive chat about the document's contents.

## Limitations

- **Entire file in prompt** -- slow and wastes tokens on irrelevant content
- **Hits token limits on large docs** -- the model's context window is finite; big documents get truncated or rejected
- **No way to find the "relevant" part** -- the entire chapter is processed whether you ask about Confederation or the Charter of Rights
- **PDF structure loss** -- tables, columns, and formatting are flattened into plain text

## What's Next

Phase 3 breaks documents into smaller chunks so we can eventually retrieve only the relevant pieces instead of stuffing everything into the prompt.
