// ============================================
// RAG PIPELINE MODULE - Phase 5
//
// RAG = Retrieval-Augmented Generation
// Instead of asking the LLM to answer from memory,
// we RETRIEVE relevant chunks from our documents,
// AUGMENT the prompt with that context, then let
// the LLM GENERATE a grounded answer.
//
// This module wires together:
//   - Chunking (Phase 3)
//   - Embeddings & Vector Store (Phase 4)
//   - Hybrid search (keyword + embedding)
//   - Query expansion (LLM expands abbreviations)
//   - Re-ranking (filter + optional LLM re-rank)
//   - Source attribution (cite where the answer came from)
// ============================================

import "dotenv/config";
import { Ollama } from "ollama";
import * as fs from "fs";
import * as path from "path";
import { parse } from "csv-parse/sync";
import mammoth from "mammoth";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { chunkText, type Chunk, type ChunkingStrategy } from "./chunker.js";
import {
  VectorStore,
  embedTexts,
  embedText,
  type SearchResult,
} from "./vector-store.js";

// Local Ollama for embeddings (fast, small, free)
const localOllama = new Ollama();

// OpenRouter chat via fetch (fast cloud GPUs, free tier)
const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

async function cloudChat(
  model: string,
  messages: { role: "system" | "user" | "assistant"; content: string }[]
): Promise<string> {
  // Free models may not support system role — merge into first user message
  const processedMessages: { role: "user" | "assistant"; content: string }[] = [];
  let systemContent = "";

  for (const msg of messages) {
    if (msg.role === "system") {
      systemContent += msg.content + "\n\n";
    } else {
      processedMessages.push({ role: msg.role, content: msg.content });
    }
  }

  // Prepend system content to first user message
  if (systemContent && processedMessages.length > 0) {
    processedMessages[0]!.content = systemContent + processedMessages[0]!.content;
  }

  let response: Response;
  try {
    response = await fetch(OPENROUTER_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
      },
      body: JSON.stringify({ model, messages: processedMessages }),
    });
  } catch (err: any) {
    const cause = err.cause ? ` | cause: ${err.cause.message ?? err.cause.code ?? err.cause}` : "";
    const keySet = !!process.env.OPENROUTER_API_KEY;
    throw new Error(`Cloud chat fetch failed (key set: ${keySet}, url: ${OPENROUTER_URL}${cause})`);
  }

  const data = (await response.json()) as any;
  if (data.error) {
    throw new Error(data.error.message ?? JSON.stringify(data.error));
  }
  return (data.choices?.[0]?.message?.content ?? "").trim();
}

// --- Types ---

export interface RAGConfig {
  chatModel: string;
  embedModel: string;
  /** How many chunks to retrieve (default: 5) */
  topK: number;
  /** Minimum hybrid score to keep (default: 0.3) */
  scoreThreshold: number;
  /** Weights for combining keyword + embedding scores */
  hybridWeights: {
    keyword: number; // default 0.3
    embedding: number; // default 0.7
  };
  /** Use LLM to expand short/ambiguous queries before searching */
  enableQueryExpansion: boolean;
  /** Use LLM to re-rank results after hybrid search */
  enableReranking: boolean;
}

export const DEFAULT_CONFIG: RAGConfig = {
  chatModel: "nvidia/nemotron-3-super-120b-a12b:free",
  embedModel: "nomic-embed-text",
  topK: 20,
  scoreThreshold: 0.3,
  hybridWeights: { keyword: 0.3, embedding: 0.7 },
  enableQueryExpansion: false,
  enableReranking: false,
};

// ============================================
// CONTEXT WINDOW CALCULATOR
// ============================================
// Calculates how many chunks fit in the model's context
// window after reserving space for system prompt,
// conversation history, and the answer.
// ============================================

/** Known context window sizes for common models */
const MODEL_CONTEXT_WINDOWS: Record<string, number> = {
  "llama3.1:8b": 128000,
  "llama3.1:70b": 128000,
  "llama3.2": 4096,
  "qwen3:32b": 131072,
  "qwen3:8b": 131072,
  "qwen3:4b": 40960,
  "gemma3:27b": 128000,
  "gemma3:4b": 128000,
  "deepseek-r1:7b": 8192,
  "openai/gpt-4o-mini": 131072,
  "nvidia/nemotron-3-super-120b-a12b:free": 262144,
};

export interface ContextBudget {
  /** Total context window in tokens */
  contextWindow: number;
  /** Tokens reserved for system prompt */
  systemPromptReserved: number;
  /** Tokens reserved for conversation history */
  historyReserved: number;
  /** Tokens reserved for the LLM's answer */
  answerReserved: number;
  /** Tokens available for chunks */
  availableForChunks: number;
  /** Average tokens per chunk */
  avgChunkTokens: number;
  /** Max chunks that fit */
  maxTopK: number;
  /** Recommended topK (80% of max, for safety margin) */
  recommendedTopK: number;
  /** Total chunks in store */
  totalChunks: number;
}

export function calculateContextBudget(
  store: VectorStore,
  chatModel: string,
  historyTokens: number = 3000
): ContextBudget {
  const entries = store.getEntries();
  const totalChunks = entries.length;

  // Get context window for the model (default 8192 if unknown)
  const contextWindow = MODEL_CONTEXT_WINDOWS[chatModel] ?? 8192;

  // Reserve tokens for non-chunk content
  const systemPromptReserved = 300;
  const historyReserved = historyTokens;
  const answerReserved = 2000;

  const availableForChunks =
    contextWindow - systemPromptReserved - historyReserved - answerReserved;

  // Calculate average chunk size in tokens
  const totalChars = entries.reduce((sum, e) => sum + e.chunk.content.length, 0);
  const avgChunkChars = totalChunks > 0 ? totalChars / totalChunks : 500;
  const avgChunkTokens = Math.ceil(avgChunkChars / 4);

  // How many chunks fit
  const maxTopK = Math.floor(availableForChunks / avgChunkTokens);
  // Cap at 15 — good coverage without overwhelming the model or cloud API
  const PRACTICAL_MAX = 5;
  const recommendedTopK = Math.min(
    totalChunks,
    PRACTICAL_MAX,
    Math.max(5, Math.floor(maxTopK * 0.8))
  );

  return {
    contextWindow,
    systemPromptReserved,
    historyReserved,
    answerReserved,
    availableForChunks,
    avgChunkTokens,
    maxTopK,
    recommendedTopK,
    totalChunks,
  };
}

export interface HybridSearchResult {
  chunk: Chunk;
  embeddingScore: number;
  keywordScore: number;
  hybridScore: number;
  rank: number;
}

export interface SourceCitation {
  chunkIndex: number;
  source: string;
  score: number;
  preview: string;
}

export interface RAGResult {
  answer: string;
  sources: SourceCitation[];
  query: string;
  expandedQuery?: string;
  retrievalTimeMs: number;
  generationTimeMs: number;
}

// ============================================
// FILE READERS (duplicated for self-containment)
// ============================================

function readTextFile(filePath: string): string {
  return fs.readFileSync(filePath, "utf-8");
}

function readJsonFile(filePath: string): string {
  const raw = fs.readFileSync(filePath, "utf-8");
  const data = JSON.parse(raw);
  return JSON.stringify(data, null, 2);
}

function readCsvFile(filePath: string): string {
  const raw = fs.readFileSync(filePath, "utf-8");
  const records = parse(raw, { columns: true }) as Record<string, string>[];
  const lines = records.map((row, i) => {
    const fields = Object.entries(row)
      .map(([key, val]) => `${key}: ${val}`)
      .join(", ");
    return `Row ${i + 1}: ${fields}`;
  });
  return `CSV Data (${records.length} rows):\n${lines.join("\n")}`;
}

async function readPdfFile(filePath: string): Promise<string> {
  const pages = await readPdfPages(filePath);
  return pages.map((p) => p.text).join("\n\n");
}

/** Read a PDF and return text per page — used for page-based chunking */
export async function readPdfPages(
  filePath: string
): Promise<{ page: number; text: string }[]> {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdf = await getDocument({ data }).promise;
  const pages: { page: number; text: string }[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map((item: any) => item.str).join(" ");
    if (pageText.trim()) {
      pages.push({ page: i, text: pageText.trim() });
    }
  }
  return pages;
}

async function readDocxFile(filePath: string): Promise<string> {
  const buffer = fs.readFileSync(filePath);
  const result = await mammoth.extractRawText({ buffer });
  return result.value;
}

export async function readDocument(filePath: string): Promise<string> {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case ".txt":
      return readTextFile(filePath);
    case ".json":
      return readJsonFile(filePath);
    case ".csv":
      return readCsvFile(filePath);
    case ".md":
      return readTextFile(filePath);
    case ".pdf":
      return await readPdfFile(filePath);
    case ".docx":
    case ".doc":
      return await readDocxFile(filePath);
    default:
      return readTextFile(filePath);
  }
}

// ============================================
// KEYWORD SEARCH (normalized scores)
// ============================================
// Same word-counting approach as Phase 3/4, but
// scores are normalized to 0-1 so they can be
// combined with cosine similarity scores.
// ============================================

export function keywordSearch(
  query: string,
  chunks: Chunk[],
  topK: number
): { chunk: Chunk; score: number }[] {
  const queryWords = query
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length > 2);

  const scored = chunks.map((chunk) => {
    const chunkLower = chunk.content.toLowerCase();
    let rawScore = 0;
    for (const word of queryWords) {
      const regex = new RegExp(`\\b${word}\\b`, "gi");
      const matches = chunkLower.match(regex);
      rawScore += matches ? matches.length : 0;
    }
    return { chunk, rawScore };
  });

  // Normalize to 0-1 by dividing by max score
  const maxScore = Math.max(...scored.map((s) => s.rawScore), 1); // avoid /0
  const normalized = scored.map((s) => ({
    chunk: s.chunk,
    score: s.rawScore / maxScore,
  }));

  normalized.sort((a, b) => b.score - a.score);
  return normalized.slice(0, topK);
}

// ============================================
// HYBRID SEARCH
// ============================================
// The core innovation of Phase 5.
// Combines keyword search (exact matches) with
// embedding search (semantic meaning) into one
// ranked list.
//
// Why hybrid? Each method has blind spots:
//   - Embedding: fails on abbreviations like "MGRS"
//   - Keyword: fails on paraphrases like "work from home" vs "remote work"
//   - Hybrid: catches both!
//
// Formula:
//   hybridScore = (w_emb * embeddingScore) + (w_key * keywordScore)
//   Default weights: 0.7 embedding + 0.3 keyword
// ============================================

export function hybridSearch(
  query: string,
  queryEmbedding: number[],
  store: VectorStore,
  chunks: Chunk[],
  config: RAGConfig
): HybridSearchResult[] {
  const retrieveCount = config.topK * 3; // retrieve more, then filter

  // Run both searches
  const embeddingResults = store.search(queryEmbedding, retrieveCount);
  const keywordResults = keywordSearch(query, chunks, retrieveCount);

  // Build a map: chunkIndex → scores
  const scoreMap = new Map<
    number,
    { chunk: Chunk; embeddingScore: number; keywordScore: number }
  >();

  for (const r of embeddingResults) {
    scoreMap.set(r.chunk.metadata.chunkIndex, {
      chunk: r.chunk,
      embeddingScore: r.score,
      keywordScore: 0,
    });
  }

  for (const r of keywordResults) {
    const idx = r.chunk.metadata.chunkIndex;
    const existing = scoreMap.get(idx);
    if (existing) {
      existing.keywordScore = r.score;
    } else {
      scoreMap.set(idx, {
        chunk: r.chunk,
        embeddingScore: 0,
        keywordScore: r.score,
      });
    }
  }

  // Compute hybrid scores
  const { embedding: wEmb, keyword: wKey } = config.hybridWeights;
  const results: HybridSearchResult[] = [];

  for (const [, entry] of scoreMap) {
    const hybridScore =
      wEmb * entry.embeddingScore + wKey * entry.keywordScore;

    if (hybridScore >= config.scoreThreshold) {
      results.push({
        chunk: entry.chunk,
        embeddingScore: entry.embeddingScore,
        keywordScore: entry.keywordScore,
        hybridScore,
        rank: 0, // filled below
      });
    }
  }

  // Sort by hybrid score, assign ranks
  results.sort((a, b) => b.hybridScore - a.hybridScore);
  const topResults = results.slice(0, config.topK);
  topResults.forEach((r, i) => (r.rank = i + 1));

  return topResults;
}

// ============================================
// QUERY EXPANSION
// ============================================
// Short or ambiguous queries like "MGRS" produce
// weak embeddings. We use the LLM to expand them
// with synonyms and full forms before searching.
//
// "MGRS" → "MGRS Monthly Global Revenue Sharing
//           bonus qualification payout shares"
//
// The expanded query is used for searching only —
// the original query is what gets sent to the LLM
// for the final answer.
// ============================================

export async function expandQuery(
  query: string,
  chatModel: string,
  documentSources?: string[]
): Promise<string> {
  // Include document context so the LLM knows the domain
  const contextHint = documentSources?.length
    ? `\nThe documents being searched are: ${documentSources.join(", ")}. Use this context to expand abbreviations correctly.`
    : "";

  const expanded = await cloudChat(chatModel, [
    {
      role: "system",
      content: `You are a search query expander. Given a short query, expand it with synonyms, full forms of abbreviations, and related terms to improve document search. Return ONLY the expanded query on a single line, nothing else. Keep it under 50 words. Do not explain or add commentary.${contextHint}`,
    },
    { role: "user", content: query },
  ]);
  return `${query} ${expanded}`;
}

// ============================================
// RE-RANKING WITH LLM
// ============================================
// After hybrid search, we can optionally ask the
// LLM to re-order the results by relevance.
// This is slow (extra LLM call) but can catch
// cases where the scoring formula ranked incorrectly.
// ============================================

export async function rerankWithLLM(
  query: string,
  results: HybridSearchResult[],
  chatModel: string
): Promise<HybridSearchResult[]> {
  if (results.length <= 1) return results;

  // Build passages text
  const passages = results
    .map((r, i) => `${i + 1}. ${r.chunk.content.slice(0, 200)}`)
    .join("\n\n");

  const text = await cloudChat(chatModel, [
    {
      role: "system",
      content: `You are a relevance judge. Given a query and numbered passages, rank them from most to least relevant. Return ONLY the numbers in order, comma-separated. Example: "3,1,2". Do not explain.`,
    },
    {
      role: "user",
      content: `Query: "${query}"\n\nPassages:\n${passages}`,
    },
  ]);
  const numbers = text
    .split(/[,\s]+/)
    .map((s) => parseInt(s))
    .filter((n) => !isNaN(n) && n >= 1 && n <= results.length);

  // If parsing failed, fall back to original order
  if (numbers.length === 0) {
    return results;
  }

  // Re-order results based on LLM ranking
  const reranked: HybridSearchResult[] = [];
  const used = new Set<number>();

  for (const num of numbers) {
    const idx = num - 1; // convert to 0-based
    if (!used.has(idx) && idx < results.length) {
      reranked.push({ ...results[idx]!, rank: reranked.length + 1 });
      used.add(idx);
    }
  }

  // Add any results the LLM missed
  for (let i = 0; i < results.length; i++) {
    if (!used.has(i)) {
      reranked.push({ ...results[i]!, rank: reranked.length + 1 });
    }
  }

  return reranked;
}

// ============================================
// PROMPT ENGINEERING FOR RAG
// ============================================

export function buildRAGPrompt(
  query: string,
  results: HybridSearchResult[],
  conversationHistory?: { role: "user" | "assistant" | "system"; content: string }[]
): { role: "system" | "user" | "assistant"; content: string }[] {
  const contextParts = results.map((r) => {
    return `[Source: ${r.chunk.metadata.source}, Chunk ${r.chunk.metadata.chunkIndex + 1} (relevance: ${r.hybridScore.toFixed(3)})]
${r.chunk.content}`;
  });

  const system = `/no_think
Answer the user's question using ONLY the context below. Be concise. If the answer is not in the context, say "Not found in documents."

--- CONTEXT ---
${contextParts.join("\n\n")}
--- END CONTEXT ---`;

  const messages: { role: "system" | "user" | "assistant"; content: string }[] = [
    { role: "system", content: system },
  ];

  // Include conversation history for multi-turn context
  if (conversationHistory) {
    for (const msg of conversationHistory) {
      // Only include user/assistant messages (skip system messages like summaries)
      if (msg.role === "user" || msg.role === "assistant") {
        messages.push({ role: msg.role, content: msg.content });
      }
    }
  }

  // Add the current query
  messages.push({ role: "user", content: query });

  return messages;
}

// ============================================
// SOURCE ATTRIBUTION
// ============================================

export function formatSources(sources: SourceCitation[]): string {
  if (sources.length === 0) return "Sources: none";

  // Group by source file
  const grouped = new Map<string, SourceCitation[]>();
  for (const s of sources) {
    if (!grouped.has(s.source)) grouped.set(s.source, []);
    grouped.get(s.source)!.push(s);
  }

  const lines: string[] = ["Sources:"];
  for (const [source, citations] of grouped) {
    lines.push(`  ${source}`);
    for (const c of citations) {
      const preview = c.preview.slice(0, 80);
      lines.push(
        `    - Chunk ${c.chunkIndex + 1} (score: ${c.score.toFixed(3)}): "${preview}..."`
      );
    }
  }

  return lines.join("\n");
}

// ============================================
// DOCUMENT INGESTION
// ============================================

export async function ingestDocuments(
  filePaths: string[],
  strategy: ChunkingStrategy,
  chunkSize: number,
  overlap: number,
  embedModel: string,
  onProgress?: (msg: string) => void
): Promise<{ store: VectorStore; chunks: Chunk[] }> {
  const log = onProgress ?? ((msg: string) => console.log(msg));
  const allChunks: Chunk[] = [];

  // Read and chunk each file
  for (const filePath of filePaths) {
    const fileName = path.basename(filePath);
    const ext = path.extname(filePath).toLowerCase();
    log(`  Reading ${fileName}...`);

    if (ext === ".pdf") {
      // PDF: chunk by page — each page becomes its own chunk
      // This preserves the natural page boundaries and keeps
      // related content together (headings + body on same page)
      const pages = await readPdfPages(filePath);
      log(`  ${pages.length} pages → one chunk per page`);

      for (const page of pages) {
        allChunks.push({
          content: page.text,
          metadata: {
            source: fileName,
            strategy: "semantic" as ChunkingStrategy,
            chunkIndex: allChunks.length,
            totalChunks: 0, // filled after all chunks collected
            startOffset: 0,
            endOffset: page.text.length,
          },
        });
      }
    } else {
      // Non-PDF: use the selected chunking strategy
      const text = await readDocument(filePath);
      log(`  ${text.length} chars → chunking with "${strategy}"...`);

      const chunks = chunkText(text, strategy, {
        chunkSize,
        overlap,
        source: fileName,
      });
      allChunks.push(...chunks);
    }

    log(`  ${allChunks.length} chunks so far`);
  }

  // Fix totalChunks count now that we know the final number
  for (const chunk of allChunks) {
    chunk.metadata.totalChunks = allChunks.length;
  }

  log(`\n  Total: ${allChunks.length} chunks from ${filePaths.length} file(s)`);

  // Embed all chunks with progress
  const store = new VectorStore(embedModel);
  const batchSize = 5;

  for (let i = 0; i < allChunks.length; i += batchSize) {
    const batch = allChunks.slice(i, i + batchSize);
    const texts = batch.map((c) => c.content);

    log(
      `  Embedding ${i + 1}-${Math.min(i + batchSize, allChunks.length)} of ${allChunks.length}...`
    );

    const embeddings = await embedTexts(texts, embedModel);
    store.addBatch(batch, embeddings);
  }

  log(`  Done! ${store.getCount()} vectors stored (${store.getMetadata().dimensions}d)`);

  return { store, chunks: allChunks };
}

// ============================================
// THE FULL RAG PIPELINE
// ============================================
// This ties everything together into one call:
//   query → expand → embed → hybrid search →
//   rerank → build prompt → stream LLM → result
// ============================================

export async function ragQuery(
  query: string,
  store: VectorStore,
  config: RAGConfig,
  conversationHistory?: { role: "user" | "assistant" | "system"; content: string }[],
  onToken?: (token: string) => void
): Promise<RAGResult> {
  const retrievalStart = Date.now();

  // Step 1: Query expansion (optional)
  let searchQuery = query;
  let expandedQuery: string | undefined;

  if (config.enableQueryExpansion) {
    const sources = store.getMetadata().sources;
    expandedQuery = await expandQuery(query, config.chatModel, sources);
    searchQuery = expandedQuery;
  }

  // Step 2: Embed the search query
  const queryEmbedding = await embedText(searchQuery, config.embedModel);

  // Step 3: Hybrid search
  const chunks = store.getEntries().map((e) => e.chunk);
  let results = hybridSearch(
    searchQuery,
    queryEmbedding,
    store,
    chunks,
    config
  );

  // Step 4: Re-rank (optional)
  if (config.enableReranking && results.length > 1) {
    results = await rerankWithLLM(query, results, config.chatModel);
  }

  const retrievalTimeMs = Date.now() - retrievalStart;

  // Step 5: Build RAG prompt (now includes conversation history)
  const messages = buildRAGPrompt(query, results, conversationHistory);

  // Step 6: Generate LLM response
  // We collect the full response first, then strip <think> blocks
  // (deepseek-r1 outputs <think>reasoning...</think>answer).
  // This is more reliable than trying to filter during streaming.
  const generationStart = Date.now();

  let answer = await cloudChat(config.chatModel, messages);

  // Strip <think>...</think> blocks if any
  answer = answer.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

  // Show the answer to the user
  if (onToken && answer) onToken(answer);

  const generationTimeMs = Date.now() - generationStart;

  // Step 7: Build source citations
  const sources: SourceCitation[] = results.map((r) => ({
    chunkIndex: r.chunk.metadata.chunkIndex,
    source: r.chunk.metadata.source,
    score: r.hybridScore,
    preview: r.chunk.content.slice(0, 100).replace(/\n/g, " "),
  }));

  return {
    answer,
    sources,
    query,
    ...(expandedQuery !== undefined ? { expandedQuery } : {}),
    retrievalTimeMs,
    generationTimeMs,
  };
}
