// ============================================
// VECTOR STORE MODULE - Phase 4
//
// What are embeddings?
// Embeddings convert text into arrays of numbers (vectors)
// that capture the *meaning* of the text. Similar texts
// produce similar vectors, even if they use different words.
//
// Example:
//   "Can I work from home?" → [0.12, -0.45, 0.78, ...]
//   "Remote work policy"    → [0.11, -0.43, 0.80, ...]
//   "Stock market crash"    → [-0.56, 0.22, -0.11, ...]
//
// The first two vectors are close together (similar meaning).
// The third points in a totally different direction.
//
// We use cosine similarity to measure this closeness.
// ============================================

import "dotenv/config";
import { Ollama } from "ollama";
import * as fs from "fs";
import type { Chunk, ChunkMetadata } from "./chunker.js";

// Embeddings always run locally — they're fast and small (nomic-embed-text is 274MB).
// Cloud is only used for the chat model (bigger, smarter models).
const ollama = new Ollama();

const DEFAULT_MODEL = "nomic-embed-text";

// --- Types ---

/** A chunk paired with its embedding vector */
export interface EmbeddedChunk {
  chunk: Chunk;
  embedding: number[];
}

/** A search result with similarity score */
export interface SearchResult {
  chunk: Chunk;
  score: number; // cosine similarity, 0 to 1
  rank: number; // 1-based
}

/** Info about the vector store */
export interface VectorStoreMetadata {
  model: string;
  dimensions: number;
  count: number;
  updatedAt: string;
  sources: string[];
}

/** Shape of the JSON file when saved to disk */
interface VectorStoreSnapshot {
  metadata: VectorStoreMetadata;
  entries: EmbeddedChunk[];
}

// ============================================
// MATH FUNCTIONS (built from scratch)
// ============================================

/**
 * Dot product of two vectors.
 *
 * Formula: A · B = a1*b1 + a2*b2 + ... + an*bn
 *
 * This is the core operation behind cosine similarity.
 * It measures how much two vectors "agree" in direction.
 */
export function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i]! * b[i]!;
  }
  return sum;
}

/**
 * Magnitude (length) of a vector.
 *
 * Formula: |A| = sqrt(a1² + a2² + ... + an²)
 *
 * Think of it as the distance from the origin to the point
 * the vector represents in N-dimensional space.
 */
export function magnitude(v: number[]): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i]! * v[i]!;
  }
  return Math.sqrt(sum);
}

/**
 * Cosine similarity between two vectors.
 *
 * Formula: cos(θ) = (A · B) / (|A| × |B|)
 *
 * Returns a value from -1 to 1:
 *   1  = identical direction (same meaning)
 *   0  = perpendicular (unrelated)
 *  -1  = opposite direction (opposite meaning)
 *
 * For embedding vectors, values are typically 0 to 1.
 *
 * Why cosine instead of Euclidean distance?
 * Cosine ignores vector magnitude (length) and only cares
 * about direction. Two texts about "remote work" will point
 * in similar directions even if one is a long paragraph and
 * the other is a short question.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const dot = dotProduct(a, b);
  const magA = magnitude(a);
  const magB = magnitude(b);

  // Avoid division by zero
  if (magA === 0 || magB === 0) return 0;

  return dot / (magA * magB);
}

// ============================================
// EMBEDDING HELPERS (wrap Ollama API)
// ============================================

/**
 * Embed multiple texts in one batch call.
 * The Ollama embed API accepts an array of inputs.
 */
export async function embedTexts(
  texts: string[],
  model: string = DEFAULT_MODEL
): Promise<number[][]> {
  const response = await ollama.embed({
    model,
    input: texts,
  });
  return response.embeddings;
}

/**
 * Embed a single text. Convenience wrapper around embedTexts.
 */
export async function embedText(
  text: string,
  model: string = DEFAULT_MODEL
): Promise<number[]> {
  const embeddings = await embedTexts([text], model);
  return embeddings[0]!;
}

// ============================================
// VECTOR STORE CLASS
// ============================================

export class VectorStore {
  private entries: EmbeddedChunk[] = [];
  private model: string;
  private dimensions: number | null = null;

  constructor(model: string = DEFAULT_MODEL) {
    this.model = model;
  }

  /** Add a single chunk with its pre-computed embedding */
  add(chunk: Chunk, embedding: number[]): void {
    if (this.dimensions === null) {
      this.dimensions = embedding.length;
    }
    this.entries.push({ chunk, embedding });
  }

  /** Add multiple chunks with their embeddings at once */
  addBatch(chunks: Chunk[], embeddings: number[][]): void {
    for (let i = 0; i < chunks.length; i++) {
      this.add(chunks[i]!, embeddings[i]!);
    }
  }

  /**
   * Search for the most similar chunks to a query embedding.
   *
   * This is the core of semantic search:
   * 1. Compare the query vector against every stored vector
   * 2. Rank by cosine similarity (highest = most relevant)
   * 3. Return the top K results
   */
  search(queryEmbedding: number[], topK: number = 3): SearchResult[] {
    const scored = this.entries.map((entry) => ({
      chunk: entry.chunk,
      score: cosineSimilarity(queryEmbedding, entry.embedding),
    }));

    // Sort by score descending
    scored.sort((a, b) => b.score - a.score);

    // Take top K and add rank
    return scored.slice(0, topK).map((item, i) => ({
      ...item,
      rank: i + 1,
    }));
  }

  /** Get metadata about the store */
  getMetadata(): VectorStoreMetadata {
    const sources = [...new Set(this.entries.map((e) => e.chunk.metadata.source))];
    return {
      model: this.model,
      dimensions: this.dimensions ?? 0,
      count: this.entries.length,
      updatedAt: new Date().toISOString(),
      sources,
    };
  }

  /** Get total number of stored chunks */
  getCount(): number {
    return this.entries.length;
  }

  /** Remove all entries */
  clear(): void {
    this.entries = [];
    this.dimensions = null;
  }

  /** Get all entries (for inspection/comparison) */
  getEntries(): EmbeddedChunk[] {
    return this.entries;
  }

  // --- Persistence ---

  /**
   * Save the vector store to a JSON file.
   * This avoids re-embedding documents on every run.
   */
  save(filePath: string): void {
    const dir = filePath.substring(0, filePath.lastIndexOf("/"));
    if (dir && !fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const snapshot: VectorStoreSnapshot = {
      metadata: this.getMetadata(),
      entries: this.entries,
    };

    fs.writeFileSync(filePath, JSON.stringify(snapshot));
    console.log(`  Saved ${this.entries.length} vectors to ${filePath}`);
  }

  /**
   * Load a vector store from a previously saved JSON file.
   */
  static load(filePath: string): VectorStore {
    const raw = fs.readFileSync(filePath, "utf-8");
    const snapshot: VectorStoreSnapshot = JSON.parse(raw);

    const store = new VectorStore(snapshot.metadata.model);
    store.dimensions = snapshot.metadata.dimensions;
    store.entries = snapshot.entries;

    console.log(
      `  Loaded ${store.entries.length} vectors (${snapshot.metadata.dimensions}d) from ${filePath}`
    );
    return store;
  }
}
