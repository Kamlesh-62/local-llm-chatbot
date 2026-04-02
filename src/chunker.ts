// ============================================
// CHUNKER MODULE - Phase 3
// Breaks documents into smaller pieces (chunks)
// for more effective LLM context injection.
//
// Why chunk? LLMs have token limits. Instead of
// stuffing an entire document into the prompt,
// we split it into chunks, find the relevant ones,
// and only send those. This improves:
//   1. Speed (less tokens to process)
//   2. Accuracy (less noise, more signal)
//   3. Scalability (works with huge documents)
// ============================================

// --- Types ---

/** Metadata attached to every chunk for tracing back to the source */
export interface ChunkMetadata {
  /** Original file path */
  source: string;
  /** Which chunking strategy produced this chunk */
  strategy: ChunkingStrategy;
  /** Zero-based index of this chunk */
  chunkIndex: number;
  /** Total number of chunks produced */
  totalChunks: number;
  /** Character offset where this chunk starts in the original text */
  startOffset: number;
  /** Character offset where this chunk ends in the original text */
  endOffset: number;
}

export interface Chunk {
  /** The actual text content */
  content: string;
  /** Metadata for tracing and debugging */
  metadata: ChunkMetadata;
}

export type ChunkingStrategy =
  | "fixed-size"
  | "sentence"
  | "paragraph"
  | "recursive"
  | "semantic";

export interface ChunkingOptions {
  /** Target chunk size in characters (default: 500) */
  chunkSize: number;
  /** Overlapping characters between adjacent chunks (default: 50) */
  overlap: number;
  /** Source file path — stored in metadata */
  source: string;
}

// --- Helper: build metadata for a chunk array ---

function buildChunks(
  pieces: { content: string; startOffset: number }[],
  strategy: ChunkingStrategy,
  source: string
): Chunk[] {
  const total = pieces.length;
  return pieces.map((piece, i) => ({
    content: piece.content,
    metadata: {
      source,
      strategy,
      chunkIndex: i,
      totalChunks: total,
      startOffset: piece.startOffset,
      endOffset: piece.startOffset + piece.content.length,
    },
  }));
}

// ============================================
// STRATEGY 1: Fixed-Size Chunks
// ============================================
// The simplest approach: slide a window of `chunkSize`
// characters across the text. Each new window starts
// at (previous start + chunkSize - overlap).
//
// Overlap ensures that if a key sentence straddles two
// chunks, it appears in both — so retrieval won't miss it.
//
// Pros: Simple, predictable chunk count
// Cons: Cuts words and sentences mid-way
// ============================================

export function chunkFixedSize(text: string, options: ChunkingOptions): Chunk[] {
  const { chunkSize, overlap, source } = options;
  const step = chunkSize - overlap;
  const pieces: { content: string; startOffset: number }[] = [];

  for (let i = 0; i < text.length; i += step) {
    pieces.push({
      content: text.slice(i, i + chunkSize),
      startOffset: i,
    });
  }

  return buildChunks(pieces, "fixed-size", source);
}

// ============================================
// STRATEGY 2: Paragraph-Based Chunks
// ============================================
// Split on double newlines (paragraph boundaries).
// Accumulate paragraphs into a chunk until adding the
// next one would exceed `chunkSize`.
//
// If a single paragraph is larger than `chunkSize`,
// we fall back to fixed-size splitting for that paragraph.
//
// Overlap: the last paragraph of chunk N is repeated
// as the first paragraph of chunk N+1.
//
// Pros: Respects document structure
// Cons: Paragraph sizes vary wildly
// ============================================

export function chunkByParagraph(text: string, options: ChunkingOptions): Chunk[] {
  const { chunkSize, overlap, source } = options;
  const paragraphs = text.split(/\n\s*\n/);
  const pieces: { content: string; startOffset: number }[] = [];

  let currentChunk = "";
  let currentStart = 0;
  let offset = 0; // tracks position in original text

  for (let i = 0; i < paragraphs.length; i++) {
    const para = paragraphs[i]!.trim();
    if (!para) {
      // Account for the split separator length
      offset += (paragraphs[i]?.length ?? 0) + 2; // +2 for \n\n
      continue;
    }

    // Would adding this paragraph exceed the chunk size?
    const wouldBe = currentChunk ? currentChunk + "\n\n" + para : para;

    if (wouldBe.length > chunkSize && currentChunk) {
      // Save the current chunk
      pieces.push({ content: currentChunk, startOffset: currentStart });

      // Overlap: start the new chunk with the previous paragraph
      if (overlap > 0) {
        const overlapText = currentChunk.slice(-overlap);
        currentChunk = overlapText + "\n\n" + para;
      } else {
        currentChunk = para;
      }
      currentStart = offset;
    } else if (para.length > chunkSize && !currentChunk) {
      // Single paragraph is bigger than chunkSize — break it down
      // using fixed-size splitting so we don't lose any data
      const step = chunkSize - overlap;
      for (let j = 0; j < para.length; j += step) {
        pieces.push({
          content: para.slice(j, j + chunkSize),
          startOffset: offset + j,
        });
      }
      currentChunk = "";
      currentStart = offset + para.length;
    } else {
      if (!currentChunk) currentStart = offset;
      currentChunk = wouldBe;
    }

    offset += para.length + 2; // +2 for the \n\n separator
  }

  // Don't forget the last chunk
  if (currentChunk) {
    pieces.push({ content: currentChunk, startOffset: currentStart });
  }

  return buildChunks(pieces, "paragraph", source);
}

// ============================================
// STRATEGY 3: Sentence-Based Chunks
// ============================================
// Split on sentence boundaries (. ! ?), then accumulate
// sentences until the chunk would exceed `chunkSize`.
//
// For overlap, we repeat the last N sentences from the
// previous chunk that fit within the `overlap` budget.
//
// NOTE: The regex /(?<=[.!?])\s+/ is simple but imperfect.
// It will break on abbreviations ("Dr. Smith"), decimal
// numbers ("3.14"), URLs, etc. Production systems use NLP
// libraries like `compromise` or `natural` for proper
// sentence tokenization.
//
// Pros: Preserves complete sentences (better meaning)
// Cons: Sentence lengths vary; regex is imperfect
// ============================================

export function chunkBySentence(text: string, options: ChunkingOptions): Chunk[] {
  const { chunkSize, overlap, source } = options;

  // Split into sentences. The lookbehind keeps the punctuation with the sentence.
  const sentences = text.split(/(?<=[.!?])\s+/).filter((s) => s.trim());
  const pieces: { content: string; startOffset: number }[] = [];

  let currentSentences: string[] = [];
  let currentLength = 0;
  let currentStart = 0;
  let offset = 0;

  for (const sentence of sentences) {
    const addedLength = currentSentences.length > 0 ? sentence.length + 1 : sentence.length;

    if (currentLength + addedLength > chunkSize && currentSentences.length > 0) {
      // Save current chunk
      pieces.push({
        content: currentSentences.join(" "),
        startOffset: currentStart,
      });

      // Overlap: carry over last N sentences that fit in `overlap` chars
      const overlapSentences: string[] = [];
      let overlapLen = 0;
      for (let j = currentSentences.length - 1; j >= 0; j--) {
        if (overlapLen + currentSentences[j]!.length + 1 > overlap) break;
        overlapSentences.unshift(currentSentences[j]!);
        overlapLen += currentSentences[j]!.length + 1;
      }

      currentSentences = [...overlapSentences, sentence];
      currentLength = currentSentences.join(" ").length;
      currentStart = offset - overlapLen;
    } else {
      if (currentSentences.length === 0) currentStart = offset;
      currentSentences.push(sentence);
      currentLength += addedLength;
    }

    offset += sentence.length + 1; // +1 for the space between sentences
  }

  // Last chunk
  if (currentSentences.length > 0) {
    pieces.push({
      content: currentSentences.join(" "),
      startOffset: currentStart,
    });
  }

  return buildChunks(pieces, "sentence", source);
}

// ============================================
// STRATEGY 4: Recursive Character Splitting
// ============================================
// This is the approach used by LangChain's
// RecursiveCharacterTextSplitter. The idea:
//
// 1. Define a hierarchy of separators from "biggest
//    natural boundary" to "smallest":
//    ["\n\n", "\n", ". ", " ", ""]
//
// 2. Try splitting on the first separator. If any
//    resulting piece is still too large, recursively
//    split that piece using the next separator.
//
// This produces chunks that respect document structure
// as much as possible — paragraphs stay together when
// they fit, otherwise we fall back to lines, then
// sentences, then words, then raw characters.
//
// Pros: Best balance of structure and size
// Cons: More complex to implement and debug
// ============================================

export function chunkRecursive(text: string, options: ChunkingOptions): Chunk[] {
  const { chunkSize, overlap, source } = options;
  const separators = ["\n\n", "\n", ". ", " ", ""];

  // Recursively split text using the separator hierarchy
  function splitRecursive(text: string, sepIndex: number): string[] {
    // Base case: text fits in one chunk
    if (text.length <= chunkSize) {
      return [text];
    }

    // Base case: no more separators — force split by characters
    if (sepIndex >= separators.length) {
      const result: string[] = [];
      for (let i = 0; i < text.length; i += chunkSize) {
        result.push(text.slice(i, i + chunkSize));
      }
      return result;
    }

    const sep = separators[sepIndex]!;

    // If separator is empty string, split by characters
    if (sep === "") {
      const result: string[] = [];
      for (let i = 0; i < text.length; i += chunkSize) {
        result.push(text.slice(i, i + chunkSize));
      }
      return result;
    }

    const parts = text.split(sep);
    const result: string[] = [];
    let current = "";

    for (const part of parts) {
      const wouldBe = current ? current + sep + part : part;

      if (wouldBe.length <= chunkSize) {
        current = wouldBe;
      } else {
        // Save what we have
        if (current) result.push(current);

        // If this part alone is too big, recurse with next separator
        if (part.length > chunkSize) {
          result.push(...splitRecursive(part, sepIndex + 1));
        } else {
          current = part;
          continue;
        }
        current = "";
      }
    }

    if (current) result.push(current);
    return result;
  }

  const rawChunks = splitRecursive(text, 0);

  // Apply overlap between adjacent chunks
  const pieces: { content: string; startOffset: number }[] = [];
  let offset = 0;

  for (let i = 0; i < rawChunks.length; i++) {
    const chunk = rawChunks[i]!;

    if (i > 0 && overlap > 0) {
      // Grab the last `overlap` chars from the previous raw chunk
      const prevChunk = rawChunks[i - 1]!;
      const overlapText = prevChunk.slice(-overlap);
      pieces.push({
        content: overlapText + chunk,
        startOffset: offset - overlap,
      });
    } else {
      pieces.push({ content: chunk, startOffset: offset });
    }

    offset += chunk.length;
  }

  return buildChunks(pieces, "recursive", source);
}

// ============================================
// STRATEGY 5: Semantic Chunking (Simplified)
// ============================================
// Real semantic chunking uses embeddings to detect when
// the "meaning" changes between consecutive sentences.
// We'll build that in Phase 4.
//
// For now, we use structural cues as a proxy for topic
// boundaries:
//   - Markdown headings (lines starting with #)
//   - Numbered sections (lines starting with 1., 2., etc.)
//   - ALL CAPS lines (often section headers)
//
// If no headings are found, falls back to paragraph-based.
//
// Pros: Respects logical document sections
// Cons: Only works on well-structured documents
// ============================================

export function chunkSemantic(text: string, options: ChunkingOptions): Chunk[] {
  const { chunkSize, source } = options;

  // Detect section boundaries using structural cues
  const lines = text.split("\n");
  const sectionStarts: number[] = [0]; // first section always starts at 0

  const headingPattern = /^(#{1,6}\s|(\d+\.)\s|[A-Z][A-Z\s]{4,}$)/;

  let charOffset = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    if (i > 0 && headingPattern.test(line.trim())) {
      sectionStarts.push(charOffset);
    }
    charOffset += line.length + 1; // +1 for \n
  }

  // If no headings found, fall back to paragraph-based chunking
  if (sectionStarts.length <= 1) {
    const chunks = chunkByParagraph(text, options);
    // Re-label the strategy as semantic
    return chunks.map((c) => ({
      ...c,
      metadata: { ...c.metadata, strategy: "semantic" as ChunkingStrategy },
    }));
  }

  // Extract sections
  const pieces: { content: string; startOffset: number }[] = [];

  for (let i = 0; i < sectionStarts.length; i++) {
    const start = sectionStarts[i]!;
    const end = i + 1 < sectionStarts.length ? sectionStarts[i + 1]! : text.length;
    let section = text.slice(start, end).trim();

    if (!section) continue;

    // If section is too large, split it further with paragraph chunking
    if (section.length > chunkSize) {
      const subChunks = chunkByParagraph(section, { ...options, source });
      for (const sub of subChunks) {
        pieces.push({
          content: sub.content,
          startOffset: start + sub.metadata.startOffset,
        });
      }
    } else {
      pieces.push({ content: section, startOffset: start });
    }
  }

  return buildChunks(pieces, "semantic", source);
}

// ============================================
// DISPATCHER - Pick a strategy by name
// ============================================

export function chunkText(
  text: string,
  strategy: ChunkingStrategy,
  options: ChunkingOptions
): Chunk[] {
  switch (strategy) {
    case "fixed-size":
      return chunkFixedSize(text, options);
    case "sentence":
      return chunkBySentence(text, options);
    case "paragraph":
      return chunkByParagraph(text, options);
    case "recursive":
      return chunkRecursive(text, options);
    case "semantic":
      return chunkSemantic(text, options);
  }
}
