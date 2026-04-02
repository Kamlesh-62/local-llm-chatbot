import { Ollama } from "ollama";
import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import { parse } from "csv-parse/sync";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import {
  chunkText,
  type Chunk,
  type ChunkingStrategy,
  type ChunkingOptions,
} from "./chunker.js";

const ollama = new Ollama();
const MODEL = "llama3.2";

// ============================================
// File Readers (duplicated from phase2)
// In a real project, you'd extract these into a
// shared module. We duplicate here to keep each
// phase self-contained for learning.
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
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdf = await getDocument({ data }).promise;
  const textParts: string[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map((item: any) => item.str).join(" ");
    textParts.push(pageText);
  }
  return textParts.join("\n\n");
}

async function readDocument(filePath: string): Promise<string> {
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
    default:
      return readTextFile(filePath);
  }
}

// ============================================
// NAIVE KEYWORD RETRIEVAL
// ============================================
// This is intentionally simple. We score each chunk by
// how many words from the query appear in it.
//
// Phase 4 replaces this with embedding-based semantic
// search (cosine similarity over vectors), which is
// MUCH better — it understands meaning, not just keywords.
// For example, "work hours" would match "schedule" with
// embeddings, but not with keyword matching.
// ============================================

function findRelevantChunks(query: string, chunks: Chunk[], topK: number = 3): Chunk[] {
  const queryWords = query
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length > 2); // skip tiny words like "a", "is", "the"

  const scored = chunks.map((chunk) => {
    const chunkLower = chunk.content.toLowerCase();
    let score = 0;
    for (const word of queryWords) {
      // Count occurrences of each query word in the chunk
      const regex = new RegExp(`\\b${word}\\b`, "gi");
      const matches = chunkLower.match(regex);
      score += matches ? matches.length : 0;
    }
    return { chunk, score };
  });

  // Sort by score descending, take top K
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK).map((s) => s.chunk);
}

// ============================================
// STREAMING CHAT with chunk context
// ============================================

async function chatWithChunks(
  question: string,
  relevantChunks: Chunk[],
  conversationMessages: { role: "system" | "user" | "assistant"; content: string }[]
): Promise<string> {
  // Build context from chunks
  const contextParts = relevantChunks.map((chunk, i) => {
    const meta = chunk.metadata;
    return `[Chunk ${meta.chunkIndex + 1} of ${meta.totalChunks} | ${meta.strategy} | chars ${meta.startOffset}-${meta.endOffset}]\n${chunk.content}`;
  });

  const systemPrompt = `You are a helpful assistant. Answer questions based ONLY on the provided context chunks.
If the answer is not in the chunks, say "I don't find that in the provided context."
Cite which chunk number(s) your answer comes from.

--- CONTEXT CHUNKS ---
${contextParts.join("\n\n")}
--- END CONTEXT ---`;

  // Build messages: system + conversation history + new question
  const messages = [
    { role: "system" as const, content: systemPrompt },
    ...conversationMessages,
    { role: "user" as const, content: question },
  ];

  console.log("  [Message sent. Waiting for model...]");
  const startTime = Date.now();

  const spinnerFrames = ["|", "/", "-", "\\"];
  let spinnerIndex = 0;
  const spinner = setInterval(() => {
    process.stdout.write(
      `\r  Thinking... ${spinnerFrames[spinnerIndex++ % spinnerFrames.length]}`
    );
  }, 100);

  const stream = await ollama.chat({ model: MODEL, messages, stream: true });

  clearInterval(spinner);
  process.stdout.write("\r                    \r");

  let reply = "";
  process.stdout.write("Assistant: ");
  for await (const chunk of stream) {
    process.stdout.write(chunk.message.content);
    reply += chunk.message.content;
  }

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n  [Done in ${elapsed}s]\n`);

  return reply;
}

// ============================================
// READLINE HELPERS
// ============================================

function createPrompt(): {
  prompt: (q: string) => Promise<string>;
  close: () => void;
} {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return {
    prompt: (q: string) => new Promise((resolve) => rl.question(q, resolve)),
    close: () => rl.close(),
  };
}

async function pickFile(
  prompt: (q: string) => Promise<string>
): Promise<string | null> {
  const docsDir = path.join(process.cwd(), "sample-docs");
  const files = fs.readdirSync(docsDir);
  console.log("\nAvailable documents:");
  files.forEach((f, i) => console.log(`  ${i + 1}. ${f}`));
  console.log();

  const choice = await prompt("Enter file number (or full path): ");
  const num = parseInt(choice);

  if (!isNaN(num) && num >= 1 && num <= files.length) {
    return path.join(docsDir, files[num - 1]!);
  }

  const filePath = choice.trim();
  if (fs.existsSync(filePath)) return filePath;

  console.log("File not found.");
  return null;
}

function pickStrategy(): ChunkingStrategy[] {
  const strategies: ChunkingStrategy[] = [
    "fixed-size",
    "sentence",
    "paragraph",
    "recursive",
    "semantic",
  ];
  return strategies;
}

async function pickOneStrategy(
  prompt: (q: string) => Promise<string>
): Promise<ChunkingStrategy> {
  const strategies = pickStrategy();
  console.log("\nChunking strategies:");
  strategies.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
  console.log();

  const choice = await prompt("Pick a strategy (1-5, default 4=recursive): ");
  const num = parseInt(choice);
  if (!isNaN(num) && num >= 1 && num <= strategies.length) {
    return strategies[num - 1]!;
  }
  return "recursive"; // default
}

// ============================================
// MODE 1: Chunk & Chat
// ============================================

async function modeChunkAndChat(prompt: (q: string) => Promise<string>) {
  const filePath = await pickFile(prompt);
  if (!filePath) return;

  console.log(`\nReading: ${path.basename(filePath)}...`);
  const text = await readDocument(filePath);
  console.log(`Document length: ${text.length} characters\n`);

  const strategy = await pickOneStrategy(prompt);

  const sizeInput = await prompt("Chunk size in chars (default 500): ");
  const chunkSize = parseInt(sizeInput) || 500;

  const overlapInput = await prompt("Overlap in chars (default 50): ");
  const overlap = parseInt(overlapInput) || 50;

  const options: ChunkingOptions = {
    chunkSize,
    overlap,
    source: path.basename(filePath),
  };

  console.log(`\nChunking with "${strategy}" (size=${chunkSize}, overlap=${overlap})...`);
  const chunks = chunkText(text, strategy, options);

  const avgSize = Math.round(
    chunks.reduce((sum, c) => sum + c.content.length, 0) / chunks.length
  );
  console.log(`Created ${chunks.length} chunks (avg ${avgSize} chars each)\n`);

  // Preview first 3 chunks
  console.log("--- Chunk Preview ---");
  for (let i = 0; i < Math.min(3, chunks.length); i++) {
    const c = chunks[i]!;
    console.log(
      `\n[Chunk ${i + 1}/${chunks.length} | chars ${c.metadata.startOffset}-${c.metadata.endOffset}]`
    );
    console.log(c.content.slice(0, 150) + (c.content.length > 150 ? "..." : ""));
  }
  console.log("\n--- End Preview ---\n");

  console.log('Ask questions about the document. Type "quit" to exit.\n');

  const conversationMessages: {
    role: "system" | "user" | "assistant";
    content: string;
  }[] = [];

  while (true) {
    const userInput = await prompt("You: ");
    if (userInput.trim().toLowerCase() === "quit") break;
    if (!userInput.trim()) continue;

    // Find relevant chunks using naive keyword search
    const relevant = findRelevantChunks(userInput, chunks, 3);
    console.log(
      `  [Found ${relevant.length} relevant chunks: ${relevant.map((c) => `#${c.metadata.chunkIndex + 1}`).join(", ")}]`
    );

    const reply = await chatWithChunks(
      userInput,
      relevant,
      conversationMessages
    );

    // Save to conversation history
    conversationMessages.push({ role: "user", content: userInput });
    conversationMessages.push({ role: "assistant", content: reply });
  }
}

// ============================================
// MODE 2: Compare Strategies
// ============================================

async function modeCompare(prompt: (q: string) => Promise<string>) {
  const filePath = await pickFile(prompt);
  if (!filePath) return;

  console.log(`\nReading: ${path.basename(filePath)}...`);
  const text = await readDocument(filePath);
  console.log(`Document length: ${text.length} characters\n`);

  const sizeInput = await prompt("Chunk size in chars (default 500): ");
  const chunkSize = parseInt(sizeInput) || 500;

  const question = await prompt("\nEnter your question: ");
  if (!question.trim()) return;

  // Run all strategies (skip semantic for speed since it falls back to paragraph)
  const strategies: ChunkingStrategy[] = [
    "fixed-size",
    "sentence",
    "paragraph",
    "recursive",
  ];

  console.log(
    `\nComparing ${strategies.length} strategies on: "${question}"\n`
  );
  console.log("=".repeat(60));

  for (const strategy of strategies) {
    const options: ChunkingOptions = {
      chunkSize,
      overlap: 50,
      source: path.basename(filePath),
    };

    const chunks = chunkText(text, strategy, options);
    const relevant = findRelevantChunks(question, chunks, 3);

    console.log(`\n=== ${strategy.toUpperCase()} (${chunkSize} chars, 50 overlap) ===`);
    console.log(`Chunks created: ${chunks.length}`);
    console.log(
      `Top chunks: ${relevant.map((c) => `#${c.metadata.chunkIndex + 1}`).join(", ")}`
    );

    const reply = await chatWithChunks(question, relevant, []);

    console.log("-".repeat(60));
  }

  console.log("\n" + "=".repeat(60));
  console.log("Comparison complete! Notice how different strategies");
  console.log("produce different chunk counts and potentially different answers.\n");
}

// ============================================
// MODE 3: Explore Chunks (no LLM)
// ============================================

async function modeExplore(prompt: (q: string) => Promise<string>) {
  const filePath = await pickFile(prompt);
  if (!filePath) return;

  console.log(`\nReading: ${path.basename(filePath)}...`);
  const text = await readDocument(filePath);
  console.log(`Document length: ${text.length} characters\n`);

  const strategy = await pickOneStrategy(prompt);

  const sizeInput = await prompt("Chunk size in chars (default 500): ");
  const chunkSize = parseInt(sizeInput) || 500;

  const overlapInput = await prompt("Overlap in chars (default 50): ");
  const overlap = parseInt(overlapInput) || 50;

  const options: ChunkingOptions = {
    chunkSize,
    overlap,
    source: path.basename(filePath),
  };

  const chunks = chunkText(text, strategy, options);
  const avgSize = Math.round(
    chunks.reduce((sum, c) => sum + c.content.length, 0) / chunks.length
  );

  console.log(`\n--- ${strategy} | ${chunks.length} chunks | avg ${avgSize} chars ---\n`);

  for (const chunk of chunks) {
    const m = chunk.metadata;
    console.log(
      `[Chunk ${m.chunkIndex + 1}/${m.totalChunks} | chars ${m.startOffset}-${m.endOffset} | ${chunk.content.length} chars]`
    );
    console.log(chunk.content);
    console.log("-".repeat(40));
  }

  console.log(`\nTotal: ${chunks.length} chunks from "${path.basename(filePath)}"`);
  console.log(`Strategy: ${strategy} | Size: ${chunkSize} | Overlap: ${overlap}\n`);
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log(`\n--- Phase 3: Chunking Strategies (${MODEL}) ---\n`);
  console.log("Modes:");
  console.log("  1. Chunk & Chat     - pick a file + strategy, ask questions");
  console.log("  2. Compare          - same file + question across ALL strategies");
  console.log("  3. Explore Chunks   - inspect what chunks look like (no LLM)");
  console.log();

  const { prompt, close } = createPrompt();

  const modeChoice = await prompt("Pick a mode (1-3): ");
  const mode = parseInt(modeChoice) || 1;

  switch (mode) {
    case 1:
      await modeChunkAndChat(prompt);
      break;
    case 2:
      await modeCompare(prompt);
      break;
    case 3:
      await modeExplore(prompt);
      break;
    default:
      console.log("Invalid mode.");
  }

  close();
  console.log("Goodbye!");
}

main();
