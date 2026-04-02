import { Ollama } from "ollama";
import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import { parse } from "csv-parse/sync";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { chunkText, type Chunk, type ChunkingStrategy } from "./chunker.js";
import {
  VectorStore,
  embedTexts,
  embedText,
  cosineSimilarity,
  type SearchResult,
} from "./vector-store.js";

const ollama = new Ollama();
const CHAT_MODEL = "llama3.2";
const EMBED_MODEL = "nomic-embed-text";
const DEFAULT_STORE_PATH = "data/vectors.json";

// ============================================
// File Readers (duplicated from phase2/3)
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
// KEYWORD SEARCH (from Phase 3, for comparison)
// ============================================

function findRelevantChunksKeyword(
  query: string,
  chunks: Chunk[],
  topK: number = 3
): { chunk: Chunk; score: number }[] {
  const queryWords = query
    .toLowerCase()
    .split(/\W+/)
    .filter((w) => w.length > 2);

  const scored = chunks.map((chunk) => {
    const chunkLower = chunk.content.toLowerCase();
    let score = 0;
    for (const word of queryWords) {
      const regex = new RegExp(`\\b${word}\\b`, "gi");
      const matches = chunkLower.match(regex);
      score += matches ? matches.length : 0;
    }
    return { chunk, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
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

async function pickStrategy(
  prompt: (q: string) => Promise<string>
): Promise<ChunkingStrategy> {
  const strategies: ChunkingStrategy[] = [
    "fixed-size",
    "sentence",
    "paragraph",
    "recursive",
    "semantic",
  ];
  console.log("\nChunking strategies:");
  strategies.forEach((s, i) => console.log(`  ${i + 1}. ${s}`));
  console.log();

  const choice = await prompt("Pick a strategy (1-5, default 4=recursive): ");
  const num = parseInt(choice);
  if (!isNaN(num) && num >= 1 && num <= strategies.length) {
    return strategies[num - 1]!;
  }
  return "recursive";
}

// ============================================
// MODE 1: Explore Embeddings
// ============================================
// No documents needed — just embed sample sentences
// and see how similar/different they are.
// ============================================

async function modeExploreEmbeddings() {
  console.log("\n--- Explore Embeddings ---");
  console.log("Embedding sample sentences to see how vectors work...\n");

  const sentences = [
    "The cat sat on the mat",
    "A kitten rested on the rug",
    "The stock market crashed today",
    "Can I work from home?",
    "What is the remote work policy?",
    "The weather is sunny and warm",
  ];

  // Embed all in one batch call
  console.log("Embedding 6 sentences...");
  const embeddings = await embedTexts(sentences, EMBED_MODEL);

  // Show dimensions and sample values
  console.log(`\nVector dimensions: ${embeddings[0]!.length}`);
  console.log("(Each sentence is now a point in 768-dimensional space)\n");

  for (let i = 0; i < sentences.length; i++) {
    const first5 = embeddings[i]!.slice(0, 5).map((v) => v.toFixed(4));
    console.log(`  "${sentences[i]}"`);
    console.log(`  → [${first5.join(", ")}, ...] (${embeddings[i]!.length} values)\n`);
  }

  // Similarity matrix
  console.log("--- Similarity Matrix ---");
  console.log("(1.0 = identical meaning, 0.0 = unrelated)\n");

  // Header
  const labels = sentences.map((_, i) => `S${i + 1}`);
  console.log("      " + labels.map((l) => l.padStart(6)).join(""));
  console.log("      " + labels.map(() => "------").join(""));

  for (let i = 0; i < sentences.length; i++) {
    const row = labels.map((_, j) => {
      const sim = cosineSimilarity(embeddings[i]!, embeddings[j]!);
      return sim.toFixed(3).padStart(6);
    });
    console.log(`  ${labels[i]}  |${row.join("")}`);
  }

  console.log("\nLegend:");
  sentences.forEach((s, i) => console.log(`  S${i + 1} = "${s}"`));

  console.log("\nNotice:");
  const catKitten = cosineSimilarity(embeddings[0]!, embeddings[1]!);
  const catStock = cosineSimilarity(embeddings[0]!, embeddings[2]!);
  const homeRemote = cosineSimilarity(embeddings[3]!, embeddings[4]!);
  console.log(`  "cat/mat" vs "kitten/rug":  ${catKitten.toFixed(4)} (similar meaning, different words!)`);
  console.log(`  "cat/mat" vs "stock crash": ${catStock.toFixed(4)} (unrelated)`);
  console.log(`  "work from home" vs "remote work policy": ${homeRemote.toFixed(4)} (semantically close!)`);
  console.log("\nThis is why embedding search beats keyword search.\n");
}

// ============================================
// MODE 2: Build Store
// ============================================

async function modeBuildStore(prompt: (q: string) => Promise<string>) {
  const filePath = await pickFile(prompt);
  if (!filePath) return;

  console.log(`\nReading: ${path.basename(filePath)}...`);
  const text = await readDocument(filePath);
  console.log(`Document: ${text.length} chars\n`);

  const strategy = await pickStrategy(prompt);

  const sizeInput = await prompt("Chunk size (default 500): ");
  const chunkSize = parseInt(sizeInput) || 500;

  // Chunk the document
  const chunks = chunkText(text, strategy, {
    chunkSize,
    overlap: 50,
    source: path.basename(filePath),
  });
  console.log(`\nCreated ${chunks.length} chunks with "${strategy}" strategy\n`);

  // Embed all chunks with progress
  const store = new VectorStore(EMBED_MODEL);
  const batchSize = 5;

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const texts = batch.map((c) => c.content);

    process.stdout.write(
      `  Embedding chunks ${i + 1}-${Math.min(i + batchSize, chunks.length)} of ${chunks.length}...`
    );

    const embeddings = await embedTexts(texts, EMBED_MODEL);
    store.addBatch(batch, embeddings);

    console.log(" done");
  }

  // Show metadata
  const meta = store.getMetadata();
  console.log(`\nVector Store Ready:`);
  console.log(`  Chunks: ${meta.count}`);
  console.log(`  Dimensions: ${meta.dimensions}`);
  console.log(`  Sources: ${meta.sources.join(", ")}`);

  // Save
  const savePath = await prompt(`\nSave to (default: ${DEFAULT_STORE_PATH}): `);
  const finalPath = savePath.trim() || DEFAULT_STORE_PATH;
  store.save(finalPath);

  console.log("\nStore built and saved! Use Mode 3 to search it.\n");
}

// ============================================
// MODE 3: Search
// ============================================

async function modeSearch(prompt: (q: string) => Promise<string>) {
  // Load store
  const loadPath = await prompt(
    `Load store from (default: ${DEFAULT_STORE_PATH}): `
  );
  const finalPath = loadPath.trim() || DEFAULT_STORE_PATH;

  if (!fs.existsSync(finalPath)) {
    console.log(`\nNo store found at ${finalPath}. Run Mode 2 first to build one.`);
    return;
  }

  const store = VectorStore.load(finalPath);
  const meta = store.getMetadata();
  console.log(`  Model: ${meta.model} | Chunks: ${meta.count} | Dims: ${meta.dimensions}\n`);

  const chatChoice = await prompt("Send results to LLM for answers? (y/n, default n): ");
  const useChat = chatChoice.trim().toLowerCase() === "y";

  console.log('Enter queries. Type "quit" to exit.\n');

  while (true) {
    const query = await prompt("Search: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    // Embed the query
    const queryEmbedding = await embedText(query, EMBED_MODEL);

    // Search
    const results = store.search(queryEmbedding, 5);

    console.log(`\n  Top ${results.length} results:\n`);
    for (const r of results) {
      const preview = r.chunk.content.slice(0, 120).replace(/\n/g, " ");
      console.log(
        `  #${r.rank} (sim: ${r.score.toFixed(4)}) [Chunk ${r.chunk.metadata.chunkIndex + 1} | ${r.chunk.metadata.source}]`
      );
      console.log(`     "${preview}..."\n`);
    }

    // Optionally get LLM answer
    if (useChat && results.length > 0) {
      const topChunks = results.slice(0, 3);
      const contextParts = topChunks.map((r) => {
        return `[Chunk ${r.chunk.metadata.chunkIndex + 1} | sim: ${r.score.toFixed(3)} | ${r.chunk.metadata.source}]\n${r.chunk.content}`;
      });

      const messages = [
        {
          role: "system" as const,
          content: `Answer based ONLY on these context chunks. Cite chunk numbers.\n\n--- CONTEXT ---\n${contextParts.join("\n\n")}\n--- END ---`,
        },
        { role: "user" as const, content: query },
      ];

      console.log("  [Asking LLM...]");
      const startTime = Date.now();

      const spinnerFrames = ["|", "/", "-", "\\"];
      let spinnerIndex = 0;
      const spinner = setInterval(() => {
        process.stdout.write(
          `\r  Thinking... ${spinnerFrames[spinnerIndex++ % spinnerFrames.length]}`
        );
      }, 100);

      const stream = await ollama.chat({
        model: CHAT_MODEL,
        messages,
        stream: true,
      });

      clearInterval(spinner);
      process.stdout.write("\r                    \r");

      process.stdout.write("  Assistant: ");
      for await (const chunk of stream) {
        process.stdout.write(chunk.message.content);
      }

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`\n  [Done in ${elapsed}s]\n`);
    }
  }
}

// ============================================
// MODE 4: Keyword vs Embedding Showdown
// ============================================

async function modeShowdown(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Keyword vs Embedding Search ---\n");

  // Load store
  const loadPath = await prompt(
    `Load store from (default: ${DEFAULT_STORE_PATH}): `
  );
  const finalPath = loadPath.trim() || DEFAULT_STORE_PATH;

  if (!fs.existsSync(finalPath)) {
    console.log(`\nNo store found at ${finalPath}. Run Mode 2 first.`);
    return;
  }

  const store = VectorStore.load(finalPath);

  // Get all chunks for keyword search
  const allChunks = store.getEntries().map((e) => e.chunk);
  console.log(`  ${allChunks.length} chunks loaded\n`);

  console.log('Enter queries to compare. Type "quit" to exit.\n');
  console.log('Try: "Can I work from home?" or "What equipment do I get?"\n');

  while (true) {
    const query = await prompt("Query: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    // --- Keyword search ---
    const keywordResults = findRelevantChunksKeyword(query, allChunks, 3);

    console.log("\n=== KEYWORD SEARCH ===");
    for (let i = 0; i < keywordResults.length; i++) {
      const r = keywordResults[i]!;
      const preview = r.chunk.content.slice(0, 100).replace(/\n/g, " ");
      console.log(
        `  #${i + 1} (score: ${r.score}) [Chunk ${r.chunk.metadata.chunkIndex + 1}] "${preview}..."`
      );
    }

    // --- Embedding search ---
    const queryEmbedding = await embedText(query, EMBED_MODEL);
    const embeddingResults = store.search(queryEmbedding, 3);

    console.log("\n=== EMBEDDING SEARCH ===");
    for (const r of embeddingResults) {
      const preview = r.chunk.content.slice(0, 100).replace(/\n/g, " ");
      console.log(
        `  #${r.rank} (sim: ${r.score.toFixed(4)}) [Chunk ${r.chunk.metadata.chunkIndex + 1}] "${preview}..."`
      );
    }

    // Compare
    const keyChunks = new Set(keywordResults.map((r) => r.chunk.metadata.chunkIndex));
    const embChunks = new Set(embeddingResults.map((r) => r.chunk.metadata.chunkIndex));
    const overlap = [...keyChunks].filter((c) => embChunks.has(c));

    console.log(`\n  Overlap: ${overlap.length}/3 chunks in common`);
    if (overlap.length < 3) {
      console.log(
        "  → Embedding search found different (often better) chunks!"
      );
      console.log(
        "  → Keyword search only matches exact words. Embedding search understands meaning."
      );
    }
    console.log();
  }
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log(`\n--- Phase 4: Embeddings & Vector Store ---`);
  console.log(`    Chat: ${CHAT_MODEL} | Embed: ${EMBED_MODEL}\n`);
  console.log("Modes:");
  console.log(
    "  1. Explore Embeddings  - see vectors & similarity matrix (no docs needed)"
  );
  console.log("  2. Build Store         - chunk + embed a document, save to disk");
  console.log("  3. Search              - query a saved store, ranked results");
  console.log(
    "  4. Keyword vs Embedding - side-by-side comparison"
  );
  console.log();

  const { prompt, close } = createPrompt();
  const modeChoice = await prompt("Pick a mode (1-4): ");
  const mode = parseInt(modeChoice) || 1;

  switch (mode) {
    case 1:
      await modeExploreEmbeddings();
      break;
    case 2:
      await modeBuildStore(prompt);
      break;
    case 3:
      await modeSearch(prompt);
      break;
    case 4:
      await modeShowdown(prompt);
      break;
    default:
      console.log("Invalid mode.");
  }

  close();
  console.log("Goodbye!");
}

main();
