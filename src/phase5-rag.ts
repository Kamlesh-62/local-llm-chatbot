import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import {
  ragQuery,
  ingestDocuments,
  hybridSearch,
  keywordSearch,
  expandQuery,
  formatSources,
  buildRAGPrompt,
  DEFAULT_CONFIG,
  calculateContextBudget,
  type RAGConfig,
  type HybridSearchResult,
} from "./rag-pipeline.js";
import { VectorStore, embedText } from "./vector-store.js";
import type { Chunk, ChunkingStrategy } from "./chunker.js";

const DEFAULT_STORE_PATH = "data/rag-store.json";

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

// Recursively find all files in a directory
function getAllFiles(dirPath: string, prefix: string = ""): { display: string; fullPath: string }[] {
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  const results: { display: string; fullPath: string }[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    const displayName = prefix ? `${prefix}/${entry.name}` : entry.name;

    if (entry.isDirectory()) {
      results.push(...getAllFiles(fullPath, displayName));
    } else {
      results.push({ display: displayName, fullPath });
    }
  }

  return results;
}

async function pickFiles(
  prompt: (q: string) => Promise<string>
): Promise<string[]> {
  const docsDir = path.join(process.cwd(), "sample-docs");
  const allFiles = getAllFiles(docsDir);

  console.log("\nAvailable documents:");
  allFiles.forEach((f, i) => console.log(`  ${i + 1}. ${f.display}`));
  console.log();
  console.log('  Tip: "all" = all files, "PDF" = all files in PDF/ folder');
  console.log('        "1-10" = range, "1,3,5" = specific, "1-5,14,15" = mix');

  const choice = await prompt(
    '\nEnter selection: '
  );

  const trimmed = choice.trim().toLowerCase();

  if (trimmed === "all") {
    return allFiles.map((f) => f.fullPath);
  }

  // Check if user typed a folder name (e.g., "PDF", "TEXT", "sections")
  const folderMatch = allFiles.filter((f) =>
    f.display.toLowerCase().startsWith(trimmed + "/")
  );
  if (folderMatch.length > 0) {
    console.log(`  Found ${folderMatch.length} files in ${choice.trim()}/`);
    return folderMatch.map((f) => f.fullPath);
  }

  // Parse numbers — supports ranges (1-10) and individual (1,3,5) and mix (1-5,14,15)
  const nums = new Set<number>();
  const parts = choice.split(",");

  for (const part of parts) {
    const trimmedPart = part.trim();
    if (trimmedPart.includes("-")) {
      // Range: "1-10"
      const [startStr, endStr] = trimmedPart.split("-");
      const start = parseInt(startStr!);
      const end = parseInt(endStr!);
      if (!isNaN(start) && !isNaN(end)) {
        for (let i = Math.min(start, end); i <= Math.max(start, end); i++) {
          if (i >= 1 && i <= allFiles.length) nums.add(i);
        }
      }
    } else {
      // Single number
      const n = parseInt(trimmedPart);
      if (!isNaN(n) && n >= 1 && n <= allFiles.length) nums.add(n);
    }
  }

  if (nums.size === 0) {
    console.log("No valid files selected.");
    return [];
  }

  const selected = [...nums].sort((a, b) => a - b).map((n) => allFiles[n - 1]!.fullPath);
  console.log(`  Selected ${selected.length} file(s)`);
  return selected;
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

  const choice = await prompt("\nPick strategy (default 4=recursive): ");
  const num = parseInt(choice);
  if (!isNaN(num) && num >= 1 && num <= strategies.length) {
    return strategies[num - 1]!;
  }
  return "recursive";
}

function loadStore(): VectorStore | null {
  if (!fs.existsSync(DEFAULT_STORE_PATH)) {
    console.log(
      `\nNo store found at ${DEFAULT_STORE_PATH}. Run Mode 1 first to ingest documents.`
    );
    return null;
  }
  return VectorStore.load(DEFAULT_STORE_PATH);
}

// ============================================
// MODE 1: Ingest Documents
// ============================================

async function modeIngest(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Ingest Documents ---\n");

  const filePaths = await pickFiles(prompt);
  if (filePaths.length === 0) return;

  // Check if all files are PDFs — if so, skip strategy/size questions
  const allPdfs = filePaths.every((f) => f.toLowerCase().endsWith(".pdf"));

  let strategy: ChunkingStrategy = "recursive";
  let chunkSize = 500;

  if (allPdfs) {
    console.log("\n  PDF detected → using page-based chunking (one chunk per page)");
  } else {
    strategy = await pickStrategy(prompt);
    const sizeInput = await prompt("Chunk size (default 2000): ");
    chunkSize = parseInt(sizeInput) || 2000;
  }

  console.log(
    `\nIngesting ${filePaths.length} file(s)...\n`
  );

  const { store } = await ingestDocuments(
    filePaths,
    strategy,
    chunkSize,
    50, // overlap
    DEFAULT_CONFIG.embedModel
  );

  store.save(DEFAULT_STORE_PATH);

  const meta = store.getMetadata();
  const budget = calculateContextBudget(store, DEFAULT_CONFIG.chatModel);

  console.log(`\nStore ready:`);
  console.log(`  Chunks: ${meta.count}`);
  console.log(`  Dimensions: ${meta.dimensions}`);
  console.log(`  Sources: ${meta.sources.join(", ")}`);
  console.log(`  Saved to: ${DEFAULT_STORE_PATH}`);

  console.log(`\nContext window budget (${DEFAULT_CONFIG.chatModel}):`);
  console.log(`  Context window:    ${budget.contextWindow.toLocaleString()} tokens`);
  console.log(`  System prompt:     -${budget.systemPromptReserved} tokens`);
  console.log(`  Conversation:      -${budget.historyReserved.toLocaleString()} tokens`);
  console.log(`  Answer reserved:   -${budget.answerReserved.toLocaleString()} tokens`);
  console.log(`  Available:         ${budget.availableForChunks.toLocaleString()} tokens`);
  console.log(`  Avg chunk size:    ~${budget.avgChunkTokens} tokens`);
  console.log(`  Max topK:          ${budget.maxTopK} chunks`);
  console.log(`  Recommended topK:  ${budget.recommendedTopK} chunks`);
  if (budget.recommendedTopK >= budget.totalChunks) {
    console.log(`  --> All ${budget.totalChunks} chunks fit! Every query sees the entire document.`);
  }
  console.log();
}

// ============================================
// MODE 2: RAG Chat
// ============================================

async function modeRAGChat(prompt: (q: string) => Promise<string>) {
  console.log("\n--- RAG Chat ---\n");

  const store = loadStore();
  if (!store) return;

  const meta = store.getMetadata();
  console.log(`  ${meta.count} chunks from: ${meta.sources.join(", ")}\n`);

  // Configure
  const expandChoice = await prompt("Enable query expansion? (y/n, default y): ");
  const rerankChoice = await prompt("Enable LLM re-ranking? (y/n, default n): ");

  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    enableQueryExpansion: expandChoice.trim().toLowerCase() !== "n",
    enableReranking: rerankChoice.trim().toLowerCase() === "y",
  };

  console.log(
    `\nConfig: expansion=${config.enableQueryExpansion}, reranking=${config.enableReranking}, topK=${config.topK}`
  );
  console.log('Ask questions. Type "quit" to exit.\n');

  while (true) {
    const query = await prompt("You: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    // Show spinner while processing (retrieval + generation)
    const spinnerFrames = ["|", "/", "-", "\\"];
    let spinnerIndex = 0;
    const spinner = setInterval(() => {
      process.stdout.write(
        `\r  Processing... ${spinnerFrames[spinnerIndex++ % spinnerFrames.length]}`
      );
    }, 100);

    const result = await ragQuery(query, store, config, undefined);

    clearInterval(spinner);
    process.stdout.write("\r                    \r");

    // Display the answer (already cleaned of <think> blocks)
    console.log(`Assistant: ${result.answer}`);

    // Show metadata
    console.log(`\n`);
    if (result.expandedQuery) {
      console.log(`  [Expanded query: "${result.expandedQuery.slice(0, 100)}"]`);
    }
    console.log(
      `  [Retrieval: ${result.retrievalTimeMs}ms | Generation: ${(result.generationTimeMs / 1000).toFixed(1)}s]`
    );
    console.log();
    console.log(formatSources(result.sources));
    console.log();
  }
}

// ============================================
// MODE 3: Search Comparison
// ============================================

async function modeSearchComparison(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Search Comparison: Keyword vs Embedding vs Hybrid ---\n");

  const store = loadStore();
  if (!store) return;

  const chunks = store.getEntries().map((e) => e.chunk);
  console.log(`  ${chunks.length} chunks loaded\n`);
  console.log('Enter queries to compare. Type "quit" to exit.\n');

  while (true) {
    const query = await prompt("Query: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    const queryEmbedding = await embedText(query, DEFAULT_CONFIG.embedModel);

    // 1. Keyword only
    const keyResults = keywordSearch(query, chunks, 5);
    console.log("\n=== KEYWORD SEARCH ===");
    for (let i = 0; i < keyResults.length; i++) {
      const r = keyResults[i]!;
      const preview = r.chunk.content.slice(0, 80).replace(/\n/g, " ");
      console.log(
        `  #${i + 1} (score: ${r.score.toFixed(3)}) [Chunk ${r.chunk.metadata.chunkIndex + 1} | ${r.chunk.metadata.source}]`
      );
      console.log(`     "${preview}..."`);
    }

    // 2. Embedding only
    const embResults = store.search(queryEmbedding, 5);
    console.log("\n=== EMBEDDING SEARCH ===");
    for (const r of embResults) {
      const preview = r.chunk.content.slice(0, 80).replace(/\n/g, " ");
      console.log(
        `  #${r.rank} (sim: ${r.score.toFixed(3)}) [Chunk ${r.chunk.metadata.chunkIndex + 1} | ${r.chunk.metadata.source}]`
      );
      console.log(`     "${preview}..."`);
    }

    // 3. Hybrid
    const hybridResults = hybridSearch(
      query,
      queryEmbedding,
      store,
      chunks,
      DEFAULT_CONFIG
    );
    console.log("\n=== HYBRID SEARCH (0.7 embedding + 0.3 keyword) ===");
    for (const r of hybridResults) {
      const preview = r.chunk.content.slice(0, 80).replace(/\n/g, " ");
      console.log(
        `  #${r.rank} (hybrid: ${r.hybridScore.toFixed(3)} = emb:${r.embeddingScore.toFixed(2)} + key:${r.keywordScore.toFixed(2)}) [Chunk ${r.chunk.metadata.chunkIndex + 1}]`
      );
      console.log(`     "${preview}..."`);
    }

    // Analysis
    const keyChunks = new Set(keyResults.map((r) => r.chunk.metadata.chunkIndex));
    const embChunks = new Set(embResults.map((r) => r.chunk.metadata.chunkIndex));
    const hybChunks = new Set(hybridResults.map((r) => r.chunk.metadata.chunkIndex));

    const onlyInHybrid = [...hybChunks].filter(
      (c) => !keyChunks.has(c) && !embChunks.has(c)
    );
    const keyInHybridNotEmb = [...hybChunks].filter(
      (c) => keyChunks.has(c) && !embChunks.has(c)
    );

    console.log("\n  Analysis:");
    if (keyInHybridNotEmb.length > 0) {
      console.log(
        `  Hybrid promoted ${keyInHybridNotEmb.length} chunk(s) that keyword found but embedding missed`
      );
    }
    if (onlyInHybrid.length > 0) {
      console.log(
        `  Hybrid found ${onlyInHybrid.length} chunk(s) neither method alone had in top 5`
      );
    }
    console.log();
  }
}

// ============================================
// MODE 4: Pipeline Breakdown
// ============================================

async function modePipelineBreakdown(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Pipeline Breakdown (see every step) ---\n");

  const store = loadStore();
  if (!store) return;

  const chunks = store.getEntries().map((e) => e.chunk);
  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    enableQueryExpansion: true,
    enableReranking: false,
  };

  console.log(`  ${chunks.length} chunks loaded`);
  console.log('Enter a query. Type "quit" to exit.\n');

  while (true) {
    const query = await prompt("Query: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    console.log("\n" + "=".repeat(50));

    // Step 1: Query Expansion
    console.log("\nSTEP 1: Query Expansion");
    console.log(`  Original: "${query}"`);
    const sources = store.getMetadata().sources;
    const expanded = await expandQuery(query, config.chatModel, sources);
    console.log(`  Expanded: "${expanded}"`);

    // Step 2: Embed
    console.log("\nSTEP 2: Embed Query");
    const queryEmbedding = await embedText(expanded, config.embedModel);
    const first5 = queryEmbedding.slice(0, 5).map((v) => v.toFixed(4));
    console.log(`  Vector: [${first5.join(", ")}, ...] (${queryEmbedding.length}d)`);

    // Step 3: Keyword search
    console.log("\nSTEP 3: Keyword Search (on expanded query)");
    const keyResults = keywordSearch(expanded, chunks, 5);
    for (let i = 0; i < Math.min(3, keyResults.length); i++) {
      const r = keyResults[i]!;
      console.log(
        `  #${i + 1} (score: ${r.score.toFixed(3)}) Chunk ${r.chunk.metadata.chunkIndex + 1}`
      );
    }

    // Step 4: Embedding search
    console.log("\nSTEP 4: Embedding Search");
    const embResults = store.search(queryEmbedding, 5);
    for (let i = 0; i < Math.min(3, embResults.length); i++) {
      const r = embResults[i]!;
      console.log(
        `  #${r.rank} (sim: ${r.score.toFixed(3)}) Chunk ${r.chunk.metadata.chunkIndex + 1}`
      );
    }

    // Step 5: Hybrid merge
    console.log("\nSTEP 5: Hybrid Merge (0.7 emb + 0.3 key)");
    const hybridResults = hybridSearch(
      expanded,
      queryEmbedding,
      store,
      chunks,
      config
    );
    for (const r of hybridResults) {
      console.log(
        `  #${r.rank} (hybrid: ${r.hybridScore.toFixed(3)}) Chunk ${r.chunk.metadata.chunkIndex + 1} [emb:${r.embeddingScore.toFixed(2)} key:${r.keywordScore.toFixed(2)}]`
      );
    }

    // Step 6: Build prompt (preview)
    console.log("\nSTEP 6: RAG Prompt (preview)");
    const { system } = buildRAGPrompt(query, hybridResults);
    console.log(`  System prompt: ${system.length} chars`);
    console.log(`  Context chunks: ${hybridResults.length}`);

    // Step 7: Generate
    console.log("\nSTEP 7: LLM Generation");
    process.stdout.write("  Answer: ");

    const result = await ragQuery(query, store, config, undefined, (token) => {
      process.stdout.write(token);
    });

    console.log(`\n`);
    console.log(
      `  [Retrieval: ${result.retrievalTimeMs}ms | Generation: ${(result.generationTimeMs / 1000).toFixed(1)}s]`
    );
    console.log();
    console.log(formatSources(result.sources));
    console.log("\n" + "=".repeat(50) + "\n");
  }
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log(`\n--- Phase 5: RAG Pipeline ---`);
  console.log(
    `    Chat: ${DEFAULT_CONFIG.chatModel} | Embed: ${DEFAULT_CONFIG.embedModel}\n`
  );
  console.log("Modes:");
  console.log("  1. Ingest Documents    - chunk + embed files, save store");
  console.log("  2. RAG Chat            - ask questions, get grounded answers");
  console.log(
    "  3. Search Comparison   - keyword vs embedding vs hybrid"
  );
  console.log(
    "  4. Pipeline Breakdown  - see every step of the RAG pipeline"
  );
  console.log();

  const { prompt, close } = createPrompt();
  const modeChoice = await prompt("Pick a mode (1-4): ");
  const mode = parseInt(modeChoice) || 2;

  switch (mode) {
    case 1:
      await modeIngest(prompt);
      break;
    case 2:
      await modeRAGChat(prompt);
      break;
    case 3:
      await modeSearchComparison(prompt);
      break;
    case 4:
      await modePipelineBreakdown(prompt);
      break;
    default:
      console.log("Invalid mode.");
  }

  close();
  console.log("Goodbye!");
}

main();
