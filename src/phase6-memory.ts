import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import {
  ragQuery,
  formatSources,
  DEFAULT_CONFIG,
  calculateContextBudget,
  type RAGConfig,
} from "./rag-pipeline.js";
import { VectorStore } from "./vector-store.js";
import {
  ConversationMemory,
  refineQueryWithHistory,
} from "./conversation.js";

const DEFAULT_STORE_PATH = "data/rag-store.json";
const CONVERSATIONS_DIR = "data/conversations";

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

function listStores(): { name: string; path: string }[] {
  const dataDir = "data";
  if (!fs.existsSync(dataDir)) return [];
  return fs.readdirSync(dataDir)
    .filter((f) => f.endsWith(".json") && f !== "conversations")
    .map((f) => ({
      name: f.replace(".json", ""),
      path: path.join(dataDir, f),
    }));
}

async function pickStore(prompt: (q: string) => Promise<string>): Promise<VectorStore | null> {
  const stores = listStores();

  if (stores.length === 0) {
    console.log("\nNo stores found in data/. Run Phase 5 Mode 1 first to ingest documents.");
    return null;
  }

  if (stores.length === 1) {
    console.log(`  Loading: ${stores[0]!.name}`);
    return VectorStore.load(stores[0]!.path);
  }

  console.log("\nAvailable document stores:");
  stores.forEach((s, i) => console.log(`  ${i + 1}. ${s.name}`));
  console.log();

  const choice = await prompt("Pick a store (number): ");
  const num = parseInt(choice);
  if (isNaN(num) || num < 1 || num > stores.length) {
    console.log("Invalid choice.");
    return null;
  }

  return VectorStore.load(stores[num - 1]!.path);
}

// ============================================
// MODE 1: RAG Chat with Memory
// ============================================

async function modeRAGChatWithMemory(prompt: (q: string) => Promise<string>) {
  console.log("\n--- RAG Chat with Conversation Memory ---\n");

  const store = await pickStore(prompt);
  if (!store) return;

  const meta = store.getMetadata();
  const budget = calculateContextBudget(store, DEFAULT_CONFIG.chatModel);
  console.log(`  ${meta.count} chunks from: ${meta.sources.join(", ")}`);
  console.log(`  Context budget: ${budget.availableForChunks.toLocaleString()} tokens → topK auto-set to ${budget.recommendedTopK}\n`);

  // Create conversation memory
  const memory = new ConversationMemory();

  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    topK: budget.recommendedTopK, // auto-calculated from context window
    enableQueryExpansion: false,
    enableReranking: false,
  };

  console.log(`  Memory: sliding window of 10 turns, auto-summarization`);
  console.log('  Ask questions. Type "quit" to exit, "save" to save conversation.\n');

  while (true) {
    const query = await prompt("You: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    if (query.trim().toLowerCase() === "save") {
      const filePath = memory.save(CONVERSATIONS_DIR);
      console.log(`  Conversation saved to ${filePath}\n`);
      continue;
    }

    // Refine query using conversation context
    // e.g., "How do I qualify for it?" → "Regarding 'What is MGRS?': How do I qualify for it?"
    const refinedQuery = await refineQueryWithHistory(query, memory.getHistory(), config.chatModel);
    if (refinedQuery !== query) {
      console.log(`  [Refined query: "${refinedQuery.slice(0, 100)}"]`);
    }

    // Get conversation context messages for the LLM
    const contextMessages = memory.getContextMessages();

    // Show spinner
    const spinnerFrames = ["|", "/", "-", "\\"];
    let spinnerIndex = 0;
    const spinner = setInterval(() => {
      process.stdout.write(
        `\r  Processing... ${spinnerFrames[spinnerIndex++ % spinnerFrames.length]}`
      );
    }, 100);

    // Run RAG query with conversation history
    let result;
    try {
      result = await ragQuery(
        refinedQuery,
        store,
        config,
        contextMessages
      );
    } catch (err: any) {
      clearInterval(spinner);
      process.stdout.write("\r                    \r");
      console.log(`  [Error: ${err.message?.slice(0, 100) ?? "Unknown error"}]`);
      console.log(`  Try again or type "quit" to exit.\n`);
      continue;
    }

    clearInterval(spinner);
    process.stdout.write("\r                    \r");

    console.log(`Assistant: ${result.answer}`);

    // Add messages to memory
    memory.addMessage("user", query); // store original query, not refined
    memory.addMessage("assistant", result.answer);

    // Check if sliding window needs maintenance
    const summarized = await memory.maintainWindow();
    if (summarized) {
      console.log(`  [Memory: summarized older messages to save context space]`);
    }

    // Show stats
    const stats = memory.getStats();
    console.log(
      `\n  [Retrieval: ${result.retrievalTimeMs}ms | Generation: ${(result.generationTimeMs / 1000).toFixed(1)}s]`
    );
    console.log(
      `  [Memory: ${stats.totalMessages} messages, ~${stats.tokenEstimate} tokens${stats.hasSummary ? ", has summary" : ""}]`
    );
    console.log();
    console.log(formatSources(result.sources));
    console.log();
  }

  // Offer to save on exit
  if (memory.getMessageCount() > 0) {
    const saveChoice = await prompt("Save this conversation? (y/n): ");
    if (saveChoice.trim().toLowerCase() === "y") {
      const filePath = memory.save(CONVERSATIONS_DIR);
      console.log(`  Saved to ${filePath}`);
    }
  }
}

// ============================================
// MODE 2: Explore Memory
// ============================================

async function modeExploreMemory(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Explore Conversation Memory ---\n");
  console.log("This mode shows how the sliding window and summarization work.");
  console.log("Chat with the bot and watch the memory state change.\n");

  const store = await pickStore(prompt);
  if (!store) return;

  const memory = new ConversationMemory({ maxTurns: 3 }); // small window for demo

  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    enableQueryExpansion: false,
  };

  console.log("  Using a small window (3 turns) to demonstrate summarization faster.");
  console.log('  Type "quit" to exit, "memory" to see full memory state.\n');

  while (true) {
    const query = await prompt("You: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    if (query.trim().toLowerCase() === "memory") {
      // Show full memory state
      console.log("\n" + "=".repeat(50));
      console.log("MEMORY STATE");
      console.log("=".repeat(50));

      const stats = memory.getStats();
      console.log(`  Total messages: ${stats.totalMessages}`);
      console.log(`  Window size: ${stats.windowSize}`);
      console.log(`  Token estimate: ~${stats.tokenEstimate}`);
      console.log(`  Has summary: ${stats.hasSummary}`);

      if (memory.getSummary()) {
        console.log(`\n  Summary of older messages:`);
        console.log(`  "${memory.getSummary()}"`);
      }

      console.log(`\n  Active messages in window:`);
      const contextMsgs = memory.getContextMessages();
      for (const msg of contextMsgs) {
        const preview = msg.content.slice(0, 80).replace(/\n/g, " ");
        console.log(`    [${msg.role}] "${preview}..."`);
      }

      console.log("=".repeat(50) + "\n");
      continue;
    }

    const refinedQuery = await refineQueryWithHistory(query, memory.getHistory(), config.chatModel);
    const contextMessages = memory.getContextMessages();

    const spinner = setInterval(() => {
      process.stdout.write("\r  Processing... ");
    }, 500);

    const result = await ragQuery(refinedQuery, store, config, contextMessages);

    clearInterval(spinner);
    process.stdout.write("\r                    \r");

    console.log(`Assistant: ${result.answer}`);

    memory.addMessage("user", query);
    memory.addMessage("assistant", result.answer);

    const summarized = await memory.maintainWindow();
    if (summarized) {
      console.log(`\n  >>> SLIDING WINDOW TRIGGERED <<<`);
      console.log(`  Older messages summarized. Type "memory" to see the summary.`);
    }

    const stats = memory.getStats();
    console.log(
      `  [Messages: ${stats.totalMessages} | Window: ${stats.windowSize} | Tokens: ~${stats.tokenEstimate}]\n`
    );
  }
}

// ============================================
// MODE 3: Load Saved Conversation
// ============================================

async function modeLoadConversation(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Load Saved Conversation ---\n");

  const saved = ConversationMemory.listSaved(CONVERSATIONS_DIR);

  if (saved.length === 0) {
    console.log("No saved conversations found. Chat in Mode 1 and save first.");
    return;
  }

  console.log("Saved conversations:");
  saved.forEach((s, i) => {
    console.log(
      `  ${i + 1}. [${s.messageCount} msgs] "${s.preview}..." (${s.updatedAt})`
    );
  });
  console.log();

  const choice = await prompt("Pick a conversation to resume (number): ");
  const num = parseInt(choice);
  if (isNaN(num) || num < 1 || num > saved.length) {
    console.log("Invalid choice.");
    return;
  }

  const selectedId = saved[num - 1]!.id;
  const filePath = path.join(CONVERSATIONS_DIR, `${selectedId}.json`);
  const memory = ConversationMemory.load(filePath);

  console.log(`\nLoaded conversation (${memory.getMessageCount()} messages)\n`);

  // Show previous messages
  console.log("--- Previous conversation ---");
  const history = memory.getHistory();
  for (const msg of history.slice(-10)) {
    // show last 10
    const prefix = msg.role === "user" ? "You" : "Assistant";
    console.log(`${prefix}: ${msg.content.slice(0, 150)}${msg.content.length > 150 ? "..." : ""}`);
  }
  console.log("--- Resuming ---\n");

  // Continue chatting
  const store = await pickStore(prompt);
  if (!store) return;

  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    enableQueryExpansion: false,
  };

  console.log('Continue the conversation. Type "quit" to exit, "save" to save.\n');

  while (true) {
    const query = await prompt("You: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    if (query.trim().toLowerCase() === "save") {
      memory.save(CONVERSATIONS_DIR);
      console.log("  Conversation saved.\n");
      continue;
    }

    const refinedQuery = await refineQueryWithHistory(query, memory.getHistory(), config.chatModel);
    const contextMessages = memory.getContextMessages();

    const spinner = setInterval(() => {
      process.stdout.write("\r  Processing... ");
    }, 500);

    const result = await ragQuery(refinedQuery, store, config, contextMessages);

    clearInterval(spinner);
    process.stdout.write("\r                    \r");

    console.log(`Assistant: ${result.answer}`);

    memory.addMessage("user", query);
    memory.addMessage("assistant", result.answer);
    await memory.maintainWindow();

    const stats = memory.getStats();
    console.log(
      `  [Messages: ${stats.totalMessages} | Tokens: ~${stats.tokenEstimate}]\n`
    );
  }

  // Save on exit
  if (memory.getMessageCount() > 0) {
    const saveChoice = await prompt("Save updated conversation? (y/n): ");
    if (saveChoice.trim().toLowerCase() === "y") {
      memory.save(CONVERSATIONS_DIR);
      console.log("  Saved.");
    }
  }
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log(`\n--- Phase 6: Conversation Memory ---`);
  console.log(`    Chat: ${DEFAULT_CONFIG.chatModel} | Embed: ${DEFAULT_CONFIG.embedModel}\n`);
  console.log("Modes:");
  console.log(
    "  1. RAG Chat with Memory   - multi-turn conversation with context"
  );
  console.log(
    "  2. Explore Memory         - see sliding window & summarization in action"
  );
  console.log(
    "  3. Load Saved Conversation - resume a previous chat session"
  );
  console.log();

  const { prompt, close } = createPrompt();
  const modeChoice = await prompt("Pick a mode (1-3): ");
  const mode = parseInt(modeChoice) || 1;

  switch (mode) {
    case 1:
      await modeRAGChatWithMemory(prompt);
      break;
    case 2:
      await modeExploreMemory(prompt);
      break;
    case 3:
      await modeLoadConversation(prompt);
      break;
    default:
      console.log("Invalid mode.");
  }

  close();
  console.log("Goodbye!");
}

main();
