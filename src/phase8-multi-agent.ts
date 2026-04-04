// ============================================
// PHASE 8: MULTI-AGENT SYSTEM
//
// Multiple specialist agents collaborate:
//   - Router Agent dispatches to specialists
//   - Research, Summarizer, QA, General agents
//
// Three orchestration patterns:
//   1. Supervisor - router picks agents
//   2. Pipeline   - research → summarize → QA
//   3. Parallel   - fan-out, then merge
// ============================================

import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import {
  DEFAULT_CONFIG,
  calculateContextBudget,
  type RAGConfig,
} from "./rag-pipeline.js";
import { VectorStore } from "./vector-store.js";
import {
  createSearchDocumentsTool,
  createReadFileTool,
  createListFilesTool,
  createCalculatorTool,
  createFetchWebPageTool,
  createWebSearchTool,
  type Tool,
} from "./agent.js";
import {
  runMultiAgent,
  runSpecialist,
  ALL_SPECIALISTS,
  summarizeBlackboard,
  type MultiAgentConfig,
  type OrchestrationMode,
  type SpecialistResult,
} from "./multi-agent.js";

const SAMPLE_DOCS_DIR = "sample-docs";

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
  return fs
    .readdirSync(dataDir)
    .filter((f) => f.endsWith(".json") && f !== "conversations")
    .map((f) => ({
      name: f.replace(".json", ""),
      path: path.join(dataDir, f),
    }));
}

async function pickStore(
  prompt: (q: string) => Promise<string>
): Promise<VectorStore | null> {
  const stores = listStores();
  if (stores.length === 0) {
    console.log("\nNo stores found. Run Phase 5 first.");
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
// MODEL PICKER
// ============================================

const CHAT_MODELS = [
  { id: "nvidia/nemotron-3-super-120b-a12b:free", label: "nemotron-120b (free)" },
  { id: "openai/gpt-4o-mini", label: "gpt-4o-mini" },
];

async function pickModel(prompt: (q: string) => Promise<string>): Promise<string> {
  console.log("\nChat models:");
  CHAT_MODELS.forEach((m, i) =>
    console.log(`  ${i + 1}. ${m.label}${i === 0 ? " (default)" : ""}`)
  );
  console.log(`  ${CHAT_MODELS.length + 1}. Custom model ID`);
  const choice = await prompt(`Pick a model (1-${CHAT_MODELS.length + 1}, default 1): `);
  const num = parseInt(choice) || 1;
  if (num >= 1 && num <= CHAT_MODELS.length) return CHAT_MODELS[num - 1]!.id;
  if (num === CHAT_MODELS.length + 1) {
    return (await prompt("Enter model ID: ")).trim() || DEFAULT_CONFIG.chatModel;
  }
  return DEFAULT_CONFIG.chatModel;
}

// ============================================
// CREATE TOOLS
// ============================================

function createTools(store: VectorStore, config: RAGConfig): Tool[] {
  return [
    createSearchDocumentsTool(store, config),
    createReadFileTool(SAMPLE_DOCS_DIR),
    createListFilesTool(SAMPLE_DOCS_DIR),
    createCalculatorTool(),
    createWebSearchTool(),
    createFetchWebPageTool(),
  ];
}

// ============================================
// PRINTING HELPERS
// ============================================

function printAgentStart(name: string): void {
  console.log(`\n  >> Calling ${name} agent...`);
}

function printAgentComplete(sr: SpecialistResult): void {
  console.log(
    `  << ${sr.agentName} agent done (${sr.result.iterations} steps, ${(sr.result.totalTimeMs / 1000).toFixed(1)}s)`
  );
  const preview = sr.result.answer.slice(0, 200);
  console.log(`     ${preview}${sr.result.answer.length > 200 ? "..." : ""}`);
}

// ============================================
// MODE 1: Supervisor Chat
// ============================================

async function modeSupervisorChat(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Supervisor Chat ---");
  console.log("  The router agent decides which specialist(s) to call.\n");

  const store = await pickStore(prompt);
  if (!store) return;
  const chatModel = await pickModel(prompt);

  const meta = store.getMetadata();
  const budget = calculateContextBudget(store, chatModel);
  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    chatModel,
    topK: budget.recommendedTopK,
  };
  const allTools = createTools(store, config);

  console.log(`\n  ${meta.count} chunks | Model: ${chatModel}`);
  console.log(`  Specialists: ${ALL_SPECIALISTS.map((s) => s.name).join(", ")}`);
  console.log('  Type "quit" to exit, "board" to see the blackboard.\n');

  const maConfig: MultiAgentConfig = {
    chatModel,
    orchestrationMode: "supervisor",
    verbose: false,
    onAgentStart: printAgentStart,
    onAgentComplete: printAgentComplete,
  };

  while (true) {
    const query = await prompt("You: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    const { answer, blackboard } = await runMultiAgent(query, allTools, maConfig);

    if (query.trim().toLowerCase() === "board") {
      console.log(`\n${summarizeBlackboard(blackboard)}\n`);
      continue;
    }

    console.log(`\n  Answer: ${answer}`);
    console.log(
      `\n  [supervisor | agents called: ${blackboard.executionLog.map((e) => e.agentName).join(", ") || "none"}]\n`
    );
  }
}

// ============================================
// MODE 2: Pipeline Demo
// ============================================

async function modePipelineDemo(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Pipeline Demo ---");
  console.log("  Sequential: research → summarize → QA validate\n");

  const store = await pickStore(prompt);
  if (!store) return;
  const chatModel = await pickModel(prompt);

  const budget = calculateContextBudget(store, chatModel);
  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    chatModel,
    topK: budget.recommendedTopK,
  };
  const allTools = createTools(store, config);

  const query = await prompt("\nYour question: ");
  if (!query.trim()) return;

  console.log("\n  Running pipeline...\n");

  const maConfig: MultiAgentConfig = {
    chatModel,
    orchestrationMode: "pipeline",
    verbose: false,
    onAgentStart: printAgentStart,
    onAgentComplete: printAgentComplete,
  };

  const { answer, blackboard } = await runMultiAgent(query, allTools, maConfig);

  console.log("\n  ============================================");
  console.log("  PIPELINE RESULT");
  console.log("  ============================================");

  // Show each stage
  for (const [name, sr] of blackboard.results) {
    console.log(`\n  --- ${name} (${(sr.result.totalTimeMs / 1000).toFixed(1)}s) ---`);
    console.log(`  ${sr.result.answer}`);
  }

  console.log("\n  --- Final Combined Answer ---");
  console.log(`  ${answer}\n`);
}

// ============================================
// MODE 3: Parallel Demo
// ============================================

async function modeParallelDemo(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Parallel Demo ---");
  console.log("  Fan-out to research + summarizer, then merge.\n");

  const store = await pickStore(prompt);
  if (!store) return;
  const chatModel = await pickModel(prompt);

  const budget = calculateContextBudget(store, chatModel);
  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    chatModel,
    topK: budget.recommendedTopK,
  };
  const allTools = createTools(store, config);

  const query = await prompt("\nYour question: ");
  if (!query.trim()) return;

  console.log("\n  Running agents in parallel...\n");

  const maConfig: MultiAgentConfig = {
    chatModel,
    orchestrationMode: "parallel",
    verbose: false,
    onAgentStart: printAgentStart,
    onAgentComplete: printAgentComplete,
  };

  const { answer, blackboard } = await runMultiAgent(query, allTools, maConfig);

  console.log("\n  ============================================");
  console.log("  PARALLEL RESULT");
  console.log("  ============================================");

  for (const [name, sr] of blackboard.results) {
    console.log(`\n  --- ${name} agent (${(sr.result.totalTimeMs / 1000).toFixed(1)}s) ---`);
    console.log(`  ${sr.result.answer.slice(0, 300)}${sr.result.answer.length > 300 ? "..." : ""}`);
  }

  console.log("\n  --- Merged Answer ---");
  console.log(`  ${answer}\n`);
}

// ============================================
// MODE 4: Agent Inspector
// ============================================

async function modeAgentInspector(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Agent Inspector ---");
  console.log("  Run a single specialist and see its full trace.\n");

  const store = await pickStore(prompt);
  if (!store) return;
  const chatModel = await pickModel(prompt);

  const budget = calculateContextBudget(store, chatModel);
  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    chatModel,
    topK: budget.recommendedTopK,
  };
  const allTools = createTools(store, config);

  console.log("\nSpecialists:");
  ALL_SPECIALISTS.forEach((s, i) =>
    console.log(`  ${i + 1}. ${s.name} (${s.mode}) — ${s.description}`)
  );

  const choice = await prompt("\nPick a specialist (number): ");
  const num = parseInt(choice);
  if (isNaN(num) || num < 1 || num > ALL_SPECIALISTS.length) {
    console.log("Invalid choice.");
    return;
  }
  const specialist = ALL_SPECIALISTS[num - 1]!;

  while (true) {
    const query = await prompt(`\n[${specialist.name}] You: `);
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    console.log(`\n  Running ${specialist.name} agent (${specialist.mode} mode)...`);

    const sr = await runSpecialist(
      specialist,
      query,
      allTools,
      chatModel,
      true // verbose
    );

    console.log(`\n  Answer: ${sr.result.answer}`);
    console.log(
      `\n  [${specialist.name} | ${specialist.mode} | ${sr.result.iterations} steps | ${(sr.result.totalTimeMs / 1000).toFixed(1)}s]\n`
    );
  }
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log(`\n--- Phase 8: Multi-Agent System ---`);
  console.log(
    `    Chat: ${DEFAULT_CONFIG.chatModel} | Embed: ${DEFAULT_CONFIG.embedModel}\n`
  );
  console.log("Modes:");
  console.log("  1. Supervisor Chat    — router dispatches to specialists");
  console.log("  2. Pipeline Demo      — research → summarize → QA (sequential)");
  console.log("  3. Parallel Demo      — fan-out to agents, merge results");
  console.log("  4. Agent Inspector    — run one specialist directly");
  console.log();

  const { prompt, close } = createPrompt();

  try {
    const modeChoice = await prompt("Pick a mode (1-4): ");
    switch (modeChoice.trim()) {
      case "1":
        await modeSupervisorChat(prompt);
        break;
      case "2":
        await modePipelineDemo(prompt);
        break;
      case "3":
        await modeParallelDemo(prompt);
        break;
      case "4":
        await modeAgentInspector(prompt);
        break;
      default:
        console.log("Invalid mode.");
    }
  } finally {
    close();
    console.log("Goodbye!");
  }
}

main().catch(console.error);
