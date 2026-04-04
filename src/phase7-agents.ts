// ============================================
// PHASE 7: AGENTS & TOOL USE
//
// This demo shows the ReAct agent in action.
// The LLM can now TAKE ACTIONS by calling tools:
//   - searchDocuments: search the vector store
//   - readFile: read raw file contents
//   - listFiles: browse available files
//   - calculator: evaluate math expressions
//
// Modes:
//   1. Agent Chat   - multi-turn agent conversation
//   2. Tool Test    - test tools directly (no LLM)
//   3. Agent Trace  - see full reasoning for one question
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
  runAgent,
  createSearchDocumentsTool,
  createReadFileTool,
  createListFilesTool,
  createCalculatorTool,
  createFetchWebPageTool,
  createWebSearchTool,
  DEFAULT_AGENT_CONFIG,
  type Tool,
  type AgentStep,
  type AgentConfig,
  type AgentMode,
} from "./agent.js";

const SAMPLE_DOCS_DIR = "sample-docs";

// ============================================
// READLINE HELPERS (same pattern as phase5/6)
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
    console.log(
      "\nNo stores found in data/. Run Phase 5 Mode 1 first to ingest documents."
    );
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
// MODEL PICKER (same pattern as phase6)
// ============================================

const CHAT_MODELS = [
  {
    id: "nvidia/nemotron-3-super-120b-a12b:free",
    label: "nemotron-120b (free)",
  },
  { id: "openai/gpt-4o-mini", label: "gpt-4o-mini" },
];

async function pickModel(
  prompt: (q: string) => Promise<string>
): Promise<string> {
  console.log("\nChat models:");
  CHAT_MODELS.forEach((m, i) =>
    console.log(`  ${i + 1}. ${m.label}${i === 0 ? " (default)" : ""}`)
  );
  console.log(`  ${CHAT_MODELS.length + 1}. Custom model ID`);

  const choice = await prompt(
    `Pick a model (1-${CHAT_MODELS.length + 1}, default 1): `
  );
  const num = parseInt(choice) || 1;

  if (num >= 1 && num <= CHAT_MODELS.length) {
    return CHAT_MODELS[num - 1]!.id;
  } else if (num === CHAT_MODELS.length + 1) {
    return (
      (await prompt("Enter model ID: ")).trim() || DEFAULT_CONFIG.chatModel
    );
  }
  return DEFAULT_CONFIG.chatModel;
}

// ============================================
// STEP PRINTER
// ============================================

function printStep(step: AgentStep, stepNumber: number): void {
  console.log(`\n  --- Step ${stepNumber} ---`);
  if (step.thought) {
    console.log(`  Thought: ${step.thought}`);
  }
  if (step.action) {
    console.log(
      `  Action: ${step.action.tool}(${JSON.stringify(step.action.params)})`
    );
  }
  if (step.observation) {
    const preview = step.observation.slice(0, 300);
    console.log(
      `  Observation: ${preview}${step.observation.length > 300 ? "..." : ""}`
    );
  }
  if (step.isFinalAnswer && step.answer) {
    console.log(`\n  Final Answer: ${step.answer}`);
  }
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
// MODE 1: Agent Chat
// ============================================

async function modeAgentChat(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Agent Chat ---\n");

  const store = await pickStore(prompt);
  if (!store) return;

  const chatModel = await pickModel(prompt);

  const meta = store.getMetadata();
  const budget = calculateContextBudget(store, chatModel);
  console.log(`\n  ${meta.count} chunks from: ${meta.sources.join(", ")}`);
  console.log(
    `  Context budget: ${budget.availableForChunks.toLocaleString()} tokens → topK auto-set to ${budget.recommendedTopK}`
  );
  console.log(`  Model: ${chatModel}`);

  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    chatModel,
    topK: budget.recommendedTopK,
    enableQueryExpansion: false,
    enableReranking: false,
  };

  const tools = createTools(store, config);

  // Mode picker
  console.log("\nAgent modes:");
  console.log("  1. fast  - single RAG lookup, direct answer (fastest)");
  console.log("  2. think - RAG + step-by-step reasoning (balanced)");
  console.log("  3. react - full tool loop, multi-step reasoning (most capable)");
  const modeChoice = await prompt("Pick a mode (1-3, default 2): ");
  const modeMap: Record<string, AgentMode> = { "1": "fast", "2": "think", "3": "react" };
  const agentMode: AgentMode = modeMap[modeChoice.trim()] ?? "think";

  console.log(`\n  Mode: ${agentMode}`);
  console.log(`  Tools: ${tools.map((t) => t.name).join(", ")}`);
  console.log(
    '\n  Ask questions. Type "quit" to exit, "mode" to switch modes.\n'
  );

  let currentMode = agentMode;

  const agentConfig: AgentConfig = {
    chatModel,
    maxIterations: 10,
    verbose: true,
    mode: currentMode,
    onStep: printStep,
  };

  let lastSteps: AgentStep[] = [];
  const history: { role: "user" | "assistant"; content: string }[] = [];

  while (true) {
    const query = await prompt("You: ");
    if (query.trim().toLowerCase() === "quit") break;
    if (!query.trim()) continue;

    if (query.trim().toLowerCase() === "steps") {
      if (lastSteps.length === 0) {
        console.log("  No previous steps to show.\n");
      } else {
        console.log(`\n  Replaying ${lastSteps.length} step(s):`);
        lastSteps.forEach((s, i) => printStep(s, i + 1));
        console.log();
      }
      continue;
    }

    if (query.trim().toLowerCase() === "mode") {
      console.log("\n  1. fast  2. think  3. react");
      const mc = await prompt("  Pick (1-3): ");
      const mm: Record<string, AgentMode> = { "1": "fast", "2": "think", "3": "react" };
      currentMode = mm[mc.trim()] ?? currentMode;
      agentConfig.mode = currentMode;
      console.log(`  Switched to: ${currentMode}\n`);
      continue;
    }

    const result = await runAgent(query, tools, agentConfig, history);
    lastSteps = result.steps;

    // Track conversation for follow-ups
    history.push({ role: "user", content: query });
    history.push({ role: "assistant", content: result.answer });

    console.log(
      `\n  [${currentMode} | ${result.iterations} iteration(s) | ${(result.totalTimeMs / 1000).toFixed(1)}s${result.hitMaxIterations ? " | HIT MAX ITERATIONS" : ""}]\n`
    );
  }
}

// ============================================
// MODE 2: Single Tool Test
// ============================================

async function modeSingleToolTest(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Single Tool Test ---");
  console.log("  Test tools directly without the LLM.\n");

  const store = await pickStore(prompt);
  if (!store) return;

  const config: RAGConfig = { ...DEFAULT_CONFIG };
  const tools = createTools(store, config);

  while (true) {
    console.log("\nAvailable tools:");
    tools.forEach((t, i) => console.log(`  ${i + 1}. ${t.name} - ${t.description}`));
    console.log(`  ${tools.length + 1}. quit`);
    console.log();

    const choice = await prompt("Pick a tool: ");
    const num = parseInt(choice);
    if (num === tools.length + 1 || choice.trim().toLowerCase() === "quit") break;
    if (isNaN(num) || num < 1 || num > tools.length) {
      console.log("  Invalid choice.");
      continue;
    }

    const tool = tools[num - 1]!;
    console.log(`\n  Tool: ${tool.name}`);
    console.log(`  Parameters:`);
    for (const p of tool.parameters) {
      console.log(
        `    - ${p.name} (${p.type}${p.required ? ", required" : ", optional"}): ${p.description}`
      );
    }
    console.log();

    // Collect parameters
    const params: Record<string, unknown> = {};
    for (const p of tool.parameters) {
      const val = await prompt(`  ${p.name}: `);
      if (val.trim()) {
        params[p.name] = p.type === "number" ? parseFloat(val) : val.trim();
      }
    }

    console.log(`\n  Executing ${tool.name}(${JSON.stringify(params)})...\n`);

    try {
      const result = await tool.execute(params);
      console.log(`  Result:\n${result}\n`);
    } catch (err: any) {
      console.log(`  Error: ${err.message}\n`);
    }
  }
}

// ============================================
// MODE 3: Agent Trace Viewer
// ============================================

async function modeAgentTrace(prompt: (q: string) => Promise<string>) {
  console.log("\n--- Agent Trace Viewer ---");
  console.log("  Ask one question and see the full reasoning trace.\n");

  const store = await pickStore(prompt);
  if (!store) return;

  const chatModel = await pickModel(prompt);

  const budget = calculateContextBudget(store, chatModel);
  const config: RAGConfig = {
    ...DEFAULT_CONFIG,
    chatModel,
    topK: budget.recommendedTopK,
  };

  const tools = createTools(store, config);
  console.log(
    `\n  Tools: ${tools.map((t) => t.name).join(", ")}\n`
  );

  const query = await prompt("Your question: ");
  if (!query.trim()) {
    console.log("  Empty question, exiting.");
    return;
  }

  console.log("\n  Running agent...\n");

  const stepTimes: number[] = [];
  let lastStepTime = Date.now();

  const agentConfig: AgentConfig = {
    chatModel,
    maxIterations: 10,
    verbose: true,
    onStep: (step, stepNumber) => {
      const now = Date.now();
      stepTimes.push(now - lastStepTime);
      lastStepTime = now;
      printStep(step, stepNumber);
    },
  };

  const result = await runAgent(query, tools, agentConfig);

  // Summary
  console.log("\n  ============================================");
  console.log("  TRACE SUMMARY");
  console.log("  ============================================");
  console.log(`  Question: ${query}`);
  console.log(`  Iterations: ${result.iterations}`);
  console.log(
    `  Total time: ${(result.totalTimeMs / 1000).toFixed(1)}s`
  );
  if (result.hitMaxIterations) {
    console.log("  WARNING: Hit max iterations limit!");
  }

  // Per-step timing
  console.log("\n  Step timings:");
  result.steps.forEach((step, i) => {
    const time = stepTimes[i] ?? 0;
    const label = step.isFinalAnswer
      ? "Final Answer"
      : step.action
        ? `${step.action.tool}()`
        : "thinking";
    console.log(`    Step ${i + 1}: ${(time / 1000).toFixed(1)}s — ${label}`);
  });

  console.log(`\n  Final Answer: ${result.answer}\n`);
}

// ============================================
// MAIN
// ============================================

async function main() {
  console.log(`\n--- Phase 7: Agents & Tool Use ---`);
  console.log(
    `    Chat: ${DEFAULT_CONFIG.chatModel} | Embed: ${DEFAULT_CONFIG.embedModel}\n`
  );
  console.log("Modes:");
  console.log(
    "  1. Agent Chat        - ask questions, watch the agent reason & use tools"
  );
  console.log(
    "  2. Single Tool Test  - test individual tools directly (no LLM)"
  );
  console.log(
    "  3. Agent Trace       - see full reasoning trace for one question"
  );
  console.log();

  const { prompt, close } = createPrompt();

  try {
    const modeChoice = await prompt("Pick a mode (1-3): ");
    switch (modeChoice.trim()) {
      case "1":
        await modeAgentChat(prompt);
        break;
      case "2":
        await modeSingleToolTest(prompt);
        break;
      case "3":
        await modeAgentTrace(prompt);
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
