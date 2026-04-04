// ============================================
// MULTI-AGENT FRAMEWORK - Phase 8
//
// Multiple specialist agents collaborate to
// answer questions. A Router Agent dispatches
// to the right specialist based on the query.
//
// Orchestration patterns:
//   - Supervisor: Router picks specialists
//   - Pipeline: research → summarize → QA
//   - Parallel: fan-out, then merge results
//
// Each specialist is a configured runAgent() call
// with its own system prompt, tool set, and mode.
// ============================================

import "dotenv/config";
import {
  runAgent,
  type Tool,
  type AgentConfig,
  type AgentResult,
  type AgentMode,
  type AgentStep,
  buildAgentSystemPrompt,
} from "./agent.js";

// ============================================
// CLOUD CHAT (private copy for synthesis calls)
// ============================================

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

async function cloudChat(
  model: string,
  messages: { role: "system" | "user" | "assistant"; content: string }[]
): Promise<string> {
  const processedMessages: { role: "user" | "assistant"; content: string }[] = [];
  let systemContent = "";

  for (const msg of messages) {
    if (msg.role === "system") {
      systemContent += msg.content + "\n\n";
    } else {
      processedMessages.push({ role: msg.role, content: msg.content });
    }
  }

  if (systemContent && processedMessages.length > 0) {
    processedMessages[0]!.content = systemContent + processedMessages[0]!.content;
  }

  let response: Response;
  try {
    response = await fetch(OPENROUTER_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
      },
      body: JSON.stringify({ model, messages: processedMessages }),
    });
  } catch (err: any) {
    const cause = err.cause
      ? ` | cause: ${err.cause.message ?? err.cause.code ?? err.cause}`
      : "";
    throw new Error(`Cloud chat fetch failed (${OPENROUTER_URL}${cause})`);
  }

  const data = (await response.json()) as any;
  if (data.error) {
    throw new Error(data.error.message ?? JSON.stringify(data.error));
  }
  return (data.choices?.[0]?.message?.content ?? "").trim();
}

// ============================================
// TYPES
// ============================================

export interface SpecialistAgent {
  name: string;
  description: string;
  mode: AgentMode;
  toolNames: string[];
  systemPromptPrefix: string;
  maxIterations: number;
}

export interface SpecialistResult {
  agentName: string;
  result: AgentResult;
  startedAt: number;
  finishedAt: number;
}

export interface Blackboard {
  query: string;
  results: Map<string, SpecialistResult>;
  context: Map<string, string>;
  executionLog: { agentName: string; action: string; timestamp: number }[];
}

export type OrchestrationMode = "supervisor" | "pipeline" | "parallel";

export interface MultiAgentConfig {
  chatModel: string;
  orchestrationMode: OrchestrationMode;
  verbose: boolean;
  onAgentStart?: (agentName: string) => void;
  onAgentComplete?: (result: SpecialistResult) => void;
}

// ============================================
// SPECIALIST DEFINITIONS
// ============================================

export const RESEARCH_AGENT: SpecialistAgent = {
  name: "research",
  description:
    "Deep document search and web research. Use for factual questions needing specific info from loaded documents or the internet.",
  mode: "react",
  toolNames: ["searchDocuments", "readFile", "listFiles", "webSearch", "fetchWebPage"],
  systemPromptPrefix:
    "You are a Research Specialist. Your job is to find specific facts, details, " +
    "and information from the loaded documents. Search thoroughly — try multiple " +
    "queries if needed. ONLY answer based on what you find in the documents. " +
    "Never make up information. Cite your sources.",
  maxIterations: 8,
};

export const SUMMARIZER_AGENT: SpecialistAgent = {
  name: "summarizer",
  description:
    "Condenses and summarizes content. Use for 'summarize', 'overview', 'main points' requests.",
  mode: "think",
  toolNames: ["searchDocuments"],
  systemPromptPrefix:
    "You are a Summarization Specialist. Produce clear, organized summaries " +
    "based ONLY on the loaded documents. Use bullet points for lists. " +
    "Group related facts. Keep it concise but complete. Never add information " +
    "that isn't in the documents.",
  maxIterations: 3,
};

export const QA_AGENT: SpecialistAgent = {
  name: "qa",
  description:
    "Validates and fact-checks answers against source documents. Use to verify claims.",
  mode: "think",
  toolNames: ["searchDocuments"],
  systemPromptPrefix:
    "You are a QA Specialist. VERIFY information against the loaded source documents. " +
    "Search for evidence that supports or contradicts the claim. Be critical. " +
    "Output a confidence level (HIGH / MEDIUM / LOW) and explain what you verified.",
  maxIterations: 3,
};

export const GENERAL_AGENT: SpecialistAgent = {
  name: "general",
  description:
    "General knowledge questions not requiring document search. Greetings, meta-questions.",
  mode: "fast",
  toolNames: [],
  systemPromptPrefix:
    "You are a helpful general assistant. Answer using your general knowledge. " +
    "If the question would benefit from searching the loaded documents, suggest that.",
  maxIterations: 1,
};

export const ALL_SPECIALISTS: SpecialistAgent[] = [
  RESEARCH_AGENT,
  SUMMARIZER_AGENT,
  QA_AGENT,
  GENERAL_AGENT,
];

// ============================================
// BLACKBOARD
// ============================================

export function createBlackboard(query: string): Blackboard {
  return {
    query,
    results: new Map(),
    context: new Map(),
    executionLog: [],
  };
}

export function recordResult(board: Blackboard, result: SpecialistResult): void {
  board.results.set(result.agentName, result);
  board.executionLog.push({
    agentName: result.agentName,
    action: "completed",
    timestamp: result.finishedAt,
  });
}

export function summarizeBlackboard(board: Blackboard): string {
  const lines: string[] = [`Query: ${board.query}`, ""];
  for (const [name, sr] of board.results) {
    lines.push(
      `--- ${name} agent (${(sr.result.totalTimeMs / 1000).toFixed(1)}s, ${sr.result.iterations} steps) ---`
    );
    lines.push(sr.result.answer);
    lines.push("");
  }
  return lines.join("\n");
}

// ============================================
// RUN A SINGLE SPECIALIST
// ============================================
// Bridges a SpecialistAgent definition to runAgent().
// Filters tools, enriches query with blackboard context,
// handles no-tool agents via direct cloudChat().
// ============================================

export async function runSpecialist(
  specialist: SpecialistAgent,
  query: string,
  allTools: Tool[],
  chatModel: string,
  verbose: boolean,
  blackboard?: Blackboard
): Promise<SpecialistResult> {
  const startedAt = Date.now();

  // Enrich query with prior findings from blackboard
  let enrichedQuery = query;
  if (blackboard && blackboard.results.size > 0) {
    const priorFindings = Array.from(blackboard.results.entries())
      .map(
        ([name, sr]) =>
          `[${name} agent found]: ${sr.result.answer.slice(0, 500)}`
      )
      .join("\n\n");
    enrichedQuery = `${query}\n\nPrevious findings from other agents:\n${priorFindings}`;
  }

  let result: AgentResult;

  if (specialist.toolNames.length === 0) {
    // No tools — direct LLM call
    const start = Date.now();
    try {
      let answer = await cloudChat(chatModel, [
        { role: "system", content: specialist.systemPromptPrefix },
        { role: "user", content: enrichedQuery },
      ]);
      answer = answer.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
      result = {
        answer,
        steps: [],
        iterations: 1,
        totalTimeMs: Date.now() - start,
        hitMaxIterations: false,
      };
    } catch (err: any) {
      result = {
        answer: `Error: ${err.message}`,
        steps: [],
        iterations: 1,
        totalTimeMs: Date.now() - start,
        hitMaxIterations: false,
      };
    }
  } else {
    // Filter tools to this specialist's subset
    const tools = allTools.filter((t) =>
      specialist.toolNames.includes(t.name)
    );

    const agentConfig: AgentConfig = {
      chatModel,
      maxIterations: specialist.maxIterations,
      verbose,
      mode: specialist.mode,
      systemPromptPrefix: specialist.systemPromptPrefix,
    };

    result = await runAgent(enrichedQuery, tools, agentConfig);
  }

  return {
    agentName: specialist.name,
    result,
    startedAt,
    finishedAt: Date.now(),
  };
}

// ============================================
// SUPERVISOR MODE
// ============================================
// The Router is a ReAct agent whose "tools" are
// the specialist agents. It decides which to call.
// ============================================

function createSpecialistTool(
  specialist: SpecialistAgent,
  allTools: Tool[],
  chatModel: string,
  verbose: boolean,
  blackboard: Blackboard,
  onAgentStart?: (name: string) => void,
  onAgentComplete?: (result: SpecialistResult) => void
): Tool {
  return {
    name: `call_${specialist.name}`,
    description: specialist.description,
    parameters: [
      {
        name: "query",
        type: "string" as const,
        description: "The question or task for this specialist",
        required: true,
      },
    ],
    execute: async (params) => {
      const query = String(params.query ?? "");
      if (!query) return "Error: query parameter is required";

      onAgentStart?.(specialist.name);

      const specialistResult = await runSpecialist(
        specialist,
        query,
        allTools,
        chatModel,
        verbose,
        blackboard
      );

      recordResult(blackboard, specialistResult);
      onAgentComplete?.(specialistResult);

      return `[${specialist.name} agent | ${specialistResult.result.iterations} steps | ${(specialistResult.result.totalTimeMs / 1000).toFixed(1)}s]\n${specialistResult.result.answer}`;
    },
  };
}

export async function runSupervisor(
  query: string,
  allTools: Tool[],
  specialists: SpecialistAgent[],
  config: MultiAgentConfig
): Promise<{ answer: string; blackboard: Blackboard }> {
  const blackboard = createBlackboard(query);

  // Create meta-tools — one per specialist
  const routerTools = specialists.map((s) =>
    createSpecialistTool(
      s,
      allTools,
      config.chatModel,
      config.verbose,
      blackboard,
      config.onAgentStart,
      config.onAgentComplete
    )
  );

  const routerConfig: AgentConfig = {
    chatModel: config.chatModel,
    maxIterations: 6,
    verbose: config.verbose,
    mode: "react",
    systemPromptPrefix: `/no_think
You are a Router Agent managing specialist agents. Analyze the user's question
and call the right specialist(s). Synthesize their findings into a final answer.

Rules:
- Factual questions → call_research
- Summarize/overview requests → call_summarizer
- Verify/fact-check → call_qa
- General/meta questions → call_general
- You CAN call multiple specialists for complex questions
- Always provide a Final Answer that synthesizes the specialists' findings`,
  };

  const routerResult = await runAgent(query, routerTools, routerConfig);

  return { answer: routerResult.answer, blackboard };
}

// ============================================
// PIPELINE MODE
// ============================================
// Sequential: research → summarize → QA validate
// Each stage feeds into the next via blackboard.
// ============================================

export async function runPipeline(
  query: string,
  allTools: Tool[],
  specialists: SpecialistAgent[],
  config: MultiAgentConfig
): Promise<{ answer: string; blackboard: Blackboard }> {
  const blackboard = createBlackboard(query);

  // Step 1: Research
  const research = specialists.find((s) => s.name === "research")!;
  config.onAgentStart?.("research");
  const researchResult = await runSpecialist(
    research, query, allTools, config.chatModel, config.verbose
  );
  recordResult(blackboard, researchResult);
  config.onAgentComplete?.(researchResult);

  // Step 2: Summarize the research findings
  const summarizer = specialists.find((s) => s.name === "summarizer")!;
  config.onAgentStart?.("summarizer");
  const summarizeResult = await runSpecialist(
    summarizer,
    `Summarize these research findings about: "${query}"\n\nResearch:\n${researchResult.result.answer}`,
    allTools,
    config.chatModel,
    config.verbose
  );
  recordResult(blackboard, summarizeResult);
  config.onAgentComplete?.(summarizeResult);

  // Step 3: QA validate the summary
  const qa = specialists.find((s) => s.name === "qa")!;
  config.onAgentStart?.("qa");
  const qaResult = await runSpecialist(
    qa,
    `Verify the accuracy of this summary about: "${query}"\n\nSummary:\n${summarizeResult.result.answer}`,
    allTools,
    config.chatModel,
    config.verbose
  );
  recordResult(blackboard, qaResult);
  config.onAgentComplete?.(qaResult);

  const finalAnswer = [
    "## Summary",
    summarizeResult.result.answer,
    "",
    "## Verification",
    qaResult.result.answer,
  ].join("\n");

  return { answer: finalAnswer, blackboard };
}

// ============================================
// PARALLEL MODE
// ============================================
// Fan-out: research + summarizer run concurrently
// Fan-in: LLM merges results into one answer
// ============================================

export async function runParallel(
  query: string,
  allTools: Tool[],
  specialists: SpecialistAgent[],
  config: MultiAgentConfig
): Promise<{ answer: string; blackboard: Blackboard }> {
  const blackboard = createBlackboard(query);

  const research = specialists.find((s) => s.name === "research")!;
  const summarizer = specialists.find((s) => s.name === "summarizer")!;

  // Fan-out
  config.onAgentStart?.("research + summarizer (parallel)");

  const [researchResult, summarizeResult] = await Promise.all([
    runSpecialist(research, query, allTools, config.chatModel, config.verbose),
    runSpecialist(summarizer, query, allTools, config.chatModel, config.verbose),
  ]);

  recordResult(blackboard, researchResult);
  recordResult(blackboard, summarizeResult);
  config.onAgentComplete?.(researchResult);
  config.onAgentComplete?.(summarizeResult);

  // Fan-in: merge with LLM
  config.onAgentStart?.("synthesis");

  let mergedAnswer: string;
  try {
    mergedAnswer = await cloudChat(config.chatModel, [
      {
        role: "system",
        content: `/no_think
You are a synthesis agent. Two specialists answered the same question independently.
Merge their answers into one coherent response. Remove duplicates. Keep all unique facts.`,
      },
      {
        role: "user",
        content: `Question: ${query}\n\n--- Research Agent ---\n${researchResult.result.answer}\n\n--- Summarizer Agent ---\n${summarizeResult.result.answer}\n\nMerge into one answer:`,
      },
    ]);
    mergedAnswer = mergedAnswer.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  } catch (err: any) {
    mergedAnswer = `Synthesis error: ${err.message}\n\nResearch: ${researchResult.result.answer}\n\nSummary: ${summarizeResult.result.answer}`;
  }

  return { answer: mergedAnswer, blackboard };
}

// ============================================
// TOP-LEVEL DISPATCH
// ============================================

export async function runMultiAgent(
  query: string,
  allTools: Tool[],
  config: MultiAgentConfig
): Promise<{ answer: string; blackboard: Blackboard }> {
  switch (config.orchestrationMode) {
    case "supervisor":
      return runSupervisor(query, allTools, ALL_SPECIALISTS, config);
    case "pipeline":
      return runPipeline(query, allTools, ALL_SPECIALISTS, config);
    case "parallel":
      return runParallel(query, allTools, ALL_SPECIALISTS, config);
  }
}
