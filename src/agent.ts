// ============================================
// AGENT FRAMEWORK - Phase 7
//
// An "agent" is an LLM that can TAKE ACTIONS,
// not just answer questions. It follows the
// ReAct pattern:
//
//   Thought  → "I need to search for X"
//   Action   → calls searchDocuments("X")
//   Observation → "Here's what I found..."
//   Thought  → "Now I can answer"
//   Final Answer → "The answer is..."
//
// The agent loops until it has enough info to
// produce a final answer, or hits max iterations.
//
// This module provides:
//   - Tool definitions and registry
//   - ReAct system prompt builder
//   - LLM output parser (Thought/Action/Final Answer)
//   - The agent loop
//   - Ready-made tool factories wrapping Phase 5 functions
// ============================================

import "dotenv/config";
import * as fs from "fs";
import * as path from "path";
import * as cheerio from "cheerio";
import {
  ragQuery,
  readDocument,
  DEFAULT_CONFIG,
  type RAGConfig,
} from "./rag-pipeline.js";
import { VectorStore } from "./vector-store.js";

// ============================================
// CLOUD CHAT (private copy)
// ============================================
// Each module keeps its own copy for self-containment.
// See rag-pipeline.ts and conversation.ts for identical copies.

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
    const keySet = !!process.env.OPENROUTER_API_KEY;
    throw new Error(
      `Cloud chat fetch failed (key set: ${keySet}, url: ${OPENROUTER_URL}${cause})`
    );
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

export interface ToolParameter {
  name: string;
  type: "string" | "number" | "boolean";
  description: string;
  required: boolean;
}

export interface Tool {
  name: string;
  description: string;
  parameters: ToolParameter[];
  execute: (params: Record<string, unknown>) => Promise<string>;
}

export interface AgentStep {
  thought: string;
  action: { tool: string; params: Record<string, unknown> } | null;
  observation: string | null;
  isFinalAnswer: boolean;
  answer: string | null;
}

export interface ParsedLLMOutput {
  thought: string;
  action: { tool: string; params: Record<string, unknown> } | null;
  answer: string | null;
}

export interface AgentResult {
  answer: string;
  steps: AgentStep[];
  iterations: number;
  totalTimeMs: number;
  hitMaxIterations: boolean;
}

/**
 * Agent execution modes:
 *   - "fast"  : Single RAG lookup → direct answer. No tool loop. Fastest.
 *   - "think" : RAG lookup → LLM reasons step-by-step before answering. No tool loop.
 *   - "react" : Full ReAct loop — LLM picks tools, observes results, loops until done.
 */
export type AgentMode = "fast" | "think" | "react";

export interface AgentConfig {
  chatModel: string;
  maxIterations: number;
  verbose: boolean;
  mode: AgentMode;
  onStep?: (step: AgentStep, stepNumber: number) => void;
  /** Optional prefix prepended to the system prompt (used by multi-agent specialists) */
  systemPromptPrefix?: string;
}

export const DEFAULT_AGENT_CONFIG: AgentConfig = {
  chatModel: DEFAULT_CONFIG.chatModel,
  maxIterations: 10,
  verbose: true,
  mode: "react",
};

// ============================================
// SYSTEM PROMPT BUILDER
// ============================================
// Generates the ReAct-format system prompt that
// teaches the LLM how to use tools. The format:
//
//   Thought: <reasoning>
//   Action: <tool name>
//   Action Input: {"param": "value"}
//
// Or when done:
//
//   Thought: <reasoning>
//   Final Answer: <answer>
// ============================================

export function buildAgentSystemPrompt(tools: Tool[]): string {
  const toolDescriptions = tools
    .map((t) => {
      const params = t.parameters
        .map(
          (p) =>
            `    - ${p.name} (${p.type}${p.required ? ", required" : ", optional"}): ${p.description}`
        )
        .join("\n");
      return `  ${t.name}: ${t.description}\n    Parameters:\n${params}`;
    })
    .join("\n\n");

  return `/no_think
You are a helpful AI assistant with access to tools. To answer the user's question, you may need to use one or more tools. Follow this exact format for every response:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <JSON object with the tool's parameters on a single line>

After you receive the tool's result as an Observation, continue with another Thought/Action/Action Input if needed.

When you have enough information to answer, respond with:

Thought: <your final reasoning>
Final Answer: <your complete answer to the user>

RULES:
1. Always start with a Thought.
2. Use exactly ONE Action per response.
3. Action Input must be valid JSON on a single line.
4. Never make up information — use tools to find facts.
5. If a tool returns an error, try a different approach.
6. When you have a complete answer, use "Final Answer:" to deliver it.

Available tools:

${toolDescriptions}

Begin!`;
}

// ============================================
// LLM OUTPUT PARSER
// ============================================
// Extracts Thought, Action, Action Input, and
// Final Answer from the LLM's text response.
//
// Handles edge cases:
//   - <think> blocks from deepseek-r1
//   - Malformed JSON in Action Input
//   - LLM ignoring the format entirely
// ============================================

export function parseLLMOutput(raw: string): ParsedLLMOutput {
  // Strip <think>...</think> blocks (deepseek-r1 compatibility)
  const text = raw.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

  let thought = "";
  let action: { tool: string; params: Record<string, unknown> } | null = null;
  let answer: string | null = null;

  // Extract Thought
  const thoughtMatch = text.match(
    /Thought:\s*([\s\S]*?)(?=\n\s*(?:Action:|Final Answer:)|$)/i
  );
  if (thoughtMatch) {
    thought = thoughtMatch[1]!.trim();
  }

  // Check for Final Answer (takes priority over Action)
  const finalAnswerMatch = text.match(/Final Answer:\s*([\s\S]*)/i);
  if (finalAnswerMatch) {
    answer = finalAnswerMatch[1]!.trim();
    return { thought, action: null, answer };
  }

  // Extract Action and Action Input
  const actionMatch = text.match(/Action:\s*(.+)/i);
  const inputMatch = text.match(/Action Input:\s*(.*)/i);

  if (actionMatch) {
    const toolName = actionMatch[1]!.trim();
    let params: Record<string, unknown> = {};

    if (inputMatch) {
      const rawInput = inputMatch[1]!.trim();
      try {
        params = JSON.parse(rawInput);
      } catch {
        // Fallback: treat raw text as a query parameter
        params = { query: rawInput };
      }
    }

    action = { tool: toolName, params };
  }

  // If we got neither action nor answer, treat whole text as final answer
  if (!action && !answer && text.length > 0) {
    answer = text;
  }

  return { thought, action, answer };
}

// ============================================
// THE AGENT LOOP
// ============================================
// This is the heart of the agent. It:
//   1. Sends the system prompt + user query to the LLM
//   2. Parses the response for Thought/Action/Final Answer
//   3. If Action → executes the tool → feeds result back
//   4. If Final Answer → returns the result
//   5. Repeats until done or max iterations
// ============================================

const MAX_OBSERVATION_LENGTH = 3000;

export async function runAgent(
  query: string,
  tools: Tool[],
  config: AgentConfig = DEFAULT_AGENT_CONFIG,
  conversationHistory?: { role: "user" | "assistant"; content: string }[]
): Promise<AgentResult> {
  switch (config.mode) {
    case "fast":
      return runFastMode(query, tools, config, conversationHistory);
    case "think":
      return runThinkMode(query, tools, config, conversationHistory);
    case "react":
      return runReActMode(query, tools, config, conversationHistory);
  }
}

// ============================================
// FAST MODE
// ============================================
// Single RAG search → direct answer. No tool loop.
// Best for simple factual questions.
// ============================================

async function runFastMode(
  query: string,
  tools: Tool[],
  config: AgentConfig,
  conversationHistory?: { role: "user" | "assistant"; content: string }[]
): Promise<AgentResult> {
  const startTime = Date.now();

  // Build the query with conversation context
  let fullQuery = query;
  if (conversationHistory && conversationHistory.length > 0) {
    const historyText = conversationHistory
      .slice(-6)
      .map((m) => `${m.role}: ${m.content.slice(0, 200)}`)
      .join("\n");
    fullQuery = `Previous conversation:\n${historyText}\n\nNew question: ${query}`;
  }

  // Use searchDocuments tool directly — skip the LLM reasoning loop
  const searchTool = tools.find(
    (t) => t.name.toLowerCase() === "searchdocuments"
  );
  if (!searchTool) {
    return {
      answer: "Error: searchDocuments tool not available.",
      steps: [],
      iterations: 0,
      totalTimeMs: Date.now() - startTime,
      hitMaxIterations: false,
    };
  }

  try {
    const observation = await searchTool.execute({ query: fullQuery });

    const step: AgentStep = {
      thought: "Fast mode: direct RAG lookup",
      action: { tool: "searchDocuments", params: { query: fullQuery } },
      observation,
      isFinalAnswer: true,
      answer: observation,
    };
    config.onStep?.(step, 1);

    return {
      answer: observation,
      steps: [step],
      iterations: 1,
      totalTimeMs: Date.now() - startTime,
      hitMaxIterations: false,
    };
  } catch (err: any) {
    return {
      answer: `Error: ${err.message}`,
      steps: [],
      iterations: 1,
      totalTimeMs: Date.now() - startTime,
      hitMaxIterations: false,
    };
  }
}

// ============================================
// THINK MODE
// ============================================
// RAG search → LLM reasons step-by-step with the
// retrieved context. No tool loop — just one
// search + one thoughtful LLM call.
// Good for questions that need reasoning over
// the retrieved facts.
// ============================================

async function runThinkMode(
  query: string,
  tools: Tool[],
  config: AgentConfig,
  conversationHistory?: { role: "user" | "assistant"; content: string }[]
): Promise<AgentResult> {
  const startTime = Date.now();
  const steps: AgentStep[] = [];

  // Step 1: Search documents
  const searchTool = tools.find(
    (t) => t.name.toLowerCase() === "searchdocuments"
  );
  if (!searchTool) {
    return {
      answer: "Error: searchDocuments tool not available.",
      steps: [],
      iterations: 0,
      totalTimeMs: Date.now() - startTime,
      hitMaxIterations: false,
    };
  }

  let searchResult: string;
  try {
    searchResult = await searchTool.execute({ query });
  } catch (err: any) {
    return {
      answer: `Search error: ${err.message}`,
      steps: [],
      iterations: 1,
      totalTimeMs: Date.now() - startTime,
      hitMaxIterations: false,
    };
  }

  const searchStep: AgentStep = {
    thought: "Think mode: searching documents first",
    action: { tool: "searchDocuments", params: { query } },
    observation: searchResult,
    isFinalAnswer: false,
    answer: null,
  };
  steps.push(searchStep);
  config.onStep?.(searchStep, 1);

  // Step 2: Ask LLM to reason over the results
  const messages: {
    role: "system" | "user" | "assistant";
    content: string;
  }[] = [
    {
      role: "system",
      content: `/no_think
${config.systemPromptPrefix ?? ""}
You are a helpful assistant. You have been given search results from a document store. Think step-by-step about the information, then provide a clear, well-reasoned answer.`,
    },
  ];

  // Add conversation history
  if (conversationHistory && conversationHistory.length > 0) {
    const historyText = conversationHistory
      .slice(-6)
      .map((m) => `${m.role}: ${m.content.slice(0, 200)}`)
      .join("\n");
    messages.push({
      role: "user",
      content: `Previous conversation:\n${historyText}\n\nSearch results:\n${searchResult}\n\nQuestion: ${query}\n\nThink step-by-step, then answer:`,
    });
  } else {
    messages.push({
      role: "user",
      content: `Search results:\n${searchResult}\n\nQuestion: ${query}\n\nThink step-by-step, then answer:`,
    });
  }

  try {
    let answer = await cloudChat(config.chatModel, messages);
    answer = answer.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    const thinkStep: AgentStep = {
      thought: "Reasoning over search results",
      action: null,
      observation: null,
      isFinalAnswer: true,
      answer,
    };
    steps.push(thinkStep);
    config.onStep?.(thinkStep, 2);

    return {
      answer,
      steps,
      iterations: 2,
      totalTimeMs: Date.now() - startTime,
      hitMaxIterations: false,
    };
  } catch (err: any) {
    return {
      answer: `LLM error: ${err.message}`,
      steps,
      iterations: 2,
      totalTimeMs: Date.now() - startTime,
      hitMaxIterations: false,
    };
  }
}

// ============================================
// REACT MODE
// ============================================
// Full ReAct loop — LLM picks tools, observes
// results, and loops until it produces a final
// answer or hits max iterations.
// Best for complex multi-step questions.
// ============================================

async function runReActMode(
  query: string,
  tools: Tool[],
  config: AgentConfig,
  conversationHistory?: { role: "user" | "assistant"; content: string }[]
): Promise<AgentResult> {
  const startTime = Date.now();
  const steps: AgentStep[] = [];

  // Build tool lookup (case-insensitive)
  const toolMap = new Map<string, Tool>();
  for (const tool of tools) {
    toolMap.set(tool.name.toLowerCase(), tool);
  }

  // Build message history
  const messages: { role: "system" | "user" | "assistant"; content: string }[] =
    [{ role: "system", content: (config.systemPromptPrefix ? config.systemPromptPrefix + "\n\n" : "") + buildAgentSystemPrompt(tools) }];

  // Add conversation history for multi-turn context
  if (conversationHistory && conversationHistory.length > 0) {
    const historyText = conversationHistory
      .slice(-6) // last 3 turns
      .map((m) => `${m.role}: ${m.content.slice(0, 200)}`)
      .join("\n");
    messages.push({
      role: "user",
      content: `Previous conversation:\n${historyText}\n\nNew question: ${query}`,
    });
  } else {
    messages.push({ role: "user", content: query });
  }

  let iterations = 0;

  while (iterations < config.maxIterations) {
    iterations++;

    // 1. Call the LLM
    let llmResponse: string;
    try {
      llmResponse = await cloudChat(config.chatModel, messages);
    } catch (err: any) {
      return {
        answer: `Agent error: Failed to call LLM: ${err.message}`,
        steps,
        iterations,
        totalTimeMs: Date.now() - startTime,
        hitMaxIterations: false,
      };
    }

    // 2. Parse the response
    const parsed = parseLLMOutput(llmResponse);

    // 3. If Final Answer → done
    if (parsed.answer !== null) {
      const step: AgentStep = {
        thought: parsed.thought,
        action: null,
        observation: null,
        isFinalAnswer: true,
        answer: parsed.answer,
      };
      steps.push(step);
      config.onStep?.(step, iterations);

      return {
        answer: parsed.answer,
        steps,
        iterations,
        totalTimeMs: Date.now() - startTime,
        hitMaxIterations: false,
      };
    }

    // 4. Execute the tool
    let observation: string;

    if (parsed.action) {
      const tool = toolMap.get(parsed.action.tool.toLowerCase());
      if (!tool) {
        observation = `Error: Unknown tool "${parsed.action.tool}". Available tools: ${tools.map((t) => t.name).join(", ")}`;
      } else {
        try {
          observation = await tool.execute(parsed.action.params);
          // Truncate long observations to stay within context window
          if (observation.length > MAX_OBSERVATION_LENGTH) {
            observation =
              observation.slice(0, MAX_OBSERVATION_LENGTH) +
              "\n...[truncated]";
          }
        } catch (err: any) {
          observation = `Error executing ${tool.name}: ${err.message ?? "Unknown error"}`;
        }
      }
    } else {
      // LLM produced thought but no action and no answer — nudge it
      observation =
        "You must either use a tool (Action + Action Input) or provide a Final Answer.";
    }

    // 5. Record the step
    const step: AgentStep = {
      thought: parsed.thought,
      action: parsed.action,
      observation,
      isFinalAnswer: false,
      answer: null,
    };
    steps.push(step);
    config.onStep?.(step, iterations);

    // 6. Append to message history for next iteration
    messages.push({ role: "assistant", content: llmResponse });
    messages.push({ role: "user", content: `Observation: ${observation}` });
  }

  // Hit max iterations — return what we have
  const lastObs = steps[steps.length - 1]?.observation ?? "";
  return {
    answer: `I was unable to complete the task within ${config.maxIterations} iterations. Here is what I found so far:\n\n${lastObs}`,
    steps,
    iterations,
    totalTimeMs: Date.now() - startTime,
    hitMaxIterations: true,
  };
}

// ============================================
// TOOL FACTORIES
// ============================================
// Each factory creates a Tool by wrapping an
// existing function from Phases 3-5.
// ============================================

// --- searchDocuments ---
// Wraps ragQuery() from rag-pipeline.ts
// The agent's main way to find information in documents.

export function createSearchDocumentsTool(
  store: VectorStore,
  config: RAGConfig
): Tool {
  return {
    name: "searchDocuments",
    description:
      "Search the loaded document store for information. Returns relevant passages with sources.",
    parameters: [
      {
        name: "query",
        type: "string",
        description: "The search query",
        required: true,
      },
    ],
    execute: async (params) => {
      const query = String(params.query ?? "");
      if (!query) return "Error: query parameter is required";

      const result = await ragQuery(query, store, config);
      if (result.sources.length === 0) {
        return "No relevant documents found for this query.";
      }

      const sources = result.sources
        .map(
          (s, i) =>
            `[${i + 1}] (${s.source}, score: ${s.score.toFixed(3)}) ${s.preview.slice(0, 100)}`
        )
        .join("\n");
      return `Answer from documents: ${result.answer}\n\nSources:\n${sources}`;
    },
  };
}

// --- readFile ---
// Wraps readDocument() from rag-pipeline.ts
// Lets the agent read raw file contents.

export function createReadFileTool(allowedDir: string): Tool {
  return {
    name: "readFile",
    description:
      "Read a file's contents. Supports .txt, .md, .json, .csv, .pdf, .docx files.",
    parameters: [
      {
        name: "path",
        type: "string",
        description: "Path to the file (relative to project root)",
        required: true,
      },
    ],
    execute: async (params) => {
      const filePath = String(params.path ?? "");
      if (!filePath) return "Error: path parameter is required";

      // Security: ensure path is within allowed directory
      const resolved = path.resolve(filePath);
      const allowed = path.resolve(allowedDir);
      if (!resolved.startsWith(allowed)) {
        return `Error: Access denied. Files must be within ${allowedDir}`;
      }

      if (!fs.existsSync(resolved)) {
        return `Error: File not found: ${filePath}`;
      }

      try {
        const content = await readDocument(resolved);
        if (content.length > MAX_OBSERVATION_LENGTH) {
          return (
            content.slice(0, MAX_OBSERVATION_LENGTH) +
            `\n...[truncated, ${content.length} total chars]`
          );
        }
        return content;
      } catch (err: any) {
        return `Error reading file: ${err.message}`;
      }
    },
  };
}

// --- listFiles ---
// Wraps fs.readdirSync()
// Lets the agent discover what files are available.

export function createListFilesTool(allowedDir: string): Tool {
  return {
    name: "listFiles",
    description:
      "List files and directories at a path. Useful for discovering available documents.",
    parameters: [
      {
        name: "directory",
        type: "string",
        description: `Directory to list (default: ${allowedDir})`,
        required: false,
      },
    ],
    execute: async (params) => {
      const dir = String(params.directory ?? allowedDir);

      // Security check
      const resolved = path.resolve(dir);
      const allowed = path.resolve(allowedDir);
      if (!resolved.startsWith(allowed)) {
        return `Error: Access denied. Can only list within ${allowedDir}`;
      }

      if (!fs.existsSync(resolved)) {
        return `Error: Directory not found: ${dir}`;
      }

      try {
        const entries = fs.readdirSync(resolved, { withFileTypes: true });
        const lines = entries.map((e) => {
          const icon = e.isDirectory() ? "[DIR] " : "[FILE]";
          return `  ${icon} ${e.name}`;
        });
        return `Contents of ${dir}:\n${lines.join("\n")}`;
      } catch (err: any) {
        return `Error listing directory: ${err.message}`;
      }
    },
  };
}

// --- calculator ---
// Evaluates math expressions safely.
// No external dependencies.

export function createCalculatorTool(): Tool {
  return {
    name: "calculator",
    description:
      "Evaluate a math expression. Supports +, -, *, /, **, %, parentheses, and Math.sqrt/round/ceil/floor/abs/pow/min/max/PI/E.",
    parameters: [
      {
        name: "expression",
        type: "string",
        description: "The math expression, e.g. '(15 * 3) + 7'",
        required: true,
      },
    ],
    execute: async (params) => {
      const expr = String(params.expression ?? "").trim();
      if (!expr) return "Error: expression parameter is required";

      // Whitelist: only allow digits, operators, parens, dots, spaces, commas, and Math.*
      const withoutMath = expr.replace(
        /Math\.(sqrt|abs|round|ceil|floor|pow|min|max|PI|E)/g,
        ""
      );
      if (!/^[0-9+\-*/().,%\s]+$/.test(withoutMath)) {
        return "Error: Unsafe expression. Only numbers, basic operators (+, -, *, /, **, %), parentheses, and Math functions are allowed.";
      }

      try {
        const fn = new Function("Math", `"use strict"; return (${expr});`);
        const result = fn(Math);
        if (typeof result !== "number" || !isFinite(result)) {
          return "Error: Expression did not produce a valid number.";
        }
        return `${expr} = ${result}`;
      } catch (err: any) {
        return `Error evaluating expression: ${err.message}`;
      }
    },
  };
}

// --- fetchWebPage ---
// Fetches a URL and converts HTML to readable text.
// Uses cheerio to strip tags, scripts, styles.

export function createFetchWebPageTool(): Tool {
  return {
    name: "fetchWebPage",
    description:
      "Fetch a web page by URL and extract its text content. Useful for reading articles, documentation, or any public web page.",
    parameters: [
      {
        name: "url",
        type: "string",
        description: "The full URL to fetch (e.g. https://example.com/page)",
        required: true,
      },
    ],
    execute: async (params) => {
      const url = String(params.url ?? "").trim();
      if (!url) return "Error: url parameter is required";
      if (!url.startsWith("http://") && !url.startsWith("https://")) {
        return "Error: URL must start with http:// or https://";
      }

      try {
        const response = await fetch(url, {
          headers: {
            "User-Agent":
              "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com)",
          },
          signal: AbortSignal.timeout(15000),
        });

        if (!response.ok) {
          return `Error: HTTP ${response.status} ${response.statusText}`;
        }

        const html = await response.text();
        const $ = cheerio.load(html);

        // Remove non-content elements
        $("script, style, nav, footer, header, iframe, noscript").remove();

        // Extract title
        const title = $("title").text().trim();

        // Extract main text content
        const text = $("body")
          .text()
          .replace(/\s+/g, " ")
          .replace(/\n{3,}/g, "\n\n")
          .trim();

        if (!text) return `Fetched ${url} but no text content found.`;

        const result = title ? `Title: ${title}\n\n${text}` : text;

        // Truncate to avoid blowing up context
        if (result.length > 4000) {
          return result.slice(0, 4000) + `\n...[truncated, ${result.length} total chars]`;
        }
        return result;
      } catch (err: any) {
        return `Error fetching ${url}: ${err.message}`;
      }
    },
  };
}

// --- webSearch ---
// Searches the web using DuckDuckGo (no API key required).
// Returns top results with titles, URLs, and snippets.

export function createWebSearchTool(): Tool {
  return {
    name: "webSearch",
    description:
      "Search the internet using DuckDuckGo. Returns top results with titles, URLs, and snippets. No API key needed.",
    parameters: [
      {
        name: "query",
        type: "string",
        description: "The search query",
        required: true,
      },
    ],
    execute: async (params) => {
      const query = String(params.query ?? "").trim();
      if (!query) return "Error: query parameter is required";

      try {
        const encodedQuery = encodeURIComponent(query);
        const response = await fetch(
          `https://html.duckduckgo.com/html/?q=${encodedQuery}`,
          {
            method: "POST",
            headers: {
              "User-Agent":
                "Mozilla/5.0 (compatible; RAGBot/1.0; +https://github.com)",
              "Content-Type": "application/x-www-form-urlencoded",
            },
            body: `q=${encodedQuery}`,
            signal: AbortSignal.timeout(15000),
          }
        );

        if (!response.ok) {
          return `Error: Search returned HTTP ${response.status}`;
        }

        const html = await response.text();
        const $ = cheerio.load(html);

        const results: string[] = [];
        $(".result").each((_i, el) => {
          if (results.length >= 5) return false; // top 5

          const title = $(el).find(".result__title").text().trim();
          const snippet = $(el).find(".result__snippet").text().trim();
          const link = $(el).find(".result__url").text().trim();

          if (title && snippet) {
            results.push(
              `[${results.length + 1}] ${title}\n    URL: ${link}\n    ${snippet}`
            );
          }
        });

        if (results.length === 0) {
          return `No results found for: "${query}"`;
        }

        return `Search results for "${query}":\n\n${results.join("\n\n")}`;
      } catch (err: any) {
        return `Error searching: ${err.message}`;
      }
    },
  };
}
