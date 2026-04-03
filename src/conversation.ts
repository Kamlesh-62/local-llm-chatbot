// ============================================
// CONVERSATION MEMORY MODULE - Phase 6
//
// Problem: In Phase 5, each question is independent.
// "What is MGRS?" → "How do I qualify for it?"
// The chatbot doesn't know "it" = MGRS.
//
// Solution: Maintain conversation history and pass
// it to the LLM so it has context from previous turns.
//
// Features:
//   1. Sliding window — keep last N turns, drop old ones
//   2. Summarization — summarize dropped turns so context isn't lost
//   3. Token budgeting — ensure history fits in the context window
//   4. Persistence — save/load conversations to disk
//   5. Query refinement — resolve pronouns using recent history
// ============================================

import "dotenv/config";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

// OpenRouter chat via fetch
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
        "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
      },
      body: JSON.stringify({ model, messages: processedMessages }),
    });
  } catch (err: any) {
    const cause = err.cause ? ` | cause: ${err.cause.message ?? err.cause.code ?? err.cause}` : "";
    const keySet = !!process.env.OPENROUTER_API_KEY;
    throw new Error(`Cloud chat fetch failed (key set: ${keySet}, url: ${OPENROUTER_URL}${cause})`);
  }

  const data = (await response.json()) as any;
  if (data.error) {
    throw new Error(data.error.message ?? JSON.stringify(data.error));
  }
  return (data.choices?.[0]?.message?.content ?? "").trim();
}

// --- Types ---

export interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface ConversationConfig {
  /** Max number of message pairs (user+assistant) to keep in window (default: 10) */
  maxTurns: number;
  /** Max estimated characters for history context (default: 4000) */
  maxChars: number;
  /** Model used for summarization */
  summarizeModel: string;
}

export const DEFAULT_CONVERSATION_CONFIG: ConversationConfig = {
  maxTurns: 10,
  maxChars: 4000,
  summarizeModel: "nvidia/nemotron-3-super-120b-a12b:free",
};

interface ConversationSnapshot {
  id: string;
  messages: Message[];
  summary: string | null;
  createdAt: string;
  updatedAt: string;
  config: ConversationConfig;
}

// ============================================
// CONVERSATION MEMORY CLASS
// ============================================

export class ConversationMemory {
  private id: string;
  private messages: Message[] = [];
  private summary: string | null = null;
  private config: ConversationConfig;
  private createdAt: string;

  constructor(config: Partial<ConversationConfig> = {}) {
    this.config = { ...DEFAULT_CONVERSATION_CONFIG, ...config };
    this.id = crypto.randomUUID();
    this.createdAt = new Date().toISOString();
  }

  /** Add a message to the conversation */
  addMessage(role: "user" | "assistant", content: string): void {
    this.messages.push({
      role,
      content,
      timestamp: Date.now(),
    });
  }

  /** Get full conversation history (all messages, unfiltered) */
  getHistory(): Message[] {
    return [...this.messages];
  }

  /** Get the conversation ID */
  getId(): string {
    return this.id;
  }

  /** Get current summary of older messages (if any) */
  getSummary(): string | null {
    return this.summary;
  }

  /** Get total number of messages */
  getMessageCount(): number {
    return this.messages.length;
  }

  /**
   * Get messages for the context window (sliding window).
   *
   * Returns the most recent messages that fit within:
   * 1. maxTurns limit (e.g., last 10 pairs)
   * 2. maxChars limit (e.g., 4000 chars total)
   *
   * If there's a summary from older messages, it's included
   * as the first message for context.
   */
  getContextMessages(): { role: "user" | "assistant" | "system"; content: string }[] {
    const result: { role: "user" | "assistant" | "system"; content: string }[] = [];

    // Include summary of older messages if available
    if (this.summary) {
      result.push({
        role: "system",
        content: `Previous conversation summary: ${this.summary}`,
      });
    }

    // Take the most recent messages within limits
    const maxMessages = this.config.maxTurns * 2; // each turn = user + assistant
    const recentMessages = this.messages.slice(-maxMessages);

    // Further trim by character budget
    let charCount = this.summary?.length ?? 0;
    const fittingMessages: Message[] = [];

    // Start from most recent, work backwards
    for (let i = recentMessages.length - 1; i >= 0; i--) {
      const msg = recentMessages[i]!;
      if (charCount + msg.content.length > this.config.maxChars) break;
      fittingMessages.unshift(msg);
      charCount += msg.content.length;
    }

    for (const msg of fittingMessages) {
      result.push({ role: msg.role, content: msg.content });
    }

    return result;
  }

  /**
   * Summarize older messages that have been pushed out of the window.
   *
   * This is called automatically when the window overflows.
   * The summary preserves key context (topics discussed, entities
   * mentioned, conclusions reached) in a compact form.
   */
  async summarizeOldMessages(): Promise<string> {
    const maxMessages = this.config.maxTurns * 2;

    // Messages that will be dropped from the window
    const oldMessages = this.messages.slice(0, -maxMessages);
    if (oldMessages.length === 0) return this.summary ?? "";

    const oldText = oldMessages
      .map((m) => `${m.role}: ${m.content}`)
      .join("\n");

    // Combine with existing summary if there is one
    const existingSummary = this.summary
      ? `Previous summary: ${this.summary}\n\n`
      : "";

    const result = await cloudChat(this.config.summarizeModel, [
      {
        role: "system",
        content:
          "Summarize this conversation in 2-3 sentences. Focus on: key topics discussed, important facts mentioned, and any questions that were answered. Be concise.",
      },
      {
        role: "user",
        content: `${existingSummary}Conversation to summarize:\n${oldText}`,
      },
    ]);

    this.summary = result.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

    return this.summary;
  }

  /**
   * Check if the window is overflowing and summarize if needed.
   * Call this after adding messages.
   */
  async maintainWindow(): Promise<boolean> {
    const maxMessages = this.config.maxTurns * 2;
    if (this.messages.length > maxMessages + 2) {
      // +2 buffer before triggering
      await this.summarizeOldMessages();
      return true; // summarization happened
    }
    return false;
  }

  /** Estimate token count (rough: 1 token ≈ 4 chars) */
  getTokenEstimate(): number {
    const historyChars = this.messages.reduce(
      (sum, m) => sum + m.content.length,
      0
    );
    const summaryChars = this.summary?.length ?? 0;
    return Math.ceil((historyChars + summaryChars) / 4);
  }

  /** Get memory stats for display */
  getStats(): {
    totalMessages: number;
    windowSize: number;
    tokenEstimate: number;
    hasSummary: boolean;
  } {
    const maxMessages = this.config.maxTurns * 2;
    return {
      totalMessages: this.messages.length,
      windowSize: Math.min(this.messages.length, maxMessages),
      tokenEstimate: this.getTokenEstimate(),
      hasSummary: this.summary !== null,
    };
  }

  /** Clear all messages and summary */
  clear(): void {
    this.messages = [];
    this.summary = null;
  }

  // --- Persistence ---

  /** Save conversation to a JSON file */
  save(dirPath: string = "data/conversations"): string {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }

    const filePath = path.join(dirPath, `${this.id}.json`);
    const snapshot: ConversationSnapshot = {
      id: this.id,
      messages: this.messages,
      summary: this.summary,
      createdAt: this.createdAt,
      updatedAt: new Date().toISOString(),
      config: this.config,
    };

    fs.writeFileSync(filePath, JSON.stringify(snapshot, null, 2));
    return filePath;
  }

  /** Load a conversation from a JSON file */
  static load(filePath: string): ConversationMemory {
    const raw = fs.readFileSync(filePath, "utf-8");
    const snapshot: ConversationSnapshot = JSON.parse(raw);

    const memory = new ConversationMemory(snapshot.config);
    memory.id = snapshot.id;
    memory.messages = snapshot.messages;
    memory.summary = snapshot.summary;
    memory.createdAt = snapshot.createdAt;

    return memory;
  }

  /** List saved conversations in a directory */
  static listSaved(
    dirPath: string = "data/conversations"
  ): { id: string; updatedAt: string; messageCount: number; preview: string }[] {
    if (!fs.existsSync(dirPath)) return [];

    const files = fs.readdirSync(dirPath).filter((f) => f.endsWith(".json"));
    return files.map((f) => {
      const raw = fs.readFileSync(path.join(dirPath, f), "utf-8");
      const snapshot: ConversationSnapshot = JSON.parse(raw);
      const firstUserMsg =
        snapshot.messages.find((m) => m.role === "user")?.content ?? "Empty";
      return {
        id: snapshot.id,
        updatedAt: snapshot.updatedAt,
        messageCount: snapshot.messages.length,
        preview: firstUserMsg.slice(0, 60),
      };
    });
  }
}

// ============================================
// QUERY REFINEMENT WITH CONVERSATION CONTEXT
// ============================================
// When the user says "How do I qualify for it?",
// "it" refers to something from the previous turn.
//
// We ask the LLM to rewrite the follow-up question
// as a standalone search query using conversation context.
//
// "How do I qualify for it?"
//   + history: [user asked about MGRS]
//   → LLM rewrites: "How do I qualify for MGRS Monthly Global Revenue Sharing?"
//
// This is much smarter than regex-based pronoun detection
// because the LLM understands what "it", "that", "the above"
// actually refer to in context.
// ============================================

export async function refineQueryWithHistory(
  query: string,
  history: Message[],
  model: string
): Promise<string> {
  if (history.length < 2) return query; // no previous context

  // Build recent conversation for the LLM
  const recentHistory = history.slice(-6); // last 3 turns
  const historyText = recentHistory
    .map((m) => `${m.role}: ${m.content.slice(0, 150)}`)
    .join("\n");

  const refined = (await cloudChat(model, [
    {
      role: "system",
      content: `Given a conversation and a follow-up question, rewrite the follow-up as a standalone search query. Replace pronouns (it, this, that, they) with the actual thing they refer to. Return ONLY the rewritten query, nothing else. If the question is already standalone, return it unchanged.`,
    },
    {
      role: "user",
      content: `Conversation:\n${historyText}\n\nFollow-up question: ${query}\n\nStandalone query:`,
    },
  ])).replace(/<think>[\s\S]*?<\/think>/g, "").trim();

  // If LLM returned something reasonable, use it; otherwise keep original
  if (refined && refined.length > 3 && refined.length < 500) {
    return refined;
  }
  return query;
}
