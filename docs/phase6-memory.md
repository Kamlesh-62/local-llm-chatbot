# Phase 6: Conversation Memory — Multi-Turn Context

## Goal

Make the chatbot remember previous exchanges so follow-up questions work. Without memory, every question is isolated — the user has to repeat context every time.

**Key source files:**
- `src/conversation.ts` — reusable ConversationMemory class
- `src/phase6-memory.ts` — interactive demo with 3 modes

---

## What You'll Learn

- Why memory is two separate problems (LLM understanding vs search retrieval)
- How LLM-based query rewriting beats regex heuristics
- How sliding window + summarization manages context length
- How token budgeting prevents exceeding the context window
- How to persist conversations across sessions
- What industry production systems use for memory

---

## The Problem

Without conversation memory:

```
User: When was Confederation?
Bot:  Confederation occurred in 1867 when the Dominion of Canada was created...

User: Who was the first Prime Minister?
Bot:  Sir John A. Macdonald was the first Prime Minister...

User: What party did he lead?
Bot:  I don't have enough information to answer that.
```

The word "he" means nothing without the previous exchange. The chatbot treats each question as if it appeared out of nowhere. Both the LLM and the search system are blind to what came before.

---

## Two Different Problems

Memory seems like one feature, but it's actually two distinct challenges that require different solutions.

### Problem 1: LLM Understanding

The LLM needs to know what "he" refers to. This is the easier problem — just pass the conversation history in the messages array:

```typescript
const messages = [
  { role: "system", content: systemPrompt },
  { role: "user", content: "Who was the first Prime Minister?" },
  { role: "assistant", content: "Sir John A. Macdonald was the first Prime Minister..." },
  { role: "user", content: "What party did he lead?" },
];

const response = await ollama.chat({ model, messages });
// LLM naturally resolves "he" = Sir John A. Macdonald from the conversation history
```

LLMs are trained on conversations. They resolve pronouns and references naturally when given the history. No special logic needed.

### Problem 2: Search Retrieval

The search system receives the raw query "What party did he lead?" and has to find relevant document chunks. Neither keyword search nor embedding search knows what "he" means — they only see the current query.

- Keyword search for "party did he lead" — matches nothing useful
- Embedding search for "What party did he lead?" — the vector is too vague to find content about Sir John A. Macdonald

This is the harder problem, and it requires **query rewriting**.

---

## LLM-Based Query Rewriting

### The Naive Approach (Don't Do This)

You might think: check if the query contains pronouns ("it", "that", "they") and substitute them from the previous exchange.

```typescript
// Fragile regex approach — DON'T DO THIS
if (query.match(/\b(he|it|that|this|they)\b/)) {
  query = query.replace("he", lastTopic);
}
```

This breaks on countless edge cases: "Is it true that...", "How does this compare to that?", questions where "he" refers to someone from 5 turns ago.

### The Better Approach: Ask the LLM

```typescript
const rewritePrompt = `Given this conversation history, rewrite the user's latest
question as a standalone search query. The rewritten query should make sense
without any conversation context.

Conversation:
User: Who was the first Prime Minister?
Assistant: Sir John A. Macdonald was the first Prime Minister of Canada...
User: What party did he lead?

Rewritten query:`;

const rewritten = await ollama.chat({
  model,
  messages: [{ role: "user", content: rewritePrompt }],
});
// Result: "What political party did Sir John A. Macdonald lead?"
```

The LLM resolves "he" to "Sir John A. Macdonald" — a query that both keyword and embedding search can work with. This approach handles pronouns, ellipsis ("And the deadline?"), topic switches, and multi-reference questions without any regex.

---

## Sliding Window

You can't keep the entire conversation history forever. Each message consumes tokens, and the context window has a hard limit.

### How It Works

Keep the last **N turns** (default: 10 turns = 20 messages). When the conversation exceeds this limit, drop the oldest messages.

```typescript
class ConversationMemory {
  private maxTurns: number = 10;
  private messages: Message[] = [];

  addTurn(userMessage: string, assistantMessage: string) {
    this.messages.push(
      { role: "user", content: userMessage },
      { role: "assistant", content: assistantMessage }
    );

    // Trim if over limit
    while (this.messages.length > this.maxTurns * 2) {
      this.messages.shift(); // Drop oldest
      this.messages.shift();
    }
  }
}
```

### The Problem

Dropping old messages means losing context. If the user asked about Confederation in turn 1 and refers back to it in turn 15, the Confederation context is gone. The LLM can no longer resolve the reference.

---

## Summarization

Before dropping old messages, ask the LLM to **summarize** them. The summary is compact but preserves the key topics and facts.

```typescript
async summarizeOldMessages(messagesToDrop: Message[]): Promise<string> {
  const response = await ollama.chat({
    model,
    messages: [
      {
        role: "system",
        content: "Summarize this conversation excerpt in 2-3 sentences. Focus on key topics, facts, and decisions.",
      },
      ...messagesToDrop,
    ],
  });

  return response.message.content;
  // "User asked about Confederation (1867) and the first Prime Minister.
  //  Sir John A. Macdonald led the Conservative Party. User also asked
  //  about the British North America Act and the founding provinces."
}
```

This summary is prepended to the system prompt so the LLM retains awareness of what was discussed earlier, even after the raw messages are dropped.

```typescript
const systemPrompt = `${basePrompt}

Previous conversation summary:
${summary}

CONTEXT:
${relevantChunks}`;
```

The result: a conversation that can run for dozens of turns without losing track of what was discussed.

---

## Token Budgeting

Every piece of the prompt competes for space in the context window:

```
Context Window (e.g., 8192 tokens)
├── System prompt:        ~200 tokens
├── Conversation summary: ~100 tokens
├── RAG chunks:           ~2000 tokens
├── Conversation history: ~3000 tokens
├── Current question:     ~50 tokens
└── Reserved for answer:  ~2800 tokens
```

The `ConversationMemory` class tracks estimated token usage and trims history to fit:

```typescript
class ConversationMemory {
  private tokenBudget: number;

  getMessagesWithinBudget(reservedTokens: number): Message[] {
    const available = this.tokenBudget - reservedTokens;
    const messages: Message[] = [];
    let tokenCount = 0;

    // Add messages from most recent to oldest
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const msgTokens = this.estimateTokens(this.messages[i].content);
      if (tokenCount + msgTokens > available) break;
      messages.unshift(this.messages[i]);
      tokenCount += msgTokens;
    }

    return messages;
  }

  private estimateTokens(text: string): number {
    // Rough estimate: ~4 characters per token for English text
    return Math.ceil(text.length / 4);
  }
}
```

If the conversation history is too long, the oldest turns get dropped (and summarized) to make room. RAG chunks and the system prompt always get priority — without them, the chatbot can't ground its answers.

---

## Persistent Chat

Conversations are saved to JSON files so you can resume where you left off:

```typescript
// Save conversation state
const state = {
  id: conversationId,
  messages: memory.getMessages(),
  summary: memory.getSummary(),
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
};

fs.writeFileSync(
  `./data/conversations/${conversationId}.json`,
  JSON.stringify(state, null, 2)
);

// Load and resume
const saved = JSON.parse(
  fs.readFileSync(`./data/conversations/${conversationId}.json`, "utf-8")
);
memory.restore(saved.messages, saved.summary);
```

---

## How It Works — Multi-Turn Flow

```
  User follow-up question: "What party did he lead?"
       |
       v
  +------------------------+
  | LLM Rewrites Query     |  Uses conversation history to resolve "he"
  | "What political party   |  Output: standalone search query
  |  did Sir John A.        |
  |  Macdonald lead?"      |
  +------------------------+
       |
       v
  +------------------------+
  | Search with Refined     |  Hybrid search now finds Macdonald chunks
  | Query                   |  because the query is explicit
  +------------------------+
       |
       v
  +------------------------+
  | Build Prompt            |  System prompt + summary + history +
  |                         |  RAG chunks + current question
  +------------------------+
       |
       v
  +------------------------+
  | LLM Answers             |  Full context: knows conversation,
  |                         |  has relevant chunks, cites sources
  +------------------------+
```

---

## The 3 Demo Modes

Run `npm run phase6` and choose a mode:

### Mode 1: RAG Chat with Memory

Your main chatbot. Ask questions about your ingested documents, and it **remembers** the conversation.

```
You: When was Confederation?
Assistant: Confederation was in 1867, when the British North America Act
          united Ontario, Quebec, Nova Scotia, and New Brunswick.
  [Memory: 2 messages, ~80 tokens]

You: Who was the first Prime Minister?
Assistant: Sir John A. Macdonald was the first Prime Minister, starting in 1867.
  [Memory: 4 messages, ~150 tokens]

You: What party did he lead?
  [Refined query: "What party did Sir John A. Macdonald lead?"]
Assistant: He led the Liberal-Conservative Party.
  [Memory: 6 messages, ~220 tokens]
```

Without memory, "he" in the third question would mean nothing. With memory:
- The **LLM** sees the full conversation history and knows "he" = Macdonald
- The **search** gets a rewritten query ("What party did Sir John A. Macdonald lead?") so it finds the right chunks

You can type `"save"` anytime to save the conversation to disk.

### Mode 2: Explore Memory (Small Window)

Same chatbot but with a **tiny window (3 turns only)** so you can **watch** the memory system work. Type `"memory"` to inspect the internal state.

```
You: When was Confederation?
Assistant: 1867...
  [Messages: 2 | Window: 2 | Tokens: ~80]

You: Who was the first PM?
Assistant: Sir John A. Macdonald...
  [Messages: 4 | Window: 4 | Tokens: ~150]

You: What is the Charter of Rights?
Assistant: The Charter was enacted in 1982...
  [Messages: 6 | Window: 6 | Tokens: ~220]

You: When did women get the right to vote?
  >>> SLIDING WINDOW TRIGGERED <<<
  Older messages summarized.
  [Messages: 8 | Window: 6 | Tokens: ~180]

You: memory

==================================================
MEMORY STATE
==================================================
  Total messages: 8
  Window size: 6
  Has summary: true

  Summary of older messages:
  "User asked about Confederation (1867) and the first
   Prime Minister (Sir John A. Macdonald)."

  Active messages in window:
    [user] "What is the Charter of Rights?"
    [assistant] "The Charter was enacted in 1982..."
    [user] "When did women get the right to vote?"
    [assistant] "Women gained federal voting rights in 1918..."
==================================================
```

The first two turns got **summarized** into one paragraph and dropped from the window. The summary is still passed to the LLM so it knows what was discussed — but the full messages are gone to save token space.

**Why 3 turns?** So you see summarization happen fast. Mode 1 uses 10 turns (normal usage).

### Mode 3: Load Saved Conversation

Resume a conversation you saved earlier from Mode 1.

```
--- Load Saved Conversation ---

Saved conversations:
  1. [6 msgs] "When was Confederation?..." (2026-04-02)
  2. [12 msgs] "Tell me about Terry Fox..." (2026-04-01)

Pick a conversation to resume: 1

--- Previous conversation ---
You: When was Confederation?
Assistant: 1867, British North America Act...
You: Who was the first PM?
Assistant: Sir John A. Macdonald...
--- Resuming ---

You: What about the second PM?
Assistant: The second Prime Minister was Sir John Abbott...
```

It loads the full history, so the LLM has all the context from the previous session. You can continue chatting and type `"save"` to update.

### Mode Comparison

| | Mode 1 | Mode 2 | Mode 3 |
|---|---|---|---|
| **Purpose** | Use the chatbot | Learn how memory works | Resume old chat |
| **Window size** | 10 turns | 3 turns (triggers fast) | 10 turns |
| **Shows memory state** | Just message count | Full breakdown on `"memory"` | No |
| **Summarization** | After ~10 turns | After ~3 turns (see it quickly) | Inherits from saved |
| **Save/Load** | Save with `"save"` | No | Load + update |

Start with **Mode 1** for normal use. Try **Mode 2** to see the sliding window and summarization in action.

---

## What Industry Uses

Memory seems simple but production systems are sophisticated. Here's how what we built compares to real-world implementations.

### Common Approaches

**Sliding window + summarization** — This is what ChatGPT, Claude, and most consumer chatbots use. Keep recent messages verbatim, summarize older ones. Simple and effective for most use cases.

**RAG on conversation history** — Enterprise support chatbots (Intercom, Zendesk) embed and search past conversations. When a user returns, the system retrieves relevant past interactions to provide context.

**Memory databases** — Tools like Mem0 and Zep provide dedicated memory layers. They extract facts from conversations ("User prefers dark mode", "User's account is Enterprise tier") and store them as structured memories that persist across sessions.

**Knowledge graphs** — Google and enterprise AI platforms build entity-relationship graphs from conversations. Instead of storing raw text, they store structured relationships: "Confederation → happened in → 1867", "User → asked about → Confederation, Prime Ministers."

**Session storage** — Redis for ephemeral short-term memory (disappears after session), PostgreSQL for persistent history (stays forever), Pinecone or Weaviate for vector-searchable conversation archives.

### Comparison: What We Built vs Industry

| Feature | Our Implementation | Industry Production |
|---------|-------------------|-------------------|
| Short-term memory | Sliding window (last N turns) | Same, often with adaptive window sizing |
| Context preservation | LLM summarization | LLM summarization + entity extraction |
| Query rewriting | LLM-based rewrite | Same, often with fine-tuned rewrite models |
| Storage | JSON files | Redis, PostgreSQL, vector databases |
| Cross-session memory | Load/save JSON | Memory databases (Mem0, Zep) |
| Entity tracking | None | Knowledge graphs |
| User preferences | None | Extracted and stored as structured facts |
| Multi-user | None | Per-user memory isolation |
| Scale | Single machine | Distributed, sharded by user/session |

Our implementation covers the core patterns. The gap between this and production is mostly about persistence, scale, and structured extraction — not fundamentally different algorithms.

---

## Run It

```bash
npm run phase6
```

Make sure you have the required models and a built vector store from Phase 5:

```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1         # or your preferred chat model
npm run phase5 -- --mode ingest # build vector store first if needed
```

---

## Limitations

- No long-term memory across sessions (each conversation starts fresh unless manually loaded)
- No knowledge graph or entity extraction
- JSON file storage won't scale beyond a single user on a single machine
- Summarization quality depends on the model — small models produce poor summaries
- Token estimation is approximate (character count / 4), not precise tokenization
- No memory isolation between users

---

## What's Next

**Phase 7** adds agents that can take actions beyond just answering questions — searching the web, reading files, performing calculations, and calling external APIs. The chatbot evolves from a question-answering system into an autonomous assistant.
