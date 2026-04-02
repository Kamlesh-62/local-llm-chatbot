# Phase 1: Foundations -- Talk to Ollama

> **Key source:** `src/phase1-chat.ts` (77 lines)

## Goal

Send prompts to a local LLM and get streaming responses back in the terminal.

## What You'll Learn

- How to connect to Ollama's chat API from TypeScript
- Streaming vs non-streaming responses
- The three message roles: system, user, assistant
- How conversation history gives an LLM "memory"
- Why that memory is fragile (in-memory only)

## How It Works

```
  You type a question
        |
        v
  +---------------------+
  | Added to messages[]  |  (role: "user")
  +---------------------+
        |
        v
  +---------------------+
  | Entire messages[]    |
  | sent to Ollama       |---> POST /api/chat
  +---------------------+
        |
        v
  +---------------------+
  | Tokens stream back   |  (printed as they arrive)
  +---------------------+
        |
        v
  +---------------------+
  | Full response added  |  (role: "assistant")
  | to messages[]        |
  +---------------------+
        |
        v
  Loop: wait for next question
```

## Key Concepts Explained

### Streaming vs Non-Streaming

| Mode | Behavior |
|------|----------|
| **Streaming** (what we use) | Tokens print to the terminal as they arrive, one by one. Feels fast and interactive. |
| **Non-streaming** | Waits until the entire response is generated, then prints it all at once. Feels slow, especially for long answers. |

Streaming is controlled by the `stream: true` option in the Ollama API call. Under the hood, the API sends a series of small JSON objects, each containing the next token.

### Message Roles

Every message in the conversation has a **role**:

- **`system`** -- Instructions that shape the LLM's behavior. Set once at the start. Example: *"You are a helpful assistant. Answer concisely."*
- **`user`** -- Your questions and prompts.
- **`assistant`** -- The LLM's responses.

The LLM sees all three roles and uses them to understand the conversation context.

### Conversation History

This is the key insight: **the entire `messages[]` array is sent on every API call.**

LLMs are stateless -- they don't remember previous calls. The only reason the model "knows" what you said earlier is because we keep sending the full history each time:

```
Call 1: [system, user1]                    --> assistant1
Call 2: [system, user1, assistant1, user2] --> assistant2
Call 3: [system, user1, assistant1, user2, assistant2, user3] --> assistant3
```

The array grows with every exchange.

### In-Memory Only

The `messages[]` array lives in a variable. When you quit the program, it's gone. There is no persistence -- every session starts fresh.

## Run It

```bash
npm run phase1
```

This starts an interactive chat loop. Type your questions, press Enter, and watch the response stream in. Type `exit` to quit.

## Model Choice Matters

The model you select dramatically affects behavior:

| Model | Type | Speed | Notes |
|-------|------|-------|-------|
| `deepseek-r1:7b` | Reasoning | Slow | Outputs `<think>` blocks showing its reasoning process. Interesting to watch but adds latency. |
| `llama3.2` | General purpose | Fast | Good balance of speed and quality for local use. |
| `gemma3:27b` | General purpose | Varies | Available on Ollama Cloud. Best quality of the three, but requires more resources. |

You can change the model in the source code or pass it as a configuration option.

## Limitations

- **Memory lost on quit** -- no persistence between sessions
- **Array grows forever** -- every message gets appended, eventually hitting the model's token limit (context window)
- **No document knowledge** -- the LLM only knows what's in its training data and the conversation history. It can't answer questions about your files.

## What's Next

Phase 2 adds file reading so the LLM can answer questions about your documents. Instead of relying only on training data, we inject document content directly into the conversation.
