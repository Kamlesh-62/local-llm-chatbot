import { Ollama } from "ollama";
import * as readline from "node:readline";

const ollama = new Ollama();
const MODEL = "deepseek-r1:7b";

// Conversation history - this is how the LLM remembers context
const messages: { role: "system" | "user" | "assistant"; content: string }[] = [
  {
    role: "system",
    content:
      "You are a helpful assistant. Answer questions clearly and concisely.",
  },
];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function prompt(question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => resolve(answer));
  });
}

async function chat(userMessage: string): Promise<string> {

  // Add user message to history
  messages.push({ role: "user", content: userMessage });

  // --- Streaming version ---
  const stream = await ollama.chat({
    model: MODEL,
    messages,
    stream: true,
  });

  let reply = "";
  process.stdout.write("\nAssistant: ");
  for await (const chunk of stream) {
    const text = chunk.message.content;
    process.stdout.write(text);
    reply += text;
  }
  console.log("\n");

  // Add assistant reply to history so LLM remembers it next turn
  messages.push({ role: "assistant", content: reply });

  return reply;
}

async function main() {
  console.log(`\n--- Phase 1: Chat with ${MODEL} ---`);
  console.log('Type "quit" to exit.\n');

  while (true) {
    const userInput = await prompt("You: ");

    if (userInput.trim().toLowerCase() === "quit") {
      console.log("Goodbye!");
      rl.close();
      break;
    }

    if (!userInput.trim()) continue;

    await chat(userInput);
  }
}

main();
