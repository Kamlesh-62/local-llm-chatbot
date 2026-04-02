import { Ollama } from "ollama";
import * as readline from "readline";
import * as fs from "fs";
import * as path from "path";
import { parse } from "csv-parse/sync";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

const ollama = new Ollama();
const MODEL = "deepseek-r1:7b";

// ============================================
// STEP 1: File Readers for different formats
// ============================================

function readTextFile(filePath: string): string {
  return fs.readFileSync(filePath, "utf-8");
}

function readJsonFile(filePath: string): string {
  const raw = fs.readFileSync(filePath, "utf-8");
  const data = JSON.parse(raw);
  // Convert JSON to readable text so the LLM can understand it easily
  return JSON.stringify(data, null, 2);
}

function readCsvFile(filePath: string): string {
  const raw = fs.readFileSync(filePath, "utf-8");
  const records = parse(raw, { columns: true }) as Record<string, string>[];

  // Convert CSV rows into readable text
  // Why? LLMs understand natural text better than raw CSV
  const lines = records.map((row, i) => {
    const fields = Object.entries(row)
      .map(([key, val]) => `${key}: ${val}`)
      .join(", ");
    return `Row ${i + 1}: ${fields}`;
  });

  return `CSV Data (${records.length} rows):\n${lines.join("\n")}`;
}

async function readPdfFile(filePath: string): Promise<string> {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const pdf = await getDocument({ data }).promise;

  const textParts: string[] = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item: any) => item.str)
      .join(" ");
    textParts.push(pageText);
  }

  return textParts.join("\n\n");
}

function readMarkdownFile(filePath: string): string {
  // Markdown is already text - just read it directly
  return fs.readFileSync(filePath, "utf-8");
}

// ============================================
// STEP 2: Auto-detect file type and read
// ============================================

async function readDocument(filePath: string): Promise<{ content: string; type: string }> {
  const ext = path.extname(filePath).toLowerCase();

  switch (ext) {
    case ".txt":
      return { content: readTextFile(filePath), type: "text" };
    case ".json":
      return { content: readJsonFile(filePath), type: "json" };
    case ".csv":
      return { content: readCsvFile(filePath), type: "csv" };
    case ".md":
      return { content: readMarkdownFile(filePath), type: "markdown" };
    case ".pdf":
      return { content: await readPdfFile(filePath), type: "pdf" };
    default:
      // Try to read as plain text
      return { content: readTextFile(filePath), type: "unknown" };
  }
}

// ============================================
// STEP 3: Chat with file context
// ============================================

const messages: { role: "system" | "user" | "assistant"; content: string }[] =
  [];

async function chat(userMessage: string): Promise<string> {
  messages.push({ role: "user", content: userMessage });
  console.log("  [Message sent. Waiting for model...]");

  const startTime = Date.now();

  // Show a spinner while waiting for the model to respond
  const spinnerFrames = ["|", "/", "-", "\\"];
  let spinnerIndex = 0;
  const spinner = setInterval(() => {
    process.stdout.write(`\r  Thinking... ${spinnerFrames[spinnerIndex++ % spinnerFrames.length]}`);
  }, 100);

  const stream = await ollama.chat({
    model: MODEL,
    messages,
    stream: true,
  });

  // Stop spinner and show response
  clearInterval(spinner);
  process.stdout.write("\r                    \r"); // clear spinner line

  let reply = "";
  process.stdout.write("Assistant: ");
  for await (const chunk of stream) {
    const text = chunk.message.content;
    process.stdout.write(text);
    reply += text;
  }
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n  [Done in ${elapsed}s]\n`);

  messages.push({ role: "assistant", content: reply });
  return reply;
}

// ============================================
// STEP 4: Main - Load file, then chat about it
// ============================================

async function main() {
  console.log(`\n--- Phase 2: File Reader Chat (${MODEL}) ---\n`);

  // List available sample docs
  const docsDir = path.join(process.cwd(), "sample-docs");
  const files = fs.readdirSync(docsDir);
  console.log("Available documents:");
  files.forEach((f, i) => console.log(`  ${i + 1}. ${f}`));
  console.log();

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const prompt = (q: string): Promise<string> =>
    new Promise((resolve) => rl.question(q, resolve));

  // Pick a file
  const choice = await prompt("Enter file number (or full path): ");
  let filePath: string;

  const num = parseInt(choice);
  if (!isNaN(num) && num >= 1 && num <= files.length) {
    filePath = path.join(docsDir, files[num - 1]!);
  } else {
    filePath = choice.trim();
  }

  // Read the file
  if (!fs.existsSync(filePath)) {
    console.log(`File not found: ${filePath}`);
    rl.close();
    return;
  }

  console.log(`\nReading: ${filePath}`);
  const doc = await readDocument(filePath);
  console.log(`Type: ${doc.type} | Length: ${doc.content.length} characters\n`);

  // Show a preview (first 300 chars)
  console.log("--- Preview ---");
  console.log(doc.content.slice(0, 300) + (doc.content.length > 300 ? "\n..." : ""));
  console.log("--- End Preview ---\n");

  // KEY CONCEPT: Inject the file content into the system prompt
  // This is the simplest form of RAG - just stuff the context into the prompt
  messages.push({
    role: "system",
    content: `You are a helpful assistant. Answer questions based ONLY on the following document content. If the answer is not in the document, say "I don't find that information in the document."

--- DOCUMENT START ---
${doc.content}
--- DOCUMENT END ---`,
  });

  // Token limit awareness
  const approxTokens = Math.ceil(doc.content.length / 4); // rough estimate: 1 token ≈ 4 chars
  console.log(`Approximate tokens used for context: ~${approxTokens}`);
  if (approxTokens > 4000) {
    console.log(
      "WARNING: Large document! May hit token limits. Phase 3 (chunking) will solve this.\n"
    );
  }
  console.log('Ask questions about the document. Type "quit" to exit.\n');

  // Chat loop
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
