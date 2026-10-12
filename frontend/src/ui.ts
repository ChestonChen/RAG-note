import type { KnowledgeDocument, Message } from "./types";

const escapeHtml = (value: string) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

export function renderMarkdown(source: string): string {
  const lines = source.replaceAll("\r\n", "\n").split("\n");
  const blocks: string[] = [];
  let paragraph: string[] = [];
  let code: string[] | null = null;

  const inline = (value: string) => {
    let rendered = escapeHtml(value);
    rendered = rendered.replace(/`([^`]+)`/g, "<code>$1</code>");
    rendered = rendered.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    rendered = rendered.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    rendered = rendered.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    return rendered;
  };

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(`<p>${inline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };

  for (const line of lines) {
    if (line.startsWith("```")) {
      if (code) {
        blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
        code = null;
      } else {
        flushParagraph();
        code = [];
      }
      continue;
    }
    if (code) {
      code.push(line);
      continue;
    }
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      flushParagraph();
      const level = heading[1].length;
      blocks.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else if (!line.trim()) {
      flushParagraph();
    } else {
      paragraph.push(line.trim());
    }
  }
  if (code) blocks.push(`<pre><code>${escapeHtml(code.join("\n"))}</code></pre>`);
  flushParagraph();
  return blocks.join("");
}

export function formatRelativeTime(value: string, now = new Date()): string {
  const time = new Date(value).getTime();
  const delta = Math.max(0, now.getTime() - time);
  const minutes = Math.floor(delta / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  if (hours < 48) return "yesterday";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
}

function normalizeSourcePath(value: string): string {
  return value.replaceAll("\\", "/").replace(/^\.\//, "").replace(/^\/+/, "").toLowerCase();
}

export function findKnowledgeDocument(source: string, documents: KnowledgeDocument[]): KnowledgeDocument | null {
  const target = normalizeSourcePath(source);
  return documents.find((document) => {
    const indexed = normalizeSourcePath(document.source_path);
    return indexed === target || target.endsWith(`/${indexed}`) || indexed.endsWith(`/${target}`);
  }) ?? null;
}

export function getLastUserMessage(messages: Message[]): Message | null {
  return [...messages].reverse().find((message) => message.role === "user") ?? null;
}

export function getQuestionForMessage(messages: Message[], messageId: number): Message | null {
  const messageIndex = messages.findIndex((message) => message.id === messageId);
  if (messageIndex < 0 || messages[messageIndex].role !== "assistant") return null;
  return [...messages.slice(0, messageIndex)].reverse().find((message) => message.role === "user") ?? null;
}

export function canStartIngestion(ingesting: boolean, uploading: boolean): boolean {
  return !ingesting && !uploading;
}
