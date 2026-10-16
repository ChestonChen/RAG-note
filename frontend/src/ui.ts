import type { KnowledgeDocument, Message } from "./types";

export const knowledgeWorkflowCopy = {
  uploadPrimary: "上传文件到知识库",
  uploadFirst: "上传第一个文件",
  upload: "上传文件",
  ingest: "开始建立索引",
  indexHint: "上传后点击建立索引，文件才会参与问答。",
  supportedFormats: "支持 Markdown 和 PDF",
} as const;

export const chatThemeTokens = {
  accent: "#4d6bfe",
  accentDark: "#3555d6",
  accentSoft: "#eef3ff",
  accentBorder: "#d5e0ff",
  userBubble: "#eef3ff",
  userBubbleBorder: "#d5e0ff",
} as const;

export function getMessageRowClassName(role: "user" | "assistant"): string {
  return `message-row ${role}`;
}

export function getMessageBubbleClassName(role: "user" | "assistant"): string {
  return `message-bubble ${role === "user" ? "user-bubble" : "assistant-bubble"}`;
}

export function getEvidencePanelClassName(open: boolean): string {
  return `evidence-pane${open ? " is-open" : ""}`;
}

export function getLatestAssistantMessage(messages: Message[]): Message | null {
  return [...messages].reverse().find((message) => message.role === "assistant") ?? null;
}

const documentStatusLabels: Record<string, string> = {
  ready: "已就绪",
  pending: "待建立索引",
  indexing: "正在建立索引",
  processing: "正在处理",
  failed: "失败",
  error: "失败",
  uploaded: "已上传",
};

export function getDocumentStatusLabel(status: string): string {
  return documentStatusLabels[status.toLowerCase()] ?? status;
}

const confidenceLabels: Record<string, string> = {
  high: "高",
  medium: "中",
  low: "低",
};

export function getConfidenceLabel(confidence: string | null): string {
  if (!confidence) return "未评级";
  return confidenceLabels[confidence.toLowerCase()] ?? confidence;
}

const retrievalModeLabels: Record<string, string> = {
  hybrid: "混合检索",
  "hybrid retrieval": "混合检索",
  kb_only: "知识库检索",
  kb_plus_web: "知识库 + 联网",
  web_only: "联网检索",
  vector: "向量检索",
  keyword: "关键词检索",
  exact: "精确匹配",
};

export function getRetrievalModeLabel(mode: string | null): string {
  if (!mode) return "混合检索";
  return retrievalModeLabels[mode.toLowerCase()] ?? mode;
}

const retrievalChannelLabels: Record<string, string> = {
  vector: "向量",
  keyword: "关键词",
  exact: "精确",
  phrase: "短语",
  neighbor: "邻近分片",
};

export function getRetrievalChannelLabel(channel: string): string {
  return retrievalChannelLabels[channel.toLowerCase()] ?? channel;
}

export interface OptimisticMessagePair {
  user: Message;
  assistant: Message;
}

/** Build messages that can be rendered while the conversation/stream is still being created. */
export function createOptimisticMessagePair(
  question: string,
  conversationId: number,
  userId: number,
  assistantId: number,
  createdAt = new Date().toISOString(),
): OptimisticMessagePair {
  const common = {
    conversation_id: conversationId,
    citations: [],
    retrieval_debug: {},
    confidence: null,
    confidence_score: null,
    mode: "",
    web_sources: [],
    error_message: null,
    created_at: createdAt,
  };
  return {
    user: { ...common, id: userId, role: "user", content: question, status: "pending" as const },
    assistant: { ...common, id: assistantId, role: "assistant", content: "", status: "streaming" as const },
  };
}

/** Patch one message while preserving the order and identity of the conversation timeline. */
export function patchMessage(messages: Message[], id: number, patch: Partial<Message>): Message[] {
  return messages.map((message) => (message.id === id ? { ...message, ...patch } : message));
}

/** Mark the optimistic pair when a stream is interrupted, preserving its order. */
export function markStreamError(
  messages: Message[],
  userId: number,
  assistantId: number,
  error: string,
): Message[] {
  return messages.map((message) => (
    message.id === userId || message.id === assistantId
      ? { ...message, status: "error", error_message: error }
      : message
  ));
}

/** Replace a temporary id with a server id, dropping an already-present duplicate. */
export function replaceMessageId(messages: Message[], temporaryId: number, persistedId: number): Message[] {
  if (temporaryId === persistedId) return messages;
  const hasTemporary = messages.some((message) => message.id === temporaryId);
  if (!hasTemporary) return messages;
  if (messages.some((message) => message.id === persistedId)) {
    return messages.filter((message) => message.id !== temporaryId);
  }
  return messages.map((message) => (message.id === temporaryId ? { ...message, id: persistedId } : message));
}

/** Insert or replace one message without allowing repeated stream events to duplicate it. */
export function upsertMessage(messages: Message[], message: Message): Message[] {
  const index = messages.findIndex((entry) => entry.id === message.id);
  if (index < 0) return [...messages, message];
  return messages.map((entry, entryIndex) => (entryIndex === index ? message : entry));
}

/** Let React commit optimistic state and let the browser paint before I/O starts. */
export function waitForPaint(): Promise<void> {
  if (typeof window !== "undefined" && typeof window.requestAnimationFrame === "function") {
    return new Promise((resolve) => window.requestAnimationFrame(() => resolve()));
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

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
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  if (hours < 48) return "昨天";
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric" }).format(new Date(value));
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

export function canStartIngestion(ingesting: boolean, uploading: boolean, hasDocuments = true): boolean {
  return !ingesting && !uploading && hasDocuments;
}
