import { describe, expect, it } from "vitest";
import type { KnowledgeDocument, Message } from "./types";
import {
  canStartIngestion,
  createOptimisticMessagePair,
  findKnowledgeDocument,
  formatRelativeTime,
  getLastUserMessage,
  getQuestionForMessage,
  getDocumentStatusLabel,
  getEvidencePanelClassName,
  getLatestAssistantMessage,
  getMessageBubbleClassName,
  getMessageRowClassName,
  chatThemeTokens,
  knowledgeWorkflowCopy,
  markStreamError,
  patchMessage,
  replaceMessageId,
  renderMarkdown,
  waitForPaint,
} from "./ui";

describe("workspace presentation helpers", () => {
  it("maps message roles to stable chat layout classes", () => {
    expect(getMessageRowClassName("user")).toBe("message-row user");
    expect(getMessageRowClassName("assistant")).toBe("message-row assistant");
    expect(getMessageBubbleClassName("user")).toBe("message-bubble user-bubble");
    expect(getMessageBubbleClassName("assistant")).toBe("message-bubble assistant-bubble");
  });

  it("exposes a restrained blue chat theme", () => {
    expect(chatThemeTokens.accent).toMatch(/^#(?:[0-9a-f]{6})$/i);
    expect(chatThemeTokens.accent.toLowerCase()).not.toBe("#227653");
    expect(chatThemeTokens.userBubble).toMatch(/^#(?:[0-9a-f]{6})$/i);
  });

  it("keeps the evidence drawer closed until it is explicitly opened", () => {
    expect(getEvidencePanelClassName(false)).toBe("evidence-pane");
    expect(getEvidencePanelClassName(true)).toBe("evidence-pane is-open");
  });

  it("finds the latest assistant answer for the evidence drawer", () => {
    const messages = [
      { id: 1, role: "user", content: "first" },
      { id: 2, role: "assistant", content: "first answer" },
      { id: 3, role: "user", content: "latest" },
      { id: 4, role: "assistant", content: "latest answer" },
    ] as Message[];

    expect(getLatestAssistantMessage(messages)?.id).toBe(4);
    expect(getLatestAssistantMessage([])).toBeNull();
  });

  it("exposes a clear Chinese knowledge-base workflow", () => {
    expect(knowledgeWorkflowCopy.uploadPrimary).toBe("上传文件到知识库");
    expect(knowledgeWorkflowCopy.ingest).toBe("开始建立索引");
    expect(knowledgeWorkflowCopy.indexHint).toContain("文件才会参与问答");
    expect(getDocumentStatusLabel("ready")).toBe("已就绪");
    expect(getDocumentStatusLabel("pending")).toBe("待建立索引");
    expect(getDocumentStatusLabel("custom-status")).toBe("custom-status");
  });

  it("renders headings, emphasis, links, and fenced code safely", () => {
    const html = renderMarkdown("## Ready\n\n**bold** and `code`\n\n```ts\nconst answer = 42;\n```");

    expect(html).toContain("<h2>Ready</h2>");
    expect(html).toContain("<strong>bold</strong>");
    expect(html).toContain("<code>code</code>");
    expect(html).toContain("const answer = 42;");
    expect(html).not.toContain("<script");
  });

  it("uses stable labels for recent timestamps", () => {
    const now = new Date("2026-10-12T10:00:00.000Z");

    expect(formatRelativeTime("2026-10-12T10:00:00.000Z", now)).toBe("刚刚");
    expect(formatRelativeTime("2026-10-12T09:30:00.000Z", now)).toBe("30 分钟前");
    expect(formatRelativeTime("2026-10-11T10:00:00.000Z", now)).toBe("昨天");
  });

  it("matches local citation paths to indexed documents", () => {
    const documents = [
      { id: 7, source_path: "guides/rag.md" },
      { id: 8, source_path: "paper.pdf" },
    ] as KnowledgeDocument[];

    expect(findKnowledgeDocument("guides\\rag.md", documents)?.id).toBe(7);
    expect(findKnowledgeDocument("knowledge/paper.pdf", documents)?.id).toBe(8);
    expect(findKnowledgeDocument("missing.md", documents)).toBeNull();
  });

  it("finds the latest user message for a stream retry", () => {
    const messages = [
      { role: "user", content: "first" },
      { role: "assistant", content: "answer" },
      { role: "user", content: "latest" },
    ] as Message[];

    expect(getLastUserMessage(messages)?.content).toBe("latest");
    expect(getLastUserMessage([])).toBeNull();
  });

  it("retries the question that belongs to the selected assistant answer", () => {
    const messages = [
      { id: 1, role: "user", content: "first question" },
      { id: 2, role: "assistant", content: "first answer" },
      { id: 3, role: "user", content: "second question" },
      { id: 4, role: "assistant", content: "second answer" },
    ] as Message[];

    expect(getQuestionForMessage(messages, 2)?.content).toBe("first question");
    expect(getQuestionForMessage(messages, 4)?.content).toBe("second question");
    expect(getQuestionForMessage(messages, 1)).toBeNull();
  });

  it("only enables indexing when files are available and no operation is running", () => {
    expect(canStartIngestion(false, false)).toBe(true);
    expect(canStartIngestion(true, false)).toBe(false);
    expect(canStartIngestion(false, true)).toBe(false);
    expect(canStartIngestion(false, false, false)).toBe(false);
  });

  it("creates visible user and assistant placeholders before a stream starts", () => {
    const pair = createOptimisticMessagePair("Explain vectors", 12, -10, -11, "2026-10-12T10:00:00.000Z");

    expect(pair.user).toMatchObject({
      id: -10,
      conversation_id: 12,
      role: "user",
      content: "Explain vectors",
      status: "pending",
    });
    expect(pair.assistant).toMatchObject({
      id: -11,
      conversation_id: 12,
      role: "assistant",
      content: "",
      status: "streaming",
    });
  });

  it("replaces temporary ids without duplicating a persisted stream message", () => {
    const pair = createOptimisticMessagePair("Explain vectors", 12, -10, -11, "2026-10-12T10:00:00.000Z");
    const messages = replaceMessageId([pair.user, pair.assistant], -10, 101);

    expect(messages.map((message) => message.id)).toEqual([101, -11]);
    const withPersisted = replaceMessageId([pair.user, pair.assistant, { ...pair.user, id: 101, status: "complete" }], -10, 101);
    expect(withPersisted).toHaveLength(2);
    expect(withPersisted.find((message) => message.id === 101)?.status).toBe("complete");
  });

  it("patches the same assistant placeholder as tokens and metadata arrive", () => {
    const pair = createOptimisticMessagePair("Explain vectors", 12, -10, -11, "2026-10-12T10:00:00.000Z");
    const withToken = patchMessage([pair.user, pair.assistant], -11, { content: "Vectors", status: "streaming" });
    const completed = patchMessage(withToken, -11, { content: "Vectors are useful.", status: "complete", confidence: "high" });

    expect(completed).toHaveLength(2);
    expect(completed[1]).toMatchObject({ id: -11, content: "Vectors are useful.", status: "complete", confidence: "high" });
  });

  it("marks only the active stream pair as interrupted without changing timeline length", () => {
    const pair = createOptimisticMessagePair("Explain vectors", 12, -10, -11, "2026-10-12T10:00:00.000Z");
    const failed = markStreamError([pair.user, pair.assistant], -10, -11, "Request stopped");

    expect(failed).toHaveLength(2);
    expect(failed[0]).toMatchObject({ id: -10, status: "error", error_message: "Request stopped" });
    expect(failed[1]).toMatchObject({ id: -11, status: "error", error_message: "Request stopped" });
  });

  it("yields asynchronously so optimistic messages can paint before network work", async () => {
    let settled = false;
    const pending = waitForPaint().then(() => { settled = true; });

    expect(settled).toBe(false);
    await pending;
    expect(settled).toBe(true);
  });
});
