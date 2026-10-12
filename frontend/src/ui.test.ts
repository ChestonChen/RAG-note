import { describe, expect, it } from "vitest";
import type { KnowledgeDocument, Message } from "./types";
import { canStartIngestion, findKnowledgeDocument, formatRelativeTime, getLastUserMessage, getQuestionForMessage, renderMarkdown } from "./ui";

describe("workspace presentation helpers", () => {
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

    expect(formatRelativeTime("2026-10-12T10:00:00.000Z", now)).toBe("just now");
    expect(formatRelativeTime("2026-10-12T09:30:00.000Z", now)).toBe("30m ago");
    expect(formatRelativeTime("2026-10-11T10:00:00.000Z", now)).toBe("yesterday");
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

  it("allows indexing an empty collection unless another operation is running", () => {
    expect(canStartIngestion(false, false)).toBe(true);
    expect(canStartIngestion(true, false)).toBe(false);
    expect(canStartIngestion(false, true)).toBe(false);
  });
});
