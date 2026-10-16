// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ConversationSummary, KnowledgeDocument } from "./types";

vi.mock("./api", () => ({
  createConversation: vi.fn(),
  deleteConversation: vi.fn(),
  deleteDocument: vi.fn(),
  documentFileUrl: (id: number) => `/api/v1/knowledge/documents/${id}/file`,
  getConversation: vi.fn(),
  ingestDocuments: vi.fn(),
  listConversations: vi.fn(),
  listDocuments: vi.fn(),
  sendMessageStream: vi.fn(),
  updateConversation: vi.fn(),
  uploadDocument: vi.fn(),
}));

import {
  deleteDocument,
  listConversations,
  listDocuments,
} from "./api";
import App from "./App";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const document: KnowledgeDocument = {
  id: 7,
  source_path: "guide.md",
  title: "Guide",
  type: "md",
  status: "ready",
  chunk_count: 1,
  updated_at: "2026-10-14T00:00:00Z",
  error_message: null,
};

const conversations: ConversationSummary[] = [];

function flushAsyncWork() {
  return new Promise<void>((resolve) => window.setTimeout(resolve, 0));
}

async function renderApp() {
  const container = window.document.createElement("div");
  window.document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => {
    root.render(<App />);
    await flushAsyncWork();
  });
  return { container, root };
}

beforeEach(() => {
  vi.mocked(listConversations).mockResolvedValue(conversations);
  vi.mocked(listDocuments).mockResolvedValue([document]);
});

afterEach(() => {
  window.document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("knowledge document deletion from the app", () => {
  it("shows the API error and restores the delete control after failure", async () => {
    let rejectDelete!: (error: Error) => void;
    vi.mocked(deleteDocument).mockReturnValue(new Promise<void>((_, reject) => {
      rejectDelete = reject;
    }));
    vi.spyOn(window, "confirm").mockReturnValue(true);

    const { container, root } = await renderApp();

    act(() => {
      (container.querySelector(".knowledge-link") as HTMLButtonElement).click();
    });
    expect(container.querySelector('[aria-label="删除 Guide"]')).not.toBeNull();

    act(() => {
      (container.querySelector('[aria-label="删除 Guide"]') as HTMLButtonElement).click();
    });
    expect(container.querySelector('[aria-label="正在删除 Guide"]')).toMatchObject({ disabled: true });

    await act(async () => {
      rejectDelete(new Error("删除接口不可用"));
      await flushAsyncWork();
    });

    expect(container.querySelector('[role="alert"]')?.textContent).toContain("删除接口不可用");
    const restoredButton = container.querySelector('[aria-label="删除 Guide"]') as HTMLButtonElement;
    expect(restoredButton).not.toBeNull();
    expect(restoredButton.disabled).toBe(false);
    expect(container.textContent).toContain("Guide");

    act(() => root.unmount());
  });
});
