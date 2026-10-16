// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { KnowledgeDocument } from "../types";
import { KnowledgePanel } from "./KnowledgePanel";

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

function renderPanel(overrides: Partial<React.ComponentProps<typeof KnowledgePanel>> = {}) {
  const container = window.document.createElement("div");
  window.document.body.appendChild(container);
  const root = createRoot(container);
  const onDelete = vi.fn();
  act(() => {
    root.render(
      <KnowledgePanel
        documents={[document]}
        loading={false}
        uploading={false}
        ingesting={false}
        ingestResult={null}
        error={null}
        onUpload={() => undefined}
        onIngest={() => undefined}
        onRefresh={() => undefined}
        onBack={() => undefined}
        deletingDocumentId={null}
        onDelete={onDelete}
        {...overrides}
      />,
    );
  });
  return { container, root, onDelete };
}

afterEach(() => {
  window.document.body.innerHTML = "";
  vi.restoreAllMocks();
});

describe("KnowledgePanel delete action", () => {
  it("confirms and sends the selected document to onDelete", () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const { container, root, onDelete } = renderPanel();

    act(() => {
      (container.querySelector('[aria-label="删除 Guide"]') as HTMLButtonElement).click();
    });

    expect(window.confirm).toHaveBeenCalledWith("确定删除“Guide”吗？删除后该文件及其索引将不再参与问答。");
    expect(onDelete).toHaveBeenCalledWith(document);
    act(() => root.unmount());
  });

  it("disables the delete button while the document is pending", () => {
    const { container, root } = renderPanel({ deletingDocumentId: 7 });
    const button = container.querySelector('[aria-label="正在删除 Guide"]') as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    act(() => root.unmount());
  });

  it("does not delete when confirmation is cancelled", () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const { container, root, onDelete } = renderPanel();

    act(() => {
      (container.querySelector('[aria-label="删除 Guide"]') as HTMLButtonElement).click();
    });

    expect(onDelete).not.toHaveBeenCalled();
    act(() => root.unmount());
  });
});
