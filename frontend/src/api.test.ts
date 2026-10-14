import { describe, expect, it } from "vitest";
import { createSseParser, deleteDocument, formatApiError } from "./api";
import { vi } from "vitest";

describe("createSseParser", () => {
  it("parses events split across arbitrary chunk boundaries", () => {
    const events: Array<{ event: string; data: unknown }> = [];
    const parser = createSseParser((event) => events.push(event));

    parser.push("event: token\ndata: {\"text\":\"hel");
    parser.push("lo\"}\n\neve");
    parser.push("nt: complete\ndata: {\"answer\":\"hello\"}\n\n");
    parser.finish();

    expect(events).toEqual([
      { event: "token", data: { text: "hello" } },
      { event: "complete", data: { answer: "hello" } },
    ]);
  });

  it("joins multiline data fields and ignores comment lines", () => {
    const events: Array<{ event: string; data: unknown }> = [];
    const parser = createSseParser((event) => events.push(event));

    parser.push(": keepalive\n:event\ndata: {\"parts\":\n");
    parser.push("data: [1,2]}\n\n");

    expect(events).toEqual([{ event: "message", data: { parts: [1, 2] } }]);
  });

  it("preserves a CRLF delimiter split across chunks", () => {
    const events: Array<{ event: string; data: unknown }> = [];
    const parser = createSseParser((event) => events.push(event));

    parser.push("event: token\r");
    parser.push("\ndata: {\"text\":\"ok\"}\r");
    parser.push("\n\r");
    parser.push("\n");
    parser.finish();

    expect(events).toEqual([{ event: "token", data: { text: "ok" } }]);
  });
});

describe("formatApiError", () => {
  it.each([
    ["plain text", "plain text"],
    ['{"detail":"invalid question"}', "invalid question"],
    ['{"detail":[{"msg":"required"}]}', '[{\"msg\":\"required\"}]'],
    ['{"message":"temporarily unavailable"}', "temporarily unavailable"],
  ])("surfaces %s response bodies", (body, expected) => {
    expect(formatApiError(422, "Unprocessable Entity", body).message).toBe(expected);
  });
});

describe("deleteDocument", () => {
  it("deletes a knowledge document with the DELETE method", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 204,
      statusText: "No Content",
      text: async () => "",
    });
    vi.stubGlobal("fetch", fetchMock);

    await deleteDocument(42);

    expect(fetchMock).toHaveBeenCalledWith("/api/v1/knowledge/documents/42", { method: "DELETE" });
    vi.unstubAllGlobals();
  });
});
