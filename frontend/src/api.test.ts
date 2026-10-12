import { describe, expect, it } from "vitest";
import { createSseParser, formatApiError } from "./api";

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
