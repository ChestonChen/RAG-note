import type {
  Conversation,
  ConversationSummary,
  IngestResult,
  KnowledgeDocument,
  StreamEventMap,
  StreamEventName,
} from "./types";

const API_PREFIX = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_PREFIX}${path}`, init);
  if (!response.ok) {
    throw formatApiError(response.status, response.statusText, await response.text());
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** Convert any FastAPI/plain-text error body into a useful browser error. */
export function formatApiError(status: number, statusText: string, body: string): Error {
  const fallback = `${status}${statusText ? ` ${statusText}` : ""}`;
  const raw = body.trim();
  if (!raw) return new Error(fallback);
  try {
    const payload: unknown = JSON.parse(raw);
    if (typeof payload === "string") return new Error(payload);
    if (payload && typeof payload === "object" && "detail" in payload) {
      const detail = (payload as { detail: unknown }).detail;
      if (typeof detail === "string") return new Error(detail);
      if (detail !== undefined) return new Error(JSON.stringify(detail));
    }
    if (payload && typeof payload === "object" && "message" in payload) {
      const message = (payload as { message: unknown }).message;
      if (typeof message === "string") return new Error(message);
      if (message !== undefined) return new Error(JSON.stringify(message));
    }
  } catch {
    // Keep a plain-text response as-is.
  }
  return new Error(raw || fallback);
}

export function listConversations(): Promise<ConversationSummary[]> {
  return request<ConversationSummary[]>("/v1/conversations");
}

export function createConversation(title = "New conversation"): Promise<Conversation> {
  return request<Conversation>("/v1/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export function getConversation(id: number): Promise<Conversation> {
  return request<Conversation>(`/v1/conversations/${id}`);
}

export function updateConversation(
  id: number,
  patch: { title?: string; archived?: boolean },
): Promise<Conversation> {
  return request<Conversation>(`/v1/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

export function deleteConversation(id: number): Promise<void> {
  return request<void>(`/v1/conversations/${id}`, { method: "DELETE" });
}

export function createSseParser(
  onEvent: (event: { event: string; data: unknown }) => void,
) {
  let buffer = "";
  let eventName = "message";
  let dataLines: string[] = [];

  const dispatch = () => {
    if (!dataLines.length) return;
    const raw = dataLines.join("\n");
    try {
      onEvent({ event: eventName, data: JSON.parse(raw) });
    } catch {
      onEvent({ event: eventName, data: raw });
    }
    eventName = "message";
    dataLines = [];
  };

  const consumeLine = (line: string) => {
    if (line === "") {
      dispatch();
      return;
    }
    if (line.startsWith(":")) return;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    let value = separator === -1 ? "" : line.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") eventName = value;
    if (field === "data") dataLines.push(value);
  };

  return {
    push(chunk: string) {
      buffer += chunk;
      while (true) {
        const lf = buffer.indexOf("\n");
        const cr = buffer.indexOf("\r");
        let end = -1;
        let width = 1;
        if (lf >= 0 && (cr < 0 || lf < cr)) end = lf;
        else if (cr >= 0) {
          // A CR at the end of a transport chunk may be the first half of CRLF.
          if (cr === buffer.length - 1) break;
          end = cr;
          width = buffer[cr + 1] === "\n" ? 2 : 1;
        }
        if (end < 0) break;
        consumeLine(buffer.slice(0, end));
        buffer = buffer.slice(end + width);
      }
    },
    finish() {
      if (buffer) consumeLine(buffer);
      dispatch();
    },
  };
}

export async function sendMessageStream(
  conversationId: number,
  question: string,
  onEvent: <T extends StreamEventName>(event: T, data: StreamEventMap[T]) => void,
  signal?: AbortSignal,
  allowWebFallback = true,
): Promise<void> {
  const response = await fetch(`${API_PREFIX}/v1/conversations/${conversationId}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, allow_web_fallback: allowWebFallback }),
    signal,
  });
  if (!response.ok) {
    throw formatApiError(response.status, response.statusText, await response.text());
  }
  if (!response.body) throw new Error("The server did not return a stream");
  const decoder = new TextDecoder();
  const knownEvents = new Set<StreamEventName>(["message_start", "token", "metadata", "complete", "error"]);
  const parser = createSseParser(({ event, data }) => {
    if (!knownEvents.has(event as StreamEventName) || typeof data !== "object" || data === null) return;
    onEvent(event as StreamEventName, data as never);
  });
  const reader = response.body.getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.finish();
  } finally {
    reader.releaseLock();
  }
}

export function listDocuments(): Promise<KnowledgeDocument[]> {
  return request<KnowledgeDocument[]>("/v1/knowledge/documents");
}

export async function uploadDocument(file: File): Promise<{ source_path: string; filename: string; size: number; status: string }> {
  const form = new FormData();
  form.append("file", file);
  return request("/v1/knowledge/upload", { method: "POST", body: form });
}

export function ingestDocuments(force = false): Promise<IngestResult> {
  return request<IngestResult>("/v1/knowledge/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  });
}

export function documentFileUrl(id: number): string {
  return `${API_PREFIX}/v1/knowledge/documents/${id}/file`;
}
