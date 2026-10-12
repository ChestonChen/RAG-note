export type Role = "user" | "assistant";

export interface Citation {
  source: string;
  title: string;
  heading: string[];
  page_from: number | null;
  page_to: number | null;
  score: number;
  citation_id: string;
}

export interface SearchHit {
  chunk_id: number;
  content: string;
  source: string;
  title: string;
  heading: string[];
  page_from: number | null;
  page_to: number | null;
  score: number;
  vector_score: number;
  keyword_score: number;
  fusion_score: number;
  matched_channels: string[];
  rank: number | null;
  is_neighbor: boolean;
  matched_terms: string[];
  phrase_score: number;
}

export interface Message {
  id: number;
  conversation_id: number;
  role: Role;
  content: string;
  status: string;
  citations: Citation[];
  retrieval_debug: Record<string, unknown>;
  confidence: string | null;
  confidence_score: number | null;
  mode: string;
  web_sources: Array<{ title: string; url: string; snippet?: string | null }>;
  error_message: string | null;
  created_at: string;
}

export interface ConversationSummary {
  id: number;
  title: string;
  archived: boolean;
  created_at: string;
  updated_at: string;
}

export interface Conversation extends ConversationSummary {
  messages: Message[];
}

export interface KnowledgeDocument {
  id: number;
  source_path: string;
  title: string;
  type: string;
  status: string;
  chunk_count: number;
  updated_at: string;
  error_message: string | null;
}

export interface IngestResult {
  scanned_files: number;
  processed_files: number;
  skipped_files: number;
  failed_files: number;
  deleted_documents: number;
  chunk_count: number;
}

export interface StreamEventMap {
  message_start: { user_message_id: number; assistant_message_id: number | null };
  token: { text?: string; content?: string };
  metadata: {
    citations: Citation[];
    confidence: string;
    confidence_score: number;
    retrieval_debug: Record<string, unknown>;
    mode: string;
    web_sources: Array<{ title: string; url: string; snippet?: string | null }>;
  };
  complete: { assistant_message_id: number; answer: string };
  error: { error: string; user_message_id?: number };
}

export type StreamEventName = keyof StreamEventMap;

