export type Page = "chat" | "files";
export type ChunkingStrategy =
  | "basic"
  | "recursive"
  | "sentence"
  | "paragraph"
  | "semantic"
  | "agentic"
  | "sliding_window";

export interface ChatSource {
  filename: string;
  chunk: number;
  excerpt: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  chunk_count: number;
  uploaded_at: string;
}

export interface ChunkingOptions {
  embedding_models: string[];
  llm_models: string[];
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
}
