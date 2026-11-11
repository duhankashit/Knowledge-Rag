export type Page = "chat" | "files";

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

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
}
