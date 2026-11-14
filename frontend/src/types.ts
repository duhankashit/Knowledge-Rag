export type Page = "chat" | "files";
export type EvaluationFramework = "ragas" | "deepeval";
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
  evaluationQuestion?: string;
  generationLatencyMs?: number;
  evaluation?: EvaluationRun;
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
  generation_latency_ms: number;
}

export interface EvaluationSummary {
  retrieval: Record<string, number | null>;
  generation: Record<string, number | null>;
  safety: Record<string, number | null>;
  performance: {
    average_latency_ms: number | null;
    input_tokens: number;
    output_tokens: number;
    estimated_cost: number | null;
  };
}

export interface EvaluationCaseResult {
  case: number;
  question: string;
  expected_answer: string | null;
  answer: string;
  retrieved_contexts: string[];
  relevant_contexts: string[];
  retrieval: Record<string, number | null>;
  generation: Record<string, number | null>;
  safety: Record<string, number | null>;
  performance: {
    latency_ms: number;
    input_tokens: number;
    output_tokens: number;
    estimated_cost: number | null;
  };
  reasons: Record<string, string>;
  failed: boolean;
}

export interface EvaluationRun {
  run_id: string;
  created_at: string;
  framework: EvaluationFramework;
  case_count: number;
  failure_count: number;
  summary: EvaluationSummary;
  cases: EvaluationCaseResult[];
}
