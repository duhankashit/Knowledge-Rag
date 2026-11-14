import type {
  ChatMessage,
  ChatResponse,
  ChunkingStrategy,
  ChunkingOptions,
  DocumentInfo,
} from "./types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "/api/v1").replace(
  /\/$/,
  "",
);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (error) {
    if (
      init?.signal?.aborted ||
      (error instanceof DOMException && error.name === "AbortError")
    ) {
      throw error;
    }
    throw new Error(
      "Couldn't reach the RAG API. Check that the backend is running and try again.",
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (body.detail !== undefined) {
        detail = JSON.stringify(body.detail);
      }
    } catch {
      // Keep the HTTP status message when the server does not return JSON.
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function getDocuments(signal?: AbortSignal): Promise<DocumentInfo[]> {
  return request<DocumentInfo[]>("/documents", { signal });
}

export function getChunkingOptions(signal?: AbortSignal): Promise<ChunkingOptions> {
  return request<ChunkingOptions>("/documents/chunking-options", { signal });
}

export function uploadDocument(
  file: File,
  options: {
    strategy: ChunkingStrategy;
    chunkSize: number;
    chunkOverlap: number;
    minChunkSize: number;
    similarityThreshold: number;
    maxChunkTokens: number;
    prompt: string;
    preserveStructure: boolean;
    windowSize: number;
    windowStride: number;
    embeddingModel: string;
    llmModel: string;
  },
): Promise<DocumentInfo> {
  const body = new FormData();
  body.append("file", file);
  body.append("chunking_strategy", options.strategy);
  body.append("chunk_size", String(options.chunkSize));
  body.append("chunk_overlap", String(options.chunkOverlap));
  body.append("min_chunk_size", String(options.minChunkSize));
  body.append("similarity_threshold", String(options.similarityThreshold));
  body.append("max_chunk_tokens", String(options.maxChunkTokens));
  body.append("chunking_prompt", options.prompt);
  body.append("preserve_structure", String(options.preserveStructure));
  body.append("window_size", String(options.windowSize));
  body.append("window_stride", String(options.windowStride));
  body.append("embedding_model", options.embeddingModel);
  body.append("llm_model", options.llmModel);
  return request<DocumentInfo>("/documents/upload", {
    method: "POST",
    body,
  });
}

export function deleteDocument(documentId: string): Promise<void> {
  return request<void>(`/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  });
}

export function sendMessage(
  question: string,
  history: ChatMessage[],
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
}
