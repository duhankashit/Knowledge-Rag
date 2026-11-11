import type { ChatMessage, ChatResponse, DocumentInfo } from "./types";

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

export function uploadDocument(file: File): Promise<DocumentInfo> {
  const body = new FormData();
  body.append("file", file);
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
