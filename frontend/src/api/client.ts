import type { DocumentRecord, EvaluationSummary, QuestionResponse } from "./types";

class ApiError extends Error {
  constructor(message: string, public readonly status: number, public readonly code?: string, public readonly requestId?: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Never leave the chat spinner running forever when a local model or API stalls.
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30_000);
  let response: Response;
  try {
    const headers = new Headers(init?.headers);
    const token = localStorage.getItem("atlas_token");
    if (token) headers.set("Authorization", `Bearer ${token}`);
    response = await fetch(path, { ...init, headers, signal: init?.signal ?? controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError("The request timed out. Check that the backend is running and try again.", 408);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
  if (!response.ok) {
    if (response.status === 401 && path !== "/api/v1/auth/login") window.dispatchEvent(new Event("atlas-session-expired"));
    const problem = await response.json().catch(() => ({}));
    throw new ApiError(problem.detail ?? "The request could not be completed.", response.status, problem.code, problem.request_id);
  }
  return response.json() as Promise<T>;
}

export const api = {
  cacheSummary: () => request<{ answer: { items: number; hits: number; misses: number } }>("/api/v1/observability/cache"),
  login: (username: string, password: string) => request<{ access_token: string }>("/api/v1/auth/login", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }),
  }),
  listDocuments: () => request<{ documents: DocumentRecord[] }>("/api/v1/documents"),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ document: DocumentRecord }>("/api/v1/documents", { method: "POST", body: form });
  },
  askQuestion: (question: string, documentIds: string[]) => request<QuestionResponse>("/api/v1/questions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, document_ids: documentIds }),
  }),
  getEvaluation: () => request<EvaluationSummary>("/api/v1/evaluation/summary"),
};

export { ApiError };
