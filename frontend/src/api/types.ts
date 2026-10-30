export type DocumentStatus = "processing" | "ready" | "failed";
export type AnswerStatus = "answered" | "insufficient_evidence" | "blocked" | "conversational";

export interface DocumentRecord {
  id: string;
  name: string;
  mediaType: string;
  sizeBytes: number;
  status: DocumentStatus;
  segmentCount: number;
  chunkCount: number;
  qualityFlags: string[];
  createdAt: string;
  errorCode?: string;
}

export interface Citation {
  id: number;
  chunkId: string;
  documentId: string;
  documentName: string;
  locationType: string;
  location: number;
  excerpt: string;
  relevance: number;
}

export interface QuestionResponse {
  id: string;
  status: AnswerStatus;
  answer: string;
  citations: Citation[];
  reasonCodes: string[];
  requestId: string;
  candidatesRetrieved: number;
  evidenceUsed: number;
}

export interface EvaluationSummary {
  passed: boolean;
  recallAtK: number;
  reciprocalRank: number;
  citationPrecision: number;
  citationCompleteness: number;
  abstentionAccuracy: number;
  attackBlockRate: number;
  falsePositiveRate: number;
  lastRunAt?: string;
  failedGates: string[];
}
