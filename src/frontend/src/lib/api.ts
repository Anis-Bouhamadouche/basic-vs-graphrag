/**
 * API client for the RAG chat application
 * Handles all HTTP communication with the backend
 */

import { RAGConfig, ChatResponse, SourceDocument } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export interface ChatRequest {
  question: string;
  collection_name: string;
  top_k: number;
  temperature: number;
  max_tokens: number;
  chat_model_name: string;
  embedding_model: string;
}

export interface ApiChatResponse {
  answer: string;
  context_documents: string[];
  metadata: {
    model: string;
    embedding_model: string;
    collection: string;
    top_k: number;
    temperature: number;
    max_tokens: number;
  };
}

export interface HealthResponse {
  status: string;
  database: string;
}

class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Generic API request handler with error handling
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const config: RequestInit = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    throw new ApiError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Send a chat message to the RAG system
 */
export const sendChatMessage = async (
  message: string,
  config: RAGConfig
): Promise<ChatResponse> => {
  const requestBody: ChatRequest = {
    question: message,
    collection_name: config.collection_name,
    top_k: config.top_k,
    temperature: config.temperature,
    max_tokens: config.max_tokens,
    chat_model_name: config.chat_model_name,
    embedding_model: config.embedding_model,
  };

  const data = await apiRequest<ApiChatResponse>('/basic-rag/chat', {
    method: 'POST',
    body: JSON.stringify(requestBody),
  });

  // Transform API response to frontend format
  const sources: SourceDocument[] = data.context_documents.map((content, index) => ({
    content,
    score: 1.0 - (index * 0.1), // Mock score since API doesn't provide it
    metadata: {
      source: `Document ${index + 1}`,
      index,
    },
  }));

  return {
    answer: data.answer,
    sources,
  };
};

/**
 * Check API health status
 */
export const checkHealth = async (): Promise<HealthResponse> => {
  return apiRequest<HealthResponse>('/health');
};

/**
 * Export for testing and advanced usage
 */
export { ApiError, apiRequest };