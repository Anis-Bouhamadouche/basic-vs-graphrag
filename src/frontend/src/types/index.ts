export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: Date;
  sources?: SourceDocument[];
  ragType?: 'basic' | 'graph'; // Add RAG type
}

export interface SourceDocument {
  content: string;
  metadata: {
    source?: string;
    page?: number;
    title?: string;
    [key: string]: any;
  };
  score?: number;
}

export interface RAGConfig {
  top_k: number;
  temperature: number;
  max_tokens: number;
  collection_name: string;
  chat_model_name: string;
  embedding_model: string;
}

export interface ChatResponse {
  answer: string;
  sources?: SourceDocument[];
  metadata?: any;
}

export interface DualRAGResponse {
  basic: ChatResponse;
  graph: ChatResponse;
}
