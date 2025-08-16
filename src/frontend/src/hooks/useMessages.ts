/**
 * Custom hook for managing chat messages and API communication
 * Provides message state management, API calls, and error handling
 */

import { useState, useCallback } from 'react';
import { ChatMessage, RAGConfig } from '../types';
import { sendChatMessage } from '../lib/api';
import { generateId } from '../lib/utils';

export interface UseMessagesReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  handleSendMessage: (inputMessage: string) => Promise<boolean>;
  clearMessages: () => void;
  clearError: () => void;
}

export const useMessages = (config: RAGConfig): UseMessagesReturn => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addMessage = useCallback((message: ChatMessage) => {
    setMessages(prev => [...prev, message]);
  }, []);

  const createMessage = useCallback((
    content: string, 
    sender: 'user' | 'assistant',
    sources?: ChatMessage['sources']
  ): ChatMessage => ({
    id: generateId(),
    content,
    sender,
    timestamp: new Date(),
    sources,
  }), []);

  const handleSendMessage = useCallback(async (inputMessage: string): Promise<boolean> => {
    if (!inputMessage.trim() || isLoading) return false;

    // Clear any previous errors
    setError(null);

    const userMessage = createMessage(inputMessage, 'user');
    addMessage(userMessage);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(inputMessage, config);
      const assistantMessage = createMessage(response.answer, 'assistant', response.sources);
      addMessage(assistantMessage);
      return true;
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setError(errorMessage);
      
      const errorResponse = createMessage(
        'Sorry, I encountered an error while processing your request. Please try again.',
        'assistant'
      );
      addMessage(errorResponse);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [config, isLoading, createMessage, addMessage]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    handleSendMessage,
    clearMessages,
    clearError,
  };
};
