/**
 * Chat input component with message sending functionality
 * Handles keyboard shortcuts and loading states
 */

import React, { useState, useCallback, KeyboardEvent } from 'react';
import { Button } from './ui/button';
import { SendIcon } from './ui/icons';

interface ChatInputProps {
  onSendMessage: (message: string) => Promise<boolean>;
  isLoading: boolean;
  placeholder?: string;
}

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  isLoading,
  placeholder = "Ask me anything about your documents..."
}) => {
  const [inputMessage, setInputMessage] = useState('');

  const handleSendMessage = useCallback(async () => {
    if (!inputMessage.trim() || isLoading) return;
    
    const success = await onSendMessage(inputMessage);
    if (success) {
      setInputMessage('');
    }
  }, [inputMessage, isLoading, onSendMessage]);

  const handleKeyPress = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage]);

  return (
    <div className="border-t border-border-light bg-background-primary/80 backdrop-blur-sm p-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex space-x-4">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder={placeholder}
            className="flex-1 px-6 py-4 bg-background-secondary border border-border-medium rounded-2xl text-text-primary placeholder-text-muted focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm resize-none min-h-[56px] max-h-32"
            disabled={isLoading}
            rows={1}
            style={{
              scrollbarWidth: 'thin',
              scrollbarColor: '#94a3b8 #e2e8f0'
            }}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            loading={isLoading}
            size="icon"
            className="rounded-2xl h-[56px] w-[56px]"
            aria-label="Send message"
          >
            <SendIcon size={18} />
          </Button>
        </div>
        <div className="mt-2 text-xs text-text-muted text-center">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};

export default ChatInput;
