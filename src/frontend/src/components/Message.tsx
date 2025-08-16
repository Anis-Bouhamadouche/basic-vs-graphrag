import React from 'react';
import { ChatMessage, SourceDocument } from '../types';
import { BotIcon, UserIcon, DocumentIcon } from './ui/icons';

interface MessageProps {
  message: ChatMessage;
  onShowSources: (sources: SourceDocument[]) => void;
}

const Message: React.FC<MessageProps> = ({ message, onShowSources }) => {
  const isUser = message.sender === 'user';
  
  // Get RAG type styling
  const getRagTypeStyle = (ragType?: string) => {
    if (!ragType) return '';
    return ragType === 'basic' 
      ? 'border-l-4 border-blue-500' 
      : 'border-l-4 border-green-500';
  };

  const getRagTypeLabel = (ragType?: string) => {
    if (!ragType) return '';
    return ragType === 'basic' ? 'Basic RAG' : 'Graph RAG';
  };

  return (
    <div className="w-full">
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
        <div className={`max-w-2xl ${
          isUser 
            ? 'bg-gray-800 text-white' 
            : `bg-gray-100 text-gray-900 ${getRagTypeStyle(message.ragType)}`
        } rounded-2xl px-6 py-4`}>
          {/* RAG Type Label */}
          {!isUser && message.ragType && (
            <div className="mb-2">
              <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full ${
                message.ragType === 'basic' 
                  ? 'bg-blue-100 text-blue-800' 
                  : 'bg-green-100 text-green-800'
              }`}>
                {getRagTypeLabel(message.ragType)}
              </span>
            </div>
          )}
          
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center ${
                isUser 
                  ? 'bg-gray-700' 
                  : message.ragType === 'basic'
                    ? 'bg-blue-600'
                    : message.ragType === 'graph'
                      ? 'bg-green-600'
                      : 'bg-primary-900'
              }`}>
                {isUser ? (
                  <UserIcon size={14} className="text-white" />
                ) : (
                  <BotIcon size={14} className="text-white" />
                )}
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {message.content}
              </p>
              {message.sources && message.sources.length > 0 && (
                <button
                  onClick={() => onShowSources(message.sources!)}
                  className={`mt-3 inline-flex items-center space-x-2 text-xs transition-colors ${
                    isUser 
                      ? 'text-gray-300 hover:text-gray-200' 
                      : 'text-gray-600 hover:text-gray-800'
                  }`}
                  aria-label={`Show ${message.sources.length} source documents`}
                >
                  <DocumentIcon size={14} />
                  <span>Show Context ({message.sources.length} sources)</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
      <div className={`mt-1 text-xs text-text-muted ${
        isUser ? 'text-right' : 'text-left'
      } px-4`}>
        {message.timestamp.toLocaleTimeString()}
      </div>
    </div>
  );
};

export default Message;
