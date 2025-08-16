import React, { useState, useRef, useEffect, useCallback } from 'react';
import { RAGConfig, SourceDocument } from './types';
import { useMessages } from './hooks/useMessages';
import SettingsSidebar from './components/SettingsSidebar';
import Message from './components/Message';
import LoadingMessage from './components/LoadingMessage';
import ChatInput from './components/ChatInput';
import { SettingsIcon, BotIcon, CloseIcon } from './components/ui/icons';

// Constants
const DEFAULT_CONFIG: RAGConfig = {
  top_k: 3,
  temperature: 0.1,
  max_tokens: 1000,
  collection_name: 'eu_ai_act',
  chat_model_name: 'gpt-4.1-mini',
  embedding_model: 'text-embedding-3-large',
};

const WELCOME_MESSAGE = {
  title: 'Welcome to Dual RAG Comparison',
  description: 'Ask me anything about your documents. I\'ll show you responses from both Basic RAG and Graph RAG side by side.',
};

const App: React.FC = () => {
  // State management
  const [config, setConfig] = useState<RAGConfig>(DEFAULT_CONFIG);
  const [showSettings, setShowSettings] = useState(false);
  const [showContext, setShowContext] = useState(false);
  const [selectedSources, setSelectedSources] = useState<SourceDocument[]>([]);
  
  // Custom hooks
  const { messages, isLoading, handleSendMessage } = useMessages(config);
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Effects
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Event handlers
  const handleConfigChange = useCallback((newConfig: RAGConfig) => {
    setConfig(newConfig);
  }, []);

  const handleShowSources = useCallback((sources: SourceDocument[]) => {
    setSelectedSources(sources);
    setShowContext(true);
  }, []);

  const toggleSettings = useCallback(() => {
    setShowSettings((prev: boolean) => !prev);
  }, []);

  // Helper function to create message pairs for display
  const getMessagePairs = () => {
    const pairs: { userMessage?: any; basicMessage?: any; graphMessage?: any }[] = [];
    let currentPair: { userMessage?: any; basicMessage?: any; graphMessage?: any } = {};
    
    for (const message of messages) {
      if (message.sender === 'user') {
        // If we have a current pair, push it and start a new one
        if (Object.keys(currentPair).length > 0) {
          pairs.push(currentPair);
        }
        currentPair = { userMessage: message };
      } else if (message.ragType === 'basic') {
        currentPair.basicMessage = message;
      } else if (message.ragType === 'graph') {
        currentPair.graphMessage = message;
      }
    }
    
    // Push the last pair if it exists
    if (Object.keys(currentPair).length > 0) {
      pairs.push(currentPair);
    }
    
    return pairs;
  };

  // Helper function to filter messages by RAG type
  const getMessagesByType = (ragType: 'basic' | 'graph' | 'user') => {
    const pairs = getMessagePairs();
    const result: any[] = [];
    
    for (const pair of pairs) {
      if (pair.userMessage) {
        result.push(pair.userMessage);
      }
      
      if (ragType === 'basic' && pair.basicMessage) {
        result.push(pair.basicMessage);
      } else if (ragType === 'graph' && pair.graphMessage) {
        result.push(pair.graphMessage);
      }
    }
    
    return result;
  };

  // Components
  const WelcomeScreen = () => (
    <div className="text-center py-16">
      <div className="w-16 h-16 bg-primary-900 rounded-full flex items-center justify-center mx-auto mb-4">
        <BotIcon size={24} className="text-white" />
      </div>
      <h3 className="text-xl font-semibold text-text-primary mb-2">{WELCOME_MESSAGE.title}</h3>
      <p className="text-text-muted max-w-md mx-auto">{WELCOME_MESSAGE.description}</p>
    </div>
  );

  const ContextModal = () => (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50">
      <div className="bg-background-primary rounded-xl border border-border-light max-w-4xl w-full max-h-[80vh] overflow-hidden shadow-2xl">
        <div className="flex items-center justify-between p-6 border-b border-border-light">
          <h2 className="text-lg font-semibold text-text-primary">Source Context</h2>
          <button
            onClick={() => setShowContext(false)}
            className="p-1 rounded-lg hover:bg-background-secondary transition-colors"
            aria-label="Close context modal"
          >
            <CloseIcon size={20} className="text-text-muted" />
          </button>
        </div>
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {selectedSources.map((source, index) => (
            <div key={index} className="mb-6 last:mb-0">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-text-primary">Source {index + 1}</h3>
                {source.score && (
                  <span className="text-xs text-text-muted bg-background-secondary px-2 py-1 rounded">
                    Score: {source.score.toFixed(3)}
                  </span>
                )}
              </div>
              <p className="text-text-secondary text-sm leading-relaxed">{source.content}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const Header = () => (
    <div className="border-b border-border-light bg-background-primary/80 backdrop-blur-sm sticky top-0 z-10">
      <div className="px-4 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-primary-900 rounded-lg flex items-center justify-center">
            <BotIcon size={16} className="text-white" />
          </div>
          <h1 className="text-xl font-semibold text-text-primary">Dual RAG Comparison</h1>
        </div>
        <button
          onClick={toggleSettings}
          className="p-2 rounded-lg bg-background-secondary hover:bg-primary-100 transition-colors border border-border-light"
          aria-label={showSettings ? 'Close settings' : 'Open settings'}
        >
          <SettingsIcon size={20} className="text-text-muted" />
        </button>
      </div>
    </div>
  );

  const RAGPane = ({ title, ragType, borderColor }: { 
    title: string; 
    ragType: 'basic' | 'graph'; 
    borderColor: string;
  }) => {
    const paneMessages = getMessagesByType(ragType);
    
    return (
      <div className={`flex-1 flex flex-col border-r ${borderColor} bg-background-primary`}>
        <div className={`px-4 py-3 border-b border-gray-200 bg-gray-50 flex-shrink-0`}>
          <h2 className="text-lg font-semibold text-gray-800">{title}</h2>
        </div>
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-full px-4 py-8 min-h-full">
            {paneMessages.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-text-muted">Send a message to see {title} response</p>
              </div>
            ) : (
              <div className="space-y-6">
                {paneMessages.map((message) => (
                  <Message
                    key={message.id}
                    message={message}
                    onShowSources={handleShowSources}
                  />
                ))}
              </div>
            )}
            {isLoading && <LoadingMessage />}
          </div>
        </div>
      </div>
    );
  };

  const DualMessagesArea = () => (
    <div className="flex-1 flex overflow-hidden min-h-0">
      {messages.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <WelcomeScreen />
        </div>
      ) : (
        <>
          <RAGPane 
            title="Basic RAG" 
            ragType="basic" 
            borderColor="border-blue-200"
          />
          <RAGPane 
            title="Graph RAG" 
            ragType="graph" 
            borderColor="border-green-200"
          />
        </>
      )}
      <div ref={messagesEndRef} />
    </div>
  );

  return (
    <div className="h-screen bg-background-primary flex overflow-hidden">
      {/* Settings Sidebar */}
      {showSettings && (
        <SettingsSidebar
          config={config}
          onConfigChange={handleConfigChange}
          onClose={() => setShowSettings(false)}
        />
      )}

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full">
        <Header />
        <DualMessagesArea />
        <div className="flex-shrink-0 border-t border-border-light">
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </div>
      </div>

      {/* Context Modal */}
      {showContext && <ContextModal />}
    </div>
  );
};

export default App;
