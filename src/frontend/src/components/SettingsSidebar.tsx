import React from 'react';
import { RAGConfig } from '../types';
import { CloseIcon } from './ui/icons';

interface SettingsSidebarProps {
  config: RAGConfig;
  onConfigChange: (config: RAGConfig) => void;
  onClose: () => void;
}

const SettingsSidebar: React.FC<SettingsSidebarProps> = ({
  config,
  onConfigChange,
  onClose,
}) => {
  const updateConfig = (updates: Partial<RAGConfig>) => {
    onConfigChange({ ...config, ...updates });
  };

  return (
    <div className="w-80 border-r border-gray-700/50 bg-gray-900/95 backdrop-blur-sm p-6 overflow-y-auto flex-shrink-0">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white">Settings</h2>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-gray-800 transition-colors"
          aria-label="Close settings"
        >
          <CloseIcon size={16} className="text-gray-400" />
        </button>
      </div>
      
      <div className="space-y-6">
        {/* Top K Results */}
        <div>
          <label htmlFor="top-k" className="block text-sm font-medium text-gray-300 mb-2">
            Top K Results
          </label>
          <input
            id="top-k"
            type="range"
            min="1"
            max="20"
            value={config.top_k}
            onChange={(e) => updateConfig({ top_k: parseInt(e.target.value) })}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>1</span>
            <span className="text-blue-400 font-medium">{config.top_k}</span>
            <span>20</span>
          </div>
        </div>

        {/* Temperature */}
        <div>
          <label htmlFor="temperature" className="block text-sm font-medium text-gray-300 mb-2">
            Temperature
          </label>
          <input
            id="temperature"
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={config.temperature}
            onChange={(e) => updateConfig({ temperature: parseFloat(e.target.value) })}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>0.0</span>
            <span className="text-blue-400 font-medium">{config.temperature}</span>
            <span>2.0</span>
          </div>
        </div>

        {/* Max Tokens */}
        <div>
          <label htmlFor="max-tokens" className="block text-sm font-medium text-gray-300 mb-2">
            Max Tokens
          </label>
          <input
            id="max-tokens"
            type="range"
            min="50"
            max="4000"
            step="50"
            value={config.max_tokens}
            onChange={(e) => updateConfig({ max_tokens: parseInt(e.target.value) })}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>50</span>
            <span className="text-blue-400 font-medium">{config.max_tokens}</span>
            <span>4000</span>
          </div>
        </div>

        {/* Collection Name */}
        <div>
          <label htmlFor="collection-name" className="block text-sm font-medium text-gray-300 mb-2">
            Collection Name
          </label>
          <input
            id="collection-name"
            type="text"
            value={config.collection_name}
            onChange={(e) => updateConfig({ collection_name: e.target.value })}
            className="w-full p-3 rounded-lg bg-gray-800 border border-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Collection name"
          />
        </div>

        {/* Chat Model */}
        <div>
          <label htmlFor="chat-model" className="block text-sm font-medium text-gray-300 mb-2">
            Chat Model
          </label>
          <select
            id="chat-model"
            value={config.chat_model_name}
            onChange={(e) => updateConfig({ chat_model_name: e.target.value })}
            className="w-full p-3 rounded-lg bg-gray-800 border border-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="gpt-4o">GPT-4o</option>
            <option value="gpt-4o-mini">GPT-4o Mini</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          </select>
        </div>

        {/* Embedding Model */}
        <div>
          <label htmlFor="embedding-model" className="block text-sm font-medium text-gray-300 mb-2">
            Embedding Model
          </label>
          <select
            id="embedding-model"
            value={config.embedding_model}
            onChange={(e) => updateConfig({ embedding_model: e.target.value })}
            className="w-full p-3 rounded-lg bg-gray-800 border border-gray-600 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="text-embedding-3-large">text-embedding-3-large</option>
            <option value="text-embedding-3-small">text-embedding-3-small</option>
            <option value="text-embedding-ada-002">text-embedding-ada-002</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default SettingsSidebar;
