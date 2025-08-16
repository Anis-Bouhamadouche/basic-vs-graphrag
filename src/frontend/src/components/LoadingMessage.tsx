import React from 'react';
import { BotIcon, LoaderIcon } from './ui/icons';

const LoadingMessage: React.FC = () => {
  return (
    <div className="w-full mt-6">
      <div className="flex justify-start">
        <div className="max-w-2xl bg-gray-800 rounded-2xl px-6 py-4">
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-6 h-6 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                <BotIcon size={14} className="text-white" />
              </div>
            </div>
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <LoaderIcon size={16} className="text-blue-400" />
                <span className="text-gray-300 text-sm">Thinking...</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoadingMessage;
