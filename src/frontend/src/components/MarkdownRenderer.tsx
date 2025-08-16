import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
        // Customize heading styles
        h1: ({ children }) => (
          <h1 className="text-xl font-bold mb-3 text-text-primary border-b border-border-light pb-2">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-lg font-semibold mb-2 text-text-primary">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-base font-medium mb-2 text-text-primary">
            {children}
          </h3>
        ),
        
        // Customize paragraph styles
        p: ({ children }) => (
          <p className="mb-3 leading-relaxed text-text-secondary">
            {children}
          </p>
        ),
        
        // Customize list styles
        ul: ({ children }) => (
          <ul className="mb-3 space-y-1 list-disc list-inside text-text-secondary">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="mb-3 space-y-1 list-decimal list-inside text-text-secondary">
            {children}
          </ol>
        ),
        li: ({ children }) => (
          <li className="leading-relaxed">
            {children}
          </li>
        ),
        
        // Customize code styles
        code: ({ className, children, ...props }) => {
          const isInline = !className;
          return isInline ? (
            <code 
              className="bg-gray-100 text-gray-800 px-1 py-0.5 rounded text-sm font-mono"
              {...props}
            >
              {children}
            </code>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        pre: ({ children }) => (
          <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto mb-3 text-sm">
            {children}
          </pre>
        ),
        
        // Customize blockquote styles
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-blue-500 pl-4 mb-3 italic text-text-muted">
            {children}
          </blockquote>
        ),
        
        // Customize table styles
        table: ({ children }) => (
          <div className="overflow-x-auto mb-3">
            <table className="min-w-full border border-border-light rounded-lg">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-background-secondary">
            {children}
          </thead>
        ),
        th: ({ children }) => (
          <th className="px-4 py-2 text-left font-medium text-text-primary border-b border-border-light">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="px-4 py-2 text-text-secondary border-b border-border-light">
            {children}
          </td>
        ),
        
        // Customize link styles
        a: ({ href, children }) => (
          <a 
            href={href} 
            className="text-blue-600 hover:text-blue-800 underline"
            target="_blank" 
            rel="noopener noreferrer"
          >
            {children}
          </a>
        ),
        
        // Customize strong/bold styles
        strong: ({ children }) => (
          <strong className="font-semibold text-text-primary">
            {children}
          </strong>
        ),
        
        // Customize emphasis/italic styles
        em: ({ children }) => (
          <em className="italic text-text-secondary">
            {children}
          </em>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
