/**
 * Reusable Button component with variants and sizes
 * Follows compound component pattern with proper accessibility
 */

import React from 'react';
import { cn } from '../../lib/utils';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  loading?: boolean;
  children: React.ReactNode;
}

const buttonVariants = {
  variant: {
    default: 'bg-primary-900 text-white hover:bg-primary-800 focus:ring-primary-700',
    destructive: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500',
    outline: 'border border-border-medium bg-transparent text-text-primary hover:bg-background-secondary focus:ring-primary-500',
    secondary: 'bg-primary-600 text-white hover:bg-primary-700 focus:ring-primary-500',
    ghost: 'bg-transparent text-text-primary hover:bg-background-secondary focus:ring-primary-500',
    link: 'bg-transparent text-primary-700 underline-offset-4 hover:underline focus:ring-primary-500',
  },
  size: {
    default: 'h-10 px-4 py-2 text-sm',
    sm: 'h-8 px-3 py-1 text-xs',
    lg: 'h-12 px-6 py-3 text-base',
    icon: 'h-10 w-10 p-0',
  },
};

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ 
    className, 
    variant = 'default', 
    size = 'default', 
    loading = false, 
    disabled,
    children, 
    ...props 
  }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          // Base styles
          'inline-flex items-center justify-center whitespace-nowrap rounded-lg font-medium transition-colors',
          'focus:outline-none focus:ring-2 focus:ring-offset-2',
          'disabled:pointer-events-none disabled:opacity-50',
          // Variant styles
          buttonVariants.variant[variant],
          // Size styles
          buttonVariants.size[size],
          className
        )}
        disabled={disabled || loading}
        {...props}
      >
        {loading && (
          <svg
            className={cn(
              'animate-spin',
              size === 'sm' ? 'h-3 w-3' : size === 'lg' ? 'h-5 w-5' : 'h-4 w-4',
              children ? 'mr-2' : ''
            )}
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="m12 2 2.09 6.26L22 9l-5.91 5.91L22 22l-8.09-1.74L12 22l-1.91-1.74L2 22l5.91-7.09L2 9l7.91-0.74L12 2z"
            />
          </svg>
        )}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

export { Button };