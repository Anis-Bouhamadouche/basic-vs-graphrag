/**
 * Centralized icon components with consistent styling
 * All icons use Lucide React style SVG paths for consistency
 */

import React from 'react';
import { cn } from '../../lib/utils';

export interface IconProps {
  size?: number;
  className?: string;
}

const createIcon = (paths: string[], viewBox = '0 0 24 24') => {
  return React.forwardRef<SVGSVGElement, IconProps>(
    ({ size = 20, className, ...props }, ref) => (
      <svg
        ref={ref}
        width={size}
        height={size}
        viewBox={viewBox}
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className={cn('shrink-0', className)}
        {...props}
      >
        {paths.map((path, index) => (
          <path key={index} d={path} />
        ))}
      </svg>
    )
  );
};

// Chat and messaging icons
export const SendIcon = createIcon(['m22 2-7 20-4-9-9-4Z', 'M22 2 11 13']);

export const BotIcon = createIcon([
  'M12 8V4H8',
  'm8 4-3 3a4 4 0 0 0 0 6l3 3v4h8v-4l3-3a4 4 0 0 0 0-6l-3-3v4Z',
  'M2 14h2',
  'M20 14h2',
  'm15 13-1 1',
  'm9 13 1 1'
]);

export const UserIcon = createIcon([
  'M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2',
  'circle cx="12" cy="7" r="4"'
]);

// UI Control icons
export const SettingsIcon = createIcon([
  'M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.39a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z',
  'circle cx="12" cy="12" r="3"'
]);

export const CloseIcon = createIcon(['m18 6-12 12', 'm6 6 12 12']);

export const LoaderIcon = createIcon([
  'M21 12a9 9 0 1 1-6.219-8.56'
]);

// Document and content icons
export const DocumentIcon = createIcon([
  'M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z',
  'M14,2 L14,8 L20,8'
]);

export const FileIcon = createIcon([
  'M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z',
  'M14,2 L14,8 L20,8',
  'M16 13H8',
  'M16 17H8',
  'M10 9H8'
]);

export const SearchIcon = createIcon([
  'circle cx="11" cy="11" r="8',
  'M21 21l-4.35-4.35'
]);

// Status and feedback icons
export const CheckIcon = createIcon(['M20 6 9 17l-5-5']);

export const AlertIcon = createIcon([
  'M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z',
  'M12 9v4',
  'M12 17h.01'
]);

export const InfoIcon = createIcon([
  'circle cx="12" cy="12" r="10"',
  'M12 16v-4',
  'M12 8h.01'
]);

// Navigation icons
export const ChevronDownIcon = createIcon(['m6 9 6 6 6-6']);

export const ChevronUpIcon = createIcon(['m18 15-6-6-6 6']);

export const ChevronLeftIcon = createIcon(['m15 18-6-6 6-6']);

export const ChevronRightIcon = createIcon(['m9 18 6-6-6-6']);

// Action icons
export const CopyIcon = createIcon([
  'rect width="14" height="14" x="8" y="8" rx="2" ry="2"',
  'path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"'
]);

export const DownloadIcon = createIcon([
  'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4',
  'M7 10l5 5 5-5',
  'M12 15V3'
]);

export const ExternalLinkIcon = createIcon([
  'M15 3h6v6',
  'M10 14 21 3',
  'M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6'
]);

export const RefreshIcon = createIcon([
  'M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8',
  'M21 3v5h-5',
  'M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16',
  'M3 21v-5h5'
]);

// Export default icons object for easy importing
export const Icons = {
  Send: SendIcon,
  Bot: BotIcon,
  User: UserIcon,
  Settings: SettingsIcon,
  Close: CloseIcon,
  Loader: LoaderIcon,
  Document: DocumentIcon,
  File: FileIcon,
  Search: SearchIcon,
  Check: CheckIcon,
  Alert: AlertIcon,
  Info: InfoIcon,
  ChevronDown: ChevronDownIcon,
  ChevronUp: ChevronUpIcon,
  ChevronLeft: ChevronLeftIcon,
  ChevronRight: ChevronRightIcon,
  Copy: CopyIcon,
  Download: DownloadIcon,
  ExternalLink: ExternalLinkIcon,
  Refresh: RefreshIcon,
} as const;