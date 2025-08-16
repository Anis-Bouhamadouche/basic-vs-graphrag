/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'sans': ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Modern black, white, gray palette
        primary: {
          50: '#f8fafc',   // Very light gray
          100: '#f1f5f9',  // Light gray
          200: '#e2e8f0',  // Light gray
          300: '#cbd5e1',  // Medium light gray
          400: '#94a3b8',  // Medium gray
          500: '#64748b',  // Base gray
          600: '#475569',  // Dark gray
          700: '#334155',  // Darker gray
          800: '#1e293b',  // Very dark gray
          900: '#0f172a',  // Near black
          950: '#020617',  // Deep black
        },
        // Semantic color mappings
        background: {
          primary: '#ffffff',    // Pure white
          secondary: '#f8fafc',  // Very light gray
          tertiary: '#f1f5f9',   // Light gray
          dark: '#0f172a',       // Near black
        },
        text: {
          primary: '#0f172a',    // Near black for main text
          secondary: '#475569',  // Dark gray for secondary text
          muted: '#64748b',      // Medium gray for muted text
          inverse: '#ffffff',    // White for dark backgrounds
        },
        border: {
          light: '#e2e8f0',      // Light gray borders
          medium: '#cbd5e1',     // Medium gray borders
          dark: '#475569',       // Dark gray borders
        }
      },
    },
  },
  plugins: [],
}
