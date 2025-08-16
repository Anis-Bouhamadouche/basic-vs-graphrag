# RAG Chat Frontend

A modern React frontend for chatting with the RAG (Retrieval-Augmented Generation) model.

## Features

- **Chat Interface**: Clean, modern chat UI with real-time messaging
- **Parameter Configuration**: Adjustable RAG parameters (top_k, temperature, etc.)
- **Context Viewing**: Optional display of source documents and context
- **Dark Theme**: Modern dark UI matching the ValueIO aesthetic
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **Axios** for API communication
- **Radix UI** for accessible components

## Development Setup

### Prerequisites

1. Install Node.js (version 20.11.0 or higher):
   ```bash
   # Using nvm (recommended)
   nvm install 20.11.0
   nvm use 20.11.0
   
   # Or download from https://nodejs.org/
   ```

2. Make sure the API is running at `http://localhost:8000`

### Local Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:5173](http://localhost:5173) in your browser

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Docker Setup

### Using Docker Compose (Recommended)

From the project root:

```bash
docker-compose up --build frontend
```

This will build and run the frontend container along with the API and databases.

### Manual Docker Build

```bash
# Build the image
docker build -t rag-frontend .

# Run the container
docker run -p 3000:3000 rag-frontend
```

## Environment Configuration

Create a `.env` file in the frontend directory:

```env
VITE_API_BASE_URL=http://localhost:8000
```

For production, update the API URL accordingly.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Project Structure

```
src/
├── components/          # Reusable UI components
│   └── Icons.tsx       # SVG icon components
├── lib/                # Utilities and API client
│   └── api.ts          # API communication
├── types/              # TypeScript type definitions
│   └── index.ts        # Type interfaces
├── App.tsx             # Main application component
├── main.tsx            # React entry point
└── index.css           # Global styles
```

## Usage

1. **Start a Chat**: Type your message in the input field and press Enter or click Send
2. **Configure Parameters**: Click the settings icon to adjust RAG parameters
3. **View Context**: Click "Show Context" to see source documents used for responses
4. **Theme**: The app uses a dark theme by default

## API Integration

The frontend communicates with the RAG API at `/api/chat/basic` endpoint. Make sure the API is running and accessible.

## Contributing

1. Follow the existing code style
2. Use TypeScript for type safety
3. Write responsive components
4. Test on multiple screen sizes
