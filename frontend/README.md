# AegisPCAP React Dashboard

Modern React 18 + TypeScript + Vite dashboard for network security threat detection and analysis.

## Tech Stack

- **React 18.2+** - UI framework
- **TypeScript 5.0** - Type safety
- **Vite 5.0** - Build tool
- **React Router v6** - Client-side routing
- **Material-UI 5.14** - Component library
- **Zustand 4.4** - UI state management
- **React Query 5.28** - Server state management
- **Axios 1.6** - HTTP client
- **Zod 3.22** - Runtime validation
- **Recharts 2.10** - Data visualization

## Project Structure

```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/          # Page components
│   ├── stores/         # Zustand stores (UI & filter state)
│   ├── hooks/          # Custom React hooks
│   ├── types/          # TypeScript type definitions
│   ├── services/       # API client & services
│   ├── utils/          # Utility functions
│   ├── theme/          # Material-UI theme
│   ├── contexts/       # React contexts (real-time)
│   ├── App.tsx         # Root component
│   ├── main.tsx        # Entry point
│   └── index.css       # Global styles
├── public/             # Static assets
├── index.html          # HTML template
├── vite.config.ts      # Vite configuration
├── tsconfig.json       # TypeScript configuration
├── package.json        # Dependencies
└── .env.example        # Environment template
```

## Installation

```bash
# Clone or navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

### Running locally

```bash
npm run dev
```

Dashboard will be available at `http://localhost:3000`

Backend API should be running at `http://localhost:8000`

### Environment Variables

Create `.env` file:

```
VITE_API_BASE_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/ws
VITE_ENV=development
```

## Key Features

### State Management Pattern

**UI State (Zustand)**
- Sidebar state, theme preference, selected items
- Lightweight, minimal boilerplate

**Server State (React Query)**
- API cache, background refetch, optimistic updates
- Automatic deduplication, error handling

**Real-time State (Context)**
- WebSocket messages, live alerts/incidents
- Minimal re-render impact

### API Client

- Automatic retry with exponential backoff
- Request/response interceptors
- Bearer token authentication
- Zod validation support

### Components

- **Layout** - Main layout with AppBar and Sidebar
- **Sidebar** - Navigation menu with active states
- (Additional components in Phase 6.2b)

## Architecture Decisions

See [PHASE_6_2_STRATEGIC_THINKING.md](../PHASE_6_2_STRATEGIC_THINKING.md) for detailed architectural decisions:

1. Zustand + React Query for state management
2. TanStack Query v5 for server state
3. Container/Presentational component pattern
4. 3-tier real-time data freshness
5. Server-side pagination + infinite scroll
6. Performance optimization (code split, virtual scroll)
7. MUI + Emotion for styling
8. React Router v6 with lazy loading
9. TypeScript strict mode + Zod validation
10. Custom AppError + exponential backoff

## Performance Targets

- Initial load: < 2 seconds
- Filter apply: < 200ms
- Page navigation: < 300ms
- Table render (1000 rows): < 500ms
- WebSocket → UI update: < 100ms
- Bundle size: < 300KB (gzipped)

## Next Steps

- Implement core pages (Dashboard, Flows, Alerts, Incidents)
- Setup real-time WebSocket updates
- Add D3.js network visualization
- Integrate with backend API

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines
